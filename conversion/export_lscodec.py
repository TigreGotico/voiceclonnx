"""Export LSCodec to ONNX for voiceclonnx (external-checkout pattern).

LSCodec (Guo et al., Interspeech 2025, https://github.com/X-LANCE/LSCodec-Inference,
MIT) is a speaker-decoupled discrete speech codec.  Its code is MIT but **not
vendored** here; this script clones it at export time (like the other
external-checkout engines) and exports three ONNX models:

  1. ``lscodec_encoder.onnx`` — raw 16 kHz audio -> 64-d means (50 Hz).
  2. ``wavlm_l6.onnx``        — reference 16 kHz audio -> WavLM-Large layer-6
     features, exported at a FIXED 4 s window (the WavLM relative-position
     attention has a dynamic Gather the legacy exporter rejects with dynamic
     length; a static window sidesteps it — the prompt is a speaker reference,
     so a fixed window is fine).
  3. ``lscodec_vocoder.onnx`` — vqvec (1,L,64) + prompt (1,Lp,1024) -> 24 kHz wav
     (the CTXVEC2WAV frontend + HiFiGAN backend).

VQ (nearest-neighbour to ``codebook.npy``) runs in numpy in the adapter.

Prereqs (a Python 3.10 env matching LSCodec's stack):
  torch==1.13.1 torchaudio==0.13.1 fairseq==0.12.2 librosa==0.8.1 scipy<1.13
  kaldiio einops pyyaml soundfile onnx onnxruntime huggingface_hub
Checkpoints (``bash download_ckpt.sh 50hz`` in the clone) + WavLM-Large.pt.

Usage::

    python -m conversion.export_lscodec --lscodec-dir /path/to/LSCodec-Inference \
        --pretrained-dir /path/to/LSCodec-Inference/pretrained --output-dir /tmp/lscodec-out
    # add --push to upload to TigreGotico/voiceclonnx-lscodec

Parity (fixed-length where noted) measured at export time:
  encoder max_abs ~6e-3 | wavlm (4 s) ~5e-4 | vocoder ~6e-6
End-to-end ONNX vs torch speaker-embedding cosine ~0.97 (see demo/SPEAKER_SIMILARITY.md).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import numpy as np

HF_REPO = "TigreGotico/voiceclonnx-lscodec"
UPSTREAM = "https://github.com/X-LANCE/LSCodec-Inference"
WAVLM_WINDOW = 16000 * 4  # fixed 4 s prompt window


def _import_torch():
    import torch
    return torch


def export_encoder(load_model, pretrained_dir: str, out: str):
    import torch
    import yaml

    cfg = yaml.load(open(f"{pretrained_dir}/encoder_config.yml"), Loader=yaml.Loader)
    cfg["pretrain_codebook"] = f"{pretrained_dir}/codebook.npy"
    model = load_model(cfg, f"{pretrained_dir}/lscodec_encoder.pt").eval()

    class Enc(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, audio):  # (1,1,L)
            feats = self.m.frontend.forward(audio, None, self.m.length_mismatch_tolerance)
            return feats.transpose(1, 2)[..., :64]  # means (1,L,64)

    w = Enc(model).eval()
    dummy = torch.randn(1, 1, 16000 * 3)
    torch.onnx.export(
        w, (dummy,), out, input_names=["audio"], output_names=["means"],
        dynamic_axes={"audio": {2: "samples"}, "means": {1: "frames"}}, opset_version=14,
    )
    return _parity(w, out, {"audio": torch.randn(1, 1, 16000 * 4)}, "means")


def export_wavlm(WavLM, WavLMConfig, pretrained_dir: str, out: str):
    import torch

    ckpt = torch.load(f"{pretrained_dir}/WavLM-Large.pt", map_location="cpu")
    cfg = WavLMConfig(ckpt["cfg"])
    m = WavLM(cfg)
    m.load_state_dict(ckpt["model"])
    m = m.eval()

    class WL(torch.nn.Module):
        def __init__(self, m, norm):
            super().__init__()
            self.m = m
            self.norm = norm

        def forward(self, wav):  # (1, WAVLM_WINDOW)
            if self.norm:
                wav = torch.nn.functional.layer_norm(wav, wav.shape)
            return self.m.extract_features(wav, output_layer=6)[0]

    w = WL(m, cfg.normalize).eval()
    dummy = torch.randn(1, WAVLM_WINDOW)
    torch.onnx.export(  # static length: avoids WavLM rel-pos dynamic Gather
        w, (dummy,), out, input_names=["wav"], output_names=["feats"],
        dynamic_axes=None, opset_version=14,
    )
    return _parity(w, out, {"wav": torch.randn(1, WAVLM_WINDOW)}, "feats")


def export_vocoder(load_vocoder, pretrained_dir: str, out: str):
    import torch
    import yaml

    cfg = yaml.load(open(f"{pretrained_dir}/vocoder_config.yml"), Loader=yaml.Loader)
    cfg["vq_codebook"] = f"{pretrained_dir}/codebook.npy"
    voc = load_vocoder(cfg, f"{pretrained_dir}/lscodec_vocoder.pt").eval()

    class Voc(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, vqvec, prompt):
            return self.m.inference(vqvec, prompt)[-1].view(1, -1)

    w = Voc(voc).eval()
    vq, pr = torch.randn(1, 150, 64), torch.randn(1, 200, 1024)
    torch.onnx.export(
        w, (vq, pr), out, input_names=["vqvec", "prompt"], output_names=["wav"],
        dynamic_axes={"vqvec": {1: "frames"}, "prompt": {1: "pframes"}, "wav": {1: "samples"}},
        opset_version=14,
    )
    return _parity(w, out, {"vqvec": torch.randn(1, 130, 64), "prompt": torch.randn(1, 180, 1024)}, "wav")


def _parity(torch_mod, onnx_path, inputs, out_name):
    import onnxruntime as ort
    import torch

    with torch.no_grad():
        ref = torch_mod(*inputs.values()).numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    got = sess.run(None, {k: v.numpy() for k, v in inputs.items()})[0]
    n = min(ref.shape[1], got.shape[1]) if ref.ndim > 1 else len(ref)
    err = float(np.max(np.abs(ref[:, :n] - got[:, :n]))) if ref.ndim > 1 else float(np.max(np.abs(ref - got)))
    print(f"  parity[{out_name}] max_abs = {err:.2e}")
    return err


def quantize(out_dir: str):
    from onnxruntime.quantization import QuantType, quantize_dynamic

    for m in ["lscodec_encoder", "wavlm_l6", "lscodec_vocoder"]:
        quantize_dynamic(f"{out_dir}/{m}.onnx", f"{out_dir}/{m}_q8.onnx", weight_type=QuantType.QInt8)
        print(f"  quantized {m}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lscodec-dir", required=True, help="path to a clone of X-LANCE/LSCodec-Inference")
    ap.add_argument("--pretrained-dir", required=True, help="pretrained/ dir (encoder/vocoder/codebook/WavLM-Large.pt)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--push", action="store_true", help="upload to " + HF_REPO)
    args = ap.parse_args(argv)

    sys.path.insert(0, args.lscodec_dir)
    from lscodec.utils import load_model, load_vocoder  # noqa: E402
    from lscodec.ssl_models.WavLM import WavLM, WavLMConfig  # noqa: E402

    out = args.output_dir
    os.makedirs(out, exist_ok=True)
    print("[lscodec] exporting encoder…")
    export_encoder(load_model, args.pretrained_dir, f"{out}/lscodec_encoder.onnx")
    print("[lscodec] exporting WavLM (fixed 4 s window)…")
    export_wavlm(WavLM, WavLMConfig, args.pretrained_dir, f"{out}/wavlm_l6.onnx")
    print("[lscodec] exporting vocoder…")
    export_vocoder(load_vocoder, args.pretrained_dir, f"{out}/lscodec_vocoder.onnx")
    shutil.copy(f"{args.pretrained_dir}/codebook.npy", f"{out}/codebook.npy")
    print("[lscodec] quantizing…")
    quantize(out)

    cfg = {
        "engine": "lscodec", "content_sr": 16000, "prompt_sr": 16000, "output_sr": 24000,
        "wavlm_prompt_window_samples": WAVLM_WINDOW, "vq_codebook_size": 300, "vq_dim": 64,
        "upstream": UPSTREAM, "license": "MIT (LSCodec code; WavLM-Large Microsoft MIT)",
    }
    json.dump(cfg, open(f"{out}/config.json", "w"), indent=2)

    if args.push:
        from huggingface_hub import HfApi, create_repo

        create_repo(HF_REPO, repo_type="model", exist_ok=True)
        HfApi().upload_folder(folder_path=out, repo_id=HF_REPO, repo_type="model",
                              allow_patterns=["*.onnx", "*.npy", "*.json", "*.md"])
        print("[lscodec] pushed to", HF_REPO)


if __name__ == "__main__":
    main()
