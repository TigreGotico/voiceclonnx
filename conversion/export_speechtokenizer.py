"""Export SpeechTokenizer (Zhang et al., ACL 2024) encoder and decoder to ONNX.

SpeechTokenizer is a hierarchical RVQ speech codec with 8 quantizers at 50 Hz
(320-sample hop at 16 kHz).  Quantizer 1 is semantically distilled via HuBERT
and captures linguistic content; quantizers 2-8 carry speaker timbre.

Voice-conversion ONNX components
---------------------------------
- ``encoder.onnx``  : waveform (1, 1, N) float32  → codes (8, 1, T) int64
- ``decoder.onnx``  : codes (8, 1, T) int64        → waveform (1, 1, N) float32

The token swap (source RVQ-1 + reference RVQ-2..8 → decode) is pure numpy
inside the voiceclonnx adapter and is not exported.

Upstream
--------
- https://github.com/ZhangXInFD/SpeechTokenizer  (Apache-2.0)
- https://huggingface.co/fnlp/SpeechTokenizer

Usage::

    python -m conversion.export_speechtokenizer --output-dir /tmp/st-out
    python -m conversion.export_speechtokenizer --output-dir /tmp/st-out --no-push
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ST_HF_REPO = "fnlp/SpeechTokenizer"
ST_UPSTREAM_URL = "https://github.com/ZhangXInFD/SpeechTokenizer"
ST_UPSTREAM_REF = "main"
ST_SR = 16000
N_QUANTIZERS = 8

APACHE2_LICENSE = """\
Apache License
Version 2.0, January 2004

Copyright 2023 ZhangXin.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


# ---------------------------------------------------------------------------
# Torch wrapper modules
# ---------------------------------------------------------------------------


def _build_encoder_wrapper(model):
    """Wrap SpeechTokenizer encoder+quantizer.

    Input : audio (1, 1, N) float32
    Output: codes (Q, 1, T) int64
    """
    import torch
    import torch.nn as nn

    class STEncoder(nn.Module):
        def __init__(self, st_model):
            super().__init__()
            self.model = st_model

        def forward(self, audio):
            # audio: (1, 1, N)
            # SpeechTokenizer.encode returns: (Q, B, T) int64
            codes = self.model.encode(audio)
            return codes  # (Q, 1, T)

    return STEncoder(model).eval()


def _build_decoder_wrapper(model):
    """Wrap SpeechTokenizer decoder.

    Input : codes (Q, 1, T) int64
    Output: waveform (1, 1, N) float32
    """
    import torch
    import torch.nn as nn

    class STDecoder(nn.Module):
        def __init__(self, st_model):
            super().__init__()
            self.model = st_model

        def forward(self, codes):
            # codes: (Q, 1, T) int64
            wav = self.model.decode(codes)  # (1, 1, N) float32
            return wav

    return STDecoder(model).eval()


# ---------------------------------------------------------------------------
# Parity check helpers
# ---------------------------------------------------------------------------


def _parity_encoder(torch_model, ort_enc_path: Path, dummy_audio):
    """Compare torch encoder vs ORT encoder on the same audio."""
    import torch
    import onnxruntime as ort
    import numpy as np

    with torch.no_grad():
        torch_codes = torch_model.encode(dummy_audio)  # (Q, 1, T)
    torch_np = torch_codes.cpu().numpy()

    sess = ort.InferenceSession(str(ort_enc_path), providers=["CPUExecutionProvider"])
    ort_codes = sess.run(None, {"audio": dummy_audio.cpu().numpy()})[0]

    max_err = float(np.abs(torch_np.astype(np.float32) - ort_codes.astype(np.float32)).max())
    print(f"[parity] encoder  max_abs_err={max_err:.2e}  "
          f"(integer equality: {np.array_equal(torch_np, ort_codes)})")
    return torch_np, ort_codes


def _parity_decoder(torch_model, ort_dec_path: Path, codes_np):
    """Compare torch decoder vs ORT decoder on the same codes."""
    import torch
    import onnxruntime as ort
    import numpy as np

    codes_t = torch.from_numpy(codes_np).long()
    with torch.no_grad():
        torch_wav = torch_model.decode(codes_t)  # (1, 1, N)
    torch_np = torch_wav.cpu().numpy()

    sess = ort.InferenceSession(str(ort_dec_path), providers=["CPUExecutionProvider"])
    ort_wav = sess.run(None, {"codes": codes_np})[0]

    max_err = float(np.abs(torch_np - ort_wav).max())
    mean_err = float(np.abs(torch_np - ort_wav).mean())
    ok = max_err <= 1e-3 and mean_err <= 1e-4
    print(f"[parity] decoder  max_abs={max_err:.2e}  mean_abs={mean_err:.2e}  "
          f"{'PASS' if ok else 'FAIL'}")
    return torch_np, ort_wav, max_err, mean_err


# ---------------------------------------------------------------------------
# Main export routine
# ---------------------------------------------------------------------------


def export_speechtokenizer(output_dir: str, no_push: bool = False) -> None:
    import torch
    import numpy as np
    from huggingface_hub import snapshot_download

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from conversion.export_base import (
        OutputLayout,
        export_model,
        write_manifest,
        write_provenance,
    )
    from conversion.quantize import quantize_model

    # ------------------------------------------------------------------
    # 1. Download checkpoint
    # ------------------------------------------------------------------
    print(f"[export] downloading SpeechTokenizer from {ST_HF_REPO} ...")
    ckpt_dir = snapshot_download(ST_HF_REPO)
    ckpt_path = Path(ckpt_dir) / "speechtokenizer.pt"
    cfg_path = Path(ckpt_dir) / "config.json"

    if not ckpt_path.exists():
        # Some HF mirror layouts put the file one level down
        candidates = list(Path(ckpt_dir).rglob("speechtokenizer.pt"))
        if not candidates:
            raise FileNotFoundError(
                f"speechtokenizer.pt not found in {ckpt_dir}. "
                "Check the HF repo layout: fnlp/SpeechTokenizer"
            )
        ckpt_path = candidates[0]
    if not cfg_path.exists():
        candidates = list(Path(ckpt_dir).rglob("config.json"))
        if candidates:
            cfg_path = candidates[0]

    # ------------------------------------------------------------------
    # 2. Load model
    # ------------------------------------------------------------------
    print("[export] loading SpeechTokenizer model ...")
    try:
        from speechtokenizer import SpeechTokenizer
        model = SpeechTokenizer.load_from_checkpoint(str(cfg_path), str(ckpt_path))
    except ImportError:
        # Fallback: install and import
        import subprocess
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "speechtokenizer", "--quiet"],
            check=True,
        )
        from speechtokenizer import SpeechTokenizer
        model = SpeechTokenizer.load_from_checkpoint(str(cfg_path), str(ckpt_path))

    model.eval()

    # ------------------------------------------------------------------
    # 3. Prepare output layout
    # ------------------------------------------------------------------
    layout = OutputLayout.for_engine("speechtokenizer", base_dir=output_dir)
    layout.makedirs()

    # ------------------------------------------------------------------
    # 4. Build dummy inputs
    # ------------------------------------------------------------------
    # 2 s of audio at 16 kHz
    dummy_audio = torch.zeros(1, 1, 32000)

    # ------------------------------------------------------------------
    # 5. Export encoder (waveform → RVQ codes)
    # ------------------------------------------------------------------
    print("[export] exporting encoder ...")
    enc_wrapper = _build_encoder_wrapper(model)
    enc_path = export_model(
        model=enc_wrapper,
        dummy_inputs=(dummy_audio,),
        output_path=layout.component_path("encoder.onnx"),
        input_names=["audio"],
        output_names=["codes"],
        dynamic_axes={
            "audio": {2: "num_samples"},
            "codes": {2: "num_frames"},
        },
        opset_version=14,
    )
    print(f"[export] encoder saved: {enc_path} ({enc_path.stat().st_size // 1024 // 1024} MB)")

    # ------------------------------------------------------------------
    # 6. Parity check — encoder
    # ------------------------------------------------------------------
    with torch.no_grad():
        torch_codes = model.encode(dummy_audio)  # (Q, 1, T)
    enc_np = torch_codes.cpu().numpy()

    import onnxruntime as ort
    sess_enc = ort.InferenceSession(str(enc_path), providers=["CPUExecutionProvider"])
    ort_codes = sess_enc.run(None, {"audio": dummy_audio.numpy()})[0]

    max_enc = float(abs(enc_np.astype(float) - ort_codes.astype(float)).max())
    exact_match = bool((enc_np == ort_codes).all())
    print(f"[parity] encoder  max_diff={max_enc:.2e}  exact_int_match={exact_match}")
    if not exact_match and max_enc > 0.5:
        print("[parity] WARNING: encoder integer mismatch exceeds 0.5 — check quantizer export")

    # ------------------------------------------------------------------
    # 7. Export decoder (RVQ codes → waveform)
    # ------------------------------------------------------------------
    print("[export] exporting decoder ...")
    dec_wrapper = _build_decoder_wrapper(model)

    # Dummy codes for tracing: (Q, 1, T)
    T_dummy = torch_codes.shape[2]
    dummy_codes = torch_codes.clone()

    dec_path = export_model(
        model=dec_wrapper,
        dummy_inputs=(dummy_codes,),
        output_path=layout.component_path("decoder.onnx"),
        input_names=["codes"],
        output_names=["waveform"],
        dynamic_axes={
            "codes": {2: "num_frames"},
            "waveform": {2: "num_samples"},
        },
        opset_version=14,
    )
    print(f"[export] decoder saved: {dec_path} ({dec_path.stat().st_size // 1024 // 1024} MB)")

    # ------------------------------------------------------------------
    # 8. Parity check — decoder
    # ------------------------------------------------------------------
    with torch.no_grad():
        torch_wav = model.decode(dummy_codes)  # (1, 1, N)
    torch_wav_np = torch_wav.cpu().numpy()

    sess_dec = ort.InferenceSession(str(dec_path), providers=["CPUExecutionProvider"])
    ort_wav = sess_dec.run(None, {"codes": enc_np})[0]

    max_dec = float(abs(torch_wav_np - ort_wav).max())
    mean_dec = float(abs(torch_wav_np - ort_wav).mean())
    dec_ok = max_dec <= 1e-3 and mean_dec <= 1e-4
    print(f"[parity] decoder  max_abs={max_dec:.2e}  mean_abs={mean_dec:.2e}  "
          f"{'PASS' if dec_ok else 'FAIL (check if acceptable)'}")

    # ------------------------------------------------------------------
    # 9. Quantize
    # ------------------------------------------------------------------
    print("[export] quantizing encoder ...")
    enc_q8_report = quantize_model(str(enc_path))
    print(enc_q8_report.summary())

    print("[export] quantizing decoder ...")
    dec_q8_report = quantize_model(str(dec_path))
    print(dec_q8_report.summary())

    # ------------------------------------------------------------------
    # 10. Write manifest and provenance
    # ------------------------------------------------------------------
    write_manifest(
        layout=layout,
        components={
            "encoder": "encoder.onnx",
            "encoder_q8": "encoder_q8.onnx",
            "decoder": "decoder.onnx",
            "decoder_q8": "decoder_q8.onnx",
        },
        sample_rates={"input": ST_SR, "output": ST_SR},
        metadata={
            "opset": 14,
            "n_quantizers": N_QUANTIZERS,
            "sample_rate": ST_SR,
            "hop_length": 320,
            "vc_recipe": (
                "source RVQ-1 tokens (content) + reference RVQ-2..8 tokens (timbre) → decode"
            ),
            "parity": {
                "encoder_exact_int_match": exact_match,
                "encoder_max_diff": max_enc,
                "decoder_max_abs": max_dec,
                "decoder_mean_abs": mean_dec,
                "decoder_pass": dec_ok,
            },
        },
        distributable=True,
    )

    write_provenance(
        engine_dir=layout.engine_dir,
        upstream_repo_url=ST_UPSTREAM_URL,
        upstream_ref=ST_UPSTREAM_REF,
        license_text=APACHE2_LICENSE,
        extra={
            "hf_model_id": ST_HF_REPO,
            "vc_recipe": "source_rvq1 + reference_rvq2_8",
        },
    )

    print(f"[export] artifacts written to {layout.engine_dir}")

    # ------------------------------------------------------------------
    # 11. Push to HF
    # ------------------------------------------------------------------
    if not no_push:
        from conversion.push_models import push_engine

        push_engine(
            engine_dir=str(layout.engine_dir),
            engine_name="speechtokenizer",
            hf_repo_id="TigreGotico/voiceclonnx-speechtokenizer",
            commit_message="export: add speechtokenizer ONNX artifacts",
        )
        print("[export] pushed to TigreGotico/voiceclonnx-speechtokenizer")
    else:
        print("[export] --no-push: skipping HF upload")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Export SpeechTokenizer to ONNX")
    ap.add_argument("--output-dir", required=True, help="Staging output directory")
    ap.add_argument("--no-push", action="store_true", help="Skip HF upload")
    args = ap.parse_args()

    export_speechtokenizer(output_dir=args.output_dir, no_push=args.no_push)


if __name__ == "__main__":
    main()
