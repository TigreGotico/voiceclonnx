"""Export-parity gate for freevc (the VITS decoder — the likely failure point).

Content `c` (WavLM) and speaker `g` (d-vector) are taken from the ONNX adapter's
own sessions, so the ONLY difference is torch SynthesizerTrn.infer vs the ONNX
decoder. infer() is deterministic (uses the prior mean), so a faithful export
should match closely on BOTH waveform and speaker embedding.
"""
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from voiceclonnx.engines.freevc import FreeVCAdapter, _load_wav, _FREEVC_SR  # noqa: E402
from conversion.export_freevc import (  # noqa: E402
    _build_architecture, _build_synthesizer_trn, _download, FREEVC_CHECKPOINT_URL,
)
from speakeronnx import SpeakerEmbedder, cosine  # noqa: E402

DEMO = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(DEMO, "source.wav")
REFS = {"aria": os.path.join(DEMO, "reference_aria.wav"),
        "sonia": os.path.join(DEMO, "reference_sonia.wav")}


def corr(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ad = FreeVCAdapter()
    ad._ensure_models()

    # build torch net_g and load the original FreeVC checkpoint
    commons_ns, modules_ns = _build_architecture()
    SynthesizerTrn = _build_synthesizer_trn(commons_ns, modules_ns)
    net_g = SynthesizerTrn()
    ckpt = torch.load(str(_download(FREEVC_CHECKPOINT_URL,
                     Path(DEMO) / "_freevc.pth")), map_location="cpu",
                     weights_only=False)
    net_g.load_state_dict(ckpt.get("model", ckpt), strict=False)
    net_g.eval()

    emb = SpeakerEmbedder(model="wespeaker-resnet34")
    src_e = emb.embed(SRC)

    # ~6 s source chunk (single pass, matches adapter short-audio path)
    src = _load_wav(SRC, _FREEVC_SR)[: 6 * _FREEVC_SR]
    c = ad._extract_content(src)                 # (T, 1024)

    print(f"\n{'target':<8}{'wav-corr':<10}{'emb torch↔onnx':<16}"
          f"{'torch→tgt':<11}{'onnx→tgt':<11}{'torch→src':<11}{'onnx→src':<10}")
    print("-" * 76)
    for tgt, refpath in REFS.items():
        g = ad._extract_speaker(_load_wav(refpath, _FREEVC_SR))   # (256,)
        onnx_wav = ad._decode(c, g)                              # (S,)
        with torch.no_grad():
            ct = torch.from_numpy(c.T[None].astype(np.float32))  # (1,1024,T)
            gt = torch.from_numpy(g[None].astype(np.float32))    # (1,256)
            torch_wav = net_g.infer(ct, gt)[0, 0].numpy().astype(np.float32)

        ot = os.path.join(DEMO, f"_onnx_freevc__{tgt}.wav")
        tt = os.path.join(DEMO, f"_torch_freevc__{tgt}.wav")
        sf.write(ot, onnx_wav, _FREEVC_SR)
        sf.write(tt, torch_wav, _FREEVC_SR)

        te, oe, re = emb.embed(tt), emb.embed(ot), emb.embed(refpath)
        print(f"{tgt:<8}{corr(torch_wav, onnx_wav):<10.3f}{cosine(te, oe):<16.3f}"
              f"{cosine(te, re):<11.3f}{cosine(oe, re):<11.3f}"
              f"{cosine(te, src_e):<11.3f}{cosine(oe, src_e):<10.3f}")

    print("\ntorch↔onnx high ⇒ decoder export faithful; torch→tgt high ⇒ the model "
          "DOES transfer timbre (so a low onnx→tgt would indict the export).")


if __name__ == "__main__":
    main()
