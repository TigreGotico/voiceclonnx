"""Export-parity gate for speechtokenizer: original torch vs ONNX adapter.

Runs the ORIGINAL torch SpeechTokenizer through the SAME RVQ token-swap recipe
the ONNX adapter uses, then compares the two converted outputs (waveform corr +
speaker embedding). The ONNX output is the already-shipped demo clip.
"""
import os
import sys

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download

# speechtokenizer/__init__ pulls in its trainer (torchaudio + tensorboard), which
# we don't need. Stub the trainer submodule so __init__ skips it; the model then
# imports normally with proper package context.
from unittest.mock import MagicMock  # noqa: E402
sys.modules["speechtokenizer.trainer"] = MagicMock()
from speechtokenizer import SpeechTokenizer  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from voiceclonnx.engines.speechtokenizer import _load_wav, _swap_rvq_tokens  # noqa: E402
from speakeronnx import SpeakerEmbedder, cosine  # noqa: E402

DEMO = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(DEMO, "source.wav")
REFS = {"aria": os.path.join(DEMO, "reference_aria.wav"),
        "sonia": os.path.join(DEMO, "reference_sonia.wav")}
ONNX_OUT = {k: os.path.join(DEMO, "outputs", f"speechtokenizer__{k}.wav") for k in REFS}
SR = 16000


def corr(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def st_codes(model, wav):
    x = torch.from_numpy(wav)[None, None].float()
    return model.encode(x).cpu().numpy()[:, 0, :]   # (Q, T)


@torch.no_grad()
def st_decode(model, codes):
    c = torch.from_numpy(codes.astype(np.int64))[:, None, :]  # (Q,1,T)
    return model.decode(c)[0, 0].cpu().numpy().astype(np.float32)


def main():
    import glob
    d = snapshot_download("fnlp/SpeechTokenizer")
    cfg = glob.glob(os.path.join(d, "**", "config.json"), recursive=True)[0]
    ckpt = glob.glob(os.path.join(d, "**", "SpeechTokenizer.pt"), recursive=True)[0]
    model = SpeechTokenizer.load_from_checkpoint(cfg, ckpt).eval()

    emb = SpeakerEmbedder(model="wespeaker-resnet34")
    src_e = emb.embed(SRC)
    src = _load_wav(SRC, SR)
    src_codes = st_codes(model, src)

    print(f"\n{'target':<8}{'wav-corr':<10}{'emb torch↔onnx':<16}"
          f"{'torch→tgt':<11}{'onnx→tgt':<11}{'torch→src':<11}{'onnx→src':<10}")
    print("-" * 76)
    for tgt, refpath in REFS.items():
        ref = _load_wav(refpath, SR)
        mixed = _swap_rvq_tokens(src_codes, st_codes(model, ref), content_layers=2)
        torch_wav = st_decode(model, mixed)
        tw = os.path.join(DEMO, f"_torch_st__{tgt}.wav")
        sf.write(tw, torch_wav, SR)

        onnx_wav, _ = sf.read(ONNX_OUT[tgt])
        te, oe, re = emb.embed(tw), emb.embed(ONNX_OUT[tgt]), emb.embed(refpath)
        print(f"{tgt:<8}{corr(torch_wav, onnx_wav.astype(np.float32)):<10.3f}"
              f"{cosine(te, oe):<16.3f}{cosine(te, re):<11.3f}{cosine(oe, re):<11.3f}"
              f"{cosine(te, src_e):<11.3f}{cosine(oe, src_e):<10.3f}")

    print("\nIf torch≈onnx (high wav-corr / emb sim) the export is faithful; if "
          "torch→tgt is ALSO low, the recipe/model is the limit, not the export.")


if __name__ == "__main__":
    main()
