"""Direct export-fidelity gate for mimi: torch vs the real ONNX sessions.

Three checks, no speaker embeddings (avoids that noise):

1. self-roundtrip sanity: encode->decode source with NO swap, in both torch and
   ONNX. A faithful codec reconstructs the source. Reports corr with the input.
2. encoder fidelity: do torch and ONNX produce the SAME integer codes for the
   same audio? (% per-stream agreement)
3. decoder fidelity: decode the SAME codes with torch and ONNX; waveform corr.

If codes + waveforms agree, the export is faithful and the poor VC is the
recipe. If they diverge, the export/adapter is the culprit.
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from voiceclonnx.engines.mimi import MimiAdapter, _load_wav, _swap_streams  # noqa: E402
from transformers import MimiModel  # noqa: E402

DEMO = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(DEMO, "source.wav")
REF = os.path.join(DEMO, "reference_aria.wav")
SR = 24000


def corr(a, b):
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


@torch.no_grad()
def t_encode(m, wav):
    # Official high-level API: handles encoder + transformer + downsample + RVQ.
    x = torch.from_numpy(wav)[None, None].float()
    out = m.encode(x, num_quantizers=32)
    return out.audio_codes.cpu().numpy()   # (1, Q, T)


@torch.no_grad()
def t_decode(m, codes):
    out = m.decode(torch.from_numpy(codes.astype(np.int64)))
    return out.audio_values[0, 0].cpu().numpy().astype(np.float32)


def main():
    import transformers
    print(f"transformers {transformers.__version__}", file=sys.stderr)
    m = MimiModel.from_pretrained("kyutai/mimi", attn_implementation="eager").eval()

    ad = MimiAdapter()
    ad._ensure_models()  # loads the real ONNX encoder/decoder from HF

    src = _load_wav(SRC, SR)
    ref = _load_wav(REF, SR)

    # ---- encoder fidelity: torch codes vs ONNX codes ----
    tc = t_encode(m, src)               # (1,Q,T)
    oc = ad._encode(src)                # (1,Q,T)
    Q = min(tc.shape[1], oc.shape[1])
    T = min(tc.shape[2], oc.shape[2])
    agree = [(tc[0, q, :T] == oc[0, q, :T]).mean() for q in range(Q)]
    print(f"\n[encoder] torch vs ONNX codes  shape torch={tc.shape} onnx={oc.shape}")
    print(f"  stream0 (semantic) agreement: {agree[0]*100:5.1f}%")
    print(f"  streams1-31 mean agreement:   {np.mean(agree[1:])*100:5.1f}%")
    print(f"  overall code agreement:       {np.mean(agree)*100:5.1f}%")

    # ---- decoder fidelity: same codes -> torch wav vs ONNX wav ----
    codes = oc[:, :, :T].astype(np.int64)
    tw = t_decode(m, codes)
    ow = ad._decode(codes)
    print(f"\n[decoder] same codes -> waveform corr(torch,onnx) = {corr(tw, ow):.3f}")

    # ---- self-roundtrip intelligibility (no swap) ----
    print(f"\n[roundtrip] corr(input, torch_decode(torch_codes)) = {corr(src, t_decode(m, tc)):.3f}")
    print(f"[roundtrip] corr(input, onnx_decode(onnx_codes))   = {corr(src, ad._decode(oc)):.3f}")

    # ---- the VC swap, both backends, decoded ----
    rc_t, rc_o = t_encode(m, ref), ad._encode(ref)
    mix_t = _swap_streams(tc, rc_t)
    mix_o = _swap_streams(oc, rc_o)
    print(f"\n[VC] corr(torch_swap_decode, onnx_swap_decode) = "
          f"{corr(t_decode(m, mix_t), ad._decode(mix_o)):.3f}")


if __name__ == "__main__":
    main()
