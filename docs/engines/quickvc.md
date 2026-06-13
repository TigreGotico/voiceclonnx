# Engine: quickvc

**Family:** Flow-matching (HuBERT-soft + VITS + MS-iSTFT)
**Sample rate:** 16 kHz
**WER:** 0%
**INT8:** available (4% WER in INT8 vs 0% fp32 — recommended for constrained deployments)
**License:** MIT
**Model:** [TigreGotico/voiceclonnx-quickvc](https://huggingface.co/TigreGotico/voiceclonnx-quickvc)

---

## Overview

QuickVC (quickvc/QuickVC-VoiceConversion) performs any-to-many voice conversion
using a HuBERT-soft content encoder and a VITS-style normalizing-flow decoder
with a Multistream-iSTFT (MS-iSTFT) generator. The MS-iSTFT vocoder synthesises
4 frequency subbands with a tiny FFT (n_fft=16) rather than a full-resolution
upsampling network, yielding notably fast CPU inference.

**~0.14× RTF** — the fastest engine in voiceclonnx. WER 0%.

## How it works

1. **HuBERT-soft content encoder** (12-layer Transformer, 94M params) — 16 kHz
   audio → 256-dim soft speech units at 50 Hz. Exported in 1-second chunks
   (fixed T=50 context window due to MHA reshape constraints).
2. **Speaker encoder** (LSTM × 1 + linear) — 80-channel log-mel spectrogram of
   reference audio → 256-dim d-vector.
3. **Posterior encoder + normalizing flow** — content features → latent space
   conditioned on speaker d-vector.
4. **Multistream-iSTFT generator** — upsamples by 20× and produces per-subband
   STFT coefficients.
5. **numpy Multistream-iSTFT** — pure-numpy center-mode OLA over 4 subbands
   (n_fft=16, hop=4). Not in ONNX because `torch.istft` cannot be traced;
   parity vs torch = 0.0 max abs error.
6. **Postnet** — learned 1-D mixing conv collapses subbands to mono waveform.

## Config / params

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `quantized` | `bool` | `False` | Load INT8 `*_q8.onnx` models. Footprint drops from ~480 MB to ~130 MB; WER is 4% (vs fp32 0%). INT8 recommended for memory-constrained deployments. |

## Model and license

**MIT.**

| File | fp32 | INT8 |
|------|------|------|
| (HuBERT-soft encoder) | ~480 MB total | ~130 MB total |

## Sample rate

**16 kHz.**

## INT8 note

`quantized=True` reduces footprint from ~480 MB to ~130 MB. INT8 WER is 4%
vs fp32's 0% — a small tradeoff. INT8 is recommended for memory-constrained
deployments where 0% WER is not critical. See [QUANTS.md](../QUANTS.md).

## WER

**0%** (fp32) / **4%** (INT8) — measured with faster-whisper `base.en` on demo
clips. See [demo/VERIFICATION.md](../../demo/VERIFICATION.md).

## CLI example

```bash
voiceclonnx clone --engine quickvc \
             --audio source.wav \
             --voice reference.wav \
             --out out.wav
```

## Python example

```python
from voiceclonnx import VoiceCloner

cloner = VoiceCloner(engine="quickvc")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 16000

# INT8 — 130 MB footprint, 4% WER
cloner = VoiceCloner(engine="quickvc", quantized=True)
```

## Troubleshooting

**Slight quality loss in INT8** — this is expected (4% WER vs 0% fp32); use
fp32 if intelligibility is critical.

**First run is slow** — ~480 MB downloads from HF Hub on first use; cached in
`~/.cache/huggingface/hub`. Subsequent runs start quickly.

## References

- Upstream: [quickvc/QuickVC-VoiceConversion](https://github.com/quickvc/QuickVC-VoiceConversion)
