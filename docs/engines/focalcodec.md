# Engine: focalcodec

FocalCodec (Della Libera et al., NeurIPS 2025) — zero-shot any-to-any voice conversion
using a kNN feature-space swap on continuous pre-quantisation features.

Architecture:
1. **WavLM encoder** (inside FocalCodec) — converts 16 kHz audio to 1024-dim feature frames at 50 Hz.
2. **kNN cosine matching** (pure numpy) — replaces each source feature frame with the weighted mean of its k nearest reference frames (cosine distance).
3. **Vocos backbone + proj** — maps matched 1024-dim features to STFT coefficients (n_fft+2 dim).
4. **numpy ISTFT** — pure numpy Hann-window overlap-add vocoder; no ONNX needed.

No separate speaker encoder is required; no discrete tokenisation during VC (all operations are in continuous feature space before the quantiser).

ONNX artifacts: [TigreGotico/voiceclonnx-focalcodec](https://huggingface.co/TigreGotico/voiceclonnx-focalcodec) (Apache-2.0 license).

Output sample rate: **16 kHz**.

---

## Install

```bash
pip install voiceclonnx
```

Dependencies pulled in: `onnxruntime`, `numpy`, `soundfile`.
Models are downloaded from HF Hub on first use (~659 MB fp32 or ~358 MB int8 combined).

---

## Config keys

| Key | Type | Default | Description |
|---|---|---|---|
| `quantized` | `bool` | `False` | Use INT8 quantized ONNX models. Reduces total footprint to ~358 MB. **int8 not recommended: WER 31% vs fp32 15% in benchmark** — use fp32 for production. See [QUANTS.md](../QUANTS.md) for full comparison. |
| `k` | `int` | `4` | Number of nearest neighbours to average in the cosine matching step. |

---

## Model sizes

| File | Size | Variant |
|---|---|---|
| `focalcodec_encoder.onnx` | 594.6 MB | fp32 |
| `focalcodec_encoder_q8.onnx` | 341.2 MB | INT8 (42.6% reduction) |
| `focalcodec_vocoder.onnx` | 64.3 MB | fp32 |
| `focalcodec_vocoder_q8.onnx` | 16.3 MB | INT8 (74.7% reduction) |

---

## Parity vs torch

| Component | max abs | mean abs |
|---|---|---|
| encoder fp32 | 4.2e-4 | 1.8e-5 |
| vocoder backbone fp32 | 2.7e-5 | 2.1e-6 |
| numpy ISTFT vs torch | 1.3e-5 | 3.3e-7 |

---

## Usage

### Python

```python
from voiceclonnx import VoiceCloner

# Default (fp32, k=4)
cloner = VoiceCloner(engine="focalcodec")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 16000

# Quantized
cloner = VoiceCloner(engine="focalcodec", quantized=True)
out = cloner.clone_voice("source.wav", "reference.wav", "out_q8.wav")

# Wider neighbourhood
cloner = VoiceCloner(engine="focalcodec", k=8)
```

### CLI

```bash
pip install voiceclonnx

voiceclonnx clone --engine focalcodec \
             --audio source.wav \
             --voice reference.wav \
             --out out.wav
```

---

## References

- Paper: [FocalCodec (Della Libera et al., NeurIPS 2025)](https://arxiv.org/abs/2502.04465)
- Upstream code: [lucadellalib/focalcodec](https://github.com/lucadellalib/focalcodec)
- Upstream checkpoint: [lucadellalib/focalcodec_50hz](https://huggingface.co/lucadellalib/focalcodec_50hz)

---

## Troubleshooting

**`ImportError: onnxruntime is required`**
Install the extras group: `pip install voiceclonnx`.

**Output sounds noisy or garbled**
Ensure the reference clip is clean, at least 5 s long, and recorded at 16 kHz (or
the adapter will resample). Cosine kNN is sensitive to short reference pools.

**Out-of-memory on large files**
Use `quantized=True` — footprint drops from ~659 MB to ~358 MB.
For very long utterances, chunk the source audio.

**First run is slow**
Models are downloaded from HF Hub on first use and cached in `~/.cache/huggingface/hub`.
