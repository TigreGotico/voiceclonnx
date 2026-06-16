# Engine: knnvc

**Family:** kNN feature-swap
**Sample rate:** 16 kHz
**WER:** 12–15%
**INT8:** available (slight quality cost; recommended for memory-constrained use)
**License:** MIT
**Model:** [TigreGotico/voiceclonnx-knn-vc](https://huggingface.co/TigreGotico/voiceclonnx-knn-vc)

---

## Overview

kNN-VC (Baas et al., Interspeech 2023) performs zero-shot any-to-any voice
conversion. It extracts WavLM-Large layer-6 features from source and reference
audio, replaces each source frame with the k-nearest-neighbour average from the
reference feature set (L2 distance), then vocoders the matched features back to
waveform with HiFi-GAN. The kNN step is pure numpy — no ONNX at match time.

Smallest INT8 footprint of any engine: ~123 MB.

## How it works

1. **WavLM-Large encoder** (layer 6, `wavlm_layer6.onnx`) — 16 kHz audio →
   1024-dim feature frames at 50 Hz.
2. **L2-kNN matching** (pure numpy) — replaces each source frame with the
   average of its k nearest reference frames by L2 distance.
3. **HiFi-GAN vocoder** (`hifigan_knnvc.onnx`) — matched features → waveform.

## Config / params

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `quantized` | `bool` | `False` | Load INT8 `*_q8.onnx` models. Reduces memory to ~123 MB total; slight quality cost. |
| `k` | `int` | `4` | Number of nearest neighbours to average. Higher values smooth the conversion; lower values preserve more source characteristics. |

## Model and license

**MIT.**

| File | fp32 | INT8 |
|------|------|------|
| `wavlm_layer6.onnx` | 386.8 MB | 97.5 MB (−74.8%) |
| `hifigan_knnvc.onnx` | 63.1 MB | 25.1 MB (−60.2%) |

Total: ~450 MB fp32 / ~123 MB INT8.

## Sample rate

**16 kHz.**

## INT8 note

`quantized=True` reduces footprint to ~123 MB — the smallest INT8 footprint
among all voiceclonnx engines. Slight quality degradation expected.
See [QUANTS.md](../QUANTS.md) for the WER comparison.

## WER

**12–15%** — measured with faster-whisper `base.en` on demo clips.
See [demo/VERIFICATION.md](../../demo/VERIFICATION.md).

## CLI example

```bash
voiceclonnx clone --engine knnvc \
             --audio source.wav \
             --voice reference.wav \
             --out out.wav
```

## Python example

```python
from voiceclonnx import VoiceCloner

cloner = VoiceCloner(engine="knnvc")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 16000

# INT8 — lowest memory footprint (~123 MB)
cloner = VoiceCloner(engine="knnvc", quantized=True)

# More neighbours — smoother conversion
cloner = VoiceCloner(engine="knnvc", k=8)
```

## Troubleshooting

**Noisy kNN matching** — use a longer reference clip (5–10 s) so the feature
set covers more speaker variation. Increasing `k` also smooths results.

**First run is slow** — WavLM (~387 MB) downloads from HF Hub on first use;
cached in `~/.cache/huggingface/hub`.

## References

- Paper: [Voice Conversion With Just Nearest Neighbours](https://arxiv.org/abs/2305.18975)
- Upstream: [bshall/knn-vc](https://github.com/bshall/knn-vc)
