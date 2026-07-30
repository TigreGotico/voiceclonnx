# Engine: focalcodec

**Family:** kNN feature-swap
**Sample rate:** 16 kHz
**WER:** 15-19%
**INT8:** degrades (WER 31% in INT8 vs 15% fp32, use fp32)
**License:** Apache-2.0
**Model:** [TigreGotico/voiceclonnx-focalcodec](https://huggingface.co/TigreGotico/voiceclonnx-focalcodec)

---

## Overview

FocalCodec (Della Libera et al., NeurIPS 2025) performs zero-shot any-to-any
voice conversion using a kNN feature-space swap on continuous pre-quantization
features from the WavLM encoder. There is no discrete tokenization during
VC: all operations happen in continuous feature space before the
quantizer.

## How it works

1. **WavLM encoder** (inside FocalCodec): 16 kHz audio to 1024-dim feature
   frames at 50 Hz.
2. **kNN cosine matching** (pure numpy): replaces each source feature frame
   with the weighted mean of its k nearest reference frames (cosine
   distance).
3. **Vocos backbone + proj**: maps matched 1024-dim features to STFT
   coefficients (n_fft+2 dim).
4. **numpy ISTFT**: a Hann-window overlap-add vocoder, needing no ONNX.

No separate speaker encoder required.

## Config / params

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `quantized` | `bool` | `False` | Load INT8 models. **Not recommended**: INT8 WER is 31% vs fp32 15%. Use fp32 for production. |
| `k` | `int` | `4` | Number of nearest neighbours to average in the cosine matching step. |

## Model and license

**Apache-2.0.**

| File | fp32 | INT8 |
|------|------|------|
| `focalcodec_encoder.onnx` | 594.6 MB | 341.2 MB (−42.6%) |
| `focalcodec_vocoder.onnx` | 64.3 MB | 16.3 MB (−74.7%) |

Total: ~659 MB fp32 / ~358 MB INT8.

## Sample rate

**16 kHz.**

## INT8 note

**INT8 is not recommended for this engine.** WER increases from 15% (fp32) to
31% (INT8). Use `quantized=False` (the default) for production.
See [QUANTS.md](../QUANTS.md).

## WER

**15-19%**: measured with faster-whisper `base.en` on demo clips.
See [demo/VERIFICATION.md](../../demo/VERIFICATION.md).

## CLI example

```bash
voiceclonnx clone --engine focalcodec \
             --audio source.wav \
             --voice reference.wav \
             --out out.wav
```

## Python example

```python
from voiceclonnx import VoiceCloner

cloner = VoiceCloner(engine="focalcodec")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 16000

# More neighbours, smoother conversion at slightly higher compute cost
cloner = VoiceCloner(engine="focalcodec", k=8)
```

## Troubleshooting

**High WER on short clips.** The cosine kNN matching needs a dense
reference feature set. Use reference clips of at least 5 s for best
matching.

**Slow on first run.** About 659 MB of encoder downloads from HF Hub and is
cached in `~/.cache/huggingface/hub`.

## References

- Paper: [FocalCodec: Low-Bitrate Speech Coding via Focal Tokens](https://arxiv.org/abs/2410.23265)

---
[← knnvc](knnvc.md) · [Home](../index.md) · [lscodec →](lscodec.md)
