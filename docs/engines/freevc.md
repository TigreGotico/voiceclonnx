# Engine: freevc

**Family:** Flow-matching (WavLM + VITS)
**Sample rate:** 16 kHz
**WER:** 12%
**INT8:** ⚠ degrades (WER 62% in INT8 vs 12% fp32 — use fp32)
**License:** MIT
**Model:** [TigreGotico/voiceclonnx-freevc](https://huggingface.co/TigreGotico/voiceclonnx-freevc)

---

## Overview

FreeVC (Qian et al., ICASSP 2023) performs zero-shot any-to-any voice conversion
without text annotations. It uses WavLM-Large content features and a VITS-based
decoder conditioned on a GE2E speaker embedding. Non-autoregressive and
CPU-friendly.

## How it works

1. **WavLM-Large encoder** (full final hidden states, `wavlm_freevc.onnx`) —
   16 kHz audio → 1024-dim feature frames at 50 Hz. Uses `extract_features()[0]`
   (final transformer output). This is a separate artifact from the kNN-VC
   WavLM export (which extracts only layer 6).
2. **GE2E speaker encoder** (LSTM × 3 + linear) — log-mel spectrogram of
   reference audio → 256-dim d-vector.
3. **VITS SynthesizerTrn** (prior encoder + normalising flow + HiFi-GAN
   generator) — decodes content features conditioned on the d-vector to
   a 16 kHz waveform.

All neural components run via onnxruntime. Fully non-autoregressive; no diffusion steps.

## Config / params

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `quantized` | `bool` | `False` | Load INT8 `*_q8.onnx` models. **Not recommended**: WER jumps from 12% to 62% in INT8. Use fp32 for production. |

## Model and license

**MIT.**

| File | fp32 | INT8 |
|------|------|------|
| `wavlm_freevc.onnx` | 1204.4 MB | 303.5 MB (−74.8%) |
| (VITS decoder + speaker encoder) | varies | varies |

Total: ~1.2 GB fp32 / ~342 MB INT8.

## Sample rate

**16 kHz.**

## INT8 note

**INT8 is not recommended for this engine.** WER increases from 12% (fp32) to
62% (INT8) — the WavLM-Large encoder is sensitive to INT8 quantization. Use
fp32 for production. See [QUANTS.md](../QUANTS.md).

## WER

**12%** — measured with faster-whisper `base.en` on demo clips.
See [demo/VERIFICATION.md](../../demo/VERIFICATION.md).

## CLI example

```bash
voiceclonnx clone --engine freevc \
             --audio source.wav \
             --voice reference.wav \
             --out out.wav
```

## Python example

```python
from voiceclonnx import VoiceCloner

cloner = VoiceCloner(engine="freevc")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 16000
```

## Troubleshooting

**WER degrades with INT8** — this is expected; see INT8 note above. Use fp32.

**First run is slow** — WavLM-Large (~1.2 GB) downloads from HF Hub on first
use; cached in `~/.cache/huggingface/hub`.

## References

- Paper: [FreeVC: Towards High-Quality Text-Free One-Shot Voice Conversion](https://arxiv.org/abs/2210.15418)
- Upstream: [OlaWod/FreeVC](https://github.com/OlaWod/FreeVC)
