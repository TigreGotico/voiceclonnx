# Engine: chatterbox

**Family:** AR codec-LM
**Sample rate:** 24 kHz
**WER:** 4–8%
**INT8:** ✅ available (8% WER, 57% smaller) — we host the quantized models
**License:** Apache-2.0
**Model:** [onnx-community/chatterbox-onnx](https://huggingface.co/onnx-community/chatterbox-onnx)

---

## Overview

Chatterbox (Resemble AI) is an autoregressive codec language model that transfers
both voice timbre and speaking style (prosody, expressiveness) from a reference
clip. The VC path uses only the speech encoder and conditional decoder — the TTS
text conditioning path is bypassed entirely.

## How it works

1. **Speech encoder** (`speech_encoder.onnx`) — encodes source and reference
   waveforms to codec token embeddings at 24 kHz.
2. **Conditional decoder** (`conditional_decoder.onnx`) — autoregressive
   generation of target codec tokens conditioned on the reference speaker embedding.
3. Codec tokens → waveform via the HiFi-GAN decoder embedded in the graph.

The `exaggeration` parameter scales the reference conditioning strength.

## Config / params

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `quantized` | `bool` | `False` | When `True`, loads INT8 variants (`*_q8.onnx`) — 8% WER, ~57% smaller. |
| `exaggeration` | `float` | `0.6` | Voice exaggeration factor. `0.5` = neutral; higher = more pronounced style transfer. |

## Model and license

| File | Source | License |
|------|--------|---------|
| `speech_encoder.onnx` | [onnx-community/chatterbox-onnx](https://huggingface.co/onnx-community/chatterbox-onnx) | Apache-2.0 |
| `conditional_decoder.onnx` | same | Apache-2.0 |

Models download automatically on first use via `huggingface_hub`.

## Sample rate

**24 kHz.**

## INT8 note

INT8 quantized variants are hosted at `TigreGotico/voiceclonnx-chatterbox`
(we quantize them ourselves; upstream `onnx-community/chatterbox-onnx`
ships fp32 only). `quantized=True` loads them — 8% WER, matching fp32, at
~57% smaller (1080 MB → 467 MB).

## WER

**4–8%** — measured with faster-whisper `base.en` on demo clips.
See [demo/VERIFICATION.md](../../demo/VERIFICATION.md).

## CLI example

```bash
voiceclonnx clone --engine chatterbox \
             --audio source.wav \
             --voice reference.wav \
             --out out.wav \
             --exaggeration 0.5
```

## Python example

```python
from voiceclonnx import VoiceCloner

cloner = VoiceCloner(engine="chatterbox")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 24000

# Tuned exaggeration
cloner = VoiceCloner(engine="chatterbox", exaggeration=0.5)
```

## Troubleshooting

**Output sounds robotic or has heavy artefacts** — try `exaggeration=0.5`
(lower value reduces the conditioning strength).

**Slow on CPU** — the AR decoder generates tokens sequentially. On a typical
laptop expect 5–30 s per utterance. The VC path is faster than TTS with the same
models (no text conditioning path is exercised).
