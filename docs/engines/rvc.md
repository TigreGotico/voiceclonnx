# Engine: rvc

**Family:** Any-to-ONE (ContentVec + RMVPE + VITS)
**Sample rate:** 40 kHz (v2-40k) or 48 kHz (v2-48k). model-dependent
**WER:** 38% (sample community model; varies by model quality)
**INT8:** available for base models (ContentVec + RMVPE); per-voice model is user-supplied
**License:** MIT (base models)
**Model:** [TigreGotico/voiceclonnx-rvc](https://huggingface.co/TigreGotico/voiceclonnx-rvc)

---

## Overview

RVC (Retrieval-based Voice Conversion) is an **any-to-ONE** engine: the target
speaker identity is baked into a per-voice voice model trained or fine-tuned
separately. `reference_voice` is **not** a reference audio file. It is the
path to an `.onnx` RVC voice model (local file or HF repo ID). Thousands of
community-trained RVC voice models exist on Hugging Face.

voiceclonnx hosts the base shared models (ContentVec encoder and RMVPE F0
predictor) at `TigreGotico/voiceclonnx-rvc`. The user supplies per-voice
`net_g` synthesizers.

## How it works

1. **ContentVec encoder** (`contentvec_768l12.onnx`): a HuBERT-based content
   encoder fine-tuned for content disentanglement, producing 768-dim content
   features.
2. **RMVPE F0 predictor** (`rmvpe.onnx`): a DeepUnet-based fundamental
   frequency estimator, producing per-frame F0 values.
3. **net_g synthesizer** (per-voice `.onnx`). VITS-based generator that
   synthesizes the waveform from content features and F0, conditioned on the
   speaker baked into the model weights.

The adapter lazy-loads and caches the `net_g` corresponding to each
`reference_voice` path.

## Config / params

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `quantized` | `bool` | `False` | Apply INT8 quantization to the base models (ContentVec + RMVPE). The per-voice `net_g` is user-supplied and unaffected. |
| `default_model` | `str` | `None` | Default voice model path used when `reference_voice` is not specified. |

## Model and license

**MIT.** Base models:

| File | fp32 | INT8 |
|------|------|------|
| `contentvec_768l12.onnx` | 360.3 MB | 90.8 MB (−74.8%) |
| `rmvpe.onnx` | 344.9 MB | 94.1 MB (−72.7%) |

Per-voice `.onnx` models are user-supplied (community `.pth` models
converted to ONNX, see below).

## Sample rate

**40 kHz** (v2-40k models) or **48 kHz** (v2-48k models). Detected automatically
from ONNX model metadata; defaults to 40000 Hz if metadata is absent.

## INT8 note

`quantized=True` applies to the shared base models only (ContentVec + RMVPE).
The per-voice `net_g` model is supplied by the user and is not quantized by
voiceclonnx. See [QUANTS.md](../QUANTS.md).

## WER

**38%** (sample community model `ozada/onnx_rvc::woman_1.onnx`). WER depends
strongly on the model. High-quality community models can reach much lower
WER. See [demo/VERIFICATION.md](../../demo/VERIFICATION.md).

## CLI example

```bash
# reference_voice = path to an RVC .onnx voice model, NOT an audio file
voiceclonnx clone --engine rvc \
             --audio source.wav \
             --voice /path/to/myvoice.onnx \
             --out converted.wav
```

## Python example

```python
from voiceclonnx import VoiceCloner

# reference_voice = path to RVC .onnx model or HF repo ID
cloner = VoiceCloner(engine="rvc")
out = cloner.clone_voice("source.wav", "/path/to/myvoice.onnx", "out.wav")
print(cloner.sample_rate)   # 40000 or 48000 depending on model

# HF repo ID with a specific file
out = cloner.clone_voice("source.wav", "ozada/onnx_rvc::woman_1.onnx", "out.wav")
```

## Converting a community voice model to ONNX

Community RVC models are typically distributed as `.pth` files. Convert to ONNX:

```bash
python -m conversion.convert_rvc_model myvoice.pth myvoice.onnx \
    --parity-report myvoice_parity.json
```

The helper embeds `sample_rate` in the ONNX model metadata for automatic
40k/48k detection.

## Troubleshooting

**`reference_voice` is a WAV file.** RVC is any-to-ONE, so `reference_voice`
must be a path to an `.onnx` voice model, not audio. Use a different engine
(for example `facodec` or `knnvc`) for any-to-any conversion from a
reference audio clip.

**Output intelligibility varies widely.** WER depends on the quality of the
per-voice model. Community models range from excellent to poor. Test with
`demo/verify_demos.py` to measure WER on your specific model.

**First run is slow.** Base models (about 705 MB fp32) download from HF Hub
on first use and are cached in `~/.cache/huggingface/hub`.

## References

- Upstream: [RVC-Project/Retrieval-based-Voice-Conversion-WebUI](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)

---
[← lscodec](lscodec.md) · [Home](../index.md)
