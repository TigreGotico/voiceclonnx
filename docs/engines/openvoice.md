# Engine: openvoice

**Family:** Flow-matching (tone-color transfer)
**Sample rate:** 22050 Hz
**WER:** 0%
**INT8:** available (slight quality cost)
**License:** MIT
**Model:** [TigreGotico/voiceclonnx-openvoice-v2](https://huggingface.co/TigreGotico/voiceclonnx-openvoice-v2)

---

## Overview

OpenVoice v2 (myshell-ai/OpenVoice) performs tone-color transfer: a reference
encoder extracts a 256-dim tone-color embedding from the reference audio, and a
VITS-style converter applies that tone color to the source waveform.

**Recommended for 22 kHz output with top intelligibility and a wide speaker
style range (0% WER).**

## How it works

1. **Reference encoder** (`tone_ref_encoder.onnx`): linear magnitude
   spectrogram `(B, T, 513)` to 256-dim tone-color embedding. Runs on both
   source and reference audio.
2. **Converter** (`tone_converter.onnx`): `(spec, spec_lengths, src_tone, tgt_tone)`
   to waveform directly. A full VITS-style flow decoder with the HiFi-GAN
   vocoder embedded in the graph, so there is no separate vocoder step.

Preprocessing is pure numpy (513-bin linear magnitude, `sqrt(Re²+Im²+1e-6)`).
Griffin-Lim iteration count is configurable via `gl_iters`.

## Config / params

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `quantized` | `bool` | `False` | Load INT8 `*_q8.onnx` models. Footprint drops from ~131 MB to ~43 MB, at a slight quality cost. |
| `gl_iters` | `int` | `32` | Griffin-Lim iterations for vocoding. Lower (16) is faster, higher (64) gives
better quality. |

Spectrogram parameters are fixed to the training configuration:

| Parameter | Value |
|-----------|-------|
| Sample rate | 22050 Hz |
| n_mels | 80 |
| n_fft / win_length | 1024 |
| hop_length | 256 |
| f_min | 0 Hz |

## Model and license

**MIT.**

| File | fp32 | INT8 |
|------|------|------|
| `tone_ref_encoder.onnx` | varies | varies |
| `tone_converter.onnx` | ~131 MB total | ~43 MB total |

## Sample rate

**22050 Hz.**

## INT8 note

`quantized=True` reduces footprint from ~131 MB to ~43 MB. Slight quality
degradation is expected. See [QUANTS.md](../QUANTS.md).

## WER

**0%**: perfectly intelligible on demo clips.
See [demo/VERIFICATION.md](../../demo/VERIFICATION.md).

## CLI example

```bash
voiceclonnx clone --engine openvoice \
             --audio source.wav \
             --voice reference.wav \
             --out out.wav
```

## Python example

```python
from voiceclonnx import VoiceCloner

cloner = VoiceCloner(engine="openvoice")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 22050

# Higher quality Griffin-Lim
cloner = VoiceCloner(engine="openvoice", gl_iters=64)

# INT8, smaller footprint
cloner = VoiceCloner(engine="openvoice", quantized=True)
```

## Troubleshooting

**Slow vocoder.** Increase `gl_iters` only if you need higher quality.
Lower values (16-24) are faster, with minimal perceptual difference.

**First run is slow.** Models download from HF Hub on first use and are
cached in `~/.cache/huggingface/hub`.

## References

- Paper: [OpenVoice: Versatile Instant Voice Cloning](https://arxiv.org/abs/2312.01479)
- Upstream: [myshell-ai/OpenVoice](https://github.com/myshell-ai/OpenVoice)

---
[← facodec](facodec.md) · [Home](../index.md) · [chatterbox →](chatterbox.md)
