# Engine: cosyvoice

**Family:** Flow-matching
**Sample rate:** 22050 Hz
**WER:** 8%
**INT8:** ⚠ degrades (WER 100% in INT8, use fp32)
**License:** Apache-2.0
**Model:** [TigreGotico/voiceclonnx-cosyvoice](https://huggingface.co/TigreGotico/voiceclonnx-cosyvoice)

---

## Overview

CosyVoice (FunAudioLLM / Alibaba DAMO) performs zero-shot cross-lingual voice
conversion via a non-autoregressive flow-matching pipeline. The autoregressive
LLM used for TTS is bypassed entirely; only the speech tokenizer, speaker
encoder, flow encoder/decoder, and HiFiGAN vocoder are used.

CPU RTF: ~0.71×.

## How it works

1. **`speech_tokenizer_v1.onnx`**: Whisper-style log-mel (128-bin, 16 kHz) →
   content token indices `(1, T_tok)` int64. Encodes phoneme sequence.
2. **`campplus.onnx`** (CAM++): Kaldi fbank (80-bin, 16 kHz) → 192-d speaker
   embedding `(1, 192)`. Encodes reference voice identity.
3. **`flow_encoder.onnx`**: 6-block Conformer encoder + `InterpolateRegulator`
   → mu `(1, 80, T_mel)`. Converts content tokens to flow conditioning.
4. **`flow_decoder.onnx`**: ODE Euler solver (default 10 steps) over mu +
   80-d projected speaker embedding → mel spectrogram `(1, 80, T_mel)`.
5. **`hifigan_f0_source.onnx`**: F0 predictor + NSF harmonic source → 1-D
   source signal `(1, 1, T_audio)`.
6. **numpy STFT/ISTFT**: `aten::stft`/`aten::istft` are unsupported at ONNX
   opset 14; the HiFiGAN is split at the STFT boundary with numpy handling.
7. **`hifigan_backbone.onnx`**: HiFTGenerator backbone → magnitude/phase STFT
   bins → numpy ISTFT → waveform @ 22050 Hz.

### Speaker projection

`spk_embed_affine_layer` (192 → 80) is extracted from `flow.pt` at export time
and saved as `spk_proj.npz`. The adapter projects the CAM++ 192-d embedding into
the 80-d flow conditioning space without the full flow model.

## Config / params

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `quantized` | `bool` | `False` | Load INT8 models. **Not recommended**: INT8 WER is 100% vs fp32 8%. Use fp32 for production. |
| `ode_steps` | `int` | `10` | ODE Euler steps for the flow decoder. Fewer steps = faster but may add metallic artefacts. Range: 5-30. |

## Model and license

Upstream: `FunAudioLLM/CosyVoice-300M` (Apache-2.0).
Training corpus Emilia is CC-BY-NC-4.0, which does not constrain the model weights.

| File | fp32 | INT8 |
|------|------|------|
| `speech_tokenizer_v1.onnx` | 499 MB | not quantized |
| `campplus.onnx` | 27 MB | not quantized |
| `flow_decoder.onnx` | 314 MB | 82 MB |
| `flow_encoder.onnx` | 108 MB | 42 MB |
| `hifigan_f0_source.onnx` | 13 MB | 3 MB |
| `hifigan_backbone.onnx` | 66 MB | 25 MB |
| `spk_proj.npz` | ~120 KB | not quantized |

## Sample rate

**22050 Hz.**

## INT8 note

**INT8 is not recommended for this engine.** The flow decoder and speech
tokenizer do not quantize well to INT8; measured WER jumps to 100% from fp32's 8%. Use `quantized=False` (the
default) for production.
See [QUANTS.md](../QUANTS.md).

## WER

**8%**: measured with faster-whisper `base.en` on demo clips.
See [demo/VERIFICATION.md](../../demo/VERIFICATION.md).

## CLI example

```bash
voiceclonnx clone --engine cosyvoice \
             --audio source.wav \
             --voice reference.wav \
             --out out.wav
```

## Python example

```python
from voiceclonnx import VoiceCloner

cloner = VoiceCloner(engine="cosyvoice")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 22050

# Fewer ODE steps for faster inference (may reduce quality)
cloner = VoiceCloner(engine="cosyvoice", ode_steps=5)
```

## Troubleshooting

**Output is silence or noise.** Use clean mono speech at or near 16 kHz for
source and reference. Very short clips (under 0.5 s) may yield degraded
output, because the speech tokenizer expects at least about 25 tokens.

**ODE artefacts.** `ode_steps=5` is about 2x faster but may introduce
metallic artefacts on complex prosody. The default `ode_steps=10` is
recommended for CPU.

**First run is slow.** About 1.1 GB downloads from HF Hub on first use and
is cached in `~/.cache/huggingface/hub`.

## References

- Paper: [CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer](https://arxiv.org/abs/2407.05407)
- Upstream: [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- Upstream checkpoint: [FunAudioLLM/CosyVoice-300M](https://huggingface.co/FunAudioLLM/CosyVoice-300M)

---
[← triaan](triaan.md) · [Home](../index.md) · [bicodec →](bicodec.md)
