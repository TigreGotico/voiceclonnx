# Engine: triaan

**Family:** Flow-matching (Triple Adaptive Attention Normalization)
**Sample rate:** 16 kHz
**WER:** 4%
**INT8:** available (slight quality cost)
**License:** MIT
**Model:** [TigreGotico/voiceclonnx-triaan-vc](https://huggingface.co/TigreGotico/voiceclonnx-triaan-vc)

---

## Overview

TriAAN-VC (ICASSP 2023) performs any-to-any voice conversion using Triple
Adaptive Attention Normalization — a fusion of time-wise, channel-wise, and
global adaptive normalization — combined with a CPC encoder and ParallelWaveGAN
vocoder. Three ONNX components per inference call.

## How it works

1. **CPC encoder** (`cpc_encoder.onnx`) — 5-layer strided Conv1d (160×
   downsample, 16 kHz → 100 Hz) + single LSTM layer → 256-dim content features.
   Architecture from [facebookresearch/CPC_audio](https://github.com/facebookresearch/CPC_audio).
2. **TriAAN-VC decoder** (`triaan_vc.onnx`) — ContentEncoder + SpeakerEncoder
   (with skip connections) + bidirectional GRU fusion + TriAANBlock decoder +
   PostNet. Combines:
   - **TAN** (Time-wise Adaptive Normalization): time-domain cross-attention on speaker features
   - **CAN** (Channel-wise Adaptive Normalization): channel-domain cross-attention
   - **GLAN** (Global Adaptive Normalization): global self-attention pooling
3. **ParallelWaveGAN vocoder** (`pwg_vocoder.onnx`) — WaveNet-style vocoder
   trained on VCTK; mel spectrogram → 16 kHz waveform.

## Config / params

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `quantized` | `bool` | `False` | Load INT8 `*_q8.onnx` models. Reduces total footprint to ~84 MB; slight quality cost. |
| `model_dir` | `str` | `None` | Local path to a directory with all ONNX files. Skips HF Hub download. |
| `hf_repo_id` | `str` | `TigreGotico/voiceclonnx-triaan-vc` | HF repository to download from. |

## Model and license

**MIT.**

| File | Variant |
|------|---------|
| `cpc_encoder.onnx` | fp32 |
| `cpc_encoder_q8.onnx` | INT8 |
| `triaan_vc.onnx` | fp32 |
| `triaan_vc_q8.onnx` | INT8 |
| `pwg_vocoder.onnx` | fp32 |
| `pwg_vocoder_q8.onnx` | INT8 |

Total: ~84 MB INT8.

## Sample rate

**16 kHz.**

## INT8 note

`quantized=True` reduces footprint to ~84 MB. Slight quality degradation
expected. See [QUANTS.md](../QUANTS.md).

## WER

**4%** — measured with faster-whisper `base.en` on demo clips.
See [demo/VERIFICATION.md](../../demo/VERIFICATION.md).

## CLI example

```bash
voiceclonnx clone --engine triaan \
             --audio source.wav \
             --voice reference.wav \
             --out out.wav
```

## Python example

```python
from voiceclonnx import VoiceCloner

vc = VoiceCloner(engine="triaan")
out = vc.clone_voice("source.wav", "reference.wav", "out.wav")
print(vc.sample_rate)   # 16000

# INT8 — small footprint (~84 MB)
vc = VoiceCloner(engine="triaan", quantized=True)

# Local weights (skip HF download)
vc = VoiceCloner(engine="triaan", model_dir="/path/to/triaan-models")
```

## Troubleshooting

**Output timbre not transferring** — the TriAANBlock decoder relies on speaker
features from the reference; use a clean, at-least-3-second reference clip.

**First run is slow** — models download from HF Hub on first use; cached in
`~/.cache/huggingface/hub`.

## References

- Paper: [TriAAN-VC: Triple Adaptive Attention Normalization for Any-to-Any Voice Conversion](https://arxiv.org/abs/2303.09057)
