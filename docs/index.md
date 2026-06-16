# voiceclonnx — documentation

Pure-ONNX multi-engine voice conversion library. Audio-to-audio only: converts
the voice in an existing speech file to sound like a reference speaker. Text-driven
synthesis is out of scope.

Runtime requirements: `onnxruntime`, `numpy`, `soundfile`, `huggingface_hub` only.
No PyTorch, no CUDA driver required for inference.

---

## Contents

| Document | What it covers |
|----------|---------------|
| [api.md](api.md) | `VoiceCloner` facade, `VoiceClonerBase`, `EngineEntry`, registry functions |
| [engines/](engines/) | Per-engine guides — config keys, model info, WER, CLI example, troubleshooting |
| [QUANTS.md](QUANTS.md) | fp32 vs INT8 WER and size comparison across all engines |
| [converting.md](converting.md) | Export → parity → quantize → push → adapter toolchain for adding engines |
| [../demo/README.md](../demo/README.md) | Listen to every engine without installing |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Contribution checklist: adding an engine, running tests, the demo/verify gate |

---

## Install

```bash
pip install voiceclonnx
```

One command installs every engine. ONNX models are downloaded on first use.

| Extra | Purpose |
|-------|---------|
| `pip install "voiceclonnx[convert]"` | Export toolchain (torch, onnx, transformers, librosa) |
| `pip install "voiceclonnx[bench]"` | Benchmark / demo generation (faster-whisper, edge-tts) |
| `pip install "voiceclonnx[test]"` | Test suite (pytest, faster-whisper, edge-tts) |

---

## Engine families

The 9 built-in engines fall into five architectural families. Understanding the
family determines which tradeoffs apply.

### kNN feature-swap (`knnvc`, `focalcodec`)

Extract dense SSL features from source and reference audio, replace each source
frame with the nearest-neighbour average from the reference feature set, then
vocoder back to waveform. The matching step is pure numpy — no ONNX at match time.

- **knnvc** — WavLM-Large layer-6 + L2-kNN + HiFi-GAN vocoder. Fast, light,
  the classic kNN-VC reference implementation.
- **focalcodec** — WavLM encoder + cosine-kNN + Vocos ISTFT vocoder. NeurIPS 2025
  architecture with continuous pre-quantization features.

### Factorized codec (`facodec`, `bicodec`)

Encode speech into factorized subspaces (content / prosody / timbre / speaker),
swap the speaker or timbre component from the reference, decode the combination.
No AR generation — single forward pass per segment.

- **facodec** — NaturalSpeech 3 factorized VQ (4 subspaces). 0% WER, highly
  recommended for quality.
- **bicodec** — SparkTTS BiCodec: semantic tokens (content) + global tokens
  (speaker). CC BY-NC-SA 4.0 weights (non-commercial).

### Flow-matching (`cosyvoice`, `openvoice`, `triaan`)

Encoder–flow-decoder architectures that model the conditional distribution from
content features and speaker embedding to waveform via normalizing flows or
flow-matching (ODE solver).

- **cosyvoice** — FunAudioLLM CosyVoice non-AR VC via ODE flow-matching at
  22 kHz. STFT/ISTFT computed in numpy (opset limitation).
- **openvoice** — MyShell OpenVoice v2 tone-color transfer. 0% WER, 22 kHz.
- **triaan** — Triple Adaptive Attention Normalization (ICASSP 2023) + CPC
  encoder + ParallelWaveGAN vocoder.

### AR codec-LM (`chatterbox`)

Autoregressive language model over audio codec tokens. Transfers both voice
timbre and speaking style (prosody, expressiveness).

- **chatterbox** — Resemble AI Chatterbox AR codec-LM. 24 kHz. INT8 available
  (no INT8 variants published upstream). Configurable exaggeration factor.

### Any-to-ONE (`rvc`)

The target speaker identity is baked into a per-voice model; `reference_voice`
is a model path (`.onnx` file or HF repo ID), not an audio file. Thousands of
community-trained RVC voice models exist on Hugging Face.

- **rvc** — ContentVec-768 content encoder + RMVPE pitch estimator + VITS
  synthesizer. 40 kHz (v2-40k) or 48 kHz (v2-48k).

---

## Engine table

| Alias | Family | Sample rate | WER | INT8 | License |
|-------|--------|-------------|-----|------|---------|
| `facodec` | Factorized codec | 16 kHz | 0% | ✅ | Apache-2.0 |
| `openvoice` | Flow-matching | 22 kHz | 0% | ✅ | MIT |
| `chatterbox` | AR codec-LM | 24 kHz | 4–8% | ✅ INT8 | Apache-2.0 |
| `triaan` | Flow-matching | 16 kHz | 4% | ✅ | MIT |
| `cosyvoice` | Flow-matching | 22 kHz | 8% | ⚠ degrades | Apache-2.0 |
| `bicodec` | Factorized codec | 16 kHz | 12% | ✅ | CC BY-NC-SA 4.0 |
| `knnvc` | kNN feature-swap | 16 kHz | 12–15% | ✅ | MIT |
| `focalcodec` | kNN feature-swap | 16 kHz | 15–19% | ⚠ degrades | Apache-2.0 |
| `rvc` | Any-to-ONE | 40/48 kHz | 38%† | ✅ (base) | MIT |

†rvc WER is model-dependent; value from sample community model.

---

## Weight-license policy

voiceclonnx never redistributes model weights without redistribution rights.

- **Distributable** — MIT / Apache / BSD / CC-BY upstream license. ONNX weights
  are published to `TigreGotico/voiceclonnx-<engine>` on HF Hub and downloaded
  automatically on first use.
- **Local-only weights** — NC/ND/unlicensed upstream. The export script runs
  locally; `push_models` refuses to upload; the adapter loads from the local
  `model_dir` config key.

Full details: [converting.md](converting.md#weight-license-policy-publish-with-the-license-stated).
