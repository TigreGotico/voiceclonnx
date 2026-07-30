# voiceclonnx documentation

voiceclonnx is a pure-ONNX, multi-engine voice conversion library. It works on
audio only: it converts the voice in an existing speech file to sound like a
reference speaker. Text-driven synthesis is out of scope.

Runtime requirements: `onnxruntime`, `numpy`, `soundfile`, and `huggingface_hub`
only. Inference needs no PyTorch and no CUDA driver.

---

## Contents

| Document | What it covers |
|----------|---------------|
| [api.md](api.md) | `VoiceCloner` facade, `VoiceClonerBase`, `EngineEntry`, registry functions |
| [engines/](engines/) | Per-engine guides: config keys, model info, WER, CLI example, troubleshooting |
| [QUANTS.md](QUANTS.md) | fp32 vs INT8 WER and size comparison across all engines |
| [converting.md](converting.md) | Export, parity check, quantize, push, and adapter toolchain for adding engines |
| [../demo/README.md](../demo/README.md) | Listen to every engine without installing |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Contribution checklist: adding an engine, running tests, the demo/verify gate |

---

## Install

```bash
pip install voiceclonnx
```

One command installs every engine. voiceclonnx downloads ONNX models on first
use.

| Extra | Purpose |
|-------|---------|
| `pip install "voiceclonnx[convert]"` | Export toolchain (torch, onnx, transformers, librosa) |
| `pip install "voiceclonnx[bench]"` | Benchmark and demo generation (faster-whisper, edge-tts) |
| `pip install "voiceclonnx[test]"` | Test suite (pytest, faster-whisper, edge-tts) |

---

## Engine families

The 10 built-in engines fall into six architectural families. The family an
engine belongs to determines which tradeoffs apply to it.

### kNN feature-swap (`knnvc`, `focalcodec`)

These engines extract dense SSL features from the source and reference audio,
replace each source frame with the nearest-neighbor average from the
reference feature set, then run a vocoder to produce the waveform. The
matching step runs in pure numpy, with no ONNX at match time.

- **knnvc** uses a WavLM-Large layer-6 encoder, L2-kNN matching, and a
  HiFi-GAN vocoder, following the classic kNN-VC reference implementation. It
  is fast and light.

- **focalcodec** uses a WavLM encoder, cosine-kNN matching, and a Vocos
  ISTFT vocoder, a NeurIPS 2025 architecture with continuous
  pre-quantization features.

### Factorized codec (`facodec`, `bicodec`)

These engines encode speech into factorized subspaces (content, prosody,
timbre, speaker), swap the speaker or timbre component from the reference,
and decode the combination. There is no AR generation: each segment needs
only a single forward pass.

- **facodec** uses NaturalSpeech 3 factorized VQ across 4 subspaces,
  reaching 0% WER. It is recommended for quality.

- **bicodec** is SparkTTS BiCodec: semantic tokens carry content and global
  tokens carry speaker identity. Its weights are CC BY-NC-SA 4.0
  (non-commercial).

### Flow-matching (`cosyvoice`, `openvoice`, `triaan`)

These engines use an encoder and flow-decoder to model the conditional
distribution from content features and a speaker embedding to a waveform,
using normalizing flows or flow-matching with an ODE solver.

- **cosyvoice** is FunAudioLLM CosyVoice, a non-AR VC engine that uses ODE
  flow-matching at 22 kHz, with STFT and ISTFT running in numpy because of
  an opset limitation.

- **openvoice** is MyShell OpenVoice v2 tone-color transfer. It reaches 0%
  WER at 22 kHz.

- **triaan** combines Triple Adaptive Attention Normalization (ICASSP 2023)
  with a CPC encoder and a ParallelWaveGAN vocoder.

### AR codec-LM (`chatterbox`)

This engine is an autoregressive language model over audio codec tokens. It
transfers both voice timbre and speaking style (prosody, expressiveness).

- **chatterbox** is Resemble AI Chatterbox, an AR codec-LM running at 24 kHz.
  An INT8 variant is available, though upstream does not publish one. It has
  a configurable exaggeration factor.

### Speaker-decoupled codec (`lscodec`)

This is a discrete codec whose content tokens are trained to be
speaker-agnostic, so resynthesis with a different speaker prompt converts
the voice directly. It uses a single codebook, with no multi-stream token
swap.

- **lscodec** combines an encoder (raw audio to 64-d tokens), a numpy VQ
  step (300-entry codebook), and a WavLM-prompt CTXVEC2WAV vocoder at
  24 kHz. It gives the strongest timbre transfer of the codec engines, at
  a cost of about 35% WER.

### Any-to-ONE (`rvc`)

For this engine, the target speaker identity is baked into a per-voice
model. `reference_voice` is a model path (an `.onnx` file or an HF repo ID),
not an audio file. Thousands of community-trained RVC voice models exist on
Hugging Face.

- **rvc** combines a ContentVec-768 content encoder, an RMVPE pitch
  estimator, and a VITS synthesizer. It runs at 40 kHz (v2-40k) or 48 kHz
  (v2-48k).

---

## Engine table

| Alias | Family | Sample rate | WER | INT8 | License |
|-------|--------|-------------|-----|------|---------|
| `facodec` | Factorized codec | 16 kHz | 0% | ✅ | Apache-2.0 |
| `openvoice` | Flow-matching | 22 kHz | 0% | ✅ | MIT |
| `chatterbox` | AR codec-LM | 24 kHz | 4-8% | ✅ INT8 | Apache-2.0 |
| `triaan` | Flow-matching | 16 kHz | 4% | ✅ | MIT |
| `cosyvoice` | Flow-matching | 22 kHz | 8% | ⚠ degrades | Apache-2.0 |
| `bicodec` | Factorized codec | 16 kHz | 12% | ✅ | CC BY-NC-SA 4.0 |
| `knnvc` | kNN feature-swap | 16 kHz | 12-15% | ✅ | MIT |
| `focalcodec` | kNN feature-swap | 16 kHz | 15-19% | ⚠ degrades | Apache-2.0 |
| `lscodec` | Speaker-decoupled codec | 24 kHz | ~35% | ✅ | MIT |
| `rvc` | Any-to-ONE | 40/48 kHz | 38%† | ✅ (base) | MIT |

†rvc WER depends on the model. The value shown comes from a sample community
model.

---

## Weight-license policy

voiceclonnx never redistributes model weights without redistribution rights.

- **Distributable weights** use an MIT, Apache, BSD, or CC-BY upstream
  license. voiceclonnx publishes the ONNX weights to
  `TigreGotico/voiceclonnx-<engine>` on HF Hub, and downloads them
  automatically on first use.

- **Local-only weights** carry an NC, ND, or unlicensed upstream license.
  The export script runs locally. `push_models` refuses to upload these
  weights. The adapter loads them from the local `model_dir` config key.

Full details: [converting.md](converting.md#weight-license-policy-publish-with-the-license-stated).
