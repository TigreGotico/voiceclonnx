# vconnx

Pure-ONNX multi-engine voice-cloning library — no PyTorch at runtime.

**Audio-to-audio only.** vconnx converts the voice in an existing speech file to
sound like a reference speaker. Text-driven synthesis (text → cloned audio) is a
TTS-engine concern and is out of scope here.

---

## Install

```bash
pip install vconnx
```

One command installs **every engine**. Core dependencies:
`onnxruntime`, `numpy`, `soundfile`, `huggingface_hub`. ONNX models are
downloaded on first use from Hugging Face Hub.

For conversion / export tooling and tests only:

| Extra | What it installs | Use |
|---|---|---|
| `convert` | `torch`, `onnx`, `transformers`, `librosa`, `onnxruntime-tools`, `huggingface_hub` | Export new ONNX models from upstream checkpoints — never needed for inference |
| `bench` | `faster-whisper`, `edge-tts` | Benchmark/demo generation |
| `test` | `pytest`, `faster-whisper`, `edge-tts` | Test suite |

See [converting.md](converting.md) for the full conversion toolchain.

---

## Quick start

```python
from vconnx import VoiceCloner

cloner = VoiceCloner(engine="chatterbox")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 24000
```

---

## Engine matrix

All engines are included in the base `pip install vconnx`.

| Alias | Sample rate | HF model repo | Notes |
|---|---|---|---|
| `chatterbox` | 24 kHz | [onnx-community/chatterbox-onnx](https://huggingface.co/onnx-community/chatterbox-onnx) | AR codec-LM; VC path only |
| `focalcodec` | 16 kHz | [TigreGotico/vconnx-focalcodec](https://huggingface.co/TigreGotico/vconnx-focalcodec) | WavLM + cosine kNN + Vocos ISTFT |
| `freevc` | 16 kHz | [TigreGotico/vconnx-freevc](https://huggingface.co/TigreGotico/vconnx-freevc) | WavLM + GE2E + VITS decoder |
| `knnvc` | 16 kHz | [TigreGotico/vconnx-knn-vc](https://huggingface.co/TigreGotico/vconnx-knn-vc) | WavLM + L2-kNN + HiFi-GAN |
| `openvoice` | 22 kHz | [TigreGotico/vconnx-openvoice-v2](https://huggingface.co/TigreGotico/vconnx-openvoice-v2) | Tone-color transfer |
| `rvc` | 40/48 kHz | [TigreGotico/vconnx-rvc](https://huggingface.co/TigreGotico/vconnx-rvc) | Any-to-ONE; voice baked into model |
| `triaan` | 16 kHz | [TigreGotico/vconnx-triaan-vc](https://huggingface.co/TigreGotico/vconnx-triaan-vc) | CPC + TriAAN decoder + PWG |

---

## CLI

```bash
# Convert a WAV file
vconnx clone --engine chatterbox \
             --audio source.wav \
             --voice reference.wav \
             --out converted.wav

# List registered engines
vconnx list
```

---

## Documentation

- [API reference](api.md) — `VoiceCloner` facade, `VoiceClonerBase`, registry functions
- [Engine guides](engines/) — per-engine config keys, model sources, CLI examples, troubleshooting
  - [chatterbox](engines/chatterbox.md)
  - [knnvc](engines/knnvc.md)
  - [openvoice](engines/openvoice.md)
- [Converting models](converting.md) — export / parity / quantize / push toolchain

---

## Weight-license policy

vconnx never redistributes model weights it has no right to. Engines fall into two
classes decided per engine:

- **Distributable** — upstream license permits redistribution (MIT / Apache / BSD /
  CC-BY). Converted ONNX models are published to the public
  `TigreGotico/vconnx-<engine>` HF repo with `LICENSE` and `PROVENANCE.md`; adapters
  download them automatically on first use.
- **Local-only weights** — upstream license does not permit redistribution (NC/ND
  variants, unlicensed repos). The conversion script runs on your own machine,
  `write_manifest(..., distributable=False)` marks the output, `push_models` refuses
  to upload, and the adapter loads from the local path supplied via the `model_dir`
  config key.

See [converting.md](converting.md) for the full policy details and conversion
workflow.
