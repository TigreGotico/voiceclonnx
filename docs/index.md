# voiceclonnx

Pure-ONNX multi-engine voice-cloning library — no PyTorch at runtime.

**Audio-to-audio only.** voiceclonnx converts the voice in an existing speech file to
sound like a reference speaker. Text-driven synthesis (text → cloned audio) is a
TTS-engine concern and is out of scope here.

---

## Install

```bash
pip install voiceclonnx
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
from voiceclonnx import VoiceCloner

cloner = VoiceCloner(engine="chatterbox")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 24000
```

---

## Engine matrix

All engines are included in the base `pip install voiceclonnx`.

| Alias | Sample rate | HF model repo | INT8 | Notes |
|---|---|---|---|---|
| `chatterbox` | 24 kHz | [onnx-community/chatterbox-onnx](https://huggingface.co/onnx-community/chatterbox-onnx) | fp32 only | AR codec-LM; VC path only |
| `facodec` | 16 kHz | [TigreGotico/voiceclonnx-facodec](https://huggingface.co/TigreGotico/voiceclonnx-facodec) | ✅ | Factorised VQ; timbre-swap VC (NaturalSpeech 3) |
| `focalcodec` | 16 kHz | [TigreGotico/voiceclonnx-focalcodec](https://huggingface.co/TigreGotico/voiceclonnx-focalcodec) | ✅ | WavLM + cosine kNN + Vocos ISTFT |
| `freevc` | 16 kHz | [TigreGotico/voiceclonnx-freevc](https://huggingface.co/TigreGotico/voiceclonnx-freevc) | ✅ | WavLM + GE2E + VITS decoder |
| `knnvc` | 16 kHz | [TigreGotico/voiceclonnx-knn-vc](https://huggingface.co/TigreGotico/voiceclonnx-knn-vc) | ✅ | WavLM + L2-kNN + HiFi-GAN |
| `mimi` | 24 kHz | [TigreGotico/voiceclonnx-mimi](https://huggingface.co/TigreGotico/voiceclonnx-mimi) | ✅ | Moshi codec; encoder-decoder token swap |
| `openvoice` | 22 kHz | [TigreGotico/voiceclonnx-openvoice-v2](https://huggingface.co/TigreGotico/voiceclonnx-openvoice-v2) | ✅ | Tone-color transfer |
| `bicodec` | 16 kHz | [TigreGotico/voiceclonnx-bicodec](https://huggingface.co/TigreGotico/voiceclonnx-bicodec) | ✅ | Semantic + global token factorization (SparkTTS) |
| `rvc` | 40/48 kHz | [TigreGotico/voiceclonnx-rvc](https://huggingface.co/TigreGotico/voiceclonnx-rvc) | ✅ (base models) | Any-to-ONE; voice baked into model |
| `speechtokenizer` | 16 kHz | [TigreGotico/voiceclonnx-speechtokenizer](https://huggingface.co/TigreGotico/voiceclonnx-speechtokenizer) | ✅ | RVQ token swap VC |
| `triaan` | 16 kHz | [TigreGotico/voiceclonnx-triaan-vc](https://huggingface.co/TigreGotico/voiceclonnx-triaan-vc) | ✅ | CPC + TriAAN decoder + PWG |

### Quantized models

Every engine accepts `quantized: bool = False`. When `True`, the adapter loads
the `*_q8.onnx` INT8 variants, which are 45–75% smaller on disk and faster on
CPU at a small quality cost. See [QUANTS.md](QUANTS.md) for the fp32 vs INT8
WER and size comparison across all engines.

**Exception**: `chatterbox` is fp32-only — `onnx-community/chatterbox-onnx`
does not publish INT8 variants. The `quantized=True` flag is accepted (uniform
API) but ignored.

---

## CLI

```bash
# Convert a WAV file
voiceclonnx clone --engine chatterbox \
             --audio source.wav \
             --voice reference.wav \
             --out converted.wav

# List registered engines
voiceclonnx list
```

---

## Documentation

- [API reference](api.md) — `VoiceCloner` facade, `VoiceClonerBase`, registry functions
- [Engine guides](engines/) — per-engine config keys, model sources, CLI examples, troubleshooting
  - [chatterbox](engines/chatterbox.md)
  - [facodec](engines/facodec.md)
  - [knnvc](engines/knnvc.md)
  - [openvoice](engines/openvoice.md)
- [Converting models](converting.md) — export / parity / quantize / push toolchain

---

## Weight-license policy

voiceclonnx never redistributes model weights it has no right to. Engines fall into two
classes decided per engine:

- **Distributable** — upstream license permits redistribution (MIT / Apache / BSD /
  CC-BY). Converted ONNX models are published to the public
  `TigreGotico/voiceclonnx-<engine>` HF repo with `LICENSE` and `PROVENANCE.md`; adapters
  download them automatically on first use.
- **Local-only weights** — upstream license does not permit redistribution (NC/ND
  variants, unlicensed repos). The conversion script runs on your own machine,
  `write_manifest(..., distributable=False)` marks the output, `push_models` refuses
  to upload, and the adapter loads from the local path supplied via the `model_dir`
  config key.

See [converting.md](converting.md) for the full policy details and conversion
workflow.
