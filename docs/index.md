# vconnx

Pure-ONNX multi-engine voice-cloning library — no PyTorch at runtime.

**Audio-to-audio only.** vconnx converts the voice in an existing speech file to
sound like a reference speaker. Text-driven synthesis (text → cloned audio) is a
TTS-engine concern and is out of scope here.

---

## Install

```bash
pip install vconnx                    # core (no engine)
pip install "vconnx[chatterbox]"      # Chatterbox AR codec-LM (default engine)
pip install "vconnx[knnvc]"           # kNN-VC — WavLM + HiFi-GAN
pip install "vconnx[openvoice]"       # OpenVoice v2 tone-color converter
```

### Extras matrix

| Extra | What it installs | Engine alias |
|---|---|---|
| `chatterbox` | `chatterbox_onnx` | `chatterbox` |
| `knnvc` | `onnxruntime`, `numpy`, `soundfile` | `knnvc` |
| `openvoice` | `onnxruntime`, `numpy`, `soundfile` | `openvoice` |
| `convert` | `torch`, `onnx`, `onnxruntime`, `onnxruntime-tools`, `huggingface_hub` | — (conversion toolchain only) |
| `test` | `pytest`, `edge-tts` | — |

The `convert` extra is only needed when exporting new ONNX artifacts from upstream
model checkpoints. See [converting.md](converting.md).

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

| Alias | Install extra | Sample rate | HF model repo | Notes |
|---|---|---|---|---|
| `chatterbox` | `vconnx[chatterbox]` | 24 kHz | [onnx-community/chatterbox-onnx](https://huggingface.co/onnx-community/chatterbox-onnx) | Default engine; AR codec-LM |
| `knnvc` | `vconnx[knnvc]` | 16 kHz | [TigreGotico/vconnx-knn-vc](https://huggingface.co/TigreGotico/vconnx-knn-vc) | WavLM + k-NN + HiFi-GAN; CPU-friendly |
| `openvoice` | `vconnx[openvoice]` | 22 kHz | [TigreGotico/vconnx-openvoice-v2](https://huggingface.co/TigreGotico/vconnx-openvoice-v2) | Tone-color transfer; Griffin-Lim vocoder |

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
