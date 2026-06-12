# vconnx

![PyPI](https://img.shields.io/pypi/v/vconnx)
![Python](https://img.shields.io/pypi/pyversions/vconnx)
![License](https://img.shields.io/pypi/l/vconnx)

Pure-ONNX multi-engine voice-cloning library — no PyTorch at runtime.

**Audio-to-audio only.** vconnx converts the voice in an existing speech file to
sound like a reference speaker. Text-driven synthesis (text → cloned audio) is a
TTS-engine concern and is explicitly out of scope; see `chatterbox_onnx` or similar
libraries for that.

---

## Install

```bash
pip install vconnx                    # core (no engine)
pip install "vconnx[chatterbox]"      # Chatterbox AR codec-LM (default)
pip install "vconnx[focalcodec]"      # FocalCodec — WavLM + kNN cosine + Vocos
pip install "vconnx[freevc]"          # FreeVC — WavLM + VITS decoder
pip install "vconnx[knnvc]"           # kNN-VC — WavLM + HiFi-GAN
pip install "vconnx[openvoice]"       # OpenVoice v2 tone-color converter
pip install "vconnx[rvc]"             # RVC any-to-one (community voice models)
pip install "vconnx[triaan]"          # TriAAN-VC — CPC + TriAAN decoder + PWG
```

---

## Quick start

```python
from vconnx import VoiceCloner

cloner = VoiceCloner(engine="chatterbox")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 24000
```

---

## CLI

```bash
# Convert a WAV file
vconnx clone --engine chatterbox \
             --audio source.wav \
             --voice reference.wav \
             --out converted.wav

# With optional engine flags
vconnx clone --engine chatterbox \
             --audio source.wav \
             --voice reference.wav \
             --out converted.wav \
             --exaggeration 0.5 \
             --max-new-tokens 1024

# List registered engines
vconnx list
```

---

## Engine matrix

| Alias | Install extra | Sample rate | Model repo | License |
|---|---|---|---|---|
| `chatterbox` | `vconnx[chatterbox]` | 24 kHz | [onnx-community/chatterbox-onnx](https://huggingface.co/onnx-community/chatterbox-onnx) | See upstream |
| `focalcodec` | `vconnx[focalcodec]` | 16 kHz | [TigreGotico/vconnx-focalcodec](https://huggingface.co/TigreGotico/vconnx-focalcodec) | Apache-2.0 |
| `freevc` | `vconnx[freevc]` | 16 kHz | [TigreGotico/vconnx-freevc](https://huggingface.co/TigreGotico/vconnx-freevc) | MIT |
| `knnvc` | `vconnx[knnvc]` | 16 kHz | [TigreGotico/vconnx-knn-vc](https://huggingface.co/TigreGotico/vconnx-knn-vc) | MIT |
| `openvoice` | `vconnx[openvoice]` | 22 kHz | [TigreGotico/vconnx-openvoice-v2](https://huggingface.co/TigreGotico/vconnx-openvoice-v2) | MIT |
| `rvc` | `vconnx[rvc]` | 40/48 kHz (per voice model) | [TigreGotico/vconnx-rvc](https://huggingface.co/TigreGotico/vconnx-rvc) | MIT |
| `triaan` | `vconnx[triaan]` | 16 kHz | [TigreGotico/vconnx-triaan-vc](https://huggingface.co/TigreGotico/vconnx-triaan-vc) | MIT |

### rvc

Any-to-ONE voice conversion based on
[RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI)
(RVC-Project, MIT license).  Architecture: ContentVec-768 encoder → RMVPE
pitch estimator → VITS-based synthesizer (``net_g``).  **Any-to-ONE**: the
target speaker is baked into the voice model; thousands of community-trained
voices are available on Hugging Face.

**Semantics note:** ``reference_voice`` is the **path to an RVC voice model**
(local ``.onnx`` or HF repo ID ``owner/repo``), not a reference audio file.
The target speaker identity is encoded in the model weights.  Use
``default_model`` in the constructor to set a fallback.

```bash
pip install "vconnx[rvc]"
```

```python
from vconnx import VoiceCloner

# reference_voice = path to RVC .onnx voice model, NOT audio
cloner = VoiceCloner(engine="rvc")
out = cloner.clone_voice("source.wav", "/path/to/myvoice.onnx", "out.wav")
print(cloner.sample_rate)   # 40000 (v2 40k) or 48000 (v2 48k)
```

```bash
vconnx clone --engine rvc \
             --audio source.wav \
             --voice /path/to/myvoice.onnx \
             --out converted.wav
```

Base ONNX artifacts (ContentVec + RMVPE): [`TigreGotico/vconnx-rvc`](https://huggingface.co/TigreGotico/vconnx-rvc) (public, MIT).

Convert a community ``.pth`` voice model to ONNX:

```bash
python -m conversion.convert_rvc_model myvoice.pth myvoice.onnx
```

---

## Adding an engine

1. Subclass `VoiceClonerBase` from `vconnx.engines.base`.
2. Implement `clone_voice(audio, reference_voice, out_path) -> str`.
3. Call `register_engine(EngineEntry(alias=..., adapter_class=...))`.
4. Add an extras group in `pyproject.toml`.

See [docs/api.md](docs/api.md) for the full API reference.

---

## Documentation

- [docs/index.md](docs/index.md) — overview, install matrix, engine table
- [docs/api.md](docs/api.md) — VoiceCloner facade, VoiceClonerBase, registry
- [docs/engines/chatterbox.md](docs/engines/chatterbox.md) — config keys, troubleshooting
- [docs/engines/freevc.md](docs/engines/freevc.md) — config keys, model sizes, WavLM note, troubleshooting
- [docs/engines/knnvc.md](docs/engines/knnvc.md) — config keys, model sizes, troubleshooting
- [docs/engines/openvoice.md](docs/engines/openvoice.md) — config keys, mel params, troubleshooting
- [docs/converting.md](docs/converting.md) — ONNX export / parity / quantize / push toolchain

## Examples

- [examples/basic_clone.py](examples/basic_clone.py) — knnvc demo with edge-tts
- [examples/cli_batch.sh](examples/cli_batch.sh) — batch convert a folder via CLI
- [examples/quantized_low_memory.py](examples/quantized_low_memory.py) — INT8 vs fp32 comparison
- [examples/local_only_engine.md](examples/local_only_engine.md) — local-only weights walkthrough

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

Model weights are governed by their upstream licenses. See
[docs/converting.md](docs/converting.md) for the weight-license policy (distributable
vs local-only).
