# voiceclonnx

![PyPI](https://img.shields.io/pypi/v/voiceclonnx)
![Python](https://img.shields.io/pypi/pyversions/voiceclonnx)
![License](https://img.shields.io/pypi/l/voiceclonnx)

Pure-ONNX multi-engine voice-cloning library — no PyTorch at runtime.

**Audio-to-audio only.** voiceclonnx converts the voice in an existing speech file to
sound like a reference speaker. Text-driven synthesis (text → cloned audio) is a
TTS-engine concern and is explicitly out of scope.

---

## Install

```bash
pip install voiceclonnx
```

That single command installs **every engine** — no per-engine extras required.
Core dependencies: `onnxruntime`, `numpy`, `soundfile`, `huggingface_hub`.
ONNX models are downloaded on first use from Hugging Face Hub.

For model conversion / export tooling only:

```bash
pip install "voiceclonnx[convert]"   # torch, onnx, transformers, librosa (conversion only)
pip install "voiceclonnx[test]"      # pytest, faster-whisper, edge-tts (testing)
```

---

## Quick start

```python
from voiceclonnx import VoiceCloner

cloner = VoiceCloner(engine="chatterbox")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 24000
```

---

## CLI

```bash
# Convert a WAV file
voiceclonnx clone --engine chatterbox \
             --audio source.wav \
             --voice reference.wav \
             --out converted.wav

# With optional engine flags
voiceclonnx clone --engine chatterbox \
             --audio source.wav \
             --voice reference.wav \
             --out converted.wav \
             --exaggeration 0.5 \
             --max-new-tokens 1024

# List registered engines
voiceclonnx list
```

---

## Engine matrix

All engines ship with `pip install voiceclonnx` — no per-engine extras needed.

| Alias | Sample rate | Model repo | License |
|---|---|---|---|
| `chatterbox` | 24 kHz | [onnx-community/chatterbox-onnx](https://huggingface.co/onnx-community/chatterbox-onnx) | Apache-2.0 |
| `facodec` | 16 kHz | [TigreGotico/voiceclonnx-facodec](https://huggingface.co/TigreGotico/voiceclonnx-facodec) | Apache-2.0 |
| `focalcodec` | 16 kHz | [TigreGotico/voiceclonnx-focalcodec](https://huggingface.co/TigreGotico/voiceclonnx-focalcodec) | Apache-2.0 |
| `freevc` | 16 kHz | [TigreGotico/voiceclonnx-freevc](https://huggingface.co/TigreGotico/voiceclonnx-freevc) | MIT |
| `knnvc` | 16 kHz | [TigreGotico/voiceclonnx-knn-vc](https://huggingface.co/TigreGotico/voiceclonnx-knn-vc) | MIT |
| `mimi` | 24 kHz | [TigreGotico/voiceclonnx-mimi](https://huggingface.co/TigreGotico/voiceclonnx-mimi) | CC BY 4.0 |
| `openvoice` | 22 kHz | [TigreGotico/voiceclonnx-openvoice-v2](https://huggingface.co/TigreGotico/voiceclonnx-openvoice-v2) | MIT |
| `rvc` | 40/48 kHz | [TigreGotico/voiceclonnx-rvc](https://huggingface.co/TigreGotico/voiceclonnx-rvc) | MIT |
| `speechtokenizer` | 16 kHz | [TigreGotico/voiceclonnx-speechtokenizer](https://huggingface.co/TigreGotico/voiceclonnx-speechtokenizer) | Apache-2.0 |
| `bicodec` | 16 kHz | [TigreGotico/voiceclonnx-bicodec](https://huggingface.co/TigreGotico/voiceclonnx-bicodec) | CC BY-NC-SA 4.0 |
| `quickvc` | 16 kHz | [TigreGotico/voiceclonnx-quickvc](https://huggingface.co/TigreGotico/voiceclonnx-quickvc) | MIT |
| `triaan` | 16 kHz | [TigreGotico/voiceclonnx-triaan-vc](https://huggingface.co/TigreGotico/voiceclonnx-triaan-vc) | MIT |

### bicodec

Zero-shot any-to-any voice conversion via explicit semantic / global token
factorization (SparkAudio/Spark-TTS, 2025, Apache-2.0 code, CC BY-NC-SA 4.0
weights).

Architecture:
- **Semantic tokens** (content): Wav2Vec2-XLSR-53 (hidden layers 11, 14, 16
  averaged) → convolutional encoder → FactorizedVQ → (1, T) int64
- **Global tokens** (speaker): mel-spectrogram (128-bin, Slaney) → ECAPA-TDNN
  + Perceiver resampler → FSQ → (1, 1, 32) int32 (fixed-length per utterance)

Voice conversion is a direct token swap: source semantic tokens + reference
global tokens → decoder → waveform.  No auto-regressive LM, single forward
pass per segment.

```python
from voiceclonnx import VoiceCloner

cloner = VoiceCloner(engine="bicodec")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 16000
```

ONNX artifacts: [`TigreGotico/voiceclonnx-bicodec`](https://huggingface.co/TigreGotico/voiceclonnx-bicodec)
(public, **CC BY-NC-SA 4.0 — non-commercial use only**).

See [docs/engines/bicodec.md](docs/engines/bicodec.md) for config keys,
parity results, and troubleshooting.

---

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

```python
from voiceclonnx import VoiceCloner

# reference_voice = path to RVC .onnx voice model, NOT audio
cloner = VoiceCloner(engine="rvc")
out = cloner.clone_voice("source.wav", "/path/to/myvoice.onnx", "out.wav")
print(cloner.sample_rate)   # 40000 (v2 40k) or 48000 (v2 48k)
```

```bash
voiceclonnx clone --engine rvc \
             --audio source.wav \
             --voice /path/to/myvoice.onnx \
             --out converted.wav
```

Base ONNX artifacts (ContentVec + RMVPE): [`TigreGotico/voiceclonnx-rvc`](https://huggingface.co/TigreGotico/voiceclonnx-rvc) (public, MIT).

Convert a community ``.pth`` voice model to ONNX:

```bash
python -m conversion.convert_rvc_model myvoice.pth myvoice.onnx
```

---

## Quantized models

All engines except `chatterbox` support `quantized=True`, which loads the
`*_q8.onnx` INT8 variants — 45–75% smaller on disk and faster on CPU:

```python
cloner = VoiceCloner(engine="knnvc", quantized=True)
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
```

See [docs/QUANTS.md](docs/QUANTS.md) for the full fp32 vs INT8 WER and size
comparison across all engines, including which are recommended in INT8 mode.

**chatterbox** is fp32-only: `onnx-community/chatterbox-onnx` does not publish
INT8 variants. `quantized=True` is accepted for API uniformity but silently
ignored.

---

## Adding an engine

1. Subclass `VoiceClonerBase` from `voiceclonnx.engines.base`.
2. Implement `clone_voice(audio, reference_voice, out_path) -> str`.
3. Call `register_engine(EngineEntry(alias=..., adapter_class=...))`.
4. Add the auto-import to `voiceclonnx/__init__.py`.

See [docs/api.md](docs/api.md) for the full API reference.

---

## Documentation

- [docs/index.md](docs/index.md) — overview, install matrix, engine table
- [docs/QUANTS.md](docs/QUANTS.md) — fp32 vs INT8 WER and size comparison across all engines
- [docs/api.md](docs/api.md) — VoiceCloner facade, VoiceClonerBase, registry
- [docs/engines/chatterbox.md](docs/engines/chatterbox.md) — config keys, troubleshooting
- [docs/engines/freevc.md](docs/engines/freevc.md) — config keys, model sizes, WavLM note, troubleshooting
- [docs/engines/knnvc.md](docs/engines/knnvc.md) — config keys, model sizes, troubleshooting
- [docs/engines/bicodec.md](docs/engines/bicodec.md) — config keys, parity, ONNX sizes, export notes
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
