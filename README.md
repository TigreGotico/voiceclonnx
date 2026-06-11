# vconnx

Pure-ONNX multi-engine voice-cloning library.  No PyTorch at runtime.

## Scope

**vconnx is audio-to-audio only.**  It converts the voice in an existing
speech file to sound like a reference speaker.  Text-to-speech synthesis
with voice cloning (text → cloned audio) is a TTS-engine concern and is
explicitly out of scope here; see `chatterbox_onnx` or similar TTS
libraries for that path.

## Install

```bash
pip install vconnx                   # core only
pip install "vconnx[chatterbox]"     # + Chatterbox ONNX engine
```

## Quick start

```python
from vconnx import VoiceCloner

cloner = VoiceCloner(engine="chatterbox")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)  # 24000
```

## CLI

```bash
vconnx clone --engine chatterbox \
             --audio source.wav \
             --voice reference.wav \
             --out converted.wav

vconnx list   # show registered engines
```

## Engine matrix

| Alias | Package | Sample rate | Status |
|---|---|---|---|
| `chatterbox` | `vconnx[chatterbox]` → `chatterbox_onnx` | 24 kHz | Supported |
| `knnvc` | `vconnx[knnvc]` | 16 kHz | Supported |
| `openvoice` | `vconnx[openvoice]` | 22 kHz | Supported |
| `seed-vc` | — | — | Planned (see issue) |
| `rvc` | `vconnx[rvc]` | 40 kHz (v2 40k) / 48 kHz (v2 48k) | Supported |

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

### knnvc

Zero-shot any-to-any voice conversion based on
[kNN-VC](https://github.com/bshall/knn-vc) (Baas et al., Interspeech 2023).
Architecture: WavLM-Large encoder (layer 6) → k-nearest-neighbour matching
(pure numpy) → HiFi-GAN vocoder.  No autoregressive decoding; fully
non-autoregressive and CPU-friendly.

```bash
pip install "vconnx[knnvc]"
```

```python
from vconnx import VoiceCloner

cloner = VoiceCloner(engine="knnvc")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 16000
```

```bash
vconnx clone --engine knnvc \
             --audio source.wav \
             --voice reference.wav \
             --out converted.wav
```

ONNX artifacts: [`TigreGotico/vconnx-knn-vc`](https://huggingface.co/TigreGotico/vconnx-knn-vc) (public).

### openvoice

Zero-shot tone-color conversion based on
[OpenVoice v2](https://github.com/myshell-ai/OpenVoice) (myshell-ai, MIT license).
Architecture: reference encoder (mel → 256-dim tone-color embedding) + flow-based
AdaIN-conditioned converter + Griffin-Lim vocoder.  MIT-licensed weights — artifacts
distributed via the public per-engine `TigreGotico/vconnx-<engine>` repos (see the vconnx HF collection).

```bash
pip install "vconnx[openvoice]"
```

```python
from vconnx import VoiceCloner

cloner = VoiceCloner(engine="openvoice")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 22050
```

```bash
vconnx clone --engine openvoice \
             --audio source.wav \
             --voice reference.wav \
             --out converted.wav
```

ONNX artifacts: `TigreGotico/vconnx-openvoice-v2` (public).

## Adding an engine

1. Subclass `VoiceClonerBase` from `vconnx.engines.base`.
2. Implement `clone_voice(audio, reference_voice, out_path) -> str`.
3. Call `register_engine(EngineEntry(alias=..., adapter_class=...))`.
4. Add an extras group in `pyproject.toml`.
