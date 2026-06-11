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
| `rvc` | — | — | Planned (see issue) |

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

ONNX artifacts: `TigreGotico/vconnx-models` (private), path `knn-vc/`.

### openvoice

Zero-shot tone-color conversion based on
[OpenVoice v2](https://github.com/myshell-ai/OpenVoice) (myshell-ai, MIT license).
Architecture: reference encoder (mel → 256-dim tone-color embedding) + flow-based
AdaIN-conditioned converter + Griffin-Lim vocoder.  MIT-licensed weights — artifacts
distributed via `TigreGotico/vconnx-models`.

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

ONNX artifacts: `TigreGotico/vconnx-models` (private), path `openvoice-v2/`.

## Adding an engine

1. Subclass `VoiceClonerBase` from `vconnx.engines.base`.
2. Implement `clone_voice(audio, reference_voice, out_path) -> str`.
3. Call `register_engine(EngineEntry(alias=..., adapter_class=...))`.
4. Add an extras group in `pyproject.toml`.
