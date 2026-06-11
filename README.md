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
| `seed-vc` | — | — | Planned (see issue) |
| `openvoice` | — | — | Planned (see issue) |
| `knn-vc` | — | — | Planned (see issue) |
| `rvc` | — | — | Planned (see issue) |

## OPM plugin

When `ovos-plugin-manager` ships the `VoiceClonePlugin` base class, the
bundled plugin activates automatically via the `opm.vc` entry-point group:

| Entry-point name | Class |
|---|---|
| `ovos-vc-plugin-chatterbox-onnx` | `vconnx.opm.ChatterboxVCPlugin` |

Until OPM ships that family, import `VoiceCloner` directly.

## Adding an engine

1. Subclass `VoiceClonerBase` from `vconnx.engines.base`.
2. Implement `clone_voice(audio, reference_voice, out_path) -> str`.
3. Call `register_engine(EngineEntry(alias=..., adapter_class=...))`.
4. Add an extras group in `pyproject.toml`.

## Attribution

Developed by [TigreGotico](https://github.com/TigreGotico) for
[OpenVoiceOS](https://github.com/OpenVoiceOS), funded by the
[NGI0 Commons Fund](https://nlnet.nl/commonsfund/) / NLnet grant 101135429.
