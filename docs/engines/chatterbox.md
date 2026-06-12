# Engine: chatterbox

Chatterbox AR codec-LM (Resemble AI) — the default voiceclonnx engine.

ONNX export via [onnx-community/chatterbox-onnx](https://huggingface.co/onnx-community/chatterbox-onnx).
Voice conversion runs at **24 kHz**.

---

## Install

```bash
pip install voiceclonnx
```

No per-engine extras required. ONNX models are downloaded on first use from
[onnx-community/chatterbox-onnx](https://huggingface.co/onnx-community/chatterbox-onnx).

---

## Config keys

| Key | Type | Default | Description |
|---|---|---|---|
| `quantized` | `bool` | `False` | Accepted for API uniformity but **ignored** — no INT8 variants exist in `onnx-community/chatterbox-onnx`. Chatterbox is fp32-only. |
| `exaggeration` | `float` | `0.6` | Voice exaggeration factor. Higher values produce a more pronounced voice style; `0.5` is a neutral starting point. |

### INT8 availability

Chatterbox is **fp32-only**. The `onnx-community/chatterbox-onnx` repository
does not publish INT8 variants of `speech_encoder.onnx` or
`conditional_decoder.onnx`. Passing `quantized=True` is silently ignored —
fp32 files are always loaded. See [QUANTS.md](../QUANTS.md) for context.

---

## Usage

### Python

```python
from voiceclonnx import VoiceCloner

# Default (fp32, exaggeration=0.6)
cloner = VoiceCloner(engine="chatterbox")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 24000

# quantized=True accepted but has no effect (fp32-only engine)
cloner = VoiceCloner(engine="chatterbox", quantized=True, exaggeration=0.5)
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
```

### CLI

```bash
voiceclonnx clone --engine chatterbox \
             --audio source.wav \
             --voice reference.wav \
             --out out.wav

# With optional flags
voiceclonnx clone --engine chatterbox \
             --audio source.wav \
             --voice reference.wav \
             --out out.wav \
             --exaggeration 0.5 \
             --max-new-tokens 1024
```

---

## Model source

| Artifact | HF repo | License |
|---|---|---|
| ONNX model files | [onnx-community/chatterbox-onnx](https://huggingface.co/onnx-community/chatterbox-onnx) | Apache-2.0 |

Models are downloaded automatically on first use via `huggingface_hub`.

---

## Troubleshooting

**Quality is robotic / artefact-heavy**
Try `exaggeration=0.5` (lower value).

**Slow on CPU**
The two ONNX sessions (speech_encoder + conditional_decoder) run on CPU via
onnxruntime. On a typical laptop CPU expect 5–30 s depending on utterance length.
The VC path does not use the LLM, so it is faster than TTS with the same models.
