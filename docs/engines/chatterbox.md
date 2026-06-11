# Engine: chatterbox

Chatterbox AR codec-LM (Resemble AI) — the default vconnx engine.

ONNX export via [onnx-community/chatterbox-onnx](https://huggingface.co/onnx-community/chatterbox-onnx).
Voice conversion runs at **24 kHz**.

---

## Install

```bash
pip install "vconnx[chatterbox]"
```

This pulls in the `chatterbox_onnx` package which handles model download and ONNX
session management internally.

---

## Config keys

| Key | Type | Default | Description |
|---|---|---|---|
| `quantized` | `bool` | `True` | Use the Q4-quantized language model. Set to `False` for full-precision (much larger and slower on CPU). |
| `exaggeration` | `float` | `0.6` | Voice exaggeration factor. Higher values produce a more pronounced voice style; `0.5` is a neutral starting point. |
| `max_new_tokens` | `int` | `512` | Maximum speech tokens to generate per call. Increase for long utterances. |

---

## Usage

### Python

```python
from vconnx import VoiceCloner

# Default (quantized=True, exaggeration=0.6)
cloner = VoiceCloner(engine="chatterbox")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 24000

# Full-precision with neutral exaggeration
cloner = VoiceCloner(engine="chatterbox", quantized=False, exaggeration=0.5)
out = cloner.clone_voice("source.wav", "reference.wav", "out_fp32.wav")
```

### CLI

```bash
vconnx clone --engine chatterbox \
             --audio source.wav \
             --voice reference.wav \
             --out out.wav

# With optional flags
vconnx clone --engine chatterbox \
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
| `chatterbox_onnx` model files | [onnx-community/chatterbox-onnx](https://huggingface.co/onnx-community/chatterbox-onnx) | See upstream repo |

Model download is handled by `chatterbox_onnx` internally — no manual step needed.

---

## Troubleshooting

**`ImportError: chatterbox_onnx is required`**
Install the extras group: `pip install "vconnx[chatterbox]"`.

**Quality is robotic / artefact-heavy**
Try `exaggeration=0.5` (lower) or increase `max_new_tokens` if the output is cut off.

**Slow on CPU**
The default `quantized=True` uses a Q4 LM which is significantly faster than
`quantized=False`. On a typical laptop CPU, expect 10–40 s per conversion depending
on utterance length.
