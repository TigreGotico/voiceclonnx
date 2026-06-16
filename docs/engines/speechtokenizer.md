# Engine: speechtokenizer

**Family:** RVQ token-swap
**Sample rate:** 16 kHz
**WER:** 4–12%
**INT8:** available (slight quality cost)
**License:** Apache-2.0
**Model:** [TigreGotico/voiceclonnx-speechtokenizer](https://huggingface.co/TigreGotico/voiceclonnx-speechtokenizer)

---

## Overview

SpeechTokenizer (Zhang et al., ACL 2024) is an EnCodec-style hierarchical RVQ
codec trained at 16 kHz with 8 quantizers at 50 Hz. The first quantizer (RVQ-1)
is semantically distilled via HuBERT and captures linguistic content; quantizers
RVQ-2 through RVQ-8 carry speaker-dependent acoustic residuals (timbre, prosody
fine-grain, recording conditions). Voice conversion swaps RVQ-1 from source with
RVQ-2..8 from reference.

## How it works

```
source audio  → encode → [S₁ | S₂ … S₈]
reference audio → encode → [R₁ | R₂ … R₈]

mixed codes = [S₁ | R₂ … R₈]   ← source content + reference timbre

mixed codes → decode → converted audio
```

The token swap is pure numpy — no additional ONNX graph required. When source
and reference lengths differ, reference tokens are truncated or tiled to match
the source sequence length.

## Config / params

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `quantized` | `bool` | `False` | Load INT8 `*_q8.onnx` models. Reduces footprint from ~485 MB to ~150 MB; slight quality cost. |

## Model and license

**Apache-2.0.** Paper: [arXiv:2308.16692](https://arxiv.org/abs/2308.16692).

| File | fp32 | INT8 |
|------|------|------|
| `encoder.onnx` | 318.4 MB | 80.0 MB (−74.9%) |
| `decoder.onnx` | 166.3 MB | 70.4 MB (−57.7%) |

Total: ~485 MB fp32 / ~150 MB INT8.

## Sample rate

**16 kHz.**

## INT8 note

`quantized=True` reduces footprint from ~485 MB to ~150 MB. Slight quality
degradation expected. See [QUANTS.md](../QUANTS.md).

## WER

**4–12%** — measured with faster-whisper `base.en` on demo clips.
See [demo/VERIFICATION.md](../../demo/VERIFICATION.md).

## CLI example

```bash
voiceclonnx clone --engine speechtokenizer \
             --audio source.wav \
             --voice reference.wav \
             --out out.wav
```

## Python example

```python
from voiceclonnx import VoiceCloner

cloner = VoiceCloner(engine="speechtokenizer")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 16000

# INT8 — smaller footprint
cloner = VoiceCloner(engine="speechtokenizer", quantized=True)
```

## Troubleshooting

**Voice timbre not transferring well** — use a longer reference clip (5–10 s)
so RVQ-2..8 tokens cover more speaker variation.

**First run is slow** — ~485 MB downloads from HF Hub on first use; cached in
`~/.cache/huggingface/hub`.

## References

- Paper: [SpeechTokenizer: Unified Speech Tokenizer for Speech Language Models](https://arxiv.org/abs/2308.16692)
- Upstream: [ZhangXInFD/SpeechTokenizer](https://github.com/ZhangXInFD/SpeechTokenizer)
