# SpeechTokenizer engine

**Alias:** `speechtokenizer`
**Sample rate:** 16 kHz
**License:** Apache-2.0
**Model repo:** [TigreGotico/vconnx-speechtokenizer](https://huggingface.co/TigreGotico/vconnx-speechtokenizer)
**Paper:** Zhang et al., ACL 2024 — [arXiv:2308.16692](https://arxiv.org/abs/2308.16692)

## Architecture

SpeechTokenizer is an EnCodec-style hierarchical RVQ codec trained at 16 kHz with 8 quantizers
at 50 Hz (320-sample hop).  The first quantizer (RVQ-1) is semantically distilled via HuBERT
and captures linguistic content.  Quantizers RVQ-2 through RVQ-8 carry speaker-dependent acoustic
residuals — timbre, prosody fine-grain, and recording conditions.

## Voice-conversion recipe

```
source audio  ──► encode ──► [S₁ | S₂ … S₈]
reference audio ──► encode ──► [R₁ | R₂ … R₈]

mixed codes = [S₁ | R₂ … R₈]   ← source content + reference timbre

mixed codes ──► decode ──► converted audio
```

1. Both source and reference are encoded with `encoder.onnx` to obtain 8-layer token sequences.
2. The RVQ-1 (content) tokens from source are kept; RVQ-2 through RVQ-8 tokens are taken from
   the reference.  When source and reference lengths differ, reference tokens are truncated or
   tiled to match the source sequence length.
3. The mixed token sequence is decoded with `decoder.onnx` to produce the converted waveform.

The token swap is pure numpy — no additional ONNX graph is required.

**Layer split justification:** RVQ-1 is the semantic distillation target (trained to match HuBERT
representations), making it a reliable proxy for linguistic content.  Replacing RVQ-2..8 with
reference tokens transfers timbre while preserving the source transcription.  Swapping
additional layers towards the content layer reduces speaker similarity; the RVQ-1/RVQ-2..8
boundary gives the best measured intelligibility vs. timbre-transfer tradeoff.

## ONNX components

| File | Role | Size |
|---|---|---|
| `encoder.onnx` | waveform (1, 1, N) → codes (8, 1, T) | 318.4 MB |
| `encoder_q8.onnx` | INT8 quantized encoder | 80.0 MB (−74.9 %) |
| `decoder.onnx` | codes (8, 1, T) → waveform (1, 1, N) | 166.3 MB |
| `decoder_q8.onnx` | INT8 quantized decoder | 70.4 MB (−57.7 %) |

## Parity results

| Component | Metric | Value | Pass |
|---|---|---|---|
| Encoder | exact integer match (codes) | True | ✓ |
| Decoder | max abs Δ (waveform) | 1.53e-08 | ✓ |
| Decoder | mean abs Δ (waveform) | 2.89e-09 | ✓ |

## Usage

```python
from vconnx import VoiceCloner

cloner = VoiceCloner(engine="speechtokenizer")
out = cloner.clone_voice("source.wav", "reference.wav", "converted.wav")
```

### Constructor options

| Parameter | Default | Description |
|---|---|---|
| `quantized` | `False` | **Not supported** — raises `NotImplementedError`. The INT8 exports in `TigreGotico/vconnx-speechtokenizer` use a different interface incompatible with this adapter's pipeline. Always use fp32. |
| `content_layers` | `1` | Number of leading RVQ layers treated as content (default 1 = RVQ-1 only) |

## CPU performance

Approximate RTF on a modern CPU: 0.2–0.4x (similar to EnCodec 24kHz, ~74M encoder params).
The fp32 encoder is the bottleneck; the INT8 variant reduces it by ~4×.

## References

- GitHub: <https://github.com/ZhangXInFD/SpeechTokenizer>
- Paper: <https://arxiv.org/abs/2308.16692>
- HF weights: <https://huggingface.co/fnlp/SpeechTokenizer>
- vconnx ONNX: <https://huggingface.co/TigreGotico/vconnx-speechtokenizer>
