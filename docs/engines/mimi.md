# Engine: mimi

**Family:** RVQ token-swap
**Sample rate:** 24 kHz
**WER:** 0%
**INT8:** available (slight quality cost)
**License:** CC BY 4.0
**Model:** [TigreGotico/voiceclonnx-mimi](https://huggingface.co/TigreGotico/voiceclonnx-mimi)

---

## Overview

Mimi is the neural audio codec powering Moshi (Kyutai, 2024). It encodes audio
with 32 Residual Vector Quantizer (RVQ) streams at 12.5 Hz and 24 kHz. Voice
conversion swaps the prosodic/speaker-style stream from the reference while
keeping the phonetic content streams from the source.

WER 0% — recommended for 24 kHz output with top intelligibility.

## How it works

The 32 RVQ streams have distinct roles:

| Stream | Distillation | VC source |
|--------|-------------|-----------|
| 0 | WavLM semantic (prosodic style) | **Reference** |
| 1–31 | Acoustic residuals (phonetic content + timbre) | **Source** |

Recipe: encode both source and reference → `[ref_stream_0 | source_streams_1-31]`
→ decode → converted waveform.

This preserves source intelligibility while injecting reference prosodic style.
Empirical testing confirmed this stream assignment is the correct one for this
codec's RVQ structure (the inverse recipe — stream 0 from source — transcribes
the reference text, not source text).

## Config / params

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `quantized` | `bool` | `False` | Load INT8 `*_q8.onnx` models. Footprint drops from ~492 MB to ~295 MB; slight quality cost. |

## Model and license

**CC BY 4.0.** Attribution required.

| File | fp32 | INT8 |
|------|------|------|
| `mimi_encoder.onnx` | 274.0 MB | 162.4 MB (−40.7%) |
| `mimi_decoder.onnx` | 217.6 MB | 132.6 MB (−39.1%) |

Total: ~492 MB fp32 / ~295 MB INT8.

## Sample rate

**24 kHz.**

## INT8 note

`quantized=True` reduces footprint from ~492 MB to ~295 MB. Slight quality
degradation expected. See [QUANTS.md](../QUANTS.md).

## WER

**0%** — perfectly intelligible on demo clips.
See [demo/VERIFICATION.md](../../demo/VERIFICATION.md).

## CLI example

```bash
voiceclonnx clone --engine mimi \
             --audio source.wav \
             --voice reference.wav \
             --out out.wav
```

## Python example

```python
from voiceclonnx import VoiceCloner

cloner = VoiceCloner(engine="mimi")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 24000

# INT8 — smaller footprint
cloner = VoiceCloner(engine="mimi", quantized=True)
```

## Troubleshooting

**Output sounds like the reference text rather than the source** — this indicates
the stream assignment is inverted (stream 0 from source instead of reference).
The current adapter uses the empirically verified correct assignment.

**First run is slow** — ~492 MB downloads from HF Hub; cached in
`~/.cache/huggingface/hub`.

## References

- Paper: [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037)
- Upstream: [kyutai/moshi](https://github.com/kyutai-labs/moshi)
