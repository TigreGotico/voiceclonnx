# Quantized-model comparison
<!-- agent-hint
covers: fp32 and int8 WER of each engine
source: demo/quants.json
generated: yes, by scripts/render_quants.py. Edit the generator or rerun demo/compare_quants.py
verify: python scripts/render_quants.py --check
-->

This table gives the word error rate (WER) of each engine's fp32 and int8 ONNX models. The ASR model is faster-whisper `base.en`. The source text is one sentence of 26 words, so one word error is 3.8 WER points.

> **Sample size.** Each WER comes from one sentence. A difference of one or two words is noise. The table gives no int8 recommendation until a larger run is available.

| Engine | fp32 WER | int8 WER | fp32 errors | int8 errors |
|--------|---------:|---------:|------------:|------------:|
| `bicodec` | 12% | 8% | 3 | 2 |
| `facodec` | 0% | 0% | 0 | 0 |
| `focalcodec` | 15% | 27% | 4 | 7 |
| `knnvc` | 12% | 12% | 3 | 3 |
| `openvoice` | 0% | 0% | 0 | 0 |
| `rvc` | 38% | 73% | 10 | 19 |
| `triaan` | 4% | 12% | 1 | 3 |
| `chatterbox` | 0% | 4% | 0 | 1 |
| `cosyvoice` | 8% | 100% | 2 | 26 |
| `lscodec` | 4% | 27% | 1 | 7 |

Model sizes are not in this table. `demo/compare_quants.py` holds them as fixed values, not as a measurement, and has no value for some engines.

## Notes

- **rvc**: `quantized=True` applies only to the shared base models (ContentVec-768 and RMVPE). The per-voice synthesizer is user-supplied.
- Shared numpy artifacts (codebooks, mel filterbanks, mel stats) are never quantized, because they are not ONNX models.

---
[← api](api.md) · [Home](index.md) · [converting →](converting.md)
