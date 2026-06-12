# Mimi engine

Mimi is the neural audio codec powering Moshi (Kyutai, 2024).  It uses 32
residual-vector-quantizer (RVQ) streams at 12.5 Hz / 24 kHz.

## Stream structure

| Stream | Distillation | Role in VC |
|---|---|---|
| 0 | WavLM semantic | Speaker prosodic style (from **reference**) |
| 1–31 | Acoustic residuals | Phonetic content + timbre (from **source**) |

## VC recipe (empirically verified)

Initial theory (from the upstream paper) suggested stream 0 carries
speaker-lean content tokens.  Empirical tests with actual ONNX inference show
the opposite for this codec's RVQ structure: streams 1–31 carry the phonetic
sequence that determines what is said, while stream 0 carries higher-level
prosodic/speaker-style information.

**Working recipe:** stream 0 from reference + streams 1–31 from source.

This preserves source intelligibility while injecting reference prosodic style.
Output WER against source text: **0% for both demo clips** (aria and sonia
reference voices).

### What was tried

| Recipe | Result |
|---|---|
| stream 0 = source, 1-31 = reference | Reference text transcribed (88% WER) |
| stream 0 = reference, 1-31 = source | Source text transcribed (**0% WER**) ✅ |

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `quantized` | `False` | Use INT8 quantized models |

## Model artifacts

HF repo: [TigreGotico/vconnx-mimi](https://huggingface.co/TigreGotico/vconnx-mimi)

| File | Size |
|---|---|
| `mimi_encoder.onnx` | 274.0 MB |
| `mimi_encoder_q8.onnx` | 162.4 MB (−40.7%) |
| `mimi_decoder.onnx` | 217.6 MB |
| `mimi_decoder_q8.onnx` | 132.6 MB (−39.1%) |

## Parity results (fp32, torch vs ONNX)

| Component | max_abs Δ | mean_abs Δ | Verdict |
|---|---|---|---|
| Encoder (discrete codes) | exact match (int64) | — | PASS |
| Decoder waveform (1s) | 5.4e-6 | 3.4e-7 | PASS |
| Decoder waveform (3s) | 4.3e-6 | 3.3e-7 | PASS |
| Decoder waveform (5.6s) | 3.5e-6 | 3.2e-7 | PASS |

## Export notes

See [`conversion/export_mimi.py`](../../conversion/export_mimi.py) for the full
export script.  Key deviations from the standard export recipe:

1. **sdpa_mask bug** — transformers 5.5.0 passes a scalar Tensor as `q_length`
   to `sdpa_mask`, causing `IndexError: tuple index out of range`.
   Patched by extracting the integer value when a 0-d Tensor is detected.
2. **find_packed_sequence_indices** — calls `torch.diff(prepend=…)` which the
   legacy TorchScript ONNX exporter cannot lower.  Patched to return `None`
   (correct for single-sequence inference).
3. **MimiEuclideanCodebook.quantize** — uses `torch.cdist` with dynamic shapes
   which cannot be traced.  Replaced with manual `‖h‖² + ‖e‖² − 2h·eᵀ`.
4. **Attention mask** — the sliding-window causal mask is computed as a
   constant of shape `(B, 1, T, T)` during tracing, causing a broadcast error
   for different-length inputs.  Replaced with full bidirectional attention
   (no mask) which is correct for offline/batch VC.
5. **Dynamic input length** — fully supported; tested at 1s, 3s, and 5.6s.

## License

Weights: **CC BY 4.0** (Kyutai).
Attribution: <https://huggingface.co/kyutai/mimi>
