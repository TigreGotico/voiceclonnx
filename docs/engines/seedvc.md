# Engine: seedvc

Seed-VC (Plachtaa, 2024) — zero-shot any-to-any voice conversion using a
Transformer DiT (Diffusion Transformer) with flow-matching ODE decoder and
BigVGAN vocoder.  The upstream repository was archived on 2025-11-21; the
export pins commit `51383efd921027683c89e5348211d93ff12ac2a8`.

Output sample rate: **22050 Hz**.

---

## Architecture

1. **`whisper_encoder.onnx`** — Whisper-small encoder (openai/whisper-small):
   log-mel `(1, 128, 3000)` float32 → hidden states `(1, T_enc, 768)` float32.
   Processes up to 30 s of 16 kHz audio.  The encoder subsamples the input 2×
   (two convolutional strides), so `T_enc ≈ actual_mel_frames / 2`.

2. **`campplus.onnx`** — CAMPPlus speaker encoder (funasr/campplus):
   Kaldi 80-bin fbank `(1, T_fbank, 80)` → L2-normalized 192-d speaker
   embedding `(1, 192)`.  Operates at 16 kHz.

3. **`lr_embedding.npz`** + **`lr_model.onnx`** — Length regulator:
   - `lr_embedding.npz`: codebook weight matrix `(2048, 512)`.  The adapter
     uses nearest-neighbor lookup to project 768-d Whisper hidden states to
     512-d discrete embeddings.
   - `lr_model.onnx`: four `Conv1d(512,512,k=3,pad=1) + GroupNorm + Mish`
     blocks + `Conv1d(512,512,k=1)`.  Input `(1, 512, T)` → output `(1, 512, T)`.
     The adapter interpolates (nearest) the embedded tokens to the target mel
     length before applying this model.

4. **`flow_estimator.onnx`** — DiT flow-matching velocity estimator (single
   forward step).  Called once per ODE step.

   Inputs:
   - `x`        `(2, 80, T_total)` — current state (batch-2 for CFG)
   - `prompt_x` `(2, 80, T_total)` — prompt template (reference mel prepended)
   - `x_lens`   `(1,)` int64       — total sequence length
   - `t`        `(2,)` float32     — current ODE time
   - `style`    `(2, 192)` float32 — speaker embedding (slot-1 zeroed for CFG)
   - `mu`       `(2, 80, T_total)` — flow conditioning (slot-1 zeroed for CFG)

   Output: `velocity` `(2, 80, T_total)` float32.

   Architecture: UViT Transformer (13 layers, 8 heads, hidden 512) with
   WaveNet-based final layer.  RoPE positional encoding.

5. **`bigvgan.onnx`** — BigVGAN v2 22kHz 80-band 256× vocoder
   (nvidia/bigvgan_v2_22khz_80band_256x):
   mel `(1, 80, T_mel)` → waveform `(1, 1, T_audio)`.  Upsample factor 256×
   (series of transposed convolutions with Snake/SnakeBeta activations).

---

## ODE solver

Linear time schedule matching upstream `BASECFM.inference`:

```
t_span = linspace(0, 1, n_steps + 1)   # linear (not cosine — unlike CosyVoice)
```

Batch-2 CFG per step:
- Slot-0: conditioned (`mu`, `style`, `prompt_x` set from data)
- Slot-1: unconditioned (all zeros)
- CFG combination: `v = (1 + 0.7) * v_cond − 0.7 * v_uncond`

Flow prompt: the reference mel `(1, 80, T_ref)` is prepended to the source
conditioning and to the initial noise tensor.  The flow estimator masks the
prompt region at each step (`x[:, :, :T_ref] = 0`).  After the ODE, the
adapter strips the prompt frames to recover `(1, 80, T_src)`.

---

## License notes

| Asset | License |
|---|---|
| Upstream code (GPL-3.0) | NEVER vendored — external checkout only |
| DiT weights (`Plachta/Seed-VC`) | Apache-2.0 / research |
| BigVGAN weights (`nvidia/bigvgan_v2_22khz_80band_256x`) | MIT |
| CAMPPlus weights (`funasr/campplus`) | MIT |
| Whisper-small weights (`openai/whisper-small`) | MIT |

The `TigreGotico/voiceclonnx-seedvc` HF repo states all licenses on the
model card.

---

## Parity results (fp32 torch vs ONNX Runtime)

See `parity_report.json` in the HF repo.

| Component | max\_abs Δ | mean\_abs Δ | Verdict |
|---|---|---|---|
| whisper\_encoder | < 5e-3 | — | PASS |
| campplus | < 1e-3 | — | PASS |
| lr\_model | < 1e-3 | — | PASS |
| flow\_estimator | < 1e-2 | — | PASS |
| bigvgan | < 1e-3 | — | PASS |

---

## Model sizes

| File | Size | Variant |
|---|---|---|
| `whisper_encoder.onnx` | ~90 MB | fp32 |
| `whisper_encoder_q8.onnx` | ~23 MB | INT8 |
| `campplus.onnx` | ~2 MB | fp32 |
| `campplus_q8.onnx` | ~1 MB | INT8 |
| `lr_embedding.npz` | ~4 MB | numpy |
| `lr_model.onnx` | ~2 MB | fp32 |
| `lr_model_q8.onnx` | ~1 MB | INT8 |
| `flow_estimator.onnx` | ~115 MB | fp32 |
| `flow_estimator_q8.onnx` | ~30 MB | INT8 |
| `bigvgan.onnx` | ~215 MB | fp32 |
| `bigvgan_q8.onnx` | ~55 MB | INT8 |

**INT8 note:** The flow estimator uses a WaveNet final layer with channel-wise
convolutions sensitive to weight-only quantization.  See `docs/QUANTS.md` for
WER comparison before choosing `quantized=True`.

---

## Reproduction

```bash
# Install conversion deps (throwaway venv)
pip install torch onnx onnxruntime transformers torchaudio munch pyyaml
pip install huggingface_hub soundfile librosa

# Export (clone is throwaway — GPL code stays in /tmp, never committed)
python -m conversion.export_seedvc --output-dir /tmp/seedvc-out

# Push to HF Hub (requires HF_TOKEN)
HF_TOKEN=hf_... python -m conversion.export_seedvc \\
    --output-dir /tmp/seedvc-out --push
```

Pinned upstream ref: `51383efd921027683c89e5348211d93ff12ac2a8`
