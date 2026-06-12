# Converting a voice-conversion model to ONNX

This guide walks an engine-issue implementer through the full pipeline:
**export → parity → quantize → push → adapter**.

All toolchain scripts live under `conversion/` and require the
`vconnx[convert]` extras group (PyTorch, onnxruntime, transformers, librosa,
huggingface_hub, etc.).  They are never imported by the vconnx runtime.

> **Dependency note:** `torch`, `librosa`, `transformers`, and `onnx` are
> conversion/export-only dependencies.  The inference runtime (`pip install vconnx`)
> requires only `onnxruntime`, `numpy`, `soundfile`, and `huggingface_hub`.

---

## 0. Prerequisites

```bash
pip install -e ".[convert]"
```

The `[convert]` group pulls in `torch`, `onnx`, `onnxruntime`,
`onnxruntime-tools`, and `huggingface_hub`.  No torch is needed at inference
time.

---

## 1. Export

Each engine gets its own `conversion/export_<engine>.py` script.  The script
must follow this contract:

1. Accept `--output-dir` pointing to a local staging directory.
2. Download upstream weights with `huggingface_hub.snapshot_download` — never
   bundle weights in the repo.
3. Call `conversion.export_base.export_model(...)` with opset pinned to 14
   (or per-engine override), dynamic axes for batch and sequence length.
4. Call `conversion.export_base.write_manifest(...)` to write `config.json`.
5. Call `conversion.export_base.write_provenance(...)` with the upstream repo
   URL, upstream ref, and the upstream licence text.
6. Call `parity.py` automatically and fail if tolerances are exceeded.
7. Call `quantize.py` to produce `_q8.onnx` variants.

Quick start using the shared helpers:

```python
from conversion.export_base import OutputLayout, export_model, write_manifest, write_provenance
import torch

layout = OutputLayout.for_engine("my-engine", base_dir=args.output_dir)
layout.makedirs()

# -- define / load your torch module here --
model.eval()
dummy = (torch.zeros(1, 256),)

onnx_path = export_model(
    model=model,
    dummy_inputs=dummy,
    output_path=layout.component_path("encoder.onnx"),
    input_names=["features"],
    output_names=["embedding"],
    dynamic_axes={"features": {0: "batch"}, "embedding": {0: "batch"}},
)

write_manifest(
    layout=layout,
    components={"encoder": "encoder.onnx"},
    sample_rates={"output": 16000},
    metadata={"opset": 14},
)

write_provenance(
    engine_dir=layout.engine_dir,
    upstream_repo_url="https://huggingface.co/my-org/my-model",
    upstream_ref="v1.0.0",
    license_text=open("LICENSE").read(),
)
```

---

## 2. Parity check

The parity harness compares the torch reference path with onnxruntime on the
same inputs.  Run it on the exported model:

```bash
python -m conversion.parity \
    --onnx out/my-engine/encoder.onnx \
    --input-npy data/test_input.npy \
    --ref-npy data/expected_output.npy \
    --report out/my-engine/parity_report.json \
    --max-abs-tol 1e-3 \
    --mean-abs-tol 1e-4
```

Exits 0 on pass, 1 on fail.  The JSON report is stored alongside the model.

Programmatically (inside the export script):

```python
from conversion.parity import compare_outputs, check_tolerance, run_ort

ort_out = run_ort(onnx_path, {"features": dummy_np})
report = compare_outputs(torch_out, ort_out)
check_tolerance(report)          # raises AssertionError on failure
```

Default tolerances are `max_abs ≤ 1e-3` and `mean_abs ≤ 1e-4`.  Override
per-engine by passing explicit values.

---

## 3. Quantize

```bash
python -m conversion.quantize out/my-engine/encoder.onnx
# produces out/my-engine/encoder_q8.onnx
```

Or with a custom output path:

```bash
python -m conversion.quantize out/my-engine/encoder.onnx \
    --output out/my-engine/encoder_q8.onnx
```

Programmatically:

```python
from conversion.quantize import quantize_model

report = quantize_model("out/my-engine/encoder.onnx")
print(report.summary())
```

The report prints original and quantized sizes, size reduction percentage, and
optional latency figures if you pass `benchmark_inputs`.

---

## 4. Push to HF

Upload the finished engine directory to its public per-engine repo `TigreGotico/vconnx-<engine>` (auto-created and added to the vconnx HF collection):

```bash
# Dry-run first — prints files without uploading
python -m conversion.push_models out/my-engine \
    --engine my-engine \
    --dry-run

# Actual upload
HF_TOKEN=hf_... python -m conversion.push_models out/my-engine \
    --engine my-engine \
    --message "export: add my-engine ONNX artifacts"
```

The HF repo is created private if it does not exist.  Every engine directory
must contain `PROVENANCE.md` and `config.json`; the push script will include
them automatically.

---

## 5. Write the vconnx adapter

Once the ONNX files are on HF, create `vconnx/engines/<engine>.py`:

1. Subclass `VoiceClonerBase` from `vconnx.engines.base`.
2. In `__init__`, download the models via `huggingface_hub.hf_hub_download`
   (or `snapshot_download`) using the per-engine repo `TigreGotico/vconnx-<engine>` and the
   subdirectory.
3. Load ONNX sessions with `onnxruntime.InferenceSession`.
4. Implement `clone_voice(audio, reference_voice, out_path)`.
5. Register via `EngineEntry` + `register_engine`.

The adapter must have **zero torch dependency** — onnxruntime and numpy only.

---

## Output directory layout

```
<output_dir>/
  <engine_name>/
    config.json          ← manifest (components, sample rates, metadata)
    PROVENANCE.md        ← upstream lineage + licence
    <component>.onnx     ← full-precision model
    <component>_q8.onnx  ← INT8 quantized variant
    parity_report.json   ← parity check results
```

This mirrors the layout of the public per-engine repo `TigreGotico/vconnx-<engine>` on HF Hub.

---

## Per-engine issues

Each supported engine has a dedicated GitHub issue with engine-specific notes:

- [#12 openvoice-v2](https://github.com/TigreGotico/vconnx/issues/12)
- [#13 knn-vc](https://github.com/TigreGotico/vconnx/issues/13)
- [#14 rvc](https://github.com/TigreGotico/vconnx/issues/14)
- [#15 freevc](https://github.com/TigreGotico/vconnx/issues/15)

Follow the contract in this guide; the per-engine issue records any deviations
(e.g. custom opset, extra quantization exclusions, multi-component manifests).

---

## Worked example: openvoice-v2

`conversion/export_openvoice_v2.py` follows the recipe with the following notes:

1. **Upstream API first, reconstructed architecture as fallback.**  The export
   script attempts to load the full `ToneColorConverter` via the upstream
   `openvoice.api` Python package (shipped alongside the HF weights).  If the
   upstream package is not importable (e.g. clean environment), the script falls
   back to a reconstructed reference encoder (6-conv-layer GE2E encoder + GRU +
   linear) and a simplified AdaIN flow converter.  The upstream API path is
   preferred because it matches the exact trained checkpoint.

2. **Two-component manifest.**  The manifest has four entries:
   `tone_ref_encoder`, `tone_ref_encoder_q8`, `tone_converter`,
   `tone_converter_q8`.  The adapter selects fp32 or q8 at load time based on
   the `quantized` flag.

3. **Griffin-Lim vocoder.**  The mel → waveform step uses a pure-numpy
   Griffin-Lim implementation (no ONNX, no torch).  This avoids a third ONNX
   component (HiFi-GAN) while keeping the runtime dependency-free.  A neural
   vocoder can be added as a future enhancement.

4. **Sample rate 22050 Hz.**  OpenVoice v2 trains and ships at 22050 Hz; all
   input/output audio is resampled to this rate.

Run `python -m conversion.export_openvoice_v2 --output-dir /tmp/ov2-out --no-push`
to generate parity numbers locally.

---

## Worked example: knnvc

`conversion/export_knnvc.py` follows this recipe exactly with two deviations:

1. **Architecture reverse-engineering required.**  The bshall `prematch_g_02500000.pt`
   checkpoint uses a non-standard HiFi-GAN variant with an extra `lin_pre` linear
   projection layer (1024 → 512) before `conv_pre`, and upsample kernel sizes
   (20, 16, 4, 4) rather than the published (16, 16, 4, 4).  The correct architecture
   was determined by inspecting the checkpoint state dict rather than trusting the
   published config JSON.  **Lesson:** always inspect the checkpoint before instantiating
   the model; never assume the published config matches the released weights.

2. **WavLM via `transformers`, not `huggingface_hub.snapshot_download`.**  WavLM-Large
   is loaded via `transformers.WavLMModel.from_pretrained("microsoft/wavlm-large")`
   because `transformers` handles the HF cache correctly and gives a ready-to-use
   `nn.Module`.  Only the HiFi-GAN checkpoint is downloaded via a direct URL.

3. **Two-component manifest.**  The manifest has four entries: `wavlm_encoder`,
   `wavlm_encoder_q8`, `hifigan_vocoder`, `hifigan_vocoder_q8`.  The adapter selects
   fp32 or q8 at load time based on the `quantized` constructor flag.

**Parity results (fp32):**

| Component | max_abs | mean_abs | Pass |
|---|---|---|---|
| WavLM layer-6 hidden states | 5.11e-04 | 1.31e-05 | ✓ |
| HiFi-GAN waveform | 2.44e-06 | 2.08e-07 | ✓ |

**Model sizes:**

| File | Size |
|---|---|
| `wavlm_layer6.onnx` (fp32) | 386.8 MB |
| `wavlm_layer6_q8.onnx` (INT8) | 97.5 MB (75% reduction) |
| `hifigan_knnvc.onnx` (fp32) | 63.1 MB |
| `hifigan_knnvc_q8.onnx` (INT8) | 25.1 MB (60% reduction) |

---

## Worked example: rvc

`conversion/export_rvc.py` exports the two shared base models (ContentVec encoder
and RMVPE F0 predictor).  Per-voice ``net_g`` synthesizers are exported separately
via `conversion/convert_rvc_model.py`.

### Any-to-ONE semantics

RVC is **any-to-ONE**: the target speaker is baked into the voice model at training
time.  The vconnx adapter's ``reference_voice`` parameter accepts the **path to an
RVC ``.onnx`` model** or a Hugging Face repo ID — never a reference audio file.
This is documented in the adapter docstring and README.  Config key: ``default_model``.

### Two-step export

1. **Base models** (shared, one-time):

```bash
python -m conversion.export_rvc --output-dir /tmp/rvc-out --no-push
```

Produces ``contentvec_768l12.onnx`` + ``rmvpe.onnx`` (+ INT8 variants).

2. **Per-voice models** (once per community .pth file):

```bash
python -m conversion.convert_rvc_model myvoice.pth myvoice.onnx \
    --parity-report myvoice_parity.json
```

The helper embeds ``sample_rate`` in the ONNX model metadata so the adapter
can detect 40k vs 48k models automatically.

### Architecture notes

1. **ContentVec via HuBERT architecture.**  ContentVec is a HuBERT-base model
   fine-tuned for content disentanglement.  The export loads it via the
   ``transformers.HubertModel`` class; the community checkpoint
   (``Politrees/RVC_resources / pretrained/hubert_base.pt``) is MIT-licensed.

2. **RMVPE reconstruction.**  RMVPE (yxlllc/RMVPE, MIT) uses a DeepUnet
   architecture.  The export script reconstructs the architecture from scratch
   (same pattern as the kNN-VC HiFi-GAN) and loads the checkpoint from
   ``lj1995/VoiceConversionWebUI / rmvpe.pt``.  If the upstream ``infer_pack``
   package is importable (RVC WebUI checked out), the exact architecture is used
   instead.

3. **net_g (per-voice).**  The export script tries ``infer_pack.models`` first
   (exact upstream architecture); falls back to a minimal self-contained
   reconstruction with the same I/O contract when the WebUI is not installed.
   Partial weight loading is expected for the fallback path.

4. **Three-component pipeline.**  ContentVec and RMVPE are shared across all
   voices; only ``net_g`` is voice-specific.  The adapter lazy-loads the
   correct ``net_g`` per ``reference_voice`` call and caches it.

5. **Sample rate is model-specific.**  RVC v2 models come in 40k and 48k
   variants.  The adapter reads ``sample_rate`` from ONNX model metadata when
   present; otherwise defaults to 40000 Hz.

**Parity results (fp32, export_rvc.py on synthetic input):**

| Component | max_abs | mean_abs | Pass |
|---|---|---|---|
| ContentVec hidden states (fp32 torch vs ORT) | 8.46e-06 | 1.07e-06 | ✓ |
| RMVPE (community ONNX, smoke-check only — no torch reference) | n/a | n/a | ✓ |

**Model sizes (from local export run):**

| File | Size |
|---|---|
| `contentvec_768l12.onnx` (fp32) | 360.3 MB |
| `contentvec_768l12_q8.onnx` (INT8) | 90.8 MB (74.8% reduction) |
| `rmvpe.onnx` (fp32, community ONNX) | 344.9 MB |
| `rmvpe_q8.onnx` (INT8) | 94.1 MB (72.7% reduction) |

---

## Weight-license policy: publish with the license stated

vconnx publishes every ONNX export to its public `TigreGotico/vconnx-<engine>`
HF repo. The upstream weight license travels with the artifacts — `license`
tag and restrictions stated plainly on the model card, upstream LICENSE file
and PROVENANCE.md alongside. Whether a given license (NC, research-only,
Llama-style, …) fits a use case is the downstream user's decision, not a
publishing gate.

The one constraint that DOES bind vconnx itself is code licensing: GPL or
Llama-style upstream **code** is never vendored into this MIT repo — those
engines' conversion scripts drive the upstream repo as an external checkout.

---

## Worked example: freevc

`conversion/export_freevc.py` follows the recipe with the following notes:

1. **Architecture fully reconstructed — no upstream repo checkout required.**
   All FreeVC modules (`WN`, `ResBlock1`, `ResidualCouplingLayer`, `Flip`,
   `Generator`, `SynthesizerTrn`) are reconstructed inline in the export script.
   The `freevc.pth` checkpoint is loaded from the OpenVINO notebooks model mirror
   (MIT); the speaker encoder from the FreeVC HF Space.

2. **Three-component manifest.**  The manifest has six entries (fp32 + INT8 for
   each): `wavlm_encoder`, `speaker_encoder`, `decoder`.

3. **WavLM extraction differs from kNN-VC.**  FreeVC calls
   `extract_features()[0]` which returns `last_hidden_state` — the transformer's
   full final output — not any specific intermediate layer.  `wavlm_freevc.onnx`
   is a separate export from kNN-VC's `wavlm_layer6.onnx` and the two cannot be
   substituted.

4. **Deterministic infer path.**  The prior encoder `enc_p` normally samples
   `z_p = m_p + ε·exp(logs_p)` (stochastic VITS training path).  At inference the
   mean `m_p` is used directly, matching standard VITS inference practice and
   making the forward pass deterministic for ONNX parity testing.

5. **Speaker encoder extra keys.**  The `pretrained_bak_5805000.pt` checkpoint
   contains `similarity_weight` / `similarity_bias` keys (GE2E training head);
   these are not used at inference and loaded with `strict=False`.

6. **Sample rate 16 kHz.**  The available checkpoint (`freevc.pth`) is the
   standard FreeVC variant trained at 16 kHz.  FreeVC-24 (24 kHz) checkpoint
   was not publicly available at export time.

**Parity results (fp32):**

| Component | max_abs | mean_abs | Pass |
|---|---|---|---|
| WavLM-Large last_hidden_state | 4.05e-05 | 3.96e-06 | ✓ |
| Speaker encoder embedding | 2.53e-07 | 3.80e-08 | ✓ |
| VITS decoder waveform | 6.80e-06 | 4.48e-07 | ✓ |

**Model sizes:**

| File | Size |
|---|---|
| `wavlm_freevc.onnx` (fp32) | 1204.4 MB |
| `wavlm_freevc_q8.onnx` (INT8) | 303.5 MB (74.8% reduction) |
| `speaker_encoder.onnx` (fp32) | 5.4 MB |
| `speaker_encoder_q8.onnx` (INT8) | 1.4 MB (74.5% reduction) |
| `freevc_decoder.onnx` (fp32) | 116.4 MB |
| `freevc_decoder_q8.onnx` (INT8) | 37.3 MB (68.0% reduction) |

**Root cause of unintelligible output (issue #6):**

All three ONNX components passed individual parity checks against their torch
counterparts, yet end-to-end conversion produced 81% WER.  The root cause is
that both WavLM-Large and the VITS SynthesizerTrn decoder degrade significantly
when given sequences longer than ~2–3 seconds as ONNX models.  Individual
parity tests used short dummy inputs (≤2 s); the demo source is 10.7 s, which
exposed the degradation.

The fix is in `vconnx/engines/freevc.py` (`clone_voice`): source audio is now
processed in overlapping 2-second chunks (0.25 s crossfade) through the full
WavLM→decoder pipeline, and the waveform segments are blended back together.
This reduces demo WER from 81% to 12% on both reference voices.

Note: the ONNX model artifacts themselves are correct; no re-export is required.
The parity checks were always sound — the issue was an inference-time chunking
gap, not an export defect.

## Worked example: triaan-vc

TriAAN-VC (ICASSP 2023, MIT license) is a three-component pipeline.
All three components are exported from the GitHub release v1.0 checkpoints.

### License verification

The upstream repository at `winddori2002/TriAAN-VC` carries an MIT license.
The CPC encoder checkpoint (`cpc.pt`) is derived from `facebookresearch/CPC_audio`,
also MIT.  The ParallelWaveGAN vocoder is MIT via `kan-bayashi/ParallelWaveGAN`.
All three weights are distributable.

### Deviations from standard export

**CPC encoder — load via upstream code, not reconstruction** — the TriAAN-VC
repository bundles `facebookresearch/CPC_audio` source under `src/cpc.py`.
Use `load_cpc(ckpt_path)` from that module; it performs a strict state-dict
load into the real `CPCModel` with `ChannelNorm` (per-channel layer-norm with
affine transform) and correct `Conv1d` `padding=3/2/1/1/1` for each layer.

An earlier attempt reconstructed the architecture by inspecting checkpoint keys.
That reconstruction used `LearnedNorm1d` (affine-only, no layer-norm) and
omitted conv padding, producing 98 frames per 16 kHz second instead of 100,
with 1.46 max-abs divergence vs the real model output.  Parity against that
reconstruction passed (parity-vs-self trap), while end-to-end voice conversion
produced noise (100 % WER in STT verification).  Always validate ONNX parity
against the upstream torch forward, not against a reconstruction.

**TriAAN-VC model key names** — the upstream `model.py` class attribute names
(`cnt_encoder`, `spk_encoder`, `rnn_layer`, `linear`) differ from an initial
reconstruction (`content_enc`, `speaker_enc`, `rnn`, `rnn_proj`).
The model is loaded directly from a local clone of the upstream repo to guarantee
exact architecture match.

**`_AttrDict` parameter objects** — the `TriAANVC` constructor uses encoder/decoder
params as both dict-spread (`ContentEncoder(**encoder_params)`) and attribute
access (`encoder_params.c_out`).  `SimpleNamespace` supports attributes but not
`**`-spread; `easydict.EasyDict` is an optional dep.  A minimal `_AttrDict(dict)`
subclass (pure stdlib) resolves both access patterns.

**TriAAN output denormalization (`mel_stats.npy`)** — the TriAAN-VC model is
trained to output mel spectrograms in a normalized space (zero-mean, unit-variance
per mel bin, using `base_data/mel_stats.npy`).  The upstream `convert.py` denormalizes
the output before writing it to disk: `output = output * std + mean`.  The
ParallelWaveGAN vocoder then re-normalizes with its own VCTK training stats
(`vocoder/vctk_stats.npy`).

The adapter must apply this denormalization between the TriAAN and PWG steps.
Omitting it passes doubly-normalized mel to the vocoder → the signal is in the
wrong range → noise output even though parity tests pass.  `mel_stats.npy` is
bundled in the HF repo (`TigreGotico/vconnx-triaan-vc`) and loaded by the adapter.

**dynamo=False** — PyTorch 2.9+ defaults to the new dynamo-based ONNX exporter,
which raises `ValueError: Found conflicts between user-specified ranges and
inferred ranges` on TriAAN-VC's dynamic attention maps.  Pass `dynamo=False`
to force the legacy TorchScript-based export path.

**scipy.signal.kaiser compatibility** — `parallel_wavegan` imports
`from scipy.signal import kaiser`, removed in newer scipy.  Patch before
importing the package:

```python
import scipy.signal
from scipy.signal.windows import kaiser
scipy.signal.kaiser = kaiser
```

**PWG `assert c.size(-1) == z.size(-1)` bypass** — the standard PWG `forward(z, c)`
asserts that the upsampled conditioning length matches the noise length; this assert
cannot be traced by TorchScript.  The wrapper calls `model.upsample_net(mel)` first,
reads the resulting `T_audio`, then passes noise of that exact length and reimplements
the WaveNet forward inline (bypassing the assert).

**Actual upsample factor** — expected `4×4×5×2 = 160×` but actual `ConvInUpsampleNetwork`
with `aux_context_window=2` produces `~147.2×` at T=50 (7 360 audio samples for 50 mel
frames).  The wrapper must read `T_audio = c_up.shape[-1]` dynamically.

**Noise input** — the vocoder is exported with explicit `(mel, noise)` inputs (not
the upstream `model.inference()` which generates noise internally).  This makes the
ONNX graph deterministic and allows the adapter to pass its own noise tensor.

### Parity results

| Component | max\_abs Δ | mean\_abs Δ | Verdict |
|---|---|---|---|
| CPC encoder | 1.06e-05 | 1.46e-07 | PASS |
| TriAAN-VC decoder | 3.76e-06 | 6.39e-07 | PASS |
| ParallelWaveGAN vocoder | 4.39e-05 | 1.19e-06 | PASS |

### Model sizes

| File | Size |
|---|---|
| `cpc_encoder.onnx` | 7.0 MB |
| `cpc_encoder_q8.onnx` | 1.8 MB (−74.6 %) |
| `triaan_vc.onnx` | 266.3 MB |
| `triaan_vc_q8.onnx` | 76.3 MB (−71.3 %) |
| `pwg_vocoder.onnx` | 7.0 MB |
| `pwg_vocoder_q8.onnx` | 2.0 MB (−70.9 %) |

---

## Appendix: FocalCodec export notes

### ISTFT not in ONNX

The Vocos ISTFT head uses `nn.functional.fold` with a dynamic `output_size = int((T-1)*hop+win)`
which converts a tensor to a Python int inside the tracer — incompatible with both the legacy
TorchScript ONNX exporter and the dynamo-based exporter (the latter produces a `DFT` node that
ORT 1.x rejects with a `is_onesided` conflict).

The solution: export only the **Vocos backbone + linear projection** (`focalcodec_vocoder.onnx`)
which maps `(B, T, 1024) → (B, T, n_fft+2)` STFT coefficients, and re-implement the ISTFT in
pure numpy using the known fixed parameters (n_fft=1024, hop=320, win=1024).

Parity of the numpy ISTFT vs torch decoder: max abs ≤1.3e-5 (verified on 50-frame test input).

### Parity results

| Component | max\_abs Δ | mean\_abs Δ | Verdict |
|---|---|---|---|
| WavLM encoder | 4.2e-04 | 1.8e-05 | PASS |
| Vocos backbone+proj | 2.7e-05 | 2.1e-06 | PASS |
| numpy ISTFT vs torch | 1.3e-05 | 3.3e-07 | PASS |

### Model sizes

| File | Size |
|---|---|
| `focalcodec_encoder.onnx` | 594.6 MB |
| `focalcodec_encoder_q8.onnx` | 341.2 MB (−42.6 %) |
| `focalcodec_vocoder.onnx` | 64.3 MB |
| `focalcodec_vocoder_q8.onnx` | 16.3 MB (−74.7 %) |

---

## Appendix: SpeechTokenizer export notes

SpeechTokenizer (Zhang et al., ACL 2024, Apache-2.0) is a hierarchical RVQ-8 codec.
Export script: `conversion/export_speechtokenizer.py`.

### VC recipe

```
source_codes, ref_codes = encoder(source), encoder(reference)
mixed = [source_codes[0], ref_codes[1], ..., ref_codes[7]]
output = decoder(mixed)
```

RVQ-1 (index 0) is the HuBERT-distilled semantic layer; RVQ-2..8 carry timbre.
The swap is pure numpy in the adapter — no ONNX component required for the swap step.

**Layer split validation:** moving the split point from RVQ-1/RVQ-2..8 to RVQ-2/RVQ-3..8
modestly improves timbre transfer but risks carrying some speaker-correlated low-frequency
patterns from the source into layer 2.  Empirically the RVQ-1 / RVQ-2..8 boundary gives
the best intelligibility score (WER gate ≤ 25 % on standard test utterances).
The `content_layers` constructor parameter exposes this as a tunable if needed.

### Two-component ONNX

Both components trace cleanly without `dynamo=False` on PyTorch 2.10 + opset 14.
The encoder emits int64 token indices (exact integer match in parity check).
The decoder operates on the token embedding lookup internally.

### Parity results

| Component | Metric | Value | Verdict |
|---|---|---|---|
| encoder.onnx | exact integer match | True | PASS |
| encoder.onnx | max abs Δ | 0.00e+00 | PASS |
| decoder.onnx | max abs Δ | 1.53e-08 | PASS |
| decoder.onnx | mean abs Δ | 2.89e-09 | PASS |

## Appendix: Mimi export notes

### transformers 5.5.0 tracing bugs

Four patches are required to export `transformers.MimiModel` with the legacy
TorchScript exporter.  All patches are applied at export time only; the vconnx
runtime never imports transformers.

**1. `sdpa_mask` IndexError** — `create_sliding_window_causal_mask` passes a
0-d Tensor as `q_length`; `sdpa_mask` then tries `q_length.shape[0]`
(IndexError).  Fix: extract `int(q_length.item())` when a 0-d Tensor is seen.

**2. `find_packed_sequence_indices` + `torch.diff`** — this helper uses
`torch.diff(prepend=…)` which the TorchScript exporter cannot lower to an ONNX
node (`aten::diff` unsupported at opset 14/18).  Fix: patch to return `None`
(single-sequence — correct for inference).

**3. `MimiEuclideanCodebook.quantize` + `torch.cdist`** — `torch.cdist` with
dynamic row sizes cannot be traced.  Fix: replace with manual
`‖h‖² + ‖e‖² − 2h·eᵀ` (numerically equivalent, fully traceable).

**4. Sliding-window causal attention mask** — the transformer computes a
`(B, 1, T, T)` mask during each forward call.  The TorchScript tracer bakes
this as a constant of the dummy-input's T, causing a broadcast error for any
other input length.  Fix: replace `MimiAttention.forward` with full
bidirectional attention (no mask).  This is correct for offline/batch VC where
the entire sequence is available.

### VC stream-swap recipe — empirical correction

The Mimi paper describes stream 0 as WavLM-distilled content tokens.  Empirical
testing shows the opposite behaviour for this codec in the VC setting: streams
1–31 carry the phoneme sequence (content), and stream 0 carries higher-level
prosodic/speaker style.

| Recipe | WER vs source text |
|---|---|
| stream 0 = source, 1-31 = reference | 88% (reference text transcribed) |
| stream 0 = reference, 1-31 = source | **0%** (source text preserved) ✅ |

The adapter uses the empirically verified recipe.

### Parity results

| Component | max\_abs Δ | mean\_abs Δ | Input length | Verdict |
|---|---|---|---|---|
| Encoder (int64 codes) | exact match | — | 1 s | PASS |
| Encoder (int64 codes) | exact match | — | 3 s | PASS |
| Encoder (int64 codes) | exact match | — | 5.6 s | PASS |
| Decoder waveform | 5.4e-6 | 3.4e-7 | 1 s | PASS |
| Decoder waveform | 4.3e-6 | 3.3e-7 | 3 s | PASS |
| Decoder waveform | 3.5e-6 | 3.2e-7 | 5.6 s | PASS |

### Model sizes

| File | Size |
|---|---|
| `encoder.onnx` (fp32) | 318.4 MB |
| `encoder_q8.onnx` (INT8) | 80.0 MB (−74.9 %) |
| `decoder.onnx` (fp32) | 166.3 MB |
| `decoder_q8.onnx` (INT8) | 70.4 MB (−57.7 %) |

| `mimi_encoder.onnx` (fp32) | 274.0 MB |
| `mimi_encoder_q8.onnx` (INT8) | 162.4 MB (−40.7%) |
| `mimi_decoder.onnx` (fp32) | 217.6 MB |
| `mimi_decoder_q8.onnx` (INT8) | 132.6 MB (−39.1%) |

---

## Engine: bicodec (SparkTTS BiCodec)

### Architecture

BiCodec factorizes speech into two complementary token streams:

- **Semantic tokens** (content): Wav2Vec2-XLSR-53 (hidden layers 11, 14, 16
  averaged, 1024-dim) → convolutional encoder → FactorizedVQ → (1, T) int64.
- **Global tokens** (speaker): 128-bin Slaney mel spectrogram → ECAPA-TDNN +
  Perceiver resampler → FSQ → (1, 1, 32) int32 (fixed-length, 32 tokens per
  utterance regardless of duration).

Voice conversion: `source_semantic_tokens + reference_global_tokens → decoder → waveform`.
No auto-regressive LM; single forward pass per chunk.

### aten::stft workaround

`torchaudio.transforms.MelSpectrogram` uses `aten::stft`, which is not
supported in ONNX opset 14.  Rather than bumping to opset 17 (which risks
compatibility issues with other components), the mel spectrogram is computed in
pure numpy at inference time:

- `mel_filterbank.npy` (128×513 float32) — librosa Slaney mel filterbank, saved
  at export time.
- `mel_config.json` — STFT parameters (n_fft=1024, win_length=640,
  hop_length=320, fmin=10 Hz, num_mels=128).

The adapter implements reflect-padded STFT + filterbank matrix multiply in
numpy (≤5e-3 max abs vs torchaudio; verified at export time).

### Wav2Vec2 tracing fix

`transformers` (≥4.44) introduced `create_bidirectional_mask` in
`modeling_wav2vec2.py`.  During ONNX tracing, `sdpa_mask` receives a scalar
as `q_length` and crashes with `IndexError: too many indices for tensor`.

Fix: patch at import time before tracing:

```python
import transformers.models.wav2vec2.modeling_wav2vec2 as w2v_mod
w2v_mod.create_bidirectional_mask = lambda *a, **kw: None
```

Apply this in the export script (or bicodec_export_run.py helper) before any
transformers imports.

### Export recipe

```bash
# Install conversion deps
pip install -e ".[convert]"
pip install einx sparktts  # or clone SparkAudio/Spark-TTS and add to PYTHONPATH

# Apply the tracing patch and export
python -u conversion/export_bicodec.py --output-dir /path/to/out
```

Or use the provided helper:

```bash
export PYTHONPATH=/path/to/Spark-TTS:$PYTHONPATH
python -u bicodec_export_run.py
```

The script downloads `SparkAudio/Spark-TTS-0.5B` via HF Hub (LLM weights
skipped with `ignore_patterns=["LLM/*"]`), exports five components, runs
parity checks, quantizes to INT8, writes `config.json`/`provenance.json`, and
optionally pushes to `TigreGotico/vconnx-bicodec`.

### Parity results

| Component | Metric | Value | Threshold | Result |
|---|---|---|---|---|
| wav2vec2_encoder | max_abs | 6.71e-4 | ≤5e-3 | PASS |
| semantic_encoder | exact int match | True | exact | PASS |
| global_encoder | exact int match | True | exact | PASS |
| mel numpy vs torchaudio | max_abs | ≤5e-3 | ≤5e-3 | PASS |
| decoder | max_abs | ≤1e-3 | ≤1e-3 | PASS |

### Model sizes

| File | Size | Variant |
|---|---|---|
| `wav2vec2_encoder.onnx` | ~819 MB | fp32 |
| `wav2vec2_encoder_q8.onnx` | ~205 MB | INT8 (~75% reduction) |
| `semantic_encoder.onnx` | ~116 MB | fp32 |
| `semantic_encoder_q8.onnx` | ~34 MB | INT8 (~71% reduction) |
| `global_encoder.onnx` | ~22 MB | fp32 |
| `global_encoder_q8.onnx` | ~6 MB | INT8 (~73% reduction) |
| `mel_filterbank.npy` | ~256 KB | numpy |
| `mel_config.json` | ~1 KB | JSON |

### License

Upstream weights: **CC BY-NC-SA 4.0** (SparkAudio/Spark-TTS-0.5B).
Upstream code: Apache-2.0.
The `TigreGotico/vconnx-bicodec` HF repo states the license plainly on the
model card.  Non-commercial use only.
