# Converting a voice-conversion model to ONNX

This guide walks an engine-issue implementer through the full pipeline:
**export → parity → quantize → push → adapter**.

All toolchain scripts live under `conversion/` and require the
`voiceclonnx[convert]` extras group (PyTorch, onnxruntime, transformers, librosa,
huggingface_hub, etc.).  They are never imported by the voiceclonnx runtime.

> **Dependency note:** `torch`, `librosa`, `transformers`, and `onnx` are
> conversion/export-only dependencies.  The inference runtime (`pip install voiceclonnx`)
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

Upload the finished engine directory to its public per-engine repo `TigreGotico/voiceclonnx-<engine>` (auto-created and added to the voiceclonnx HF collection):

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

## 5. Write the voiceclonnx adapter

Once the ONNX files are on HF, create `voiceclonnx/engines/<engine>.py`:

1. Subclass `VoiceClonerBase` from `voiceclonnx.engines.base`.
2. In `__init__`, download the models via `huggingface_hub.hf_hub_download`
   (or `snapshot_download`) using the per-engine repo `TigreGotico/voiceclonnx-<engine>` and the
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

This mirrors the layout of the public per-engine repo `TigreGotico/voiceclonnx-<engine>` on HF Hub.

---

## Per-engine issues

Each supported engine has a dedicated GitHub issue with engine-specific notes:

- [#12 openvoice-v2](https://github.com/TigreGotico/voiceclonnx/issues/12)
- [#13 knn-vc](https://github.com/TigreGotico/voiceclonnx/issues/13)
- [#14 rvc](https://github.com/TigreGotico/voiceclonnx/issues/14)
- [#15 freevc](https://github.com/TigreGotico/voiceclonnx/issues/15)

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
time.  The voiceclonnx adapter's ``reference_voice`` parameter accepts the **path to an
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

voiceclonnx publishes every ONNX export to its public `TigreGotico/voiceclonnx-<engine>`
HF repo. The upstream weight license travels with the artifacts — `license`
tag and restrictions stated plainly on the model card, upstream LICENSE file
and PROVENANCE.md alongside. Whether a given license (NC, research-only,
Llama-style, …) fits a use case is the downstream user's decision, not a
publishing gate.

The one constraint that DOES bind voiceclonnx itself is code licensing: GPL or
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

The fix is in `voiceclonnx/engines/freevc.py` (`clone_voice`): source audio is now
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
bundled in the HF repo (`TigreGotico/voiceclonnx-triaan-vc`) and loaded by the adapter.

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
TorchScript exporter.  All patches are applied at export time only; the voiceclonnx
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

## Appendix: FACodec export notes

FACodec (NaturalSpeech 3, Amphion / Microsoft Research, ICML 2024) disentangles
speech into four subspaces: content, prosody, timbre, acoustic detail.  Voice
conversion swaps the timbre embedding only — no per-speaker training required.

### License verification

The HF repo `amphion/naturalspeech3_facodec` carries `license: apache-2.0` in
its YAML front-matter (verified via HuggingFace Hub API).  The Amphion GitHub
repository (`open-mmlab/Amphion`) is Apache-2.0 at repository level; per-module
headers additionally carry MIT.  ONNX artifacts are published under Apache-2.0
with provenance stated on the model card.

### Four-component split

The V2 VC path requires four ONNX components:

| File | I/O | Description |
|---|---|---|
| `facodec_encoder.onnx` | wav(1,1,N) → enc_feats(1,256,T) | Convolutional encoder (hop=200) |
| `facodec_timbre.onnx` | enc_feats(1,256,T) → spk_embs(1,256) | 4-layer Transformer → mean-pool |
| `facodec_quantize.onnx` | (enc_feats,mel_20) → vq_ids(6,1,T) | Hierarchical VQ-6 |
| `facodec_decoder.onnx` | (vq_ids,spk_embs) → wav(1,1,N) | vq2emb + AdaIN + conv decoder |

### Prosody mel in numpy

`FACodecEncoderV2.get_prosody_feature(wav)` returns the first 20 mel bins
of a standard log-mel spectrogram (n_fft=1024, hop=200, win=800, n_mels=80,
sr=16000).  This is implemented in pure numpy in the adapter via
`_compute_prosody_mel` — no ONNX component needed.

### Amphion clone

The export script performs a sparse-checkout of `models/codec/ns3_codec` from
the Amphion GitHub repository to load the upstream model classes, then loads
the V2 checkpoint from HF Hub.  The `einops` package is required.

### Export quirks

1. **alias_free_torch UpSample1d/LowPassFilter1d** — both modules call
   `self.filter.expand(C, -1, -1)` where C is read from the runtime input shape.
   The TorchScript exporter cannot export convolutions whose kernel shape depends
   on a dynamic dimension.  Fix: instrument the modules with a capturing wrapper,
   run one dummy forward to record C, then pre-expand the filter as a static buffer
   and replace `forward` with a version using the buffer.

2. **nn.MultiheadAttention dynamic T** — `nn.MultiheadAttention` with
   `batch_first=True` internally reshapes `(B, T, H)` to `(B*n_heads, T, head_dim)`
   using the TorchScript tracer, which bakes T from the dummy input.  Fix: replace
   each `nn.MultiheadAttention` with a `DynamicMHA` that uses
   `F.scaled_dot_product_attention` directly — accepting fully dynamic shapes.
   Both `timbre_encoder` and `melspec_encoder` in the decoder are affected.

3. **Prosody mel T vs encoder T** — the STFT-based mel spectrogram (computed in
   numpy in the adapter) and the convolutional encoder produce slightly different
   frame counts for the same audio length.  The adapter trims or pads `mel_20` to
   match the encoder output T before passing to the quantize component.

### Parity results (fp32 torch vs ORT)

| Component | max_abs Δ | mean_abs Δ | Verdict |
|---|---|---|---|
| facodec_encoder | 1.62e-05 | 2.36e-06 | PASS |
| facodec_timbre | 1.43e-06 | 6.40e-08 | PASS |
| facodec_quantize | exact int64 match | — | PASS |
| facodec_decoder | 7.50e-09 | 1.46e-09 | PASS |

### Model sizes

| File | Size |
|---|---|
| `facodec_encoder.onnx` (fp32) | 16.5 MB |
| `facodec_encoder_q8.onnx` (INT8) | 4.7 MB (−71.5%) |
| `facodec_timbre.onnx` (fp32) | 33.0 MB |
| `facodec_timbre_q8.onnx` (INT8) | 12.1 MB (−63.3%) |
| `facodec_quantize.onnx` (fp32) | 33.4 MB |
| `facodec_quantize_q8.onnx` (INT8) | 12.5 MB (−62.6%) |
| `facodec_decoder.onnx` (fp32) | 66.2 MB |
| `facodec_decoder_q8.onnx` (INT8) | 36.8 MB (−44.4%) |

### Intelligibility gate (WER ≤ 25%)

| Reference voice | WER | Verdict |
|---|---|---|
| en-US-AriaNeural | 0% | PASS |
| en-GB-SoniaNeural | 0% | PASS |

### VC recipe

```
vq_post_emb_src = decoder_v2.vq2emb(vq_ids_src, use_residual=False)
wav_out = decoder_v2.inference(vq_post_emb_src, spk_embs_ref)
```

Source prosody and content codes (VQ-1..3) are preserved; the reference
timbre embedding is injected via AdaIN-style conditioning in the decoder.
Residual codes are excluded (`use_residual=False`) to maximise timbre transfer.

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
optionally pushes to `TigreGotico/voiceclonnx-bicodec`.

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
The `TigreGotico/voiceclonnx-bicodec` HF repo states the license plainly on the
model card.  Non-commercial use only.

---

## Appendix: Chatterbox quantization notes

### Source models

The fp32 ONNX artifacts are re-hosted from
[`onnx-community/chatterbox-onnx`](https://huggingface.co/onnx-community/chatterbox-onnx)
(Apache-2.0).  No PyTorch export is needed; this is a pure quantization pass.

### speech_encoder.onnx — Gemm(transB=1) preprocessing bug

`speech_encoder.onnx` (opset 20) contains two `Gemm(transB=1)` nodes in the
S3 RVQ codebook (`s3.quantizer._codebook.project_down`).  When
`onnxruntime.quantization.quantize_dynamic` runs, its internal preprocessor
decomposes every `Gemm` into `MatMul + Add` *before* applying INT8
quantization.  For `Gemm(A, B, bias, transB=1)` the weight B has shape
`(8, 1280)` (output_dim × input_dim).  The preprocessor emits
`MatMul(A, B) + bias` without transposing B, producing a `MatMul((N,1280), (8,1280))`
node whose K-dimensions are incompatible.  ORT's session initializer catches
this via shape inference and raises:

```
[ShapeInferenceError] Incompatible dimensions
```

The `nodes_to_exclude` parameter does not prevent this because the
decomposition runs before the quantization pass, not during it.

**Fix** (`conversion/export_chatterbox.py`): patch the fp32 model before
calling `quantize_dynamic` — transpose the `project_down.weight` initializer
from `(8, 1280)` to `(1280, 8)` and rewrite the two `Gemm` nodes as
`MatMul(A, B_T) + bias`.  The resulting graph is numerically identical to the
original; the shape is now correct for `MatMul((N,1280), (1280,8)) = (N,8)`.

### conditional_decoder.onnx — If-subgraph hang

`conditional_decoder.onnx` (opset 17) has 23,934 nodes and 20 `If` nodes with
subgraph `MatMul` ops.  When `quantize_dynamic` quantizes `MatMul` ops inside
`If` subgraphs, ORT's session initializer hangs indefinitely during graph
optimization (not a crash — no error is raised; the process simply never
returns from `InferenceSession(...)`).

**Fix**: enumerate all node names inside `If` subgraphs and pass them to
`nodes_to_exclude`.  Only main-graph `MatMul` nodes are quantized (4,585 of
them); the 40 subgraph nodes are left at fp32.

### WER results

Measured with `faster-whisper base.en` on the voiceclonnx reference demo clip
(10.7 s, `source.wav` → `reference_aria.wav`):

| Variant | WER | Total size (MB) |
|---|---|---|
| fp32 | 8% | 1 080.3 |
| INT8 | 8% | 467.0 |

**57% size reduction, identical WER.  INT8 recommended.**

The VQ codebook token selection (discrete token indices from the S3 encoder)
is slightly different between fp32 and INT8 runs (the quantization error in
the continuous 1280-dim transformer embeddings shifts a small fraction of
codebook lookups), but this does not affect intelligibility on the tested clips.

### Stitch analysis: can speech_encoder + conditional_decoder merge into one graph?

**Verdict: stitchable in principle, impractical due to opset conflict.**

The VC pipeline calls the encoder *twice*:

```
enc(ref_audio) -> tgt_tokens, x_vector, prompt_feat
enc(src_audio) -> src_tokens
speech_tokens  = concat([tgt_tokens, src_tokens], axis=1)   # <-- this is the only glue
dec(speech_tokens, x_vector, prompt_feat) -> waveform
```

The sole inter-session operation (`np.concatenate`) is a plain ONNX `Concat`
node — no non-ONNX control flow, no dynamic dispatch, no Python-side
computation.  A stitched graph taking `(src_audio, ref_audio)` as dual inputs
is structurally sound.

**Why it does not work in practice:**

1. **Opset conflict.**  `speech_encoder.onnx` uses opset 20 (requires `STFT`
   op available at opset 17+).  `conditional_decoder.onnx` uses opset 17 and
   contains `ReduceL2` with `axes` as an attribute (deprecated in opset 18).
   A merged ONNX graph has one opset version; there is no setting where both
   models are valid simultaneously.

2. **Double weight footprint.**  The VC pipeline needs two independent encoder
   runs (on different audio inputs).  A naive `onnx.compose` merge duplicates
   all encoder initializers (~1.1 GB shared weights × 2 = 2.2 GB), defeating
   any session-startup benefit.  De-duplicating initializers after merging is
   possible but requires custom graph surgery since both copies share the same
   weight names in the original model.

3. **`onnx.compose.merge_models` is two-component only.**  Stitching three
   components (enc_src + enc_ref + dec) requires two separate merge passes or
   manual graph assembly.

The two-model adapter path is the correct design.  Introducing `stitched=True`
as an adapter option would add complexity for zero runtime benefit (ORT already
amortises weight loading across sessions sharing a cache).

**Recommendation:** keep the two-session path.  If a single-file distribution
is needed, pack the two fp32 models into a ZIP/tar alongside the adapter; do
not attempt a merged ONNX graph.

---

## Appendix: quickvc — ISTFT tracing notes

QuickVC uses a Multistream-iSTFT (MS-iSTFT) generator that calls `torch.istft`
inside `TorchSTFT.inverse`.  `torch.istft` cannot be traced to ONNX.

**Parameters:** `gen_istft_n_fft=16`, `gen_istft_hop_size=4`, `subbands=4`.
The ISTFT is tiny (16-point FFT), making a pure-numpy re-implementation both
simple and exact.

**numpy re-implementation** (`_numpy_ms_istft` in `quickvc.py`):
- Per-subband center-mode OLA (matching `torch.istft` default `center=True`).
- Hann window: `np.hanning(n_fft+1)[:-1]` ≡ `torch.hann_window(n_fft)`.
- Output trim: strip `n_fft//2` from both ends of the raw OLA output.
- Verified parity: 0.0 max abs error vs `torch.istft` (exactly reproducible).

**MHA reshape constraint:**
The HuBERT-soft content encoder (`nn.TransformerEncoderLayer` with
`batch_first=True`) has a `Reshape` node in the ONNX graph that freezes the
sequence length (T=50) at export time.  `dynamic_axes` for the sequence
dimension cannot be propagated through the MHA internal reshape.

Mitigation: export with a fixed 1-second dummy (T=50) and chunk audio at the
adapter level.  Audio of duration N seconds is split into ceil(N) non-overlapping
1-second windows; features are concatenated before the decoder.

Upstream reference:
- https://github.com/quickvc/QuickVC-VoiceConversion (MIT)
- https://github.com/bshall/hubert (MIT, HuBERT-soft checkpoint)

## Appendix: CosyVoice export notes

### Non-AR VC path

CosyVoice-300M (`FunAudioLLM/CosyVoice-300M`, Apache-2.0) supports voice
conversion without the autoregressive LLM by using three components:

| Role | Checkpoint | ONNX artifact |
|---|---|---|
| Source content tokenizer | `speech_tokenizer_v1.onnx` (upstream) | upstream |
| Reference speaker encoder | `campplus.onnx` (upstream, CAM++) | upstream |
| Flow encoder (tokens → mu) | `flow.pt` | `flow_encoder.onnx` |
| ODE flow decoder (mu → mel) | `flow.decoder.estimator.fp32.onnx` (upstream) | `flow_decoder.onnx` |
| HiFiGAN F0 + NSF source | `hift.pt` | `hifigan_f0_source.onnx` |
| HiFiGAN backbone | `hift.pt` | `hifigan_backbone.onnx` |
| Speaker affine layer | extracted from `flow.pt` | `spk_proj.npz` |

Upstream artifacts are downloaded from `FunAudioLLM/CosyVoice-300M` via
`hf_hub_download`; exported artifacts are produced by
`conversion/export_cosyvoice.py`.

### STFT/ISTFT limitation at opset 14

`aten::stft` / `aten::istft` are not representable at opset 14.  The HiFiGAN
`decode()` method calls both internally.  Solution: split HiFiGAN into two
separate ONNX subgraphs at the STFT boundary.

- **`hifigan_f0_source.onnx`**: takes mel `(1, 80, T_mel)`, outputs 1-D NSF
  source signal `(1, 1, T_audio)`.  Contains `ConvRNNF0Predictor` +
  `SourceModuleHnNSF` upsampling; no STFT involved.
- **`hifigan_backbone.onnx`**: takes `(mel, source_stft)` where `source_stft`
  is `(1, 18, T_stft)` (9 real + 9 imaginary bins from numpy STFT).  Outputs
  `(magnitude, phase)` in the same STFT domain; no STFT inside the graph.
- Numpy STFT and ISTFT are computed in the adapter (`_stft` / `_istft` in
  `voiceclonnx/engines/cosyvoice.py`) using scipy's `get_window("hann")` with
  `n_fft=16`, `hop_len=4`, center-pad = `n_fft // 2`.

Parameters (`n_fft=16`, `hop_len=4`, Hann window, center-pad) must match
exactly between the export script and the adapter.  Parity verified at export
time: `STFT max_abs ≤ 1.4e-6`, `ISTFT roundtrip max_abs ≤ 5e-7`.

### NSF source noise tolerance

`SineGen` injects Gaussian noise (`std=0.003`) at every forward pass.  When
exported to ONNX, this noise is baked as a constant in the initializer.  At
inference time the ONNX graph uses the baked constant; the torch reference uses
freshly sampled noise.  The resulting parity gap (`max_abs ≈ 0.115`) is
expected — the amplitude of the noise (~3% of signal) is well within
perceptually irrelevant territory.  Export tolerance is set to `0.15`; a value
above `0.5` would indicate a structural error.

### Dimension matching for backbone export

The backbone's `source_downs` upsampling path requires that `T_stft` (the STFT
frame count of the source signal) matches the upsampled activation length
exactly.  When constructing the dummy input for export, compute the actual
source STFT from a real forward pass of `hifigan_f0_source.onnx`:

```python
src_1d = f0_src_wrapper(dummy_mel).squeeze().numpy()   # (T_audio,)
src_real, src_imag = _numpy_stft(src_1d, n_fft=16, hop_len=4)
dummy_stft = torch.from_numpy(
    np.concatenate([src_real, src_imag], axis=0)[np.newaxis].astype(np.float32)
)  # (1, 18, T_stft)  — correct T_stft
```

Using a formula-derived `T_stft` may be off by ±1 frame due to STFT padding,
causing a `RuntimeError: tensor size mismatch` inside the source_resblocks.

### Speaker projection

The `spk_embed_affine_layer` (192 → 80) is a `Linear(192, 80)` inside the
flow model.  It maps the CAM++ 192-d embedding into the 80-d conditioning space
expected by the flow decoder.  Weights are extracted at export time and saved as
`spk_proj.npz` (`weight: (80, 192)`, `bias: (80,)`) so the adapter can apply
the projection in pure numpy without loading the full flow model.

### Reproduce the export

```bash
# Clone CosyVoice source (needed for model class definitions)
git clone https://github.com/FunAudioLLM/CosyVoice /tmp/CosyVoice
pip install /tmp/CosyVoice/third_party/AcademiCodec \
            /tmp/CosyVoice/third_party/Matcha-TTS

# Run export (torch + onnx required)
PYTHONPATH=/tmp/CosyVoice python3 conversion/export_cosyvoice.py \
    --out /tmp/vc-cosy-out --push

# Push to HF Hub (requires HF_TOKEN with write access to TigreGotico org)
huggingface-cli upload TigreGotico/voiceclonnx-cosyvoice /tmp/vc-cosy-out/cosyvoice
```

The `--push` flag skips re-exporting already-present ONNX files (checks by
filename) and calls `huggingface_hub.upload_folder` directly after export.

### Root-cause record: adapter silent-noise regression

Six bugs in the adapter caused 100% WER (pure noise output) despite all seven
ONNX components passing per-component parity:

1. **Baked mel length in flow\_encoder.onnx** — the original export used a
   single `F.interpolate(..., size=mel_len)` where `mel_len` was derived from
   `tokens.shape[1]` inside the traced model.  ONNX tracing bakes the Resize
   `sizes` input as a constant (e.g. `86`), so every input length would produce
   `(1, 80, 86)` regardless of token count.  Fix: split the encoder into
   `flow_encoder_conformer.onnx` (conformer only, explicit `token_len` input)
   plus a numpy `InterpolateRegulator` applied post-ONNX with `lr_weights.npz`.

2. **Baked attention mask in conformer** — a first re-export attempt derived
   `token_len` from `tokens.shape[1]` inside the model; ONNX tracing baked the
   padding mask as a constant.  Fix: pass `token_len` as a named ONNX input so
   it flows through the graph at runtime.

3. **Wrong CFG wiring** — both slots of the batch-2 ODE input were set to the
   conditioned signal.  The correct idiom is slot 0 = conditioned (`mu`, `spks`
   set), slot 1 = unconditioned (zeros for `mu`/`spks`/`cond`); velocity
   combined as `(1 + cfg_rate) * v_cond − cfg_rate * v_uncond`
   (`cfg_rate = 0.7`).

4. **Linear vs cosine ODE time schedule** — the adapter used
   `t_span = linspace(0, 1, n+1)` whereas upstream `ConditionalCFM.solve_euler`
   uses `t_span[i] = 1 − cos(i/n · π/2)`.

5. **Phase `arcsin` error in HiFiGAN ISTFT** — the backbone outputs the phase
   directly as an angle (via `sin(raw)` inside the network); the adapter was
   applying `arcsin(clip(phi, −1, 1))` before `_istft`, introducing severe phase
   distortion.  Fix: use `phi` directly as the angle.

6. **Kaldi fbank mismatch** — the numpy fbank approximation had different
   frequency warping and length compared with `torchaudio.compliance.kaldi.fbank`
   (dither=0).  Fix: `_kaldi_fbank_compat` calls torchaudio when available,
   falling back to numpy only when torch is absent.

After all six fixes: cosyvoice fp32 WER = 8% on both reference clips (gate ≤ 40%).

**INT8 note**: `quantize_dynamic` on `flow_decoder.onnx` produces weight-only
INT8 with correlation ≈ 0.19 vs fp32 output (WER 100%).  The transformer
attention architecture in the flow-matching estimator is sensitive to
weight-only quantization without activation calibration.  Cosyvoice is listed
in `docs/QUANTS.md` with INT8 flagged ⚠.  Activation-calibrated (static) INT8
via ONNX Runtime calibration tools may recover quality but requires a
representative dataset and is out of scope for this release.

---

## Appendix E: External-checkout pattern (LinaCodec / future engines)

Some upstream model codebases carry licenses that are incompatible with
vendoring into the MIT-licensed voiceclonnx repository.  LinaCodec is the
first engine to use this pattern:

- The LinaCodec Transformer backbone derives from Meta's Llama-3
  (Llama 3 Community License).
- The distill_wavlm module derives from torchaudio (BSD-2-Clause).

Neither license prohibits *export* or *distribution of ONNX weights*, but
vendoring the source code into a MIT repository would misrepresent the
licensing of the resulting library.

### Pattern: external clone at export time

```
conversion/
  export_linacodec.py        ← export script; clones upstream at runtime
voiceclonnx/
  engines/linacodec.py       ← runtime adapter; ZERO upstream code
```

The export script (`export_linacodec.py`) does:

1. Clones the upstream repository to a **throwaway path** (`/tmp/LinaCodec`)
   using `git clone`.  The clone is never committed to voiceclonnx.
2. Adds `<clone>/src` to `sys.path` at runtime.
3. Imports and instantiates the upstream models.
4. Exports ONNX artifacts.
5. The clone is discarded after export.

```python
def _ensure_linacodec_clone(dest: str = "/tmp/LinaCodec") -> None:
    import subprocess, sys
    if not Path(dest).exists():
        subprocess.run(
            ["git", "clone", "--depth=1",
             "https://github.com/ysharma3501/LinaCodec", dest],
            check=True,
        )
    sys.path.insert(0, str(Path(dest) / "src"))
```

The runtime adapter (`voiceclonnx/engines/linacodec.py`) contains:
- Pure `onnxruntime` + `numpy` — no imports from the upstream repository.
- No upstream source code whatsoever.

### ONNX weight licensing

The exported ONNX artifacts ARE published to HF Hub
(`TigreGotico/voiceclonnx-linacodec`).  The model card states the upstream
licenses plainly: Llama 3 Community License (Transformer backbone) + BSD-2-Clause
(distill_wavlm).  Users who download the weights should review those licenses.

### Adapting this pattern for other engines

Use this pattern whenever an upstream codebase carries a license that is
incompatible with the MIT-labeled voiceclonnx source, but the ONNX weights
may be re-distributed under their upstream license:

1. Create `conversion/export_<engine>.py` with a `_ensure_<engine>_clone()`
   helper that clones to `/tmp/`.
2. Keep `voiceclonnx/engines/<engine>.py` free of all upstream source code.
3. State the upstream license(s) explicitly in the adapter module docstring
   and in the HF model card.
4. Document the pattern in this appendix so future maintainers understand why
   the export script clones externally.

This pattern is now used by **SeedVC** (issue #2) as described in the appendix below.

## Appendix G: Seed-VC export notes

### License

Seed-VC code is GPL-3.0.  The EXTERNAL-CHECKOUT pattern applies: the export
script (`conversion/export_seedvc.py`) clones
`github.com/Plachtaa/seed-vc` (pinned to `51383efd`, archived Nov 2025) into
`/tmp/seed-vc` at export time.  No GPL code is committed to voiceclonnx.  The
ONNX weights are published to `TigreGotico/voiceclonnx-seedvc`; the model card
states the upstream code license (GPL-3.0) and the weight license separately.

### Architecture (non-F0 path, 22050 Hz)

| Component | ONNX artifact | I/O |
|---|---|---|
| Whisper-small encoder | `whisper_encoder.onnx` | log-mel (1,128,3000) → hidden (1,T_enc,768) |
| CAMPPlus speaker encoder | `campplus.onnx` | fbank (1,T,80) → embedding (1,192) |
| LR embedding | `lr_embedding.npz` | codebook (2048,512) — numpy lookup |
| LR conv model | `lr_model.onnx` | (1,512,T) → (1,512,T) |
| DiT flow estimator | `flow_estimator.onnx` | (x,prompt_x,x_lens,t,style,mu) → velocity |
| BigVGAN vocoder | `bigvgan.onnx` | mel (1,80,T) → wav (1,1,T_audio) |

### ODE schedule: LINEAR (not cosine)

Unlike CosyVoice (cosine schedule), Seed-VC uses a linear time schedule:

```python
t_span = np.linspace(0.0, 1.0, n_steps + 1)   # upstream BASECFM.inference
```

The CFG formula and batch-2 idiom are identical to CosyVoice (slot-0 = cond,
slot-1 = uncond zeros; `v = (1 + 0.7) * v_cond - 0.7 * v_uncond`).

### Flow prompt

The reference mel is prepended to the source conditioning before the ODE.
The total sequence is `T_total = T_ref + T_src`.  The estimator zeroes the
prompt region at each step; the adapter strips those frames after the ODE.

### flow_estimator.onnx tracing notes

The DiT uses `setup_caches(max_batch_size=2, max_seq_length=8192)` for
KV-cache pre-allocation.  Call this before export.  The WaveNet final layer
uses grouped convolutions that trace correctly at opset 14.

### INT8 quantization

The flow_estimator WaveNet final layer is likely sensitive to weight-only INT8
(similar to CosyVoice's flow_decoder).  Verify WER before deploying
`quantized=True`.  See `docs/QUANTS.md` for the comparison table once E2E
results are recorded.

### Parity results

| Component | max\_abs Δ | mean\_abs Δ | Verdict |
|---|---|---|---|
| whisper\_encoder | < 5e-3 | — | PASS |
| campplus | < 1e-3 | — | PASS |
| lr\_model | < 1e-3 | — | PASS |
| flow\_estimator | < 1e-2 | — | PASS |
| bigvgan | < 1e-3 | — | PASS |

## Appendix F: LinaCodec onset artifact — root cause and fix (issue #24)

### Symptom

`linacodec` scored 27%/27% WER on both demo clips.  The body of each sentence
transcribed correctly, but the onset was garbled: "The quick brown fox" →
"But the plot-bound fox" / "The clip brushbox".

### Root cause: two interacting artifacts

**1. WavLM left-context starvation at chunk boundaries.**
WavLM Base Plus uses a convolutional front-end with hop=320 at 16 kHz.  When
audio starts at sample 0 with no left-context, the first analysis windows are
computed over incomplete frames.  The distill_wavlm_encoder ONNX was exported
with symmetric zero-padding applied on both sides (upstream
`_calculate_waveform_padding`), so the model expects valid left-context
even at the nominal t=0.  Without this padding the first 2–3 content tokens
are wrong, corrupting the sentence onset.

**2. mel_decoder self-attention quality degrades for long sequences.**
The mel_decoder is a full-sequence self-attention Transformer.  When the
content sequence exceeds ~3 seconds (~37 tokens), quality of later tokens
degrades because the attention distributes over a much longer key–value
matrix.  This produced the systematic "Voice conversion" → "Those conversion"
confusion at the start of the second sentence in the 10.7 s source clip.
The effect is distinct from the first artifact: it appears mid-utterance at
every location where a new chunk boundary would naturally fall.

### Fix

Two complementary changes in `voiceclonnx/engines/linacodec.py`:

1. **Chunked processing (3 s chunks, 0.5 s crossfade).**
   Source audio is split into 3-second chunks processed independently through
   the full SSL → content_encoder → mel_decoder → Vocos pipeline.  Consecutive
   decoded chunks are crossfaded with a 0.5-second linear fade.  This keeps
   each content sequence short enough for the mel_decoder to maintain quality.

2. **Per-chunk reflect-padding (8 token onset pad).**
   Each chunk's 16 kHz waveform is reflect-padded by
   `_ONSET_PAD_TOKENS × 4 × 320 = 10240 samples (≈ 0.64 s)` before SSL
   extraction.  After content encoding, the leading `_ONSET_PAD_TOKENS = 8`
   content tokens are discarded.  This gives the WavLM convolutional extractor
   sufficient left-context at each chunk boundary.

### Verification

Before fix: 27%/27% WER (both clips).
After fix: 8% (aria) / 15% (sonia) WER — both well under the ≤25% gate.

The INT8 quantized variants are unaffected by this fix but degrade to ~27%
WER regardless; `quantized=False` (default fp32) is the supported path.
See `demo/QUANTS.md` for the full comparison table.
