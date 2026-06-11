# Converting a voice-conversion model to ONNX

This guide walks an engine-issue implementer through the full pipeline:
**export → parity → quantize → push → adapter**.

All toolchain scripts live under `conversion/` and require the
`vconnx[convert]` extras group (PyTorch, onnxruntime, huggingface_hub, etc.).
They are never imported by the vconnx runtime.

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

## Weight-license policy: distributable vs local-only

vconnx never redistributes model weights it has no right to. Two classes of
engine, decided per-engine in its tracking issue:

- **distributable** — upstream license permits redistribution (MIT/Apache/
  BSD/CC-BY): converted models are pushed to the public `TigreGotico/vconnx-<engine>` HF repo with
  the upstream LICENSE file and PROVENANCE.md alongside; adapters download
  them automatically.
- **local-only-weights** — upstream license does not permit redistribution
  (NC/ND variants, unlicensed repos) but inference and private conversion are
  fine: the conversion script runs on YOUR machine, `write_manifest(...,
  distributable=False)` marks the output, `push_models` refuses to upload it,
  and the adapter loads from the local path (`model_dir` config key) instead
  of HF. GPL upstreams additionally require the conversion script to invoke
  the upstream repo as an external checkout (never vendor GPL code here).

The engine's tracking issue carries the `local-only-weights` label when the
second class applies.

## Worked example: triaan-vc

TriAAN-VC (ICASSP 2023, MIT license) is a three-component pipeline.
All three components are exported from the GitHub release v1.0 checkpoints.

### License verification

The upstream repository at `winddori2002/TriAAN-VC` carries an MIT license.
The CPC encoder checkpoint (`cpc.pt`) is derived from `facebookresearch/CPC_audio`,
also MIT.  The ParallelWaveGAN vocoder is MIT via `kan-bayashi/ParallelWaveGAN`.
All three weights are distributable.

### Deviations from standard export

**CPC encoder architecture reconstruction** — there is no public Python model
definition for the exact CPC checkpoint used by TriAAN-VC (`gEncoder` + `gAR`
key layout).  The encoder was reconstructed by inspecting checkpoint keys and
shapes:

- 5-layer `Conv1d` with `bias=True`, 256 channels throughout
- `LearnedNorm1d` (per-channel affine, shape `(1, C, 1)`) instead of standard
  `BatchNorm1d` — required because the stored key shape differs from BatchNorm
- Single-layer LSTM (not GRU) — verified from `weight_ih_l0` shape `(1024, 256)` = 4×256 LSTM gates

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
