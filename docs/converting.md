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

Upload the finished engine directory to `TigreGotico/vconnx-models`:

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
   (or `snapshot_download`) using `TigreGotico/vconnx-models` and the engine
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

This mirrors the layout on HF Hub under `TigreGotico/vconnx-models/<engine>/`.

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

## Weight-license policy: distributable vs local-only

vconnx never redistributes model weights it has no right to. Two classes of
engine, decided per-engine in its tracking issue:

- **distributable** — upstream license permits redistribution (MIT/Apache/
  BSD/CC-BY): converted models are pushed to the `vconnx-models` HF repo with
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
