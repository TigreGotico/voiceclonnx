# Local-only weights walkthrough

Some engines cannot have their ONNX artifacts redistributed because the upstream
model license does not permit it (e.g. non-commercial / no-derivatives variants).
For those engines the conversion script runs on **your own machine** and the adapter
loads from a local path instead of downloading from HF Hub.

---

## 1. Identify the engine class

An engine with non-redistributable weights has its tracking issue labelled
`local-only-weights`. The manifest written by the export script will contain
`"distributable": false`, and the `push_models` script will refuse to upload it.

---

## 2. Run the conversion script yourself

```bash
# Install the conversion toolchain extras
pip install -e ".[convert]"

# Run the per-engine export script
# (Replace <engine> with the actual engine name, e.g. rvc)
python -m conversion.export_<engine> \
    --output-dir ~/vconnx-models/<engine> \
    --no-push
```

The script:
1. Downloads the upstream weights to your local HF cache.
2. Exports ONNX artifacts to `~/vconnx-models/<engine>/`.
3. Runs parity checks and fails if tolerances are exceeded.
4. Produces `_q8.onnx` quantized variants.
5. Writes `config.json` and `PROVENANCE.md`.

The `--no-push` flag skips the HF Hub upload step (and is required for
`distributable=False` engines — the push script enforces this).

---

## 3. Point the adapter at your local models

Pass `model_dir` when constructing `VoiceCloner`:

```python
from vconnx import VoiceCloner

cloner = VoiceCloner(
    engine="rvc",                           # or whichever local-only engine
    model_dir="~/vconnx-models/rvc",        # path to the export output directory
    quantized=True,                         # optional: use _q8.onnx variants
)
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
```

Or via the CLI (engine-specific flag — check the engine guide):

```bash
vconnx clone --engine rvc \
             --model-dir ~/vconnx-models/rvc \
             --audio source.wav \
             --voice reference.wav \
             --out out.wav
```

---

## 4. Expected directory layout after export

```
~/vconnx-models/<engine>/
  config.json         ← manifest (components, sample rates, distributable: false)
  PROVENANCE.md       ← upstream lineage + license text
  <component>.onnx    ← full-precision model
  <component>_q8.onnx ← INT8 quantized variant
  parity_report.json  ← parity check results
```

The adapter reads `config.json` to discover component paths, so the directory
layout must match exactly.

---

## 5. Sharing with others

Because the weights are non-redistributable you cannot share the converted ONNX
files. You **can** share your conversion script changes (MIT-licensed under vconnx)
so others can run the conversion themselves. The `PROVENANCE.md` records the exact
upstream checkpoint and license, making attribution clear.
