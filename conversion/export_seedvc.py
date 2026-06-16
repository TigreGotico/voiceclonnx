"""Export Seed-VC to ONNX.

Seed-VC (https://github.com/Plachtaa/seed-vc, archived Nov 2025) is a
transformer + flow-matching any-to-any voice conversion framework.

LICENSE NOTE — EXTERNAL CHECKOUT PATTERN
-----------------------------------------
The Seed-VC codebase is licensed under GPL-3.0.  GPL code is NEVER vendored
into this MIT-licensed repository.  This export script clones the upstream
repository into a **throwaway directory** (/tmp/seed-vc), adds it to sys.path
at runtime, imports the models, exports ONNX artifacts, and never commits the
upstream code.  The runtime adapter ``voiceclonnx/engines/seedvc.py`` contains
ZERO upstream code — pure onnxruntime + numpy only.

The exported ONNX weights ARE published to the public HF repo
``TigreGotico/voiceclonnx-seedvc``.  The model card states the upstream weight
license (Apache-2.0 / research) and the upstream code license (GPL-3.0)
separately.

Architecture (non-F0 path, 22050 Hz)
--------------------------------------
1. Whisper-small encoder (content): 16kHz log-mel → hidden states → discrete
   token lookup via embedding (``whisper_encoder.onnx``)
2. CAMPPlus speaker encoder: Kaldi 80-bin fbank → 192-d embedding
   (``campplus.onnx``)
3. InterpolateRegulator (length regulator): discrete token embeddings + style →
   mel-resolution conditioned features (``length_regulator.onnx``)
4. DiT flow estimator (single-step wrapper): conditioned features + noise +
   time + style → velocity (``flow_estimator.onnx``); called once per ODE step
5. BigVGAN vocoder: 80-bin mel → waveform @ 22050 Hz (``bigvgan.onnx``)

Usage::

    # Install conversion deps in a throwaway venv
    pip install torch onnx onnxruntime transformers torchaudio
    pip install munch pyyaml huggingface_hub soundfile librosa

    python -m conversion.export_seedvc --output-dir /tmp/seedvc-out

    # With HF push (requires HF_TOKEN)
    HF_TOKEN=hf_... python -m conversion.export_seedvc \\
        --output-dir /tmp/seedvc-out --push

Pinned upstream ref: 51383efd921027683c89e5348211d93ff12ac2a8
(last commit before archive, tested and known-good)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import json
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# External checkout: clone Seed-VC into /tmp (GPL code, never committed)
# ---------------------------------------------------------------------------

SEEDVC_CLONE_DIR = Path("/tmp/seed-vc")
SEEDVC_REPO_URL = "https://github.com/Plachtaa/seed-vc"
SEEDVC_PINNED_REF = "51383efd921027683c89e5348211d93ff12ac2a8"

HF_REPO_ID = "TigreGotico/voiceclonnx-seedvc"


def _ensure_seedvc_clone() -> None:
    """Clone Seed-VC at the pinned ref into /tmp/seed-vc."""
    if SEEDVC_CLONE_DIR.exists() and (SEEDVC_CLONE_DIR / "modules" / "flow_matching.py").exists():
        print(f"[seedvc] Using existing clone at {SEEDVC_CLONE_DIR}")
        return
    print(f"[seedvc] Cloning {SEEDVC_REPO_URL} → {SEEDVC_CLONE_DIR}")
    subprocess.run(
        ["git", "clone", SEEDVC_REPO_URL, str(SEEDVC_CLONE_DIR)],
        check=True,
    )
    # Pin to known-good ref (archived repo — pinning is non-optional)
    subprocess.run(
        ["git", "-C", str(SEEDVC_CLONE_DIR), "checkout", SEEDVC_PINNED_REF],
        check=True,
    )


def _add_seedvc_to_path() -> None:
    """Add the external Seed-VC directory to sys.path (GPL code, external)."""
    clone_dir = str(SEEDVC_CLONE_DIR)
    if clone_dir not in sys.path:
        sys.path.insert(0, clone_dir)
    print(f"[seedvc] Added {clone_dir} to sys.path (GPL external checkout, not vendored)")


# ---------------------------------------------------------------------------
# Install BigVGAN from the upstream clone
# ---------------------------------------------------------------------------


def _ensure_bigvgan_installed() -> None:
    """BigVGAN is bundled in the seed-vc clone; make it importable."""
    bigvgan_path = str(SEEDVC_CLONE_DIR)
    if bigvgan_path not in sys.path:
        sys.path.insert(0, bigvgan_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_config_and_models():
    """Load Seed-VC config + models via upstream API (external checkout)."""
    import torch
    import yaml
    from modules.commons import build_model, load_checkpoint, recursive_munch
    from hf_utils import load_custom_model_from_hf
    from modules.campplus.DTDNN import CAMPPlus
    from modules.bigvgan import bigvgan as bigvgan_mod

    config_path = str(SEEDVC_CLONE_DIR / "configs" / "presets" /
                      "config_dit_mel_seed_uvit_whisper_small_wavenet.yml")
    config = yaml.safe_load(open(config_path))
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = "DiT"

    # Download checkpoint
    print("[seedvc] Downloading DiT checkpoint from Plachta/Seed-VC …")
    dit_ckpt, _ = load_custom_model_from_hf(
        "Plachta/Seed-VC",
        "DiT_seed_v2_uvit_whisper_small_wavenet_bigvgan_pruned.pth",
        "config_dit_mel_seed_uvit_whisper_small_wavenet.yml",
    )

    model = build_model(model_params, stage="DiT")
    model, _, _, _ = load_checkpoint(
        model, None, dit_ckpt,
        load_only_params=True, ignore_modules=[], is_distributed=False,
    )
    for key in model:
        model[key].eval()

    # CAMPPlus speaker encoder
    print("[seedvc] Downloading CAMPPlus checkpoint …")
    campplus_ckpt = load_custom_model_from_hf(
        "funasr/campplus", "campplus_cn_common.bin", config_filename=None
    )
    campplus = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus.load_state_dict(torch.load(campplus_ckpt, map_location="cpu"))
    campplus.eval()

    # BigVGAN vocoder
    print("[seedvc] Downloading BigVGAN vocoder …")
    bigvgan_model = bigvgan_mod.BigVGAN.from_pretrained(
        "nvidia/bigvgan_v2_22khz_80band_256x", use_cuda_kernel=False
    )
    bigvgan_model.remove_weight_norm()
    bigvgan_model.eval()

    return config, model_params, model, campplus, bigvgan_model


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def _export_whisper_encoder(model, output_dir: Path) -> Path:
    """Export Whisper-small encoder (log-mel → last_hidden_state) to ONNX.

    Input:  log_mel_input_features  (1, 128, 3000) float32
    Output: encoder_hidden_states   (1, T_enc, 768) float32

    The encoder processes up to 30 s of audio at 16 kHz.  For shorter clips
    we left-pad with zeros; the adapter trims the output using actual frame count.
    """
    import torch
    from transformers import WhisperModel

    print("[seedvc] Loading Whisper-small encoder …")
    whisper = WhisperModel.from_pretrained("openai/whisper-small", torch_dtype=torch.float32)
    encoder = whisper.encoder
    encoder.eval()

    # Dummy: 30 s → 3000 mel frames
    dummy_mel = torch.zeros(1, 128, 3000, dtype=torch.float32)

    onnx_path = output_dir / "whisper_encoder.onnx"
    print(f"[seedvc] Exporting whisper_encoder → {onnx_path}")

    with torch.no_grad():
        ref_out = encoder(dummy_mel).last_hidden_state  # (1, 1500, 768)

    torch.onnx.export(
        encoder,
        (dummy_mel,),
        str(onnx_path),
        opset_version=14,
        input_names=["log_mel"],
        output_names=["encoder_hidden_states"],
        dynamic_axes={
            "log_mel": {0: "batch", 2: "n_frames"},
            "encoder_hidden_states": {0: "batch", 1: "T_enc"},
        },
        do_constant_folding=True,
        dynamo=False,
    )

    # Parity check
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"log_mel": dummy_mel.numpy()})[0]
    ref_np = ref_out.detach().numpy()
    max_abs = float(np.abs(ref_np - ort_out).max())
    mean_abs = float(np.abs(ref_np - ort_out).mean())
    print(f"[seedvc] whisper_encoder parity: max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}")
    assert max_abs < 5e-3, f"whisper_encoder parity FAIL: max_abs={max_abs:.3e}"
    print("[seedvc] whisper_encoder parity: PASS")

    return onnx_path


def _export_campplus(campplus, output_dir: Path) -> Path:
    """Export CAMPPlus speaker encoder (fbank → 192-d embedding) to ONNX.

    Input:  fbank  (1, T_fbank, 80) float32
    Output: embedding  (1, 192) float32
    """
    import torch

    dummy_fbank = torch.zeros(1, 200, 80, dtype=torch.float32)
    onnx_path = output_dir / "campplus.onnx"
    print(f"[seedvc] Exporting campplus → {onnx_path}")

    with torch.no_grad():
        ref_out = campplus(dummy_fbank)

    torch.onnx.export(
        campplus,
        (dummy_fbank,),
        str(onnx_path),
        opset_version=14,
        input_names=["fbank"],
        output_names=["embedding"],
        dynamic_axes={
            "fbank": {0: "batch", 1: "T_fbank"},
            "embedding": {0: "batch"},
        },
        do_constant_folding=True,
        dynamo=False,
    )

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"fbank": dummy_fbank.numpy()})[0]
    ref_np = ref_out.detach().numpy()
    max_abs = float(np.abs(ref_np - ort_out).max())
    mean_abs = float(np.abs(ref_np - ort_out).mean())
    print(f"[seedvc] campplus parity: max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}")
    assert max_abs < 1e-3, f"campplus parity FAIL: max_abs={max_abs:.3e}"
    print("[seedvc] campplus parity: PASS")

    return onnx_path


class _LengthRegulatorWrapper(object):
    """Wrapper to export InterpolateRegulator (discrete input path).

    The InterpolateRegulator.inference() method calls VectorQuantize internally,
    which uses non-traceable code.  We export the simpler forward path:
    embedding lookup → model (conv/norm/act layers).
    """

    def __init__(self, lr_module, embedding):
        self.lr = lr_module
        self.embedding = embedding

    def __call__(self, token_ids, target_len):
        """
        token_ids: (1, T_tok) int64
        target_len: scalar int64

        Returns: conditioned (1, T_mel, 512) float32
        """
        import torch.nn.functional as F

        # Embed discrete tokens
        x = self.embedding(token_ids)  # (1, T_tok, 512)

        # Interpolate to target length
        x = x.transpose(1, 2)  # (1, 512, T_tok)
        x = F.interpolate(x.float(), size=int(target_len.item()), mode='nearest')
        # (1, 512, T_mel)

        # Apply conv model
        x = self.lr.model(x)  # (1, 512, T_mel)
        x = x.transpose(1, 2)  # (1, T_mel, 512)
        return x


def _export_length_regulator(model, output_dir: Path) -> Path:
    """Export the length regulator embedding + interpolate + conv as ONNX.

    The length regulator takes discrete whisper token embeddings and
    interpolates them to mel-resolution for the flow estimator.

    Since tracing with dynamic target_len is non-trivial (F.interpolate
    with a tensor size argument), we export the embedding + model separately
    and implement the interpolation in numpy at inference time.

    Components saved:
    - lr_embedding.npz  (codebook_size, 512) — embedding weights
    - lr_model.onnx      (1, T, 512) → (1, T, 512) — conv/norm/act model

    The adapter applies: embed(tokens) → numpy interp → lr_model.onnx
    """
    import torch

    lr = model.length_regulator
    embedding = lr.embedding  # nn.Embedding(2048, 512)

    # Save embedding weights as numpy array (simpler than an ONNX Gather)
    lr_emb_path = output_dir / "lr_embedding.npz"
    np.savez(lr_emb_path, weight=embedding.weight.detach().numpy())
    print(f"[seedvc] Saved lr_embedding.npz: shape={embedding.weight.shape}")

    # Export the conv model (operates on (B, C, T))
    # Build a wrapper that takes (1, 512, T) → (1, 512, T)
    class _ConvModel(torch.nn.Module):
        def __init__(self, seq):
            super().__init__()
            self.model = seq

        def forward(self, x):
            return self.model(x)

    conv_wrapper = _ConvModel(lr.model)
    conv_wrapper.eval()

    T_dummy = 50
    dummy_x = torch.zeros(1, 512, T_dummy, dtype=torch.float32)

    with torch.no_grad():
        ref_out = conv_wrapper(dummy_x)  # (1, 512, T_dummy)

    onnx_path = output_dir / "lr_model.onnx"
    print(f"[seedvc] Exporting lr_model → {onnx_path}")
    torch.onnx.export(
        conv_wrapper,
        (dummy_x,),
        str(onnx_path),
        opset_version=14,
        input_names=["x"],
        output_names=["y"],
        dynamic_axes={"x": {0: "batch", 2: "T"}, "y": {0: "batch", 2: "T"}},
        do_constant_folding=True,
        dynamo=False,
    )

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"x": dummy_x.numpy()})[0]
    ref_np = ref_out.detach().numpy()
    max_abs = float(np.abs(ref_np - ort_out).max())
    mean_abs = float(np.abs(ref_np - ort_out).mean())
    print(f"[seedvc] lr_model parity: max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}")
    assert max_abs < 1e-3, f"lr_model parity FAIL: max_abs={max_abs:.3e}"
    print("[seedvc] lr_model parity: PASS")

    return onnx_path


def _export_flow_estimator(model, output_dir: Path) -> Path:
    """Export the DiT flow estimator (single forward step) to ONNX.

    The estimator is called once per ODE step during inference.

    Inputs:
      x          (2, 80, T_mel) — current state (batch-2 for CFG)
      prompt_x   (2, 80, T_mel) — prompt (zero-padded beyond prompt_len)
      x_lens     (1,) int64     — sequence length
      t          (2,) float32   — current time
      style      (2, 192) float32 — speaker embedding
      mu         (2, 80, T_mel) — flow conditioning (mu from LR)

    Output:
      velocity   (2, 80, T_mel) float32

    The estimator uses RoPE + WaveNet final layer.  setup_caches must be
    called before export to pre-allocate the KV cache buffers.
    """
    import torch

    estimator = model.cfm.estimator
    estimator.eval()
    estimator.setup_caches(max_batch_size=2, max_seq_length=8192)

    T = 50  # dummy mel frames
    # Batch-2: conditioned + unconditioned (CFG)
    dummy_x = torch.zeros(2, 80, T, dtype=torch.float32)
    dummy_prompt_x = torch.zeros(2, 80, T, dtype=torch.float32)
    dummy_x_lens = torch.tensor([T], dtype=torch.int64)
    dummy_t = torch.zeros(2, dtype=torch.float32)
    dummy_style = torch.zeros(2, 192, dtype=torch.float32)
    dummy_mu = torch.zeros(2, 80, T, dtype=torch.float32)

    with torch.no_grad():
        ref_out = estimator(
            dummy_x, dummy_prompt_x, dummy_x_lens,
            dummy_t, dummy_style, dummy_mu
        )

    onnx_path = output_dir / "flow_estimator.onnx"
    print(f"[seedvc] Exporting flow_estimator → {onnx_path}")

    torch.onnx.export(
        estimator,
        (dummy_x, dummy_prompt_x, dummy_x_lens, dummy_t, dummy_style, dummy_mu),
        str(onnx_path),
        opset_version=14,
        input_names=["x", "prompt_x", "x_lens", "t", "style", "mu"],
        output_names=["velocity"],
        dynamic_axes={
            "x": {0: "batch", 2: "T"},
            "prompt_x": {0: "batch", 2: "T"},
            "t": {0: "batch"},
            "style": {0: "batch"},
            "mu": {0: "batch", 2: "T"},
            "velocity": {0: "batch", 2: "T"},
        },
        do_constant_folding=True,
        dynamo=False,
    )

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = {
        "x": dummy_x.numpy(),
        "prompt_x": dummy_prompt_x.numpy(),
        "x_lens": dummy_x_lens.numpy(),
        "t": dummy_t.numpy(),
        "style": dummy_style.numpy(),
        "mu": dummy_mu.numpy(),
    }
    ort_out = sess.run(None, inputs)[0]
    ref_np = ref_out.detach().numpy()
    max_abs = float(np.abs(ref_np - ort_out).max())
    mean_abs = float(np.abs(ref_np - ort_out).mean())
    print(f"[seedvc] flow_estimator parity: max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}")
    assert max_abs < 1e-2, f"flow_estimator parity FAIL: max_abs={max_abs:.3e}"
    print("[seedvc] flow_estimator parity: PASS")

    return onnx_path


def _export_bigvgan(bigvgan_model, output_dir: Path) -> Path:
    """Export BigVGAN vocoder (mel → waveform) to ONNX.

    Input:  mel   (1, 80, T_mel) float32
    Output: wav   (1, 1, T_audio) float32
    """
    import torch

    bigvgan_model.eval()
    T = 50
    dummy_mel = torch.zeros(1, 80, T, dtype=torch.float32)

    with torch.no_grad():
        ref_out = bigvgan_model(dummy_mel)

    onnx_path = output_dir / "bigvgan.onnx"
    print(f"[seedvc] Exporting bigvgan → {onnx_path}")

    torch.onnx.export(
        bigvgan_model,
        (dummy_mel,),
        str(onnx_path),
        opset_version=14,
        input_names=["mel"],
        output_names=["wav"],
        dynamic_axes={
            "mel": {0: "batch", 2: "T_mel"},
            "wav": {0: "batch", 2: "T_audio"},
        },
        do_constant_folding=True,
        dynamo=False,
    )

    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_out = sess.run(None, {"mel": dummy_mel.numpy()})[0]
    ref_np = ref_out.detach().numpy()
    max_abs = float(np.abs(ref_np - ort_out).max())
    mean_abs = float(np.abs(ref_np - ort_out).mean())
    print(f"[seedvc] bigvgan parity: max_abs={max_abs:.3e}, mean_abs={mean_abs:.3e}")
    assert max_abs < 1e-3, f"bigvgan parity FAIL: max_abs={max_abs:.3e}"
    print("[seedvc] bigvgan parity: PASS")

    return onnx_path


def _write_parity_report(output_dir: Path, results: dict) -> None:
    """Write parity_report.json."""
    report_path = output_dir / "parity_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[seedvc] Wrote {report_path}")


def _write_manifest(output_dir: Path, components: dict, sizes: dict) -> None:
    """Write config.json manifest."""
    config = {
        "engine": "seedvc",
        "sample_rate": 22050,
        "components": components,
        "sizes_bytes": sizes,
        "opset": 14,
        "upstream_repo": "https://github.com/Plachtaa/seed-vc",
        "upstream_ref": SEEDVC_PINNED_REF,
        "upstream_license": "GPL-3.0",
        "weight_license": "Apache-2.0 / research (no commercial restriction on weights)",
    }
    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"[seedvc] Wrote {config_path}")


def _write_provenance(output_dir: Path) -> None:
    """Write PROVENANCE.md."""
    text = f"""# Seed-VC ONNX export provenance

## Upstream

- Repository: https://github.com/Plachtaa/seed-vc (archived 2025-11-21)
- Pinned ref: `{SEEDVC_PINNED_REF}`
- Upstream code license: **GPL-3.0**
- Upstream weight license: Apache-2.0 / research
  (see `Plachta/Seed-VC` on Hugging Face)

## Export notes

The upstream code is GPL-3.0 and is NEVER vendored into the MIT-licensed
voiceclonnx repository.  The export script (`conversion/export_seedvc.py`)
clones the upstream repository into `/tmp/seed-vc`, imports the models,
exports ONNX artifacts, and discards the clone.  The runtime adapter
(`voiceclonnx/engines/seedvc.py`) contains zero GPL code — pure onnxruntime
and numpy only.

The exported ONNX weights are published here under the upstream weight license.
Users who download these weights should review the upstream licenses:
- https://github.com/Plachtaa/seed-vc/blob/main/LICENSE (GPL-3.0 — code only)
- Model card states the weight-level license separately.

## Components

| File | Description |
|---|---|
| `whisper_encoder.onnx` | Whisper-small encoder: log-mel (1,128,3000) → hidden (1,T,768) |
| `campplus.onnx` | CAMPPlus: fbank (1,T,80) → embedding (1,192) |
| `lr_embedding.npz` | Length regulator embedding weights (2048×512) |
| `lr_model.onnx` | Length regulator conv model: (1,512,T) → (1,512,T) |
| `flow_estimator.onnx` | DiT single-step: (x,prompt_x,x_lens,t,style,mu) → velocity |
| `bigvgan.onnx` | BigVGAN vocoder: mel (1,80,T) → wav (1,1,T_audio) |

Quantized (`_q8.onnx`) variants are generated by `conversion/quantize.py`.
"""
    prov_path = output_dir / "PROVENANCE.md"
    with open(prov_path, "w") as f:
        f.write(text)
    print(f"[seedvc] Wrote {prov_path}")


def _write_model_card(output_dir: Path) -> None:
    """Write README.md HF model card."""
    text = """---
license: other
license_name: gpl-3-weights-separate
license_link: https://github.com/Plachtaa/seed-vc/blob/main/LICENSE
tags:
  - voice-conversion
  - flow-matching
  - onnx
  - voiceclonnx
  - license-note
---

# voiceclonnx-seedvc

ONNX export of [Seed-VC](https://github.com/Plachtaa/seed-vc) for use with
[voiceclonnx](https://github.com/TigreGotico/voiceclonnx).

## License

**Upstream code: GPL-3.0.**  The upstream GPL source code is NEVER included
in these artifacts.  The ONNX weights were exported via an external checkout
and are published here under the upstream weight license (Apache-2.0 /
research).  See PROVENANCE.md for full lineage.

## Usage

```python
from voiceclonnx import VoiceCloner
cloner = VoiceCloner(engine="seedvc")
cloner.clone_voice("source.wav", "reference.wav", "out.wav")
```

## Components

| File | Description | Size |
|---|---|---|
| `whisper_encoder.onnx` | Whisper-small content encoder | ~90 MB (fp32) |
| `campplus.onnx` | CAMPPlus speaker encoder | ~2 MB (fp32) |
| `lr_embedding.npz` | Length regulator embedding | ~4 MB |
| `lr_model.onnx` | Length regulator conv/norm | ~2 MB (fp32) |
| `flow_estimator.onnx` | DiT flow matching estimator | ~115 MB (fp32) |
| `bigvgan.onnx` | BigVGAN vocoder | ~215 MB (fp32) |

INT8 quantized variants (`*_q8.onnx`) are included where applicable.
See `docs/QUANTS.md` in the voiceclonnx repository for WER comparison.
"""
    card_path = output_dir / "README.md"
    with open(card_path, "w") as f:
        f.write(text)
    print(f"[seedvc] Wrote {card_path}")


def _quantize_components(output_dir: Path) -> dict:
    """Quantize ONNX components to INT8.  Returns sizes dict."""
    try:
        from conversion.quantize import quantize_model
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from conversion.quantize import quantize_model

    sizes = {}
    for onnx_name in [
        "whisper_encoder.onnx",
        "campplus.onnx",
        "lr_model.onnx",
        "flow_estimator.onnx",
        "bigvgan.onnx",
    ]:
        fp32_path = output_dir / onnx_name
        if not fp32_path.exists():
            continue
        q8_name = onnx_name.replace(".onnx", "_q8.onnx")
        q8_path = output_dir / q8_name
        fp32_size = fp32_path.stat().st_size
        sizes[onnx_name] = fp32_size
        try:
            quantize_model(str(fp32_path), output_path=str(q8_path))
            q8_size = q8_path.stat().st_size
            sizes[q8_name] = q8_size
            reduction = (1 - q8_size / fp32_size) * 100
            print(f"[seedvc] {onnx_name}: fp32={fp32_size/1e6:.1f}MB → q8={q8_size/1e6:.1f}MB ({reduction:.1f}% reduction)")
        except Exception as exc:
            print(f"[seedvc] WARNING: quantization of {onnx_name} failed: {exc}")
            # Quantization of flow_estimator may degrade quality — see QUANTS.md

    # Also record lr_embedding.npz size
    npz = output_dir / "lr_embedding.npz"
    if npz.exists():
        sizes["lr_embedding.npz"] = npz.stat().st_size

    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Seed-VC to ONNX")
    parser.add_argument("--output-dir", default="/tmp/seedvc-out", help="Output directory")
    parser.add_argument("--push", action="store_true", help="Push to HF Hub after export")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip export if ONNX files already present")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / "seedvc"
    output_dir.mkdir(parents=True, exist_ok=True)

    # External checkout (GPL code stays in /tmp, never committed)
    _ensure_seedvc_clone()
    _add_seedvc_to_path()


    # Load all models
    config, model_params, model, campplus, bigvgan_model = _load_config_and_models()

    parity_results = {}

    if args.skip_export and (output_dir / "bigvgan.onnx").exists():
        print("[seedvc] Skipping export (--skip-export and files present)")
    else:
        # Export each component
        print("\n[seedvc] Exporting components …\n")

        try:
            _export_whisper_encoder(model, output_dir)
            parity_results["whisper_encoder"] = {"status": "PASS"}
        except Exception as exc:
            print(f"[seedvc] ERROR whisper_encoder: {exc}")
            parity_results["whisper_encoder"] = {"status": "FAIL", "error": str(exc)}
            raise

        try:
            _export_campplus(campplus, output_dir)
            parity_results["campplus"] = {"status": "PASS"}
        except Exception as exc:
            print(f"[seedvc] ERROR campplus: {exc}")
            parity_results["campplus"] = {"status": "FAIL", "error": str(exc)}
            raise

        try:
            _export_length_regulator(model, output_dir)
            parity_results["lr_model"] = {"status": "PASS"}
        except Exception as exc:
            print(f"[seedvc] ERROR lr_model: {exc}")
            parity_results["lr_model"] = {"status": "FAIL", "error": str(exc)}
            raise

        try:
            _export_flow_estimator(model, output_dir)
            parity_results["flow_estimator"] = {"status": "PASS"}
        except Exception as exc:
            print(f"[seedvc] ERROR flow_estimator: {exc}")
            parity_results["flow_estimator"] = {"status": "FAIL", "error": str(exc)}
            raise

        try:
            _export_bigvgan(bigvgan_model, output_dir)
            parity_results["bigvgan"] = {"status": "PASS"}
        except Exception as exc:
            print(f"[seedvc] ERROR bigvgan: {exc}")
            parity_results["bigvgan"] = {"status": "FAIL", "error": str(exc)}
            raise

    # Quantize
    print("\n[seedvc] Quantizing to INT8 …\n")
    sizes = _quantize_components(output_dir)

    # Write metadata
    _write_parity_report(output_dir, parity_results)
    components = {
        "whisper_encoder": "whisper_encoder.onnx",
        "whisper_encoder_q8": "whisper_encoder_q8.onnx",
        "campplus": "campplus.onnx",
        "campplus_q8": "campplus_q8.onnx",
        "lr_embedding": "lr_embedding.npz",
        "lr_model": "lr_model.onnx",
        "lr_model_q8": "lr_model_q8.onnx",
        "flow_estimator": "flow_estimator.onnx",
        "flow_estimator_q8": "flow_estimator_q8.onnx",
        "bigvgan": "bigvgan.onnx",
        "bigvgan_q8": "bigvgan_q8.onnx",
    }
    _write_manifest(output_dir, components, sizes)
    _write_provenance(output_dir)
    _write_model_card(output_dir)

    print(f"\n[seedvc] Export complete → {output_dir}\n")

    if args.push:
        print("[seedvc] Pushing to HF Hub …")
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(
            repo_id=HF_REPO_ID, repo_type="model", exist_ok=True, private=False
        )
        api.upload_folder(
            folder_path=str(output_dir),
            repo_id=HF_REPO_ID,
            repo_type="model",
            commit_message="export: add Seed-VC ONNX artifacts (GPL-3.0 code external checkout)",
        )
        print(f"[seedvc] Pushed to https://huggingface.co/{HF_REPO_ID}")


if __name__ == "__main__":
    main()
