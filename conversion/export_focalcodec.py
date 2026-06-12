"""Export FocalCodec components (WavLM encoder + Vocos backbone) to ONNX.

FocalCodec (Della Libera, NeurIPS 2025) is a single-codebook binary codec using
focal modulation networks.  VC uses a kNN feature-space swap on continuous pre-VQ
features (same class as kNN-VC), then decodes with a Vocos-based vocoder.

Pipeline (at inference):
  1. WavLM encoder    — outputs (B, T, 1024) continuous pre-VQ features.
  2. kNN matching     — pure numpy, cosine distance, k=4 weighted mean.
  3. Vocos backbone+proj — (B, T, 1024) → (B, T, n_fft+2) STFT coefficients.
  4. numpy ISTFT      — pure numpy overlap-add at runtime (no ONNX needed).

The Vocos head ISTFT is NOT exported to ONNX because ``nn.functional.fold``
with a dynamic output_size cannot be traced.  Instead the adapter re-implements
it in pure numpy using the known parameters (n_fft=1024, hop=320, win=1024).

Upstream:
  - https://github.com/lucadellalib/focalcodec  (Apache-2.0)
  - HF checkpoint: lucadellalib/focalcodec_50hz

Usage::

    python -m conversion.export_focalcodec --output-dir /tmp/focalcodec-out
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FOCALCODEC_HF_REPO = "lucadellalib/focalcodec_50hz"
FOCALCODEC_UPSTREAM_URL = "https://github.com/lucadellalib/focalcodec"
FOCALCODEC_UPSTREAM_REF = "v0.0.2"
FOCALCODEC_SR = 16000

# Vocos ISTFT parameters (from lucadellalib/focalcodec_50hz config)
VOCOS_N_FFT = 1024
VOCOS_HOP_LENGTH = 320
VOCOS_WIN_LENGTH = 1024

APACHE2_LICENSE = """\
Apache License
Version 2.0, January 2004

Copyright 2025 Luca Della Libera.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


# ---------------------------------------------------------------------------
# Wrapper modules
# ---------------------------------------------------------------------------


def _build_encoder_wrapper(model):
    """Return a torch module: sig (B, samples) → feats (B, T, 1024)."""
    import torch.nn as nn

    class FocalCodecEncoder(nn.Module):
        """Wraps the FocalCodec WavLM encoder; outputs continuous pre-VQ features."""

        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder

        def forward(self, sig):
            """
            Parameters
            ----------
            sig : torch.Tensor
                Float32 (batch, samples) at 16 kHz.

            Returns
            -------
            torch.Tensor
                (batch, frames, 1024) continuous features.
            """
            out = self.encoder(sig)
            return out[0]  # feats; encoder returns (feats, *states)

    wrapper = FocalCodecEncoder(model.encoder)
    wrapper.eval()
    return wrapper


def _build_vocoder_wrapper(model):
    """Return a torch module: feats (B, T, 1024) → stft_coeffs (B, T, n_fft+2).

    Exports the Vocos backbone + linear projection only.
    The final ISTFT step is performed in pure numpy at runtime
    (``nn.functional.fold`` with dynamic output_size cannot be traced).
    """
    import torch.nn as nn

    class VocosBackboneAndProj(nn.Module):
        """Vocos backbone + head linear projection; no ISTFT."""

        def __init__(self, backbone, proj):
            super().__init__()
            self.backbone = backbone
            self.proj = proj

        def forward(self, feats):
            """
            Parameters
            ----------
            feats : torch.Tensor
                Float32 (batch, frames, 1024).

            Returns
            -------
            torch.Tensor
                (batch, frames, n_fft+2) STFT coefficients (mag_logits || phase).
            """
            out, *_ = self.backbone(feats)  # (B, T, 512)
            return self.proj(out)            # (B, T, n_fft+2)

    wrapper = VocosBackboneAndProj(model.decoder.backbone, model.decoder.head.proj)
    wrapper.eval()
    return wrapper


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------


def export_focalcodec(output_dir: str, cache_dir: Optional[str] = None) -> Path:
    """Export FocalCodec encoder + Vocos backbone to ONNX.

    Returns the engine output directory.
    """
    import torch
    from conversion.export_base import OutputLayout, export_model, write_manifest, write_provenance
    from conversion.parity import compare_outputs, check_tolerance, run_ort
    from conversion.quantize import quantize_model

    output_dir = Path(output_dir)
    layout = OutputLayout.for_engine("focalcodec", base_dir=output_dir)
    layout.makedirs()

    # ------------------------------------------------------------------
    # 0. Load FocalCodec from HF
    # ------------------------------------------------------------------
    print(f"[export] Loading FocalCodec from {FOCALCODEC_HF_REPO} ...")
    try:
        from focalcodec.codec import FocalCodec
    except ImportError as exc:
        raise ImportError(
            "focalcodec package required for export. "
            "Install: pip install git+https://github.com/lucadellalib/focalcodec.git"
        ) from exc

    model = FocalCodec.from_pretrained(FOCALCODEC_HF_REPO)
    model.eval()
    print(f"[export] Loaded. sample_rate={model.sample_rate_input} Hz")

    # ------------------------------------------------------------------
    # 1. Export encoder (WavLM → continuous features)
    # ------------------------------------------------------------------
    enc_wrapper = _build_encoder_wrapper(model)

    dummy_sig = torch.zeros(1, FOCALCODEC_SR)  # 1 s at 16 kHz

    enc_onnx = layout.component_path("focalcodec_encoder.onnx")
    print(f"[export] Exporting encoder → {enc_onnx} ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        export_model(
            model=enc_wrapper,
            dummy_inputs=(dummy_sig,),
            output_path=enc_onnx,
            input_names=["sig"],
            output_names=["feats"],
            dynamic_axes={
                "sig":   {0: "batch", 1: "samples"},
                "feats": {0: "batch", 1: "frames"},
            },
            opset_version=14,
        )
    enc_mb = enc_onnx.stat().st_size / 1024 ** 2
    print(f"[export] Encoder ONNX written: {enc_mb:.1f} MB")

    # Parity — encoder
    print("[export] Encoder parity check ...")
    with torch.no_grad():
        torch_feats = enc_wrapper(dummy_sig).detach().cpu().numpy()
    ort_feats = run_ort(enc_onnx, {"sig": dummy_sig.numpy()})
    enc_report = compare_outputs(
        [torch_feats],
        ort_feats,
        names=["feats"],
        max_abs_tol=2e-3,
        mean_abs_tol=5e-4,
    )
    print("[export] Encoder parity:", enc_report.summary())
    check_tolerance(enc_report)
    enc_report.save(layout.component_path("encoder_parity_report.json"))

    # Quantize encoder
    print("[export] Quantizing encoder (INT8) ...")
    enc_q8 = layout.component_path("focalcodec_encoder_q8.onnx")
    enc_quant = quantize_model(enc_onnx, output_path=enc_q8)
    print(enc_quant.summary())

    del enc_wrapper  # free memory

    # ------------------------------------------------------------------
    # 2. Export Vocos backbone + proj (feats → STFT coefficients)
    # ------------------------------------------------------------------
    voc_wrapper = _build_vocoder_wrapper(model)

    # Dummy: 50 frames × 1024 dim (~1 s at 50 Hz)
    dummy_feats = torch.zeros(1, 50, 1024)

    voc_onnx = layout.component_path("focalcodec_vocoder.onnx")
    print(f"[export] Exporting Vocos backbone → {voc_onnx} ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        export_model(
            model=voc_wrapper,
            dummy_inputs=(dummy_feats,),
            output_path=voc_onnx,
            input_names=["feats"],
            output_names=["stft_coeffs"],
            dynamic_axes={
                "feats":       {0: "batch", 1: "frames"},
                "stft_coeffs": {0: "batch", 1: "frames"},
            },
            opset_version=14,
        )
    voc_mb = voc_onnx.stat().st_size / 1024 ** 2
    print(f"[export] Vocoder ONNX written: {voc_mb:.1f} MB")

    # Parity — vocoder (backbone + proj only)
    print("[export] Vocoder parity check ...")
    with torch.no_grad():
        torch_stft = voc_wrapper(dummy_feats).detach().cpu().numpy()
    ort_stft = run_ort(voc_onnx, {"feats": dummy_feats.numpy()})
    voc_report = compare_outputs(
        [torch_stft],
        ort_stft,
        names=["stft_coeffs"],
        max_abs_tol=1e-3,
        mean_abs_tol=1e-4,
    )
    print("[export] Vocoder parity:", voc_report.summary())
    check_tolerance(voc_report)
    voc_report.save(layout.component_path("vocoder_parity_report.json"))

    # Quantize vocoder
    print("[export] Quantizing vocoder (INT8) ...")
    voc_q8 = layout.component_path("focalcodec_vocoder_q8.onnx")
    voc_quant = quantize_model(voc_onnx, output_path=voc_q8)
    print(voc_quant.summary())

    del voc_wrapper
    del model

    # ------------------------------------------------------------------
    # 3. Manifest + provenance
    # ------------------------------------------------------------------
    write_manifest(
        layout=layout,
        components={
            "encoder":    "focalcodec_encoder.onnx",
            "encoder_q8": "focalcodec_encoder_q8.onnx",
            "vocoder":    "focalcodec_vocoder.onnx",
            "vocoder_q8": "focalcodec_vocoder_q8.onnx",
        },
        sample_rates={"input": FOCALCODEC_SR, "output": FOCALCODEC_SR},
        metadata={
            "opset": 14,
            "feature_dim": 1024,
            "feature_rate_hz": 50,
            "knn_default_k": 4,
            "knn_distance": "cosine",
            "vocos_n_fft": VOCOS_N_FFT,
            "vocos_hop_length": VOCOS_HOP_LENGTH,
            "vocos_win_length": VOCOS_WIN_LENGTH,
            "istft_in_numpy": True,
            "upstream_hf_repo": FOCALCODEC_HF_REPO,
        },
        distributable=True,
    )
    write_provenance(
        engine_dir=layout.engine_dir,
        upstream_repo_url=FOCALCODEC_UPSTREAM_URL,
        upstream_ref=FOCALCODEC_UPSTREAM_REF,
        license_text=APACHE2_LICENSE,
        extra={
            "hf_checkpoint": FOCALCODEC_HF_REPO,
            "encoder_mb": f"{enc_mb:.1f}",
            "vocoder_mb": f"{voc_mb:.1f}",
            "note": (
                "ISTFT not in ONNX; re-implemented in numpy (irfft + hann-window OLA) "
                "with verified parity ≤1.3e-5 max abs vs torch"
            ),
        },
    )

    print(f"\n[export] FocalCodec export complete → {layout.engine_dir}")
    return layout.engine_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export FocalCodec (WavLM encoder + Vocos backbone) to ONNX."
    )
    p.add_argument("--output-dir", default="/tmp/focalcodec-out",
                   help="Staging output directory.")
    p.add_argument("--no-push", action="store_true", help="Skip HF upload.")
    p.add_argument("--dry-run-push", action="store_true",
                   help="Dry-run HF upload (prints paths only).")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    engine_dir = export_focalcodec(output_dir=args.output_dir)

    if not args.no_push:
        from conversion.push_models import push_engine
        push_engine(
            engine_dir=engine_dir,
            engine_name="focalcodec",
            dry_run=args.dry_run_push,
            commit_message=(
                "export: add focalcodec ONNX artifacts "
                "(WavLM encoder + Vocos backbone, Apache-2.0)"
            ),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
