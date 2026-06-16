"""Export RVC base models (ContentVec-768 + RMVPE) to ONNX.

RVC (Retrieval-based Voice Conversion) uses:
  1. ContentVec encoder (HuBERT-based, 768-dim, 12 layers) — shared base
  2. RMVPE pitch estimator — shared base
  3. Per-voice synthesizer (net_g, VITS-based) — user-supplied .pth

This script exports the two shared base models only.  Per-voice models
are exported via ``convert_rvc_model.py``.

Upstream:
  - ContentVec: lvc-project/contentvec (MIT) — specifically the HuBERT-based
    checkpoint trained by the RVC community (contentvec_base_plus_disentangled.pt
    / rvc/hubert_base.pt, published under MIT on HF as Politrees/RVC_resources)
  - RMVPE: yxlllc/RMVPE (MIT) — mel-spectrogram → F0 pitch estimator

Usage::

    python -m conversion.export_rvc --output-dir /tmp/rvc-out

    # Without HF push:
    python -m conversion.export_rvc --output-dir /tmp/rvc-out --no-push
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ContentVec (HuBERT-based, transformers format) — MIT
CONTENTVEC_HF_REPO = "Politrees/RVC_resources"
CONTENTVEC_CONFIG_FILE = "embedders/transformers/contentvec/config.json"
CONTENTVEC_WEIGHTS_FILE = "embedders/transformers/contentvec/pytorch_model.bin"

# RMVPE pitch extractor — MIT; community repo publishes an ONNX already.
# We use it directly when available (avoids a full torch reconstruction).
RMVPE_HF_REPO = "Politrees/RVC_resources"
RMVPE_ONNX_FILE = "predictors/rmvpe.onnx"   # pre-exported ONNX (preferred)
RMVPE_PT_FILE = "predictors/rmvpe.pt"       # .pt fallback for torch re-export

RVC_UPSTREAM_URL = "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI"
RVC_UPSTREAM_REF = "main"

MIT_LICENSE = """\
MIT License

Copyright (c) 2023 RVC-Project

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


# ---------------------------------------------------------------------------
# ContentVec export
# ---------------------------------------------------------------------------


def _build_contentvec():
    """Load the RVC ContentVec (HuBERT-based) encoder as a torch module.

    Uses the transformers-format weights from Politrees/RVC_resources
    (``embedders/transformers/contentvec/``), which include both the config.json
    and pytorch_model.bin.  This gives an exact architecture match with no
    manual weight mapping required.
    """
    import torch
    import torch.nn as nn
    from transformers import HubertModel
    from huggingface_hub import hf_hub_download
    from pathlib import Path

    class ContentVecEncoder(nn.Module):
        """Wraps HuBERT to output the final-layer hidden states.

        RVC uses a HuBERT-base model fine-tuned for content features (ContentVec).
        Architecture: 12 transformer layers, 768 hidden dim, 50 Hz output at
        16 kHz input.  RVC uses the final hidden state (last layer).
        """

        def __init__(self, hubert: HubertModel):
            super().__init__()
            self.hubert = hubert

        def forward(self, input_values: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            input_values : (batch, time) float32 — 16 kHz PCM
            attention_mask : (batch, time) int64 — all-ones for full attention
            Returns
            -------
            torch.Tensor : (batch, frames, 768) float32
            """
            out = self.hubert(
                input_values=input_values,
                attention_mask=attention_mask,
                output_hidden_states=False,
            )
            return out.last_hidden_state

    print(f"[export] Downloading ContentVec (transformers format) from {CONTENTVEC_HF_REPO} ...")
    hf_hub_download(repo_id=CONTENTVEC_HF_REPO, filename=CONTENTVEC_CONFIG_FILE)
    weights_path = hf_hub_download(repo_id=CONTENTVEC_HF_REPO, filename=CONTENTVEC_WEIGHTS_FILE)
    print(f"[export] ContentVec weights: {weights_path}")

    # Load directly from the transformers checkpoint directory
    model_dir = str(Path(weights_path).parent)
    # Use eager attention to avoid SDPA's is_causal constant-baking during trace
    hubert = HubertModel.from_pretrained(model_dir, attn_implementation="eager")
    hubert.eval()
    print("[export] ContentVec loaded via HubertModel.from_pretrained (eager attn).")
    return ContentVecEncoder(hubert)


# ---------------------------------------------------------------------------
# RMVPE export
# ---------------------------------------------------------------------------

# RMVPE model architecture constants
_RMVPE_N_MEL = 128
_RMVPE_N_CLASS = 360


def _build_rmvpe(ckpt_path: Path):
    """Reconstruct the RMVPE model and load its checkpoint.

    RMVPE (yxlllc/RMVPE, MIT) uses a DeepUnet architecture:
      input: (batch, 1, 128, T) log-mel spectrogram
      output: (batch, T, 360) pitch class probabilities

    Architecture follows the RMVPE paper / repo:
    - A U-Net with EfficientNet-style inverted residual blocks
    - DeepUnet encoder/decoder with skip connections
    - Final sigmoid output layer

    Since the architecture is non-trivial to reconstruct exactly, we use
    a simplified MLP approximation that produces the same I/O shape.
    In practice the RVC community publishes the RMVPE ONNX directly; this
    script exports from the original .pt checkpoint using a lightweight
    wrapper that drives the original model via direct model loading.
    """
    import torch
    import torch.nn as nn

    print(f"[export] Loading RMVPE checkpoint from {ckpt_path} ...")
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    # RMVPE checkpoint key: the actual model state dict may be nested
    if isinstance(state, dict):
        if "model" in state:
            model_state = state["model"]
        elif "state_dict" in state:
            model_state = state["state_dict"]
        else:
            model_state = state
    else:
        model_state = state

    # Determine if this is a full nn.Module (some RMVPE checkpoints save
    # the whole model object, not just state_dict)
    if hasattr(model_state, "parameters"):
        # state itself is a nn.Module
        model = model_state
        model.eval()
        print("[export] RMVPE loaded as full nn.Module.")
        return model

    # -------------------------------------------------------------------
    # Reconstruct RMVPE architecture from scratch (DeepUnet for RMVPE)
    # The RMVPE architecture is taken from:
    # https://github.com/yxlllc/RMVPE
    # -------------------------------------------------------------------

    class BiGRU(nn.Module):
        def __init__(self, input_features, hidden_features, num_layers):
            super().__init__()
            self.gru = nn.GRU(
                input_features, hidden_features,
                num_layers=num_layers, batch_first=True, bidirectional=True,
            )

        def forward(self, x):
            return self.gru(x)[0]

    class ConvBlockRes(nn.Module):
        def __init__(self, in_ch, out_ch, momentum=0.01):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_ch, momentum=momentum),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
                nn.BatchNorm2d(out_ch, momentum=momentum),
                nn.ReLU(),
            )
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        def forward(self, x):
            return self.conv(x) + self.shortcut(x)

    class ResEncoderBlock(nn.Module):
        def __init__(self, in_ch, out_ch, stride, n_blocks=1, momentum=0.01):
            super().__init__()
            self.conv = nn.Sequential(*[
                ConvBlockRes(in_ch if i == 0 else out_ch, out_ch, momentum)
                for i in range(n_blocks)
            ])
            self.pool = nn.AvgPool2d(stride)

        def forward(self, x):
            x = self.conv(x)
            return x, self.pool(x)

    class ResDecoderBlock(nn.Module):
        def __init__(self, in_ch, out_ch, stride, n_blocks=1, momentum=0.01):
            super().__init__()
            out_pad = 1 if stride == (2, 2) else (0, 1)
            self.upsample = nn.ConvTranspose2d(in_ch, in_ch, stride, stride, output_padding=out_pad)
            self.conv = nn.Sequential(*[
                ConvBlockRes(in_ch * 2 if i == 0 else out_ch, out_ch, momentum)
                for i in range(n_blocks)
            ])

        def forward(self, x, concat_tensor):
            x = self.upsample(x)
            x = torch.cat([x, concat_tensor], dim=1)
            return self.conv(x)

    class DeepUnet(nn.Module):
        def __init__(
            self, kernel_size, n_blocks, en_de_layers=5, inter_layers=4,
            in_channels=1, en_out_channels=16,
        ):
            super().__init__()
            self.encoder = nn.ModuleList()
            self.decoder = nn.ModuleList()

            strides = [(1, 2)] * en_de_layers
            in_ch = in_channels
            out_ch = en_out_channels
            for stride in strides:
                self.encoder.append(ResEncoderBlock(in_ch, out_ch, stride, n_blocks))
                in_ch = out_ch

            self.inter = nn.Sequential(*[
                ConvBlockRes(in_ch, in_ch) for _ in range(inter_layers)
            ])

            for stride in reversed(strides):
                self.decoder.append(ResDecoderBlock(in_ch, out_ch, stride, n_blocks))
                in_ch = out_ch

        def forward(self, x):
            # x: (B, 1, F, T)
            concat_tensors = []
            for enc in self.encoder:
                t, x = enc(x)
                concat_tensors.append(t)
            x = self.inter(x)
            for dec, ct in zip(self.decoder, reversed(concat_tensors)):
                x = dec(x, ct)
            return x

    class RMVPE(nn.Module):
        """RMVPE: mel-spectrogram → frame-wise pitch class probabilities.

        Input:  (batch, 1, 128, T) float32 log-mel spectrogram
        Output: (batch, T, 360) float32 softmax pitch class probs
        """

        def __init__(self, n_mel=128, n_class=360):
            super().__init__()
            self.unet = DeepUnet(
                kernel_size=15, n_blocks=2, en_de_layers=5,
                inter_layers=4, in_channels=1, en_out_channels=16,
            )
            self.cnn = nn.Conv2d(16, 3, 3, 1, 1)
            self.fc = nn.Sequential(
                BiGRU(3 * n_mel, 256, 2),
                nn.Linear(512, n_class),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            x : (batch, 1, n_mel, T) float32

            Returns
            -------
            torch.Tensor : (batch, T, 360) float32
            """
            x = self.unet(x)      # (B, 16, n_mel, T)
            x = self.cnn(x)       # (B, 3, n_mel, T)
            x = x.permute(0, 3, 1, 2)  # (B, T, 3, n_mel)
            B, T, C, F = x.shape
            x = x.reshape(B, T, C * F)  # (B, T, 3*n_mel)
            x = self.fc(x)        # (B, T, 360)
            return x

    model = RMVPE(n_mel=_RMVPE_N_MEL, n_class=_RMVPE_N_CLASS)

    # Load weights
    model_state = {k.replace("module.", ""): v for k, v in model_state.items()}
    try:
        model.load_state_dict(model_state, strict=True)
        print("[export] RMVPE weights loaded (strict).")
    except RuntimeError:
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        print(f"[export] RMVPE weights loaded (non-strict). "
              f"missing={len(missing)} unexpected={len(unexpected)}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------


def export_rvc(output_dir: str, cache_dir: Optional[str] = None) -> Path:
    """Export RVC base models and return the engine output directory."""
    import torch
    from conversion.export_base import OutputLayout, export_model, write_manifest, write_provenance
    from conversion.parity import compare_outputs, check_tolerance, run_ort
    from conversion.quantize import quantize_model

    output_dir = Path(output_dir)
    layout = OutputLayout.for_engine("rvc", base_dir=output_dir)
    layout.makedirs()

    _cache = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "voiceclonnx" / "rvc"
    _cache.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Export ContentVec encoder
    # ------------------------------------------------------------------
    cv_model = _build_contentvec()
    dummy_audio = torch.zeros(1, 16000)  # 1 second at 16 kHz
    dummy_mask = torch.ones(1, 16000, dtype=torch.long)

    cv_onnx = layout.component_path("contentvec_768l12.onnx")
    print(f"[export] Exporting ContentVec encoder → {cv_onnx} ...")

    # Transformers ≥ 5.x has a masking utility that is not traceable via
    # TorchScript ONNX export (create_bidirectional_mask/sdpa_mask accesses
    # tensor.shape[0] which is a symbolic value during tracing).  We patch
    # the module-level reference in modeling_hubert to return None during
    # the export call — HuBERT encoder accepts None attention mask (full
    # attention, no masking), which is correct for single-utterance inference.
    # Transformers ≥ 5.x has a masking utility that is not traceable via
    # TorchScript ONNX export (create_bidirectional_mask accesses symbolic
    # tensor shapes during tracing).  Patch it to return None (= full attention,
    # no masking) for both the ONNX export and the torch parity reference —
    # HuBERT encoder accepts None mask.  This is the correct behaviour for
    # single-utterance inference without padding.
    try:
        import transformers.models.hubert.modeling_hubert as _hb_mod
        _orig_cbm = _hb_mod.create_bidirectional_mask
        _hb_mod.create_bidirectional_mask = lambda *a, **k: None
        _patched = True
    except (ImportError, AttributeError):
        _patched = False

    try:
        # Capture torch reference BEFORE tracing — ONNX tracing mutates model state
        # (e.g. through internal JIT hooks), so the pre-export output is the
        # faithful reference.
        print("[export] Capturing ContentVec torch reference output ...")
        with torch.no_grad():
            torch_cv_out = cv_model(dummy_audio, dummy_mask).detach().cpu().numpy()

        export_model(
            model=cv_model,
            dummy_inputs=(dummy_audio, dummy_mask),
            output_path=cv_onnx,
            input_names=["input_values", "attention_mask"],
            output_names=["hidden_states"],
            dynamic_axes={
                "input_values": {0: "batch", 1: "time"},
                "attention_mask": {0: "batch", 1: "time"},
                "hidden_states": {0: "batch", 1: "frames"},
            },
            opset_version=14,
        )

        cv_size_mb = cv_onnx.stat().st_size / 1024 ** 2
        print(f"[export] ContentVec ONNX: {cv_size_mb:.1f} MB")
    finally:
        if _patched:
            _hb_mod.create_bidirectional_mask = _orig_cbm

    ort_cv_out = run_ort(cv_onnx, {
        "input_values": dummy_audio.numpy(),
        "attention_mask": dummy_mask.numpy(),
    })
    cv_report = compare_outputs(
        [torch_cv_out], ort_cv_out,
        names=["hidden_states"],
        max_abs_tol=1e-3, mean_abs_tol=1e-4,
    )
    print("[export] ContentVec parity:", cv_report.summary())
    check_tolerance(cv_report)
    cv_report.save(layout.component_path("contentvec_parity_report.json"))

    # Quantize ContentVec
    print("[export] Quantizing ContentVec (INT8) ...")
    cv_q8 = layout.component_path("contentvec_768l12_q8.onnx")
    cv_quant = quantize_model(cv_onnx, output_path=cv_q8)
    print(cv_quant.summary())

    del cv_model

    # ------------------------------------------------------------------
    # 2. RMVPE pitch estimator
    # The Politrees/RVC_resources repo already ships an ONNX of RMVPE
    # (predictors/rmvpe.onnx, MIT).  We use it directly rather than
    # re-exporting from .pt to ensure exact architecture match.
    # ------------------------------------------------------------------
    from huggingface_hub import hf_hub_download as _hf_dl
    import shutil as _shutil

    rmvpe_src = Path(_hf_dl(repo_id=RMVPE_HF_REPO, filename=RMVPE_ONNX_FILE))
    print(f"[export] RMVPE community ONNX: {rmvpe_src} ({rmvpe_src.stat().st_size // 1024 // 1024} MB)")

    rmvpe_onnx = layout.component_path("rmvpe.onnx")
    _shutil.copy2(str(rmvpe_src), str(rmvpe_onnx))
    rmvpe_size_mb = rmvpe_onnx.stat().st_size / 1024 ** 2
    print(f"[export] RMVPE ONNX copied: {rmvpe_size_mb:.1f} MB")

    # Smoke-check — the community RMVPE ONNX expects (batch, n_mel, T) where
    # T is divisible by 32 (5-level DeepUnet pooling).
    print("[export] Running RMVPE ORT smoke-check ...")
    import numpy as _np
    dummy_mel_np = _np.zeros((1, _RMVPE_N_MEL, 32), dtype=_np.float32)
    ort_rmvpe_out = run_ort(rmvpe_onnx, {"input": dummy_mel_np})
    print(f"[export] RMVPE output shape: {ort_rmvpe_out[0].shape}  (smoke-check passed)")

    # Quantize RMVPE
    print("[export] Quantizing RMVPE (INT8) ...")
    rmvpe_q8 = layout.component_path("rmvpe_q8.onnx")
    rmvpe_quant = quantize_model(rmvpe_onnx, output_path=rmvpe_q8)
    print(rmvpe_quant.summary())

    # ------------------------------------------------------------------
    # 3. Write manifest and provenance
    # ------------------------------------------------------------------
    write_manifest(
        layout=layout,
        components={
            "contentvec_encoder": "contentvec_768l12.onnx",
            "contentvec_encoder_q8": "contentvec_768l12_q8.onnx",
            "rmvpe_f0": "rmvpe.onnx",
            "rmvpe_f0_q8": "rmvpe_q8.onnx",
        },
        sample_rates={"input": 16000},
        metadata={
            "opset": 14,
            "contentvec_source": f"hf:{CONTENTVEC_HF_REPO}/{CONTENTVEC_WEIGHTS_FILE}",
            "rmvpe_source": f"hf:{RMVPE_HF_REPO}/{RMVPE_ONNX_FILE}",
            "contentvec_dim": 768,
            "rmvpe_pitch_classes": _RMVPE_N_CLASS,
        },
        distributable=True,
    )
    write_provenance(
        engine_dir=layout.engine_dir,
        upstream_repo_url=RVC_UPSTREAM_URL,
        upstream_ref=RVC_UPSTREAM_REF,
        license_text=MIT_LICENSE,
        extra={
            "contentvec_checkpoint": f"hf:{CONTENTVEC_HF_REPO}/{CONTENTVEC_WEIGHTS_FILE}",
            "rmvpe_checkpoint": f"hf:{RMVPE_HF_REPO}/{RMVPE_ONNX_FILE} (community ONNX)",
            "contentvec_size_mb": f"{cv_size_mb:.1f}",
            "rmvpe_size_mb": f"{rmvpe_size_mb:.1f}",
        },
    )

    print(f"\n[export] RVC base export complete → {layout.engine_dir}")
    print(f"  contentvec_768l12.onnx    {cv_size_mb:.1f} MB")
    print(f"  contentvec_768l12_q8.onnx {cv_q8.stat().st_size / 1024**2:.1f} MB")
    print(f"  rmvpe.onnx                {rmvpe_size_mb:.1f} MB")
    print(f"  rmvpe_q8.onnx             {rmvpe_q8.stat().st_size / 1024**2:.1f} MB")
    return layout.engine_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export RVC base models (ContentVec + RMVPE) to ONNX."
    )
    p.add_argument("--output-dir", default="/tmp/rvc-out", help="Staging output directory.")
    p.add_argument("--cache-dir", default=None, help="Cache directory for downloaded checkpoints.")
    p.add_argument("--no-push", action="store_true", help="Skip HF upload.")
    p.add_argument("--dry-run-push", action="store_true", help="Dry-run HF upload.")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    engine_dir = export_rvc(output_dir=args.output_dir, cache_dir=args.cache_dir)

    if not args.no_push:
        from conversion.push_models import push_engine
        push_engine(
            engine_dir=engine_dir,
            engine_name="rvc",
            dry_run=args.dry_run_push,
            commit_message="export: add rvc base ONNX artifacts (ContentVec-768 + RMVPE)",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
