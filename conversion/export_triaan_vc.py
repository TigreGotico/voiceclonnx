"""Export TriAAN-VC components to ONNX.

TriAAN-VC (winddori2002, ICASSP 2023) — "Triple Adaptive Attention Normalization
for Any-to-Any Voice Conversion."

Architecture exported here (all verified from checkpoint keys):

1. **CPC encoder** — 5-layer strided Conv1d (bias=True) + learned per-channel
   affine norm + single-layer LSTM(256→256); downsamples 16 kHz audio by 160×
   to 256-dim features at 100 Hz.  Loaded from facebookresearch/CPC_audio
   checkpoint ``cpc.pt``.

2. **TriAAN-VC main model** — SpeakerEncoder + ContentEncoder + bidirectional
   GRU fusion + TriAAN decoder + PostNet; takes
   ``(src_cpc, src_lf0, trg_cpc) → mel (B, 80, T)``.
   Loaded from upstream ``model.py`` (cloned from GitHub).

3. **ParallelWaveGAN vocoder** — WaveNet-style conditional generator loaded via
   the ``parallel_wavegan`` package; takes ``(mel, noise) → waveform``.
   Exported with explicit noise input so the ONNX graph is deterministic.

Export strategy
---------------
- CPC encoder: loaded via upstream ``src/cpc.load_cpc`` from a local clone
  of the winddori2002/TriAAN-VC repo (which bundles facebookresearch/CPC_audio
  code).  Uses the real ``ChannelNorm`` (per-channel layer-norm with affine)
  and correct ``Conv1d`` padding, producing 100 frames per 16 kHz second.
  A reconstructed architecture without padding produced 98 frames and 1.46
  max-abs divergence from the real model (the parity-vs-self trap).
- TriAAN-VC: loaded from a local clone of winddori2002/TriAAN-VC so the
  architecture exactly matches the checkpoint.  Uses legacy TorchScript-based
  ``torch.onnx.export(dynamo=False)``.
- Vocoder: loaded via ``parallel_wavegan`` package with a thin wrapper that
  removes the ``assert`` and accepts explicit noise for deterministic export.

Upstream: https://github.com/winddori2002/TriAAN-VC (MIT license)
Weights:  GitHub release v1.0 — model-cpc-split.pth, cpc.pt, vocoder.pkl

Usage::

    python -m conversion.export_triaan_vc --output-dir /tmp/triaan-out

    # Skip HF push (local parity check only)
    python -m conversion.export_triaan_vc --output-dir /tmp/triaan-out --no-push
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import urllib.request
from pathlib import Path
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UPSTREAM_REPO_URL = "https://github.com/winddori2002/TriAAN-VC"
UPSTREAM_REF = "v1.0"
ENGINE_NAME = "triaan-vc"
HF_REPO_ID = "TigreGotico/voiceclonnx-triaan-vc"
SAMPLE_RATE = 16000
N_MELS = 80
N_FFT = 400
HOP_LENGTH = 160
WIN_LENGTH = 400
CPC_HIDDEN = 256
CPC_DOWNSAMPLE = 160
TRIAAN_CLONE_URL = "https://github.com/winddori2002/TriAAN-VC.git"

# GitHub release v1.0 download URLs
_BASE_URL = "https://github.com/winddori2002/TriAAN-VC/releases/download/v1.0"
CPC_CHECKPOINT_URL = f"{_BASE_URL}/cpc.pt"
MODEL_CHECKPOINT_URL = f"{_BASE_URL}/model-cpc-split.pth"
VOCODER_CHECKPOINT_URL = f"{_BASE_URL}/vocoder.pkl"

MIT_LICENSE = """\
MIT License

Copyright (c) 2022 winddori2002

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

Portions derived from:
  facebookresearch/CPC_audio (MIT) — CPC encoder architecture and checkpoint
  kan-bayashi/ParallelWaveGAN (MIT) — vocoder (via parallel_wavegan package)
  Wendison/VQMIVC (MIT) — vocoder checkpoint
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path) -> Path:
    if dest.exists():
        print(f"[export] cached: {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[export] downloading {url} → {dest}")
    urllib.request.urlretrieve(url, str(dest))
    mb = dest.stat().st_size / 1024 / 1024
    print(f"[export] downloaded {mb:.1f} MB")
    return dest


def _clone_upstream(clone_dir: Path) -> Path:
    """Clone TriAAN-VC repo if not already present."""
    if (clone_dir / "model" / "model.py").exists():
        print(f"[export] upstream already cloned at {clone_dir}")
        return clone_dir
    import subprocess
    print(f"[export] cloning {TRIAAN_CLONE_URL} → {clone_dir}")
    clone_dir.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--depth=1", TRIAAN_CLONE_URL, str(clone_dir)],
        check=True,
    )
    return clone_dir


def _add_to_path(path: Path) -> None:
    s = str(path)
    if s not in sys.path:
        sys.path.insert(0, s)


def _add_conversion_to_path() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    _add_to_path(repo_root)


def _fix_scipy_kaiser() -> None:
    """Patch scipy.signal.kaiser for parallel_wavegan compatibility."""
    import scipy.signal
    if not hasattr(scipy.signal, "kaiser"):
        from scipy.signal.windows import kaiser
        scipy.signal.kaiser = kaiser


# ---------------------------------------------------------------------------
# CPC encoder
# ---------------------------------------------------------------------------


def _load_cpc_model(clone_dir: Path, ckpt_path: Path):
    """Load CPC encoder from upstream checkpoint via upstream src/cpc.py.

    Uses the real CPCModel from facebookresearch/CPC_audio (cloned inside
    the TriAAN-VC repo), with exact ChannelNorm + padding that matches the
    checkpoint.  Wraps the model to return only cFeature (B, T, 256) for
    ONNX export — the same shape the adapter transposes to (B, 256, T).

    The previous reconstruction used LearnedNorm1d (affine-only) instead of
    ChannelNorm (per-channel layer-norm) and omitted conv padding, causing
    100 vs 98 frame counts and 1.46 max-abs divergence from the real output.
    """
    import torch
    import torch.nn as nn

    _add_to_path(clone_dir)
    from src.cpc import load_cpc  # upstream loader: strict state-dict load

    cpc_model = load_cpc(str(ckpt_path))
    cpc_model.eval()

    class CPCWrapper(nn.Module):
        """Returns only cFeature (first of three outputs) for ONNX tracing."""
        def __init__(self, cpc):
            super().__init__()
            self.cpc = cpc

        def forward(self, audio: torch.Tensor) -> torch.Tensor:
            # audio: (B, 1, T) → cFeature: (B, T_frames, 256)
            cFeature, _, _ = self.cpc(audio, None)
            return cFeature

    wrapper = CPCWrapper(cpc_model)
    wrapper.eval()
    print(f"[export/cpc] loaded upstream CPCModel via load_cpc (strict)")
    return wrapper


# ---------------------------------------------------------------------------
# TriAAN-VC main model (using upstream code from cloned repo)
# ---------------------------------------------------------------------------


class _AttrDict(dict):
    """Dict with attribute access — matches upstream easydict param objects."""
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)


def _load_triaan_model(clone_dir: Path, ckpt_path: Path):
    """Load TriAANVC from upstream model.py with exact checkpoint weights."""
    import torch
    _add_to_path(clone_dir)

    from model.model import TriAANVC  # upstream code, exact architecture

    enc_params = _AttrDict(c_in=CPC_HIDDEN, c_h=512, c_out=4, num_layer=6)
    dec_params = _AttrDict(c_in=4, c_h=512, c_out=N_MELS, num_layer=6)

    model = TriAANVC(encoder_params=enc_params, decoder_params=dec_params)
    state = torch.load(str(ckpt_path), map_location="cpu")["state_dict"]
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"[export/triaan] loaded (strict=True)")
    return model


def _export_triaan_onnx(model, output_path: Path, T: int = 50):
    """Export TriAANVC to ONNX using TorchScript-based exporter (dynamo=False)."""
    import torch

    dummy_src = torch.zeros(1, CPC_HIDDEN, T)
    dummy_lf0 = torch.zeros(1, T)
    dummy_trg = torch.zeros(1, CPC_HIDDEN, T)

    with torch.no_grad():
        torch_out = model(dummy_src, dummy_lf0, dummy_trg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy_src, dummy_lf0, dummy_trg),
        str(output_path),
        opset_version=14,
        input_names=["src_cpc", "src_lf0", "trg_cpc"],
        output_names=["mel"],
        dynamic_axes={
            "src_cpc": {0: "batch", 2: "src_frames"},
            "src_lf0": {0: "batch", 1: "src_frames"},
            "trg_cpc": {0: "batch", 2: "trg_frames"},
            "mel": {0: "batch", 2: "src_frames"},
        },
        dynamo=False,
    )
    print(f"[export/triaan] exported to {output_path}")
    return torch_out, output_path


# ---------------------------------------------------------------------------
# ParallelWaveGAN vocoder
# ---------------------------------------------------------------------------


def _load_pwg_model(clone_dir: Path, ckpt_path: Path):
    """Load ParallelWaveGAN generator with scipy compatibility fix."""
    import torch
    import yaml

    _fix_scipy_kaiser()
    from parallel_wavegan.models.parallel_wavegan import ParallelWaveGANGenerator

    config_path = clone_dir / "vocoder" / "config.yml"
    with open(str(config_path)) as f:
        config = yaml.load(f, Loader=yaml.Loader)

    gen_params = dict(config["generator_params"])
    model = ParallelWaveGANGenerator(**gen_params)

    state = torch.load(str(ckpt_path), map_location="cpu")
    model.load_state_dict(state["model"]["generator"])

    stats_path = clone_dir / "vocoder" / "vctk_stats.npy"
    if stats_path.exists():
        model.register_stats(str(stats_path))
        print(f"[export/pwg] stats loaded from {stats_path}")

    model.remove_weight_norm()
    model.eval()
    print("[export/pwg] loaded ParallelWaveGAN generator")
    return model


def _build_pwg_wrapper(model):
    """Build a nn.Module that takes (mel, noise) → waveform.

    The upstream ``forward(z, c)`` has an ``assert`` on sizes which breaks
    tracing.  This wrapper handles upsampling + WaveNet forward manually.
    """
    import torch
    import torch.nn as nn

    class PWGWrapper(nn.Module):
        def __init__(self, gen):
            super().__init__()
            self.gen = gen
            self.register_buffer("mean", gen.mean.clone().detach().float())
            self.register_buffer("scale", gen.scale.clone().detach().float())

        def forward(self, mel, noise):
            """
            mel   : (B, 80, T_mel)   mel spectrogram
            noise : (B, 1, T_audio)  Gaussian noise

            Returns
            -------
            waveform : (B, 1, T_audio)
            """
            # Mel-normalization (subtract training stats)
            c = (mel - self.mean.unsqueeze(-1)) / self.scale.unsqueeze(-1)
            # Upsample mel → audio resolution
            c = self.gen.upsample_net(c)  # (B, 80, T_audio)
            # WaveNet forward without assert
            x = self.gen.first_conv(noise)
            skips = 0
            for f in self.gen.conv_layers:
                x, h = f(x, c)
                skips = skips + h
            skips = skips * math.sqrt(1.0 / len(self.gen.conv_layers))
            x = skips
            for f in self.gen.last_conv_layers:
                x = f(x)
            return x

    return PWGWrapper(model)


def _export_pwg_onnx(wrapper, model, output_path: Path, T: int = 50):
    """Export the PWG wrapper to ONNX."""
    import torch

    # Determine T_audio from actual upsample result
    test_mel = torch.zeros(1, N_MELS, T)
    with torch.no_grad():
        c_up = model.upsample_net(test_mel)
    T_audio = c_up.shape[-1]

    dummy_mel = torch.zeros(1, N_MELS, T)
    dummy_noise = torch.randn(1, 1, T_audio)

    with torch.no_grad():
        torch_out = wrapper(dummy_mel, dummy_noise)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        (dummy_mel, dummy_noise),
        str(output_path),
        opset_version=14,
        input_names=["mel", "noise"],
        output_names=["waveform"],
        dynamic_axes={
            "mel": {0: "batch", 2: "mel_frames"},
            "noise": {0: "batch", 2: "audio_samples"},
            "waveform": {0: "batch", 2: "audio_samples"},
        },
        dynamo=False,
    )
    print(f"[export/pwg] exported to {output_path}")
    return torch_out, dummy_mel, dummy_noise, output_path


# ---------------------------------------------------------------------------
# Main export routine
# ---------------------------------------------------------------------------


def export_triaan_vc(output_dir: str, no_push: bool = False,
                     clone_dir: str = "/tmp/triaan-vc-src") -> None:
    _add_conversion_to_path()
    from conversion.export_base import OutputLayout, export_model, write_manifest, write_provenance
    from conversion.parity import compare_outputs, check_tolerance, run_ort
    from conversion.quantize import quantize_model

    import torch
    import numpy as np

    out_dir = Path(output_dir)
    layout = OutputLayout.for_engine(ENGINE_NAME, base_dir=out_dir)
    layout.makedirs()

    cache_dir = out_dir / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    clone_path = Path(clone_dir)

    # -----------------------------------------------------------------------
    # 1. Download checkpoints + clone upstream
    # -----------------------------------------------------------------------
    print("\n=== Step 1: Prepare upstream ===")
    cpc_ckpt = _download(CPC_CHECKPOINT_URL, cache_dir / "cpc.pt")
    triaan_ckpt = _download(MODEL_CHECKPOINT_URL, cache_dir / "model-cpc-split.pth")
    vocoder_ckpt = _download(VOCODER_CHECKPOINT_URL, cache_dir / "vocoder.pkl")
    _clone_upstream(clone_path)

    # -----------------------------------------------------------------------
    # 2. CPC encoder
    # -----------------------------------------------------------------------
    print("\n=== Step 2: Export CPC encoder ===")
    cpc_model = _load_cpc_model(clone_path, cpc_ckpt)

    dummy_audio = torch.zeros(1, 1, SAMPLE_RATE)
    with torch.no_grad():
        torch_cpc_out = cpc_model(dummy_audio)
    print(f"[export/cpc] torch output shape: {torch_cpc_out.shape}")

    cpc_onnx = export_model(
        model=cpc_model,
        dummy_inputs=(dummy_audio,),
        output_path=layout.component_path("cpc_encoder.onnx"),
        input_names=["audio"],
        output_names=["features"],
        dynamic_axes={"audio": {0: "batch", 2: "samples"}, "features": {0: "batch", 1: "frames"}},
        opset_version=14,
    )

    ort_cpc_out = run_ort(str(cpc_onnx), {"audio": dummy_audio.numpy()})
    report_cpc = compare_outputs([torch_cpc_out.numpy()], ort_cpc_out, names=["features"])
    _cpc_c = report_cpc.components[0]
    print(f"[parity/cpc] max_abs={_cpc_c.max_abs_delta:.2e}  mean_abs={_cpc_c.mean_abs_delta:.2e}")
    check_tolerance(report_cpc)
    print("[parity/cpc] PASS")

    q8_cpc = quantize_model(str(cpc_onnx))
    print(f"[quantize/cpc] {q8_cpc.summary()}")

    # -----------------------------------------------------------------------
    # 3. TriAAN-VC main model
    # -----------------------------------------------------------------------
    print("\n=== Step 3: Export TriAAN-VC main model ===")
    triaan_model = _load_triaan_model(clone_path, triaan_ckpt)

    T = 50  # short sequence for export tracing; dynamic axes handle variable T
    triaan_onnx_path = layout.component_path("triaan_vc.onnx")
    torch_mel_out, triaan_onnx = _export_triaan_onnx(triaan_model, triaan_onnx_path, T=T)
    print(f"[export/triaan] torch output shape: {torch_mel_out.shape}")

    dummy_src = torch.zeros(1, CPC_HIDDEN, T)
    dummy_lf0 = torch.zeros(1, T)
    dummy_trg = torch.zeros(1, CPC_HIDDEN, T)
    ort_mel_out = run_ort(
        str(triaan_onnx),
        {"src_cpc": dummy_src.numpy(), "src_lf0": dummy_lf0.numpy(), "trg_cpc": dummy_trg.numpy()},
    )
    report_triaan = compare_outputs([torch_mel_out.numpy()], ort_mel_out, names=["mel"])
    _triaan_c = report_triaan.components[0]
    print(f"[parity/triaan] max_abs={_triaan_c.max_abs_delta:.2e}  mean_abs={_triaan_c.mean_abs_delta:.2e}")
    check_tolerance(report_triaan)
    print("[parity/triaan] PASS")

    q8_triaan = quantize_model(str(triaan_onnx))
    print(f"[quantize/triaan] {q8_triaan.summary()}")

    # -----------------------------------------------------------------------
    # 4. ParallelWaveGAN vocoder
    # -----------------------------------------------------------------------
    print("\n=== Step 4: Export ParallelWaveGAN vocoder ===")
    pwg_model = _load_pwg_model(clone_path, vocoder_ckpt)
    pwg_wrapper = _build_pwg_wrapper(pwg_model)
    pwg_wrapper.eval()

    pwg_onnx_path = layout.component_path("pwg_vocoder.onnx")
    torch_wav_out, dummy_mel, dummy_noise, pwg_onnx = _export_pwg_onnx(
        pwg_wrapper, pwg_model, pwg_onnx_path, T=T
    )
    print(f"[export/pwg] torch output shape: {torch_wav_out.shape}")

    ort_wav_out = run_ort(
        str(pwg_onnx),
        {"mel": dummy_mel.numpy(), "noise": dummy_noise.numpy()},
    )
    report_pwg = compare_outputs([torch_wav_out.numpy()], ort_wav_out, names=["waveform"])
    _pwg_c = report_pwg.components[0]
    print(f"[parity/pwg] max_abs={_pwg_c.max_abs_delta:.2e}  mean_abs={_pwg_c.mean_abs_delta:.2e}")
    check_tolerance(report_pwg, max_abs_tol=5e-3, mean_abs_tol=1e-3)
    print("[parity/pwg] PASS")

    q8_pwg = quantize_model(str(pwg_onnx))
    print(f"[quantize/pwg] {q8_pwg.summary()}")

    # -----------------------------------------------------------------------
    # 5. Manifest + provenance
    # -----------------------------------------------------------------------
    print("\n=== Step 5: Write manifest and provenance ===")
    write_manifest(
        layout=layout,
        components={
            "cpc_encoder": "cpc_encoder.onnx",
            "cpc_encoder_q8": "cpc_encoder_q8.onnx",
            "triaan_vc": "triaan_vc.onnx",
            "triaan_vc_q8": "triaan_vc_q8.onnx",
            "pwg_vocoder": "pwg_vocoder.onnx",
            "pwg_vocoder_q8": "pwg_vocoder_q8.onnx",
        },
        sample_rates={"output": SAMPLE_RATE},
        metadata={
            "opset": 14,
            "cpc_hidden": CPC_HIDDEN,
            "cpc_downsample": CPC_DOWNSAMPLE,
            "n_mels": N_MELS,
            "n_fft": N_FFT,
            "hop_length": HOP_LENGTH,
            "win_length": WIN_LENGTH,
            "parity": {
                "cpc_encoder": {"max_abs": _cpc_c.max_abs_delta, "mean_abs": _cpc_c.mean_abs_delta},
                "triaan_vc": {"max_abs": _triaan_c.max_abs_delta, "mean_abs": _triaan_c.mean_abs_delta},
                "pwg_vocoder": {"max_abs": _pwg_c.max_abs_delta, "mean_abs": _pwg_c.mean_abs_delta},
            },
        },
    )

    write_provenance(
        engine_dir=layout.engine_dir,
        upstream_repo_url=UPSTREAM_REPO_URL,
        upstream_ref=UPSTREAM_REF,
        license_text=MIT_LICENSE,
    )

    # -----------------------------------------------------------------------
    # 6. Print size summary
    # -----------------------------------------------------------------------
    print("\n=== Model sizes ===")
    for onnx_file in sorted(layout.engine_dir.glob("*.onnx")):
        mb = onnx_file.stat().st_size / 1024 / 1024
        print(f"  {onnx_file.name:44s}  {mb:.1f} MB")

    # -----------------------------------------------------------------------
    # 7. Push to HF
    # -----------------------------------------------------------------------
    if not no_push:
        print("\n=== Step 6: Push to HF Hub ===")
        from conversion.push_models import push_engine

        push_engine(
            engine_dir=layout.engine_dir,
            engine_name=ENGINE_NAME,
            commit_message="export: add triaan-vc ONNX artifacts (CPC encoder + TriAAN decoder + PWG vocoder)",
        )
        print(f"[push] uploaded to https://huggingface.co/{HF_REPO_ID}")
    else:
        print("\n[export] --no-push: skipping HF upload")

    print("\n=== Export complete ===")
    print(f"Output directory: {layout.engine_dir}")
    print(f"\nParity summary:")
    print(f"  CPC encoder:   max_abs={_cpc_c.max_abs_delta:.2e}  mean_abs={_cpc_c.mean_abs_delta:.2e}  PASS")
    print(f"  TriAAN-VC:     max_abs={_triaan_c.max_abs_delta:.2e}  mean_abs={_triaan_c.mean_abs_delta:.2e}  PASS")
    print(f"  PWG vocoder:   max_abs={_pwg_c.max_abs_delta:.2e}  mean_abs={_pwg_c.mean_abs_delta:.2e}  PASS")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Export TriAAN-VC components to ONNX.")
    parser.add_argument("--output-dir", default="/tmp/triaan-vc-out",
                        help="Staging directory for ONNX files (default: /tmp/triaan-vc-out)")
    parser.add_argument("--no-push", action="store_true",
                        help="Skip upload to Hugging Face Hub")
    parser.add_argument("--clone-dir", default="/tmp/triaan-vc-src",
                        help="Path to local TriAAN-VC git clone (auto-cloned if missing)")
    args = parser.parse_args()

    export_triaan_vc(args.output_dir, no_push=args.no_push, clone_dir=args.clone_dir)


if __name__ == "__main__":
    main()
