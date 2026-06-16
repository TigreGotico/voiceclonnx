"""Export kNN-VC components (WavLM-Large encoder + HiFi-GAN vocoder) to ONNX.

kNN-VC architecture (Baas et al., Interspeech 2023):
  1. WavLM-Large encoder — layer-6 hidden states used as features.
  2. k-NN matching — pure numpy, no ONNX needed.
  3. HiFi-GAN vocoder — fully convolutional, straightforward export.

Upstream:
  - https://github.com/bshall/knn-vc  (MIT, Stellenbosch University 2023)
  - WavLM-Large weights: microsoft/wavlm-large on HF (MIT)
  - HiFi-GAN prematched checkpoint: bshall/knn-vc GitHub release v0.1

Usage::

    python -m conversion.export_knnvc --output-dir /tmp/knnvc-out

    # Or with explicit staging dir and no HF push
    python -m conversion.export_knnvc --output-dir /tmp/knnvc-out --no-push
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WAVLM_HF_MODEL = "microsoft/wavlm-large"
KNNVC_RELEASE_URL = "https://github.com/bshall/knn-vc/releases/download/v0.1/prematch_g_02500000.pt"
KNNVC_UPSTREAM_URL = "https://github.com/bshall/knn-vc"
KNNVC_UPSTREAM_REF = "v0.1"
WAVLM_LAYER = 6

MIT_LICENSE = """\
MIT License

Copyright (c) 2023 bshall (Stellenbosch University)
Copyright (c) 2021 Microsoft Corporation (WavLM-Large weights)

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
# WavLM layer-6 wrapper
# ---------------------------------------------------------------------------


def _build_wavlm_layer6():
    """Return a torch module that outputs WavLM-Large layer-6 hidden states."""
    import torch
    import torch.nn as nn
    from transformers import WavLMModel

    class WavLMLayer6(nn.Module):
        """WavLM-Large truncated to output the layer-6 hidden state."""

        def __init__(self, wavlm: WavLMModel):
            super().__init__()
            self.wavlm = wavlm

        def forward(self, input_values: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            input_values:
                Float32 tensor of shape (batch, time) — 16 kHz PCM, normalised.

            Returns
            -------
            torch.Tensor
                Hidden states from layer 6, shape (batch, frames, 1024).
            """
            out = self.wavlm(
                input_values=input_values,
                output_hidden_states=True,
            )
            # hidden_states is a tuple of (num_layers+1) tensors; index 7 = layer 6
            # (index 0 is the CNN feature extractor output, 1..N are transformer layers)
            return out.hidden_states[WAVLM_LAYER + 1]

    print(f"[export] Loading {WAVLM_HF_MODEL} ...")
    wavlm = WavLMModel.from_pretrained(WAVLM_HF_MODEL)
    wavlm.eval()
    model = WavLMLayer6(wavlm)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# HiFi-GAN wrapper
# ---------------------------------------------------------------------------


def _download_hifigan(cache_dir: Path) -> Path:
    """Download the prematched HiFi-GAN checkpoint from bshall GitHub release."""
    dest = cache_dir / "prematch_g_02500000.pt"
    if dest.exists():
        print(f"[export] HiFi-GAN checkpoint already cached at {dest}")
        return dest
    print(f"[export] Downloading HiFi-GAN from {KNNVC_RELEASE_URL} ...")
    cache_dir.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(KNNVC_RELEASE_URL, str(dest))
    print(f"[export] Downloaded to {dest} ({dest.stat().st_size // 1024 // 1024} MB)")
    return dest


def _build_hifigan(ckpt_path: Path):
    """Load the bshall kNN-VC HiFi-GAN vocoder from a checkpoint file.

    The bshall prematched checkpoint uses a custom variant of HiFi-GAN:
    - lin_pre: Linear(1024 → 512) projects WavLM features before the CNN
    - conv_pre: Conv1d(512, 512, 7)
    - upsample_rates: (8, 8, 2, 2) → total 256x upsampling (50 Hz → 12800 Hz ≈ 16 kHz)
    - upsample_kernel_sizes: (20, 16, 4, 4)
    - upsample_initial_channel: 512
    - resblock_kernel_sizes: (3, 7, 11), dilation: ((1,3,5), (1,3,5), (1,3,5))
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class ResBlock(nn.Module):
        def __init__(self, channels: int, kernel_size: int = 3, dilation=(1, 3, 5)):
            super().__init__()
            self.convs1 = nn.ModuleList([
                nn.utils.weight_norm(nn.Conv1d(
                    channels, channels, kernel_size, 1,
                    dilation=d, padding=(kernel_size * d - d) // 2,
                )) for d in dilation
            ])
            self.convs2 = nn.ModuleList([
                nn.utils.weight_norm(nn.Conv1d(
                    channels, channels, kernel_size, 1,
                    dilation=1, padding=(kernel_size - 1) // 2,
                )) for _ in dilation
            ])

        def forward(self, x):
            for c1, c2 in zip(self.convs1, self.convs2):
                xt = F.leaky_relu(x, 0.1)
                xt = c1(xt)
                xt = F.leaky_relu(xt, 0.1)
                xt = c2(xt)
                x = xt + x
            return x

    class HiFiGANGenerator(nn.Module):
        """bshall kNN-VC HiFi-GAN variant with lin_pre projection layer."""

        def __init__(
            self,
            wavlm_dim: int = 1024,
            upsample_rates=(8, 8, 2, 2),
            upsample_kernel_sizes=(20, 16, 4, 4),
            upsample_initial_channel: int = 512,
            resblock_kernel_sizes=(3, 7, 11),
            resblock_dilation_sizes=((1, 3, 5), (1, 3, 5), (1, 3, 5)),
        ):
            super().__init__()
            self.num_kernels = len(resblock_kernel_sizes)
            self.num_upsamples = len(upsample_rates)

            # bshall adds a linear projection before conv_pre
            self.lin_pre = nn.Linear(wavlm_dim, upsample_initial_channel)
            self.conv_pre = nn.utils.weight_norm(
                nn.Conv1d(upsample_initial_channel, upsample_initial_channel, 7, 1, padding=3)
            )
            self.ups = nn.ModuleList()
            ch = upsample_initial_channel
            for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
                self.ups.append(nn.utils.weight_norm(
                    nn.ConvTranspose1d(ch, ch // 2, k, u, padding=(k - u) // 2)
                ))
                ch //= 2

            self.resblocks = nn.ModuleList()
            ch = upsample_initial_channel
            for i in range(len(upsample_rates)):
                ch //= 2
                for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                    self.resblocks.append(ResBlock(ch, k, d))

            self.conv_post = nn.utils.weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            x:
                (batch, 1024, frames) — WavLM feature frames (post-kNN matching).

            Returns
            -------
            torch.Tensor
                (batch, 1, samples) — 16 kHz waveform.
            """
            # lin_pre expects (batch, frames, wavlm_dim) — transpose in/out
            x = x.transpose(1, 2)   # (B, T, 1024)
            x = self.lin_pre(x)     # (B, T, 512)
            x = x.transpose(1, 2)   # (B, 512, T)

            x = self.conv_pre(x)
            for i, up in enumerate(self.ups):
                x = F.leaky_relu(x, 0.1)
                x = up(x)
                xs = None
                for j in range(self.num_kernels):
                    if xs is None:
                        xs = self.resblocks[i * self.num_kernels + j](x)
                    else:
                        xs += self.resblocks[i * self.num_kernels + j](x)
                x = xs / self.num_kernels
            x = F.leaky_relu(x)
            x = self.conv_post(x)
            x = torch.tanh(x)
            return x

    print(f"[export] Loading HiFi-GAN checkpoint from {ckpt_path} ...")
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    gen_state = state["generator"] if "generator" in state else state

    model = HiFiGANGenerator()
    model.load_state_dict(gen_state)

    # Remove weight_norm for clean ONNX export
    for module in model.modules():
        if hasattr(module, "weight_g"):
            try:
                nn.utils.remove_weight_norm(module)
            except Exception:
                pass

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------


def export_knnvc(output_dir: str, cache_dir: Optional[str] = None) -> Path:
    """Export kNN-VC components and return the engine output directory."""
    import torch
    from conversion.export_base import OutputLayout, export_model, write_manifest, write_provenance
    from conversion.parity import compare_outputs, check_tolerance, run_ort
    from conversion.quantize import quantize_model

    output_dir = Path(output_dir)
    layout = OutputLayout.for_engine("knn-vc", base_dir=output_dir)
    layout.makedirs()

    _cache = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "voiceclonnx" / "knnvc"
    _cache.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Export WavLM-Large layer-6
    # ------------------------------------------------------------------
    wavlm_model = _build_wavlm_layer6()

    # Dummy input: 1 second of 16 kHz audio → 16000 samples
    dummy_audio = torch.zeros(1, 16000)

    wavlm_onnx = layout.component_path("wavlm_layer6.onnx")
    print(f"[export] Exporting WavLM-Large layer-6 → {wavlm_onnx} ...")
    export_model(
        model=wavlm_model,
        dummy_inputs=(dummy_audio,),
        output_path=wavlm_onnx,
        input_names=["input_values"],
        output_names=["hidden_states"],
        dynamic_axes={
            "input_values": {0: "batch", 1: "time"},
            "hidden_states": {0: "batch", 1: "frames"},
        },
        opset_version=14,
    )
    print(f"[export] WavLM ONNX written: {wavlm_onnx.stat().st_size / 1024**2:.1f} MB")

    # Parity check — WavLM
    print("[export] Running WavLM parity check ...")
    with torch.no_grad():
        torch_wavlm_out = wavlm_model(dummy_audio).detach().cpu().numpy()
    ort_wavlm_out = run_ort(wavlm_onnx, {"input_values": dummy_audio.numpy()})
    wavlm_report = compare_outputs(
        [torch_wavlm_out],
        ort_wavlm_out,
        names=["hidden_states"],
        max_abs_tol=1e-3,
        mean_abs_tol=1e-4,
    )
    print("[export] WavLM parity:", wavlm_report.summary())
    check_tolerance(wavlm_report)
    wavlm_report.save(layout.component_path("wavlm_parity_report.json"))

    # Quantize WavLM
    print("[export] Quantizing WavLM (INT8) ...")
    wavlm_q8_path = layout.component_path("wavlm_layer6_q8.onnx")
    wavlm_quant = quantize_model(wavlm_onnx, output_path=wavlm_q8_path)
    print(wavlm_quant.summary())

    del wavlm_model  # free memory before loading HiFi-GAN

    # ------------------------------------------------------------------
    # 2. Export HiFi-GAN vocoder
    # ------------------------------------------------------------------
    hifigan_ckpt = _download_hifigan(_cache)
    hifigan_model = _build_hifigan(hifigan_ckpt)

    # Dummy input: 100 frames of 1024-dim features (matches WavLM hidden dim)
    dummy_features = torch.zeros(1, 1024, 100)

    hifigan_onnx = layout.component_path("hifigan_knnvc.onnx")
    print(f"[export] Exporting HiFi-GAN vocoder → {hifigan_onnx} ...")
    export_model(
        model=hifigan_model,
        dummy_inputs=(dummy_features,),
        output_path=hifigan_onnx,
        input_names=["features"],
        output_names=["waveform"],
        dynamic_axes={
            "features": {0: "batch", 2: "frames"},
            "waveform": {0: "batch", 2: "samples"},
        },
        opset_version=14,
    )
    print(f"[export] HiFi-GAN ONNX written: {hifigan_onnx.stat().st_size / 1024**2:.1f} MB")

    # Parity check — HiFi-GAN
    print("[export] Running HiFi-GAN parity check ...")
    with torch.no_grad():
        torch_hifigan_out = hifigan_model(dummy_features).detach().cpu().numpy()
    ort_hifigan_out = run_ort(hifigan_onnx, {"features": dummy_features.numpy()})
    hifigan_report = compare_outputs(
        [torch_hifigan_out],
        ort_hifigan_out,
        names=["waveform"],
        max_abs_tol=1e-3,
        mean_abs_tol=1e-4,
    )
    print("[export] HiFi-GAN parity:", hifigan_report.summary())
    check_tolerance(hifigan_report)
    hifigan_report.save(layout.component_path("hifigan_parity_report.json"))

    # Quantize HiFi-GAN
    print("[export] Quantizing HiFi-GAN (INT8) ...")
    hifigan_q8_path = layout.component_path("hifigan_knnvc_q8.onnx")
    hifigan_quant = quantize_model(hifigan_onnx, output_path=hifigan_q8_path)
    print(hifigan_quant.summary())

    del hifigan_model

    # ------------------------------------------------------------------
    # 3. Write manifest and provenance
    # ------------------------------------------------------------------
    write_manifest(
        layout=layout,
        components={
            "wavlm_encoder": "wavlm_layer6.onnx",
            "wavlm_encoder_q8": "wavlm_layer6_q8.onnx",
            "hifigan_vocoder": "hifigan_knnvc.onnx",
            "hifigan_vocoder_q8": "hifigan_knnvc_q8.onnx",
        },
        sample_rates={"input": 16000, "output": 16000},
        metadata={
            "opset": 14,
            "wavlm_layer": WAVLM_LAYER,
            "knn_default_k": 4,
            "wavlm_source": WAVLM_HF_MODEL,
            "hifigan_source": KNNVC_RELEASE_URL,
        },
    )
    write_provenance(
        engine_dir=layout.engine_dir,
        upstream_repo_url=KNNVC_UPSTREAM_URL,
        upstream_ref=KNNVC_UPSTREAM_REF,
        license_text=MIT_LICENSE,
        extra={
            "wavlm_model": WAVLM_HF_MODEL,
            "hifigan_checkpoint": "prematch_g_02500000.pt",
            "wavlm_layer_extracted": str(WAVLM_LAYER),
        },
    )

    print(f"\n[export] kNN-VC export complete → {layout.engine_dir}")
    return layout.engine_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export kNN-VC (WavLM-Large + HiFi-GAN) to ONNX."
    )
    p.add_argument("--output-dir", default="/tmp/knnvc-out", help="Staging output directory.")
    p.add_argument("--cache-dir", default=None, help="Cache directory for downloaded checkpoints.")
    p.add_argument("--no-push", action="store_true", help="Skip HF upload.")
    p.add_argument("--dry-run-push", action="store_true", help="Dry-run HF upload (prints paths).")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    engine_dir = export_knnvc(output_dir=args.output_dir, cache_dir=args.cache_dir)

    if not args.no_push:
        from conversion.push_models import push_engine
        push_engine(
            engine_dir=engine_dir,
            engine_name="knn-vc",
            dry_run=args.dry_run_push,
            commit_message="export: add knn-vc ONNX artifacts (WavLM-Large layer-6 + HiFi-GAN)",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
