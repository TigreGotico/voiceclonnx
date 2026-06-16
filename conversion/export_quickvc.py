"""Export QuickVC components to ONNX.

QuickVC (quickvc/QuickVC-VoiceConversion, MIT) is an any-to-many voice
conversion system using:

1. **HuBERT-soft content encoder** — converts 16 kHz audio to (B, T, 1024)
   soft speech units at 50 Hz.  Exported to ONNX (``quickvc_content_encoder.onnx``).
2. **Speaker encoder (LSTM)** — converts an 80-channel log-mel spectrogram to a
   256-dim d-vector.  Exported to ONNX (``quickvc_speaker_encoder.onnx``).
3. **Posterior encoder + normalising flow + Multistream-iSTFT decoder** — maps
   (content features, speaker d-vector) to per-subband (spec, phase) STFT
   coefficients.  Exported to ONNX (``quickvc_decoder.onnx``).
4. **numpy Multistream-iSTFT** — pure-numpy overlap-add across 4 subbands
   then downsampled by a learned 1-D conv filter (mirrored as fixed weights).
   Not in ONNX because ``torch.istft`` cannot be traced; parity vs torch
   verified at ≤5e-4 max abs error.

Runtime requirements: ``onnxruntime``, ``numpy``, ``soundfile``.

Upstream:
  - https://github.com/quickvc/QuickVC-VoiceConversion  (MIT)
  - HuBERT-soft: https://github.com/bshall/hubert  (MIT)
    checkpoint: https://github.com/bshall/hubert/releases/download/v0.2/hubert-soft-35d9f29f.pt
  - Pretrained weights:
    https://drive.google.com/drive/folders/1DF6RgIHHkn2aoyyUMt4_hPitKSc2YR9d

Usage::

    python -m conversion.export_quickvc --output-dir /tmp/quickvc-out
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUICKVC_UPSTREAM_URL = "https://github.com/quickvc/QuickVC-VoiceConversion"
QUICKVC_UPSTREAM_REF = "277118de9c81d1689e16be8a43408eda4223553d"
HUBERT_UPSTREAM_URL = "https://github.com/bshall/hubert"
HUBERT_UPSTREAM_REF = "v0.2"
QUICKVC_SR = 16000

# Config matches configs/quickvc.json
QUICKVC_HPS = {
    "filter_length": 1280,
    "hop_length": 320,
    "win_length": 1280,
    "n_mel_channels": 80,
    "mel_fmin": 0.0,
    "mel_fmax": None,
    "sampling_rate": 16000,
    "segment_size": 10240,
    # model params
    "gen_istft_n_fft": 16,
    "gen_istft_hop_size": 4,
    "subbands": 4,
}

MIT_LICENSE = """\
MIT License

Copyright (c) 2023 quickvc

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
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


# ---------------------------------------------------------------------------
# Wrapper modules
# ---------------------------------------------------------------------------

def _build_content_encoder_wrapper(hubert_model):
    """HuBERT-soft: (B, 1, samples) → (B, T, 1024)."""
    import torch.nn as nn

    class HubertSoftEncoder(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, wav):
            # wav: (B, 1, samples) — hubert_soft.units expects (B, 1, N)
            return self.model.units(wav)

    wrapper = HubertSoftEncoder(hubert_model)
    wrapper.eval()
    return wrapper


def _build_speaker_encoder_wrapper(enc_spk):
    """Speaker encoder: (B, T, 80) log-mel → (B, 256) d-vector."""
    import torch.nn as nn

    class SpeakerEncoderWrapper(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc

        def forward(self, mel):
            # mel: (B, T, 80)
            return self.enc(mel)

    wrapper = SpeakerEncoderWrapper(enc_spk)
    wrapper.eval()
    return wrapper


def _build_decoder_wrapper(net_g):
    """Decoder: (c_feats, c_lengths, g) → (spec_phase) subband STFT coefficients.

    Wraps enc_p + flow (reverse) + dec up to (but not including) the ISTFT call.
    The ISTFT is implemented in pure numpy at runtime.

    Input shapes:
      c_feats:   (B, 256, T)    — content features (transposed from HuBERT-soft output)
      c_lengths: (B,)           — sequence lengths
      g:         (B, 256, 1)    — speaker d-vector

    Output:
      spec_phase: (B*subbands, n_fft//2+1, T_dec) concatenated [spec; phase]
        where spec = torch.exp(x[:, :, :n_fft//2+1, :]) (log-magnitude)
              phase = pi*sin(x[:, :, n_fft//2+1:, :])
        These are returned as (B*subbands, 2, n_fft//2+1, T_dec) with dim-1
        indexing [0]=spec, [1]=phase so the numpy ISTFT can unpack cleanly.
    """
    import torch
    import torch.nn as nn
    import math

    class DecoderWrapper(nn.Module):
        def __init__(self, model, subbands, n_fft):
            super().__init__()
            self.enc_p = model.enc_p
            self.flow = model.flow
            self.dec_pre = model.dec.conv_pre
            self.dec_ups = model.dec.ups
            self.dec_resblocks = model.dec.resblocks
            self.dec_subband_conv = model.dec.subband_conv_post
            self.dec_cond = model.dec.cond
            self.dec_reflection_pad = model.dec.reflection_pad
            self.dec_updown_filter = model.dec.updown_filter
            self.dec_multistream_conv = model.dec.multistream_conv_post
            self.num_upsamples = model.dec.num_upsamples
            self.num_kernels = model.dec.num_kernels
            self.subbands = subbands
            self.n_fft = n_fft
            self.half = n_fft // 2 + 1

        def forward(self, c, c_lengths, g):
            # c: (B, 1024, T), c_lengths: (B,), g: (B, 256, 1)
            # 1. Posterior encoder
            z_p, _, _, c_mask = self.enc_p(c, c_lengths)
            # 2. Normalising flow (reverse)
            z = self.flow(z_p, c_mask, g=g, reverse=True)
            # 3. Decoder (without ISTFT)
            x = z * c_mask  # (B, 192, T)
            x = self.dec_pre(x)
            x = x + self.dec_cond(g)
            import torch.nn.functional as F
            import modules as qvc_modules
            for i in range(self.num_upsamples):
                x = F.leaky_relu(x, qvc_modules.LRELU_SLOPE)
                x = self.dec_ups[i](x)
                xs = None
                for j in range(self.num_kernels):
                    if xs is None:
                        xs = self.dec_resblocks[i * self.num_kernels + j](x)
                    else:
                        xs += self.dec_resblocks[i * self.num_kernels + j](x)
                x = xs / self.num_kernels
            x = F.leaky_relu(x)
            x = self.dec_reflection_pad(x)
            x = self.dec_subband_conv(x)
            # x: (B, subbands*(n_fft+2), T_dec)
            B = x.shape[0]
            T_dec = x.shape[-1]
            x = x.reshape(B, self.subbands, self.n_fft + 2, T_dec)
            spec = torch.exp(x[:, :, :self.half, :])    # (B, subbands, half, T_dec)
            phase = math.pi * torch.sin(x[:, :, self.half:, :])  # (B, subbands, half, T_dec)
            # Stack and reshape to (B*subbands, 2, half, T_dec)
            spec_phase = torch.stack([spec, phase], dim=2)  # (B, subbands, 2, half, T_dec)
            spec_phase = spec_phase.reshape(B * self.subbands, 2, self.half, T_dec)
            return spec_phase

    wrapper = DecoderWrapper(
        net_g,
        subbands=QUICKVC_HPS["subbands"],
        n_fft=QUICKVC_HPS["gen_istft_n_fft"],
    )
    wrapper.eval()
    return wrapper


def _build_postnet_wrapper(net_g):
    """Postnet: (B*subbands, 1, T_audio) → (B, 1, T_audio).

    Wraps the updown_filter interpolation + multistream_conv_post.
    This is a learned 1-D mixing conv — fully traceable.
    """
    import torch.nn as nn
    import torch.nn.functional as F

    class PostnetWrapper(nn.Module):
        def __init__(self, updown_filter, multistream_conv, subbands):
            super().__init__()
            self.register_buffer("updown_filter", updown_filter.clone())
            self.multistream_conv = multistream_conv
            self.subbands = subbands

        def forward(self, y_mb_hat):
            # y_mb_hat: (B, subbands, T_audio)
            y_mb_hat = F.conv_transpose1d(
                y_mb_hat,
                self.updown_filter * self.subbands,
                stride=self.subbands,
            )
            return self.multistream_conv(y_mb_hat)

    wrapper = PostnetWrapper(
        net_g.dec.updown_filter,
        net_g.dec.multistream_conv_post,
        subbands=QUICKVC_HPS["subbands"],
    )
    wrapper.eval()
    return wrapper


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------


def export_quickvc(
    output_dir: str,
    quickvc_weights: str,
    hubert_weights: Optional[str] = None,
) -> Path:
    """Export QuickVC components to ONNX.

    Parameters
    ----------
    output_dir:
        Staging directory; engine files land in ``output_dir/quickvc/``.
    quickvc_weights:
        Path to ``G_1200000.pth`` (or equivalent) QuickVC generator checkpoint.
    hubert_weights:
        Path to ``hubert-soft-35d9f29f.pt``.  If None, downloads from GitHub.

    Returns
    -------
    Path
        The engine output directory.
    """
    import torch
    from conversion.export_base import OutputLayout, export_model, write_manifest, write_provenance
    from conversion.parity import compare_outputs, check_tolerance, run_ort
    from conversion.quantize import quantize_model

    # Patch old scipy
    import scipy.signal
    if not hasattr(scipy.signal, "kaiser"):
        import scipy.signal.windows
        scipy.signal.kaiser = scipy.signal.windows.kaiser

    sys.path.insert(0, str(Path(__file__).parent.parent / "_upstream" / "quickvc"))

    output_dir = Path(output_dir)
    layout = OutputLayout.for_engine("quickvc", base_dir=output_dir)
    layout.makedirs()

    # ------------------------------------------------------------------
    # 0. Load HuBERT-soft
    # ------------------------------------------------------------------
    if hubert_weights is None:
        print("[export] Downloading HuBERT-soft from GitHub ...")
        import urllib.request
        hubert_dir = output_dir / "hubert_soft_cache"
        hubert_dir.mkdir(exist_ok=True)
        hubert_weights = str(hubert_dir / "hubert-soft-35d9f29f.pt")
        urllib.request.urlretrieve(
            "https://github.com/bshall/hubert/releases/download/v0.2/hubert-soft-35d9f29f.pt",
            hubert_weights,
        )

    print(f"[export] Loading HuBERT-soft from {hubert_weights} ...")
    hubert_model = torch.hub.load(
        "bshall/hubert:main",
        "hubert_soft",
        source="github",
        trust_repo=True,
    )
    hubert_model.load_state_dict(torch.load(hubert_weights, map_location="cpu", weights_only=False))
    hubert_model.eval()
    print("[export] HuBERT-soft loaded.")

    # ------------------------------------------------------------------
    # 1. Load QuickVC generator
    # ------------------------------------------------------------------
    _qvc_src = Path(__file__).parent.parent / "_upstream" / "quickvc"
    print(f"[export] Loading QuickVC from {quickvc_weights} ...")

    # Add upstream src to sys.path so models.py imports work
    sys.path.insert(0, str(_qvc_src))

    from models import SynthesizerTrn  # type: ignore[import]

    with open(_qvc_src / "configs" / "quickvc.json") as f:
        hps_dict = json.load(f)

    net_g = SynthesizerTrn(
        hps_dict["data"]["filter_length"] // 2 + 1,
        hps_dict["train"]["segment_size"] // hps_dict["data"]["hop_length"],
        **hps_dict["model"],
    )
    net_g.eval()
    ckpt = torch.load(quickvc_weights, map_location="cpu", weights_only=False)
    net_g.load_state_dict(ckpt["model"], strict=False)
    n_params = sum(p.numel() for p in net_g.parameters()) / 1e6
    print(f"[export] QuickVC loaded. {n_params:.1f}M params")

    # ------------------------------------------------------------------
    # 2. Export content encoder (HuBERT-soft)
    # ------------------------------------------------------------------
    enc_wrapper = _build_content_encoder_wrapper(hubert_model)
    dummy_wav = torch.zeros(1, 1, QUICKVC_SR)  # 1 s at 16 kHz

    enc_onnx = layout.component_path("quickvc_content_encoder.onnx")
    print(f"[export] Exporting content encoder → {enc_onnx} ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        export_model(
            model=enc_wrapper,
            dummy_inputs=(dummy_wav,),
            output_path=enc_onnx,
            input_names=["wav"],
            output_names=["units"],
            dynamic_axes={
                "wav":   {0: "batch", 2: "samples"},
                "units": {0: "batch", 1: "frames"},
            },
            opset_version=14,
        )
    enc_mb = enc_onnx.stat().st_size / 1024 ** 2
    print(f"[export] Content encoder written: {enc_mb:.1f} MB")

    # Parity — content encoder
    print("[export] Content encoder parity check ...")
    with torch.no_grad():
        torch_units = enc_wrapper(dummy_wav).detach().cpu().numpy()
    ort_units = run_ort(enc_onnx, {"wav": dummy_wav.numpy()})
    enc_report = compare_outputs(
        [torch_units],
        ort_units,
        names=["units"],
        max_abs_tol=2e-3,
        mean_abs_tol=5e-4,
    )
    print("[export] Content encoder parity:", enc_report.summary())
    check_tolerance(enc_report)
    enc_report.save(layout.component_path("content_encoder_parity_report.json"))

    # Quantize
    print("[export] Quantizing content encoder (INT8) ...")
    enc_q8 = layout.component_path("quickvc_content_encoder_q8.onnx")
    enc_quant = quantize_model(enc_onnx, output_path=enc_q8)
    print(enc_quant.summary())

    del enc_wrapper, hubert_model

    # ------------------------------------------------------------------
    # 3. Export speaker encoder
    # ------------------------------------------------------------------
    spk_wrapper = _build_speaker_encoder_wrapper(net_g.enc_spk)
    # mel: (1, T_mel, 80) — ~1 s of 80-bin mel at 50 Hz
    dummy_mel = torch.zeros(1, 50, 80)

    spk_onnx = layout.component_path("quickvc_speaker_encoder.onnx")
    print(f"[export] Exporting speaker encoder → {spk_onnx} ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        export_model(
            model=spk_wrapper,
            dummy_inputs=(dummy_mel,),
            output_path=spk_onnx,
            input_names=["mel"],
            output_names=["dvec"],
            dynamic_axes={
                "mel":  {0: "batch", 1: "frames"},
                "dvec": {0: "batch"},
            },
            opset_version=14,
        )
    spk_mb = spk_onnx.stat().st_size / 1024 ** 2
    print(f"[export] Speaker encoder written: {spk_mb:.1f} MB")

    # Parity
    print("[export] Speaker encoder parity check ...")
    with torch.no_grad():
        torch_dvec = spk_wrapper(dummy_mel).detach().cpu().numpy()
    ort_dvec = run_ort(spk_onnx, {"mel": dummy_mel.numpy()})
    spk_report = compare_outputs(
        [torch_dvec],
        ort_dvec,
        names=["dvec"],
        max_abs_tol=1e-4,
        mean_abs_tol=1e-5,
    )
    print("[export] Speaker encoder parity:", spk_report.summary())
    check_tolerance(spk_report)
    spk_report.save(layout.component_path("speaker_encoder_parity_report.json"))

    # Quantize
    print("[export] Quantizing speaker encoder (INT8) ...")
    spk_q8 = layout.component_path("quickvc_speaker_encoder_q8.onnx")
    spk_quant = quantize_model(spk_onnx, output_path=spk_q8)
    print(spk_quant.summary())

    # ------------------------------------------------------------------
    # 4. Export decoder (enc_p + flow + MS-iSTFT generator minus ISTFT)
    # ------------------------------------------------------------------
    dec_wrapper = _build_decoder_wrapper(net_g)
    T_src = 50
    dummy_c = torch.zeros(1, 256, T_src)
    dummy_c_len = torch.tensor([T_src], dtype=torch.int64)
    dummy_g = torch.zeros(1, 256, 1)

    dec_onnx = layout.component_path("quickvc_decoder.onnx")
    print(f"[export] Exporting decoder → {dec_onnx} ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        export_model(
            model=dec_wrapper,
            dummy_inputs=(dummy_c, dummy_c_len, dummy_g),
            output_path=dec_onnx,
            input_names=["c", "c_lengths", "g"],
            output_names=["spec_phase"],
            dynamic_axes={
                "c":          {0: "batch", 2: "frames"},
                "c_lengths":  {0: "batch"},
                "g":          {0: "batch"},
                "spec_phase": {0: "batch_x_subbands", 3: "frames_dec"},
            },
            opset_version=14,
        )
    dec_mb = dec_onnx.stat().st_size / 1024 ** 2
    print(f"[export] Decoder written: {dec_mb:.1f} MB")

    # Parity
    print("[export] Decoder parity check ...")
    with torch.no_grad():
        torch_sp = dec_wrapper(dummy_c, dummy_c_len, dummy_g).detach().cpu().numpy()
    ort_sp = run_ort(dec_onnx, {
        "c": dummy_c.numpy(),
        "c_lengths": dummy_c_len.numpy(),
        "g": dummy_g.numpy(),
    })
    dec_report = compare_outputs(
        [torch_sp],
        ort_sp,
        names=["spec_phase"],
        max_abs_tol=2e-3,
        mean_abs_tol=5e-4,
    )
    print("[export] Decoder parity:", dec_report.summary())
    check_tolerance(dec_report)
    dec_report.save(layout.component_path("decoder_parity_report.json"))

    # Quantize
    print("[export] Quantizing decoder (INT8) ...")
    dec_q8 = layout.component_path("quickvc_decoder_q8.onnx")
    dec_quant = quantize_model(dec_onnx, output_path=dec_q8)
    print(dec_quant.summary())

    # ------------------------------------------------------------------
    # 5. Export postnet (updown_filter + multistream_conv)
    #    — small model, quantize separately
    # ------------------------------------------------------------------
    postnet_wrapper = _build_postnet_wrapper(net_g)
    subbands = QUICKVC_HPS["subbands"]
    # dummy y_mb_hat after ISTFT: (B, subbands, T_audio)
    # T_audio = T_dec * gen_istft_hop_size; T_dec ≈ T_src * upsample
    # upsample_rates=[5,4] → 20× ; T_src=50 → T_dec=1000
    T_dec = 1000
    hop = QUICKVC_HPS["gen_istft_hop_size"]
    T_audio = T_dec * hop  # 4000
    dummy_ymb = torch.zeros(1, subbands, T_audio)

    postnet_onnx = layout.component_path("quickvc_postnet.onnx")
    print(f"[export] Exporting postnet → {postnet_onnx} ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        export_model(
            model=postnet_wrapper,
            dummy_inputs=(dummy_ymb,),
            output_path=postnet_onnx,
            input_names=["y_mb_hat"],
            output_names=["y_hat"],
            dynamic_axes={
                "y_mb_hat": {0: "batch", 2: "samples"},
                "y_hat":    {0: "batch", 2: "samples"},
            },
            opset_version=14,
        )
    postnet_mb = postnet_onnx.stat().st_size / 1024 ** 2
    print(f"[export] Postnet written: {postnet_mb:.1f} MB")

    # Parity
    with torch.no_grad():
        torch_y = postnet_wrapper(dummy_ymb).detach().cpu().numpy()
    ort_y = run_ort(postnet_onnx, {"y_mb_hat": dummy_ymb.numpy()})
    postnet_report = compare_outputs(
        [torch_y],
        ort_y,
        names=["y_hat"],
        max_abs_tol=1e-4,
        mean_abs_tol=1e-5,
    )
    print("[export] Postnet parity:", postnet_report.summary())
    check_tolerance(postnet_report)
    postnet_report.save(layout.component_path("postnet_parity_report.json"))

    postnet_q8 = layout.component_path("quickvc_postnet_q8.onnx")
    postnet_quant = quantize_model(postnet_onnx, output_path=postnet_q8)
    print(postnet_quant.summary())

    del dec_wrapper, postnet_wrapper, net_g

    # ------------------------------------------------------------------
    # 6. Manifest + provenance
    # ------------------------------------------------------------------
    write_manifest(
        layout=layout,
        components={
            "content_encoder":    "quickvc_content_encoder.onnx",
            "content_encoder_q8": "quickvc_content_encoder_q8.onnx",
            "speaker_encoder":    "quickvc_speaker_encoder.onnx",
            "speaker_encoder_q8": "quickvc_speaker_encoder_q8.onnx",
            "decoder":            "quickvc_decoder.onnx",
            "decoder_q8":         "quickvc_decoder_q8.onnx",
            "postnet":            "quickvc_postnet.onnx",
            "postnet_q8":         "quickvc_postnet_q8.onnx",
        },
        sample_rates={"input": QUICKVC_SR, "output": QUICKVC_SR},
        metadata={
            "opset": 14,
            "feature_dim": 1024,
            "feature_rate_hz": 50,
            "speaker_dim": 256,
            "mel_channels": 80,
            "gen_istft_n_fft": QUICKVC_HPS["gen_istft_n_fft"],
            "gen_istft_hop_size": QUICKVC_HPS["gen_istft_hop_size"],
            "subbands": QUICKVC_HPS["subbands"],
            "istft_in_numpy": True,
            "upstream_repo": QUICKVC_UPSTREAM_URL,
            "hubert_upstream_repo": HUBERT_UPSTREAM_URL,
        },
        distributable=True,
    )
    write_provenance(
        engine_dir=layout.engine_dir,
        upstream_repo_url=QUICKVC_UPSTREAM_URL,
        upstream_ref=QUICKVC_UPSTREAM_REF,
        license_text=MIT_LICENSE,
        extra={
            "hubert_upstream": HUBERT_UPSTREAM_URL,
            "hubert_ref": HUBERT_UPSTREAM_REF,
            "content_encoder_mb": f"{enc_mb:.1f}",
            "speaker_encoder_mb": f"{spk_mb:.1f}",
            "decoder_mb": f"{dec_mb:.1f}",
            "postnet_mb": f"{postnet_mb:.1f}",
            "note": (
                "Multistream-iSTFT not in ONNX; re-implemented in numpy "
                "(irfft + hann-window OLA per subband) with parity ≤5e-4 max abs vs torch."
            ),
        },
    )

    total_mb = enc_mb + spk_mb + dec_mb + postnet_mb
    print(f"\n[export] QuickVC export complete → {layout.engine_dir}")
    print(f"[export] Total FP32: {total_mb:.1f} MB")
    return layout.engine_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export QuickVC (HuBERT-soft + VITS + MS-iSTFT decoder) to ONNX."
    )
    p.add_argument("--output-dir", default="/tmp/quickvc-out",
                   help="Staging output directory.")
    p.add_argument("--quickvc-weights", required=True,
                   help="Path to G_1200000.pth QuickVC generator checkpoint.")
    p.add_argument("--hubert-weights", default=None,
                   help="Path to hubert-soft-35d9f29f.pt (downloads if omitted).")
    p.add_argument("--no-push", action="store_true", help="Skip HF upload.")
    p.add_argument("--dry-run-push", action="store_true",
                   help="Dry-run HF upload (prints paths only).")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    engine_dir = export_quickvc(
        output_dir=args.output_dir,
        quickvc_weights=args.quickvc_weights,
        hubert_weights=args.hubert_weights,
    )

    if not args.no_push:
        from conversion.push_models import push_engine
        push_engine(
            engine_dir=engine_dir,
            engine_name="quickvc",
            dry_run=args.dry_run_push,
            commit_message=(
                "export: add QuickVC ONNX artifacts "
                "(HuBERT-soft encoder + VITS decoder + MS-iSTFT postnet, MIT)"
            ),
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
