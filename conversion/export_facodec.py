"""Export FACodec (NaturalSpeech 3) encoder and decoder to ONNX for voice conversion.

FACodec (Amphion / Microsoft Research, ICML 2024) disentangles speech into four
independently controllable subspaces: content, prosody, timbre, acoustic detail.
Voice conversion is zero-shot: encode source → quantize → swap-in reference timbre
embedding → decode.

Architecture (V2 VC path)
--------------------------
The V2 models expose a clean timbre-injection path:

1. Encode source and reference audio with FACodecEncoderV2 → (1, 256, T)
2. Extract prosody mel (first 20 mel bins at hop=200) from source.
3. Run FACodecDecoderV2 in quantize mode → VQ token ids (6, 1, T) + spk_embs.
4. VC: ``vq2emb(source_vq_ids, use_residual=False)`` → emb, then
   ``inference(emb, reference_spk_embs)`` → wav.

ONNX components
---------------
- ``facodec_encoder.onnx``   : wav (1,1,N) float32  → enc_feats (1,256,T) float32
- ``facodec_timbre.onnx``    : enc_feats (1,256,T)  → spk_embs (1,256) float32
- ``facodec_quantize.onnx``  : (enc_feats (1,256,T), mel_20 (1,20,T)) → vq_ids (6,1,T) int64
- ``facodec_decoder.onnx``   : (vq_ids (6,1,T), spk_embs (1,256)) → wav (1,1,N) float32

Mel prosody is computed in numpy by the adapter (standard STFT mel with fixed
parameters; hop=200, win=800, n_fft=1024, n_mels=80 → first 20 bins).

Upstream weights: amphion/naturalspeech3_facodec (Apache-2.0).
Code: https://github.com/open-mmlab/Amphion (Apache-2.0, MIT per module header).

Usage::

    python -m conversion.export_facodec --output-dir /tmp/facodec-out [--no-push]

Requires: ``torch``, ``onnx``, ``onnxruntime``, ``librosa``, ``huggingface_hub``,
``einops``. Never imported at voiceclonnx runtime.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FACODEC_HF_REPO = "amphion/naturalspeech3_facodec"
_FACODEC_UPSTREAM_URL = "https://github.com/open-mmlab/Amphion"
_FACODEC_UPSTREAM_REF = "main"
_FACODEC_SR = 16000
_N_QUANTIZERS = 6  # 1 prosody + 2 content + 3 residual

_APACHE2_LICENSE = """\
Apache License
Version 2.0, January 2004
http://www.apache.org/licenses/

TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
<...full Apache-2.0 text; see https://www.apache.org/licenses/LICENSE-2.0>

Copyright 2023 Amphion Authors (open-mmlab/Amphion).
Weights: amphion/naturalspeech3_facodec on Hugging Face (apache-2.0).
"""

# FACodecEncoderV2 constructor kwargs
_ENC_KWARGS = dict(ngf=32, up_ratios=[2, 4, 5, 5], out_channels=256)

# FACodecDecoderV2 constructor kwargs (match checkpoint)
_DEC_KWARGS = dict(
    in_channels=256,
    upsample_initial_channel=1024,
    ngf=32,
    up_ratios=[5, 5, 4, 2],
    vq_num_q_c=2,
    vq_num_q_p=1,
    vq_num_q_r=3,
    vq_dim=256,
    codebook_dim=8,
    codebook_size_prosody=10,
    codebook_size_content=10,
    codebook_size_residual=10,
    use_gr_x_timbre=True,
    use_gr_residual_f0=True,
    use_gr_residual_phone=True,
)


# ---------------------------------------------------------------------------
# Amphion clone / import helpers
# ---------------------------------------------------------------------------


def _get_amphion_models(
    output_dir: Path,
) -> "tuple[FACodecEncoderV2, FACodecDecoderV2]":  # noqa: F821
    """Clone Amphion (if needed) and return loaded V2 encoder/decoder."""
    import subprocess
    import torch

    # Store Amphion in HOME to avoid filling /tmp (it's ~100 MB with sparse checkout)
    import os as _os
    amphion_dir = Path(_os.path.expanduser("~/.cache/voiceclonnx_amphion"))
    if not (amphion_dir / "models").exists():
        print(f"[facodec] Cloning Amphion → {amphion_dir} …", flush=True)
        amphion_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "git",
                "clone",
                "--depth=1",
                "--filter=blob:none",
                "--sparse",
                "https://github.com/open-mmlab/Amphion.git",
                str(amphion_dir),
            ],
            check=True,
        )
        subprocess.run(
            ["git", "sparse-checkout", "set", "models/codec/ns3_codec"],
            cwd=str(amphion_dir),
            check=True,
        )

    # Install minimal deps (einops required by facodec.py)
    for pkg in ("einops",):
        try:
            __import__(pkg)
        except ImportError:
            subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=True)

    # Add Amphion to path so ``from models.codec.ns3_codec import …`` works
    if str(amphion_dir) not in sys.path:
        sys.path.insert(0, str(amphion_dir))

    from models.codec.ns3_codec import FACodecEncoderV2, FACodecDecoderV2
    from huggingface_hub import hf_hub_download

    print("[facodec] Loading FACodecEncoderV2 …", flush=True)
    encoder = FACodecEncoderV2(**_ENC_KWARGS)
    enc_ckpt = hf_hub_download(repo_id=_FACODEC_HF_REPO, filename="ns3_facodec_encoder_v2.bin")
    encoder.load_state_dict(torch.load(enc_ckpt, map_location="cpu"))
    encoder.eval()

    print("[facodec] Loading FACodecDecoderV2 …", flush=True)
    decoder = FACodecDecoderV2(**_DEC_KWARGS)
    dec_ckpt = hf_hub_download(repo_id=_FACODEC_HF_REPO, filename="ns3_facodec_decoder_v2.bin")
    decoder.load_state_dict(torch.load(dec_ckpt, map_location="cpu"))
    decoder.eval()

    return encoder, decoder


# ---------------------------------------------------------------------------
# Wrapper modules for ONNX export
# ---------------------------------------------------------------------------


def _patch_multihead_attention_for_onnx(root_module):
    """Replace nn.MultiheadAttention.forward with a dynamic-T ONNX-compatible version.

    ``nn.MultiheadAttention`` with ``batch_first=True`` internally reshapes
    ``(B, T, H)`` to ``(B*n_heads, T, head_dim)`` using the TorchScript tracer,
    which bakes T as a constant from the dummy input.  This causes a Reshape
    error at inference time when T differs from the tracing T.

    Fix: replace each ``nn.MultiheadAttention`` with a ``DynamicMHA`` that uses
    ``F.scaled_dot_product_attention`` directly — it accepts dynamic shapes.
    Weights are preserved.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import math

    class DynamicMHA(nn.Module):
        """Dynamic-T multi-head attention using F.scaled_dot_product_attention."""

        def __init__(self, orig_mha: nn.MultiheadAttention):
            super().__init__()
            self.embed_dim = orig_mha.embed_dim
            self.num_heads = orig_mha.num_heads
            self.head_dim = orig_mha.head_dim
            self.in_proj_weight = orig_mha.in_proj_weight   # (3*E, E)
            self.in_proj_bias = orig_mha.in_proj_bias       # (3*E,) or None
            self.out_proj = orig_mha.out_proj

        def forward(self, query, key, value, key_padding_mask=None, attn_mask=None, **kw):
            # query/key/value: (B, T, E)  batch_first=True assumed
            B, T, E = query.shape
            H, D = self.num_heads, self.head_dim

            # Compute Q, K, V projections
            if self.in_proj_bias is not None:
                qkv = F.linear(query, self.in_proj_weight, self.in_proj_bias)  # (B, T, 3E)
            else:
                qkv = F.linear(query, self.in_proj_weight)
            q, k, v = qkv.chunk(3, dim=-1)  # each (B, T, E)

            # Reshape to (B, H, T, D)
            q = q.view(B, T, H, D).transpose(1, 2)
            k = k.view(B, T, H, D).transpose(1, 2)
            v = v.view(B, T, H, D).transpose(1, 2)

            # Attention
            attn_out = F.scaled_dot_product_attention(q, k, v)  # (B, H, T, D)

            # Merge heads
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, E)
            out = self.out_proj(attn_out)

            # MultiheadAttention returns (attn_output, attn_weights) — we return (out, None)
            return out, None

    for name, mod in list(root_module.named_modules()):
        if isinstance(mod, nn.MultiheadAttention):
            parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
            parent = root_module if parent_name == "" else dict(root_module.named_modules())[parent_name]
            setattr(parent, child_name, DynamicMHA(mod))


def _patch_alias_free_torch_for_onnx(root_module):
    """Patch all UpSample1d / LowPassFilter1d submodules of *root_module* for ONNX export.

    Both modules use ``self.filter.expand(C, -1, -1)`` where C is read from the
    runtime input tensor shape.  The TorchScript ONNX exporter cannot export
    convolutions whose kernel shape depends on a dynamic tensor dimension.

    This function must be called **after** the module has been run with realistic
    inputs at least once (via ``_warm_module``), so that the correct C is captured.
    It then pre-expands the filters and replaces the forward method with a static
    version that uses the stored ``filter_expanded`` buffer.

    Call pattern::

        _warm_module(enc_wrapper, dummy_wav)
        _patch_alias_free_torch_for_onnx(enc_wrapper)
    """
    import torch.nn.functional as F

    try:
        from models.codec.ns3_codec.alias_free_torch.resample import UpSample1d
        from models.codec.ns3_codec.alias_free_torch.filter import LowPassFilter1d
    except ImportError:
        return

    for m in root_module.modules():
        if isinstance(m, UpSample1d) and hasattr(m, "_captured_C"):
            C = m._captured_C
            if C is None:
                continue
            expanded = m.filter.expand(C, -1, -1).contiguous()
            m.register_buffer("filter_expanded", expanded)
            pad, pad_left, pad_right = m.pad, m.pad_left, m.pad_right
            ratio, stride = m.ratio, m.stride

            def make_up(mod, ratio=ratio, stride=stride, pad=pad, pl=pad_left, pr=pad_right):
                def forward(x):
                    x = F.pad(x, (pad, pad), mode="replicate")
                    x = ratio * F.conv_transpose1d(x, mod.filter_expanded, stride=stride, groups=mod.filter_expanded.shape[0])
                    x = x[..., pl:-pr]
                    return x
                return forward

            m.forward = make_up(m)

        elif isinstance(m, LowPassFilter1d) and hasattr(m, "_captured_C"):
            C = m._captured_C
            if C is None:
                continue
            expanded = m.filter.expand(C, -1, -1).contiguous()
            m.register_buffer("filter_expanded", expanded)
            pad_l, pad_r, stride = m.pad_left, m.pad_right, m.stride
            padding, padding_mode = m.padding, m.padding_mode

            def make_lp(mod, pl=pad_l, pr=pad_r, stride=stride, do_pad=padding, pm=padding_mode):
                def forward(x):
                    if do_pad:
                        x = F.pad(x, (pl, pr), mode=pm)
                    return F.conv1d(x, mod.filter_expanded, stride=stride, groups=mod.filter_expanded.shape[0])
                return forward

            m.forward = make_lp(m)


def _warm_and_patch(root_module, warm_fn):
    """Instrument, warm (one forward pass via warm_fn), then patch alias_free_torch modules.

    *warm_fn* is a zero-argument callable that triggers a forward pass through *root_module*.
    """
    import torch.nn.functional as F

    try:
        from models.codec.ns3_codec.alias_free_torch.resample import UpSample1d
        from models.codec.ns3_codec.alias_free_torch.filter import LowPassFilter1d
    except ImportError:
        return

    # Step 1: instrument to capture C
    for m in root_module.modules():
        if isinstance(m, (UpSample1d, LowPassFilter1d)):
            m._captured_C = None

    orig_forwards = {}
    for m in root_module.modules():
        if isinstance(m, UpSample1d):
            orig_forwards[id(m)] = type(m).forward

            def _capture(x, _m=m, _orig=type(m).forward):
                _, C, _ = x.shape
                _m._captured_C = C
                return _orig(_m, x)

            m.forward = _capture

        elif isinstance(m, LowPassFilter1d):
            orig_forwards[id(m)] = type(m).forward

            def _capture(x, _m=m, _orig=type(m).forward):
                _, C, _ = x.shape
                _m._captured_C = C
                return _orig(_m, x)

            m.forward = _capture

    # Step 2: warm (one forward pass)
    import torch
    with torch.no_grad():
        warm_fn()

    # Step 3: patch
    _patch_alias_free_torch_for_onnx(root_module)


def _build_encoder_wrapper(encoder):
    """wav (1,1,N) → enc_feats (1,256,T)."""
    import torch.nn as nn

    class EncoderWrapper(nn.Module):
        def __init__(self, enc):
            super().__init__()
            self.enc = enc

        def forward(self, wav):
            return self.enc(wav)

    return EncoderWrapper(encoder).eval()


def _build_timbre_wrapper(decoder):
    """enc_feats (1,256,T) → spk_embs (1,256).

    Runs the timbre TransformerEncoder and mean-pools over time.
    """
    import torch
    import torch.nn as nn

    class TimbreWrapper(nn.Module):
        def __init__(self, dec):
            super().__init__()
            self.timbre_encoder = dec.timbre_encoder

        def forward(self, enc_feats):
            # enc_feats: (1, 256, T)
            x = enc_feats.transpose(1, 2)   # (1, T, 256)
            x = self.timbre_encoder(x, None, None)  # (1, T, 256)
            x = x.transpose(1, 2)           # (1, 256, T)
            spk_embs = torch.mean(x, dim=2)  # (1, 256)
            return spk_embs

    return TimbreWrapper(decoder).eval()


def _build_quantize_wrapper(decoder):
    """(enc_feats (1,256,T), mel_20 (1,20,T)) → vq_ids (6,1,T) int64.

    Runs the full V2 quantize step (prosody RVQ-1 + content RVQ-2 + residual RVQ-3).
    """
    import torch.nn as nn

    class QuantizeWrapper(nn.Module):
        def __init__(self, dec):
            super().__init__()
            self.quantizer = dec.quantizer
            self.melspec_linear = dec.melspec_linear
            self.melspec_encoder = dec.melspec_encoder
            self.vq_num_q_p = dec.vq_num_q_p
            self.vq_num_q_c = dec.vq_num_q_c
            self.vq_num_q_r = dec.vq_num_q_r

        def forward(self, enc_feats, mel_20):
            # enc_feats: (1, 256, T)   mel_20: (1, 20, T)
            # --- prosody RVQ ---
            f0_in = mel_20.transpose(1, 2)  # (1, T, 20)
            f0_in = self.melspec_linear(f0_in)  # (1, T, 256)
            f0_in = self.melspec_encoder(f0_in, None, None)  # (1, T, 256)
            f0_in = f0_in.transpose(1, 2)  # (1, 256, T)
            _, q_p, _, q_buf_p = self.quantizer[0](f0_in)

            # --- content RVQ ---
            _, q_c, _, q_buf_c = self.quantizer[1](enc_feats)

            # --- residual RVQ ---
            residual = enc_feats - (q_buf_p.sum(0) + q_buf_c.sum(0)).detach()
            _, q_r, _, _ = self.quantizer[2](residual)

            # Concatenate all VQ ids → (6, 1, T)
            import torch
            vq_ids = torch.cat([q_p, q_c, q_r], dim=0)  # (6, 1, T)
            return vq_ids

    return QuantizeWrapper(decoder).eval()


def _build_decoder_wrapper(decoder):
    """(vq_ids (6,1,T), spk_embs (1,256)) → wav (1,1,N).

    Runs vq2emb (without residual) then the synthesis inference path.
    """
    import torch.nn as nn

    class DecoderWrapper(nn.Module):
        def __init__(self, dec):
            super().__init__()
            self.quantizer = dec.quantizer
            self.timbre_linear = dec.timbre_linear
            self.timbre_norm = dec.timbre_norm
            self.model = dec.model
            self.vq_num_q_p = dec.vq_num_q_p
            self.vq_num_q_c = dec.vq_num_q_c
            self.vq_num_q_r = dec.vq_num_q_r

        def forward(self, vq_ids, spk_embs):
            # vq_ids: (6, 1, T)   spk_embs: (1, 256)
            # vq2emb without residual: prosody + content only
            emb = 0
            emb = emb + self.quantizer[0].vq2emb(vq_ids[: self.vq_num_q_p])
            emb = emb + self.quantizer[1].vq2emb(
                vq_ids[self.vq_num_q_p : self.vq_num_q_p + self.vq_num_q_c]
            )
            # Apply timbre conditioning (inference path)
            style = self.timbre_linear(spk_embs).unsqueeze(2)  # (1, 512, 1)
            gamma, beta = style.chunk(2, 1)   # each (1, 256, 1)
            x = emb.transpose(1, 2)           # (1, T, 256)
            x = self.timbre_norm(x)           # (1, T, 256)
            x = x.transpose(1, 2)             # (1, 256, T)
            x = x * gamma + beta
            wav = self.model(x)               # (1, 1, N)
            return wav

    return DecoderWrapper(decoder).eval()


# ---------------------------------------------------------------------------
# Parity helpers
# ---------------------------------------------------------------------------


def _parity_check(torch_out, ort_out, name: str, tol_abs: float = 1e-3, tol_mean: float = 1e-4):
    import numpy as np

    if isinstance(torch_out, (list, tuple)):
        torch_out = torch_out[0]
    ref = torch_out.detach().cpu().numpy()
    hyp = ort_out if isinstance(ort_out, np.ndarray) else ort_out[0]
    max_abs = float(np.abs(ref - hyp).max())
    mean_abs = float(np.abs(ref - hyp).mean())
    ok_max = max_abs <= tol_abs
    ok_mean = mean_abs <= tol_mean
    status = "PASS" if (ok_max and ok_mean) else "FAIL"
    print(
        f"  [{name}] max_abs={max_abs:.3e}  mean_abs={mean_abs:.3e}  → {status}",
        flush=True,
    )
    if not (ok_max and ok_mean):
        raise AssertionError(
            f"Parity check FAILED for {name}: "
            f"max_abs={max_abs:.3e} (tol={tol_abs}), "
            f"mean_abs={mean_abs:.3e} (tol={tol_mean})"
        )
    return {"max_abs": max_abs, "mean_abs": mean_abs, "pass": True}


def _run_ort(onnx_path: str, feed: dict):
    import onnxruntime as ort
    import numpy as np

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feed_np = {k: (v.detach().cpu().numpy() if hasattr(v, "detach") else np.array(v)) for k, v in feed.items()}
    return sess.run(None, feed_np)


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------


def export(output_dir: str, no_push: bool = False) -> None:
    """Full export pipeline: load → export → parity → quantize → (push)."""
    import json
    import torch

    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        raise ImportError("onnxruntime required: pip install onnxruntime")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from conversion.export_base import OutputLayout, export_model, write_manifest, write_provenance
    from conversion.quantize import quantize_model

    layout = OutputLayout.for_engine("facodec", base_dir=output_dir)
    layout.makedirs()

    # ---- Load models ----
    encoder, decoder = _get_amphion_models(output_path)

    # ---- Dummy inputs ----
    T_samp = 16000  # 1 s at 16 kHz
    T_frame = T_samp // 200  # hop=200 → 80 frames
    dummy_wav = torch.zeros(1, 1, T_samp)

    # Patch alias_free_torch resample modules to use static filter sizes for ONNX export
    print("[facodec] Patching encoder alias_free_torch modules …", flush=True)
    enc_wrapper_tmp = _build_encoder_wrapper(encoder)
    _warm_and_patch(enc_wrapper_tmp, lambda: enc_wrapper_tmp(dummy_wav))
    # Also patch the encoder module in-place (enc_wrapper shares params)
    _patch_alias_free_torch_for_onnx(encoder)

    with torch.no_grad():
        dummy_enc_feats = encoder(dummy_wav)           # (1, 256, 80)
        dummy_mel_20 = encoder.get_prosody_feature(dummy_wav)  # (1, 20, 80)

    # Patch decoder alias_free_torch modules using the synthesis (inference) path
    print("[facodec] Patching decoder alias_free_torch modules …", flush=True)
    with torch.no_grad():
        _, qs, _, qbufs, spk = decoder(dummy_enc_feats, dummy_mel_20, eval_vq=True, vq=True)
        dummy_vq_emb = decoder.vq2emb(qs, use_residual=False)
    dec_wrapper_tmp = _build_decoder_wrapper(decoder)
    _warm_and_patch(
        dec_wrapper_tmp,
        lambda: dec_wrapper_tmp(qs.detach(), spk.detach()),
    )
    _patch_alias_free_torch_for_onnx(decoder)

    # ---- Export 1: encoder ----
    print("[facodec] Exporting facodec_encoder.onnx …", flush=True)
    enc_wrapper = _build_encoder_wrapper(encoder)
    enc_path = layout.component_path("facodec_encoder.onnx")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        export_model(
            model=enc_wrapper,
            dummy_inputs=(dummy_wav,),
            output_path=str(enc_path),
            input_names=["wav"],
            output_names=["enc_feats"],
            dynamic_axes={"wav": {0: "batch", 2: "samples"}, "enc_feats": {0: "batch", 2: "frames"}},
        )
    with torch.no_grad():
        torch_enc = enc_wrapper(dummy_wav)
    ort_enc = _run_ort(str(enc_path), {"wav": dummy_wav})
    parity_enc = _parity_check(torch_enc, ort_enc[0], "encoder", tol_abs=1e-3)

    # Patch nn.MultiheadAttention in decoder (timbre_encoder + melspec_encoder) for dynamic T
    print("[facodec] Patching MultiheadAttention in decoder for dynamic T …", flush=True)
    _patch_multihead_attention_for_onnx(decoder)

    # ---- Export 2: timbre extractor ----
    print("[facodec] Exporting facodec_timbre.onnx …", flush=True)
    timbre_wrapper = _build_timbre_wrapper(decoder)
    timbre_path = layout.component_path("facodec_timbre.onnx")
    with torch.no_grad():
        torch_timbre = timbre_wrapper(dummy_enc_feats)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        export_model(
            model=timbre_wrapper,
            dummy_inputs=(dummy_enc_feats,),
            output_path=str(timbre_path),
            input_names=["enc_feats"],
            output_names=["spk_embs"],
            dynamic_axes={"enc_feats": {0: "batch", 2: "frames"}, "spk_embs": {0: "batch"}},
        )
    ort_timbre = _run_ort(str(timbre_path), {"enc_feats": dummy_enc_feats})
    parity_timbre = _parity_check(torch_timbre, ort_timbre[0], "timbre", tol_abs=1e-3)

    # ---- Export 3: quantize ----
    print("[facodec] Exporting facodec_quantize.onnx …", flush=True)
    quant_wrapper = _build_quantize_wrapper(decoder)
    quant_path = layout.component_path("facodec_quantize.onnx")
    with torch.no_grad():
        torch_vq_ids = quant_wrapper(dummy_enc_feats, dummy_mel_20)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        export_model(
            model=quant_wrapper,
            dummy_inputs=(dummy_enc_feats, dummy_mel_20),
            output_path=str(quant_path),
            input_names=["enc_feats", "mel_20"],
            output_names=["vq_ids"],
            dynamic_axes={
                "enc_feats": {0: "batch", 2: "frames"},
                "mel_20": {0: "batch", 2: "frames"},
                "vq_ids": {1: "batch", 2: "frames"},
            },
        )
    ort_vq = _run_ort(str(quant_path), {"enc_feats": dummy_enc_feats, "mel_20": dummy_mel_20})
    import numpy as np
    vq_match = bool(np.array_equal(torch_vq_ids.cpu().numpy(), ort_vq[0]))
    print(f"  [quantize] exact int64 match: {vq_match}", flush=True)
    parity_quant = {"exact_match": vq_match, "pass": vq_match}
    if not vq_match:
        raise AssertionError("Quantize parity check FAILED: VQ token IDs do not match")

    # ---- Export 4: decoder ----
    print("[facodec] Exporting facodec_decoder.onnx …", flush=True)
    dec_wrapper = _build_decoder_wrapper(decoder)
    dec_path = layout.component_path("facodec_decoder.onnx")
    dummy_spk_embs = torch_timbre.detach()
    with torch.no_grad():
        torch_dec = dec_wrapper(torch_vq_ids.detach(), dummy_spk_embs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        export_model(
            model=dec_wrapper,
            dummy_inputs=(torch_vq_ids.detach(), dummy_spk_embs),
            output_path=str(dec_path),
            input_names=["vq_ids", "spk_embs"],
            output_names=["wav"],
            dynamic_axes={
                "vq_ids": {1: "batch", 2: "frames"},
                "spk_embs": {0: "batch"},
                "wav": {0: "batch", 2: "samples"},
            },
        )
    ort_dec = _run_ort(str(dec_path), {"vq_ids": torch_vq_ids.detach(), "spk_embs": dummy_spk_embs})
    parity_dec = _parity_check(torch_dec, ort_dec[0], "decoder", tol_abs=1e-3)

    # ---- Quantize to INT8 ----
    print("[facodec] Quantizing models to INT8 …", flush=True)
    sizes = {}
    for name, path in [
        ("facodec_encoder", enc_path),
        ("facodec_timbre", timbre_path),
        ("facodec_quantize", quant_path),
        ("facodec_decoder", dec_path),
    ]:
        q_path = str(path).replace(".onnx", "_q8.onnx")
        report = quantize_model(str(path), output_path=q_path)
        fp32_mb = Path(path).stat().st_size / 1024**2
        q8_mb = Path(q_path).stat().st_size / 1024**2
        sizes[name] = {"fp32_mb": round(fp32_mb, 1), "q8_mb": round(q8_mb, 1)}
        print(f"  {name}: {fp32_mb:.1f} MB → {q8_mb:.1f} MB", flush=True)

    # ---- Manifest ----
    write_manifest(
        layout=layout,
        components={
            "facodec_encoder": "facodec_encoder.onnx",
            "facodec_encoder_q8": "facodec_encoder_q8.onnx",
            "facodec_timbre": "facodec_timbre.onnx",
            "facodec_timbre_q8": "facodec_timbre_q8.onnx",
            "facodec_quantize": "facodec_quantize.onnx",
            "facodec_quantize_q8": "facodec_quantize_q8.onnx",
            "facodec_decoder": "facodec_decoder.onnx",
            "facodec_decoder_q8": "facodec_decoder_q8.onnx",
        },
        sample_rates={"output": _FACODEC_SR},
        metadata={
            "opset": 14,
            "upstream_repo": _FACODEC_HF_REPO,
            "license": "apache-2.0",
            "n_quantizers": _N_QUANTIZERS,
            "vq_num_q_p": _DEC_KWARGS["vq_num_q_p"],
            "vq_num_q_c": _DEC_KWARGS["vq_num_q_c"],
            "vq_num_q_r": _DEC_KWARGS["vq_num_q_r"],
            "hop_length": 200,
            "parity": {
                "encoder": parity_enc,
                "timbre": parity_timbre,
                "quantize": parity_quant,
                "decoder": parity_dec,
            },
            "sizes_mb": sizes,
        },
    )

    write_provenance(
        engine_dir=layout.engine_dir,
        upstream_repo_url=_FACODEC_UPSTREAM_URL,
        upstream_ref=_FACODEC_UPSTREAM_REF,
        license_text=_APACHE2_LICENSE,
    )

    print(f"[facodec] Export complete → {layout.engine_dir}", flush=True)

    if not no_push:
        from conversion.push_models import push
        push(str(layout.engine_dir), engine="facodec")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args():
    p = argparse.ArgumentParser(description="Export FACodec V2 to ONNX")
    p.add_argument("--output-dir", required=True, help="Staging directory for ONNX files.")
    p.add_argument("--no-push", action="store_true", help="Skip HF upload.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    export(args.output_dir, no_push=args.no_push)
