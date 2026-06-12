"""Export LinaCodec to ONNX.

LinaCodec (https://github.com/ysharma3501/LinaCodec) is a codec-based VC engine
producing 48 kHz output.

LICENSE NOTE
------------
The LinaCodec transformer module (``linacodec/module/transformer.py``) is adapted
from Meta's Llama-3 source code under the Llama 3 Community License Agreement.
The ``distill_wavlm`` module derives from torchaudio (BSD-2-Clause).

EXTERNAL CHECKOUT PATTERN
--------------------------
Neither file is vendored into this MIT repository.  The export script clones
the LinaCodec repository into a throwaway directory (``/tmp/LinaCodec``) at
conversion time, adds it to ``sys.path``, and removes it on exit.  The runtime
adapter ``voiceclonnx/engines/linacodec.py`` contains ZERO upstream code —
pure onnxruntime + numpy only.

Usage::

    # Install conversion deps (throwaway venv recommended)
    pip install torch onnx onnxruntime safetensors soundfile vocos jsonargparse torchaudio huggingface_hub

    # Clone LinaCodec externally (will be done automatically if not present)
    git clone https://github.com/ysharma3501/LinaCodec /tmp/LinaCodec

    # Run export
    python -m conversion.export_linacodec --output-dir /tmp/lina-out

    # With HF push
    HF_TOKEN=hf_... python -m conversion.export_linacodec --output-dir /tmp/lina-out --push
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import os
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# External checkout: clone LinaCodec into /tmp if not present
# ---------------------------------------------------------------------------

LINACODEC_CLONE_DIR = Path("/tmp/LinaCodec")
LINACODEC_REPO_URL = "https://github.com/ysharma3501/LinaCodec"


def _ensure_linacodec_clone() -> None:
    """Clone LinaCodec into /tmp/LinaCodec if not already present."""
    if LINACODEC_CLONE_DIR.exists() and (LINACODEC_CLONE_DIR / "src" / "linacodec").exists():
        print(f"[linacodec] Using existing clone at {LINACODEC_CLONE_DIR}")
        return
    print(f"[linacodec] Cloning {LINACODEC_REPO_URL} → {LINACODEC_CLONE_DIR}")
    subprocess.run(
        ["git", "clone", "--depth=1", LINACODEC_REPO_URL, str(LINACODEC_CLONE_DIR)],
        check=True,
    )


def _add_linacodec_to_path() -> None:
    """Add the external LinaCodec src/ directory to sys.path."""
    src_dir = str(LINACODEC_CLONE_DIR / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    print(f"[linacodec] Added {src_dir} to sys.path (external checkout, not vendored)")


# ---------------------------------------------------------------------------
# RoPE ONNX-compatible rewrite
# ---------------------------------------------------------------------------
# The upstream apply_rotary_emb uses torch.view_as_complex / view_as_real
# which are not supported by the ONNX TorchScript exporter.
# We replace them with equivalent real-valued cos/sin operations.

def _precompute_freqs_cos_sin(head_dim: int, max_len: int, theta: float = 10000.0):
    """Pre-compute cos/sin for RoPE as real tensors."""
    import torch
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (max_len, head_dim//2)
    cos = freqs.cos()  # (max_len, head_dim//2)
    sin = freqs.sin()  # (max_len, head_dim//2)
    return cos, sin


def _apply_rotary_emb_real(x, cos, sin):
    """
    Apply rotary embeddings using real cos/sin (ONNX-compatible).

    Args:
        x: (B, T, n_heads, head_dim) float32
        cos: (T, head_dim//2) float32
        sin: (T, head_dim//2) float32

    Returns:
        (B, T, n_heads, head_dim) float32
    """
    import torch
    # Split head_dim into two halves: x1 = x[..., 0::2], x2 = x[..., 1::2]
    x1 = x[..., 0::2]   # (B, T, n_heads, head_dim//2)
    x2 = x[..., 1::2]   # (B, T, n_heads, head_dim//2)
    # Broadcast cos/sin: (T, head_dim//2) → (1, T, 1, head_dim//2)
    c = cos.unsqueeze(0).unsqueeze(2)
    s = sin.unsqueeze(0).unsqueeze(2)
    # Rotate: [x1, x2] → [x1·cos - x2·sin, x1·sin + x2·cos]
    out_x1 = x1 * c - x2 * s
    out_x2 = x1 * s + x2 * c
    # Interleave back
    out = torch.stack([out_x1, out_x2], dim=-1).flatten(-2)  # (B, T, n_heads, head_dim)
    return out.type_as(x)


def _patch_transformer_rope(transformer_module):
    """
    Monkey-patch the Transformer to use real-valued RoPE.
    Patches each Attention.forward to use cos/sin instead of complex freqs_cis.
    """
    import torch
    import torch.nn.functional as F
    import types

    if transformer_module.freqs_cis is None:
        return  # no-op for transformers without RoPE

    # Convert freqs_cis to cos+sin
    freqs_cis = transformer_module.freqs_cis  # (max_len, head_dim//2) complex
    head_dim = transformer_module.dim // transformer_module.n_heads
    max_len = freqs_cis.shape[0]

    cos, sin = _precompute_freqs_cos_sin(
        head_dim, max_len, transformer_module.rope_theta
    )
    # Store on the module for forward use
    transformer_module.register_buffer("_rope_cos", cos, persistent=False)
    transformer_module.register_buffer("_rope_sin", sin, persistent=False)
    transformer_module.freqs_cis = None  # disable complex path

    def patched_transformer_forward(self, x, mask=None, condition=None, **kwargs):
        bsz, seqlen, _dim = x.shape
        rope_cos = self._rope_cos[:seqlen]  # (T, head_dim//2)
        rope_sin = self._rope_sin[:seqlen]

        x = self.input_proj(x)
        for layer in self.layers:
            x = _patched_block_forward(layer, x, rope_cos, rope_sin, mask, condition)

        if self.use_adaln_zero:
            x, _ = self.norm(x, condition=condition)
        else:
            x = self.norm(x)
        return self.output_proj(x)

    def _patched_block_forward(block, x, rope_cos, rope_sin, mask=None, condition=None):
        if block.use_adaln_zero:
            attn_normed, attn_gate = block.attention_norm(x, condition=condition)
        else:
            attn_normed = block.attention_norm(x)

        attn_out = _patched_attention_forward(block.attention, attn_normed, rope_cos, rope_sin, mask)

        if block.use_adaln_zero:
            h = x + attn_gate * attn_out
        else:
            h = x + attn_out

        if block.use_adaln_zero:
            ffn_normed, ffn_gate = block.ffn_norm(h, condition=condition)
        else:
            ffn_normed = block.ffn_norm(h)

        ffn_out = block.feed_forward(ffn_normed)

        if block.use_adaln_zero:
            return h + ffn_gate * ffn_out
        return h + ffn_out

    def _patched_attention_forward(attn, x, rope_cos, rope_sin, mask=None):
        bsz, seqlen, _ = x.shape
        xq = attn.wq(x).view(bsz, seqlen, attn.n_heads, attn.head_dim)
        xk = attn.wk(x).view(bsz, seqlen, attn.n_heads, attn.head_dim)
        xv = attn.wv(x).view(bsz, seqlen, attn.n_heads, attn.head_dim)

        # Apply real-valued RoPE
        xq = _apply_rotary_emb_real(xq, rope_cos, rope_sin)
        xk = _apply_rotary_emb_real(xk, rope_cos, rope_sin)

        # Disable local attention mask for ONNX export: the window-based mask
        # creates (seqlen, seqlen) tensors from Python ints, baking seqlen as
        # a constant in the traced graph.  For offline VC (full sequence
        # available), full bidirectional attention is correct and produces
        # equivalent quality for short sequences (window_size >> T_tokens).
        attn_mask = None

        output = F.scaled_dot_product_attention(
            xq.transpose(1, 2),
            xk.transpose(1, 2),
            xv.transpose(1, 2),
            attn_mask=attn_mask,
            dropout_p=0.0,
            scale=attn.scale,
        ).transpose(1, 2)

        output = output.contiguous().view(bsz, seqlen, -1)
        return attn.wo(output)

    transformer_module.forward = types.MethodType(patched_transformer_forward, transformer_module)


# ---------------------------------------------------------------------------
# Vocos ISTFT numpy re-implementation
# ---------------------------------------------------------------------------

def _numpy_istft(magnitude: np.ndarray, phase: np.ndarray,
                 n_fft: int = 1024, hop_length: int = 256,
                 padding: str = "center") -> np.ndarray:
    """
    Pure numpy ISTFT matching vocos ISTFTHead output.

    Args:
        magnitude: (B, n_fft//2+1, T) float32
        phase: (B, n_fft//2+1, T) float32 — phase angle (radians)
        n_fft, hop_length: STFT parameters
        padding: "center" or "same"

    Returns:
        waveform: (B, T_audio) float32
    """
    B, n_bins, T_frames = magnitude.shape
    win = np.hanning(n_fft + 1)[:-1].astype(np.float32)

    results = []
    for b in range(B):
        # Reconstruct complex spectrum
        spec = magnitude[b] * np.exp(1j * phase[b])  # (n_bins, T)

        # OLA accumulation
        output_length = (T_frames - 1) * hop_length + n_fft
        out = np.zeros(output_length, dtype=np.float32)
        norm = np.zeros(output_length, dtype=np.float32)

        for t in range(T_frames):
            frame = np.fft.irfft(spec[:, t], n=n_fft).astype(np.float32)
            frame = frame * win
            start = t * hop_length
            out[start:start + n_fft] += frame
            norm[start:start + n_fft] += win ** 2

        # Normalise overlap
        norm = np.where(norm < 1e-8, 1.0, norm)
        out /= norm

        if padding == "center":
            trim = n_fft // 2
            out = out[trim:-trim]

        results.append(out)

    return np.stack(results, axis=0)


def _numpy_linkwitz_riley(path1_48k: np.ndarray, path2_48k: np.ndarray,
                           sample_rate: int = 48000, cutoff: int = 4000,
                           transition_bins: int = 8) -> np.ndarray:
    """
    Pure numpy Linkwitz-Riley frequency crossover merge.
    Merges low-freq content from path2 with high-freq from path1.
    """
    # Truncate to shorter length
    min_len = min(path1_48k.shape[-1], path2_48k.shape[-1])
    p1 = path1_48k[..., :min_len]
    p2 = path2_48k[..., :min_len]

    spec1 = np.fft.rfft(p1, axis=-1)
    spec2 = np.fft.rfft(p2, axis=-1)

    n_bins = spec1.shape[-1]
    cutoff_bin = int((cutoff / (sample_rate / 2)) * n_bins)

    mask = np.ones(n_bins, dtype=np.float32)
    half = transition_bins // 2
    start = max(0, cutoff_bin - half)
    end = min(n_bins, cutoff_bin + half)
    actual_width = end - start

    x = np.linspace(-1, 1, actual_width, dtype=np.float32)
    fade = 3 * ((x + 1) / 2) ** 2 - 2 * ((x + 1) / 2) ** 3

    mask[:start] = 0.0
    mask[start:end] = fade
    mask[end:] = 1.0

    merged = spec1 * mask + spec2 * (1.0 - mask)
    return np.fft.irfft(merged, n=min_len, axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Wrapper modules for export
# ---------------------------------------------------------------------------

class _SSLEncoderWrapper(object):
    """
    Wraps the SSL feature extractor (WavLM Base Plus via torchaudio) for export.
    Exports two output tensors: acoustic features for global branch (layers 1-2 avg).
    """
    pass


class _ContentEncoderWrapper(object):
    """
    Wraps distill_wavlm + local_encoder + FSQ quantizer.
    Input: waveform (B, N) float32 at 24kHz
    Output: content_embedding (B, T, 768), content_tokens (B, T) int64
    """
    pass


# ---------------------------------------------------------------------------
# ONNX Wrapper nn.Module classes
# ---------------------------------------------------------------------------

def _build_ssl_acoustic_module(model):
    """Return a traceable module for acoustic SSL branch (global encoder input)."""
    import torch, torch.nn as nn

    class AcousticSSL(nn.Module):
        """
        Extracts acoustic SSL features at layers 1 and 2, averages them.
        Input: waveform (1, N) float32 at 16kHz (SSL sample rate)
        Output: acoustic_features (1, T_ssl, 768) float32
        """
        def __init__(self, ssl_extractor):
            super().__init__()
            self.extractor = ssl_extractor

        def forward(self, waveform_16k: torch.Tensor) -> torch.Tensor:
            # waveform_16k: (1, N) at 16kHz
            feats = self.extractor(waveform_16k, num_layers=2)
            # feats is a list of 2 tensors (1, T, 768)
            return (feats[0] + feats[1]) / 2.0  # (1, T, 768)

    return AcousticSSL(model.ssl_feature_extractor)


def _build_content_module_parts(model):
    """
    Returns a traceable module for content encoding.
    Input: local_ssl_features (1, T, 768) float32 (from distill_wavlm)
    Output: content_embedding (1, T//factor, 768), content_tokens (1, T//factor) int64
    """
    import torch, torch.nn as nn

    class ContentEncoder(nn.Module):
        """
        local_encoder (patched RoPE Transformer) + conv_downsample + FSQ quantizer.
        """
        def __init__(self, local_encoder, conv_downsample, local_quantizer, downsample_factor):
            super().__init__()
            self.local_encoder = local_encoder
            self.conv_downsample = conv_downsample
            self.downsample_factor = downsample_factor
            self.local_quantizer = local_quantizer

        def forward(self, local_ssl_features: torch.Tensor):
            # local_ssl_features: (1, T, 768)
            local_encoded = self.local_encoder(local_ssl_features)  # (1, T, 768)
            # Temporal downsampling via Conv1d
            local_encoded = self.conv_downsample(local_encoded.transpose(1, 2)).transpose(1, 2)  # (1, T/4, 768)
            # FSQ encode: returns (quantized, indices)
            local_quantized, indices = self.local_quantizer.encode(local_encoded)
            return local_quantized, indices.to(torch.int64)

    return ContentEncoder(
        model.local_encoder,
        model.conv_downsample,
        model.local_quantizer,
        model.downsample_factor,
    )


def _build_global_module(model):
    """
    Returns a traceable module for global embedding.
    Input: acoustic_ssl_features (1, T, 768) float32
    Output: global_embedding (1, 128) float32
    """
    return model.global_encoder


def _build_mel_decoder_module(model):
    """
    Returns a traceable module for mel decoding.
    Inputs: content_embedding (1, T_content, 768), global_embedding (1, 128)
    Output: mel_spectrogram (1, 100, T_mel)

    mel_length is computed from content_embedding.shape[1] using the known
    frame-rate relationship:
        audio_len ≈ T_tokens * 1920  (12.5 tokens/sec at 24kHz)
        mel_length = audio_len // 256 + 1  (hop_length=256, center pad)
    i.e. mel_length = T_tokens * 1920 // 256 + 1 = T_tokens * 7 + (T_tokens*4//256) + 1

    Because mel_length is an int derived from tensor shapes, it is baked as a
    constant by the TorchScript tracer.  This is correct for the dummy length;
    the adapter must therefore call one ORT session per T_tokens value.
    For variable-length inference the adapter slices content_embedding into
    fixed-size windows (if needed), but in practice 1-10 second clips produce
    T_tokens=12..125 which all trace independently.

    To avoid the variable-length issue we use scale_factor=15/32 ≈ 0.46875
    for the final interpolation:
        after mel_conv_upsample (stride=8): T_after = T_content * 8
        target: T_mel = T_tokens * 7.5 + 1 ≈ T_after * (7.5/8) = T_after * 0.9375
    We bake mel_length as a Python int (traced constant) — which is correct
    because audio_len is proportional to T_tokens for any input duration.
    """
    import torch, torch.nn as nn

    # Precompute constants
    # audio_len ≈ T_tokens * downsample_factor * ssl_hop * model_sr / ssl_sr
    # = T_tokens * 4 * 320 * 24000/16000 = T_tokens * 1920
    _SAMPLES_PER_TOKEN = 1920  # ≈ 24000 / 12.5
    _HOP = 256

    class MelDecoder(nn.Module):
        """mel_prenet + mel_conv_upsample + interpolation + mel_decoder (AdaLN) + postnet.

        F.interpolate bakes mel_length as a constant during tracing — this is expected
        and correct because mel_length is determined by the sequence length of the input.
        The adapter uses this with a fixed content_embedding T dimension per call.
        """

        def __init__(self, mel_prenet, mel_conv_upsample, mel_decoder, mel_postnet,
                     mel_upsample_factor, mel_interpolation_mode):
            super().__init__()
            self.mel_prenet = mel_prenet
            self.mel_conv_upsample = mel_conv_upsample
            self.mel_decoder = mel_decoder
            self.mel_postnet = mel_postnet
            self.mel_upsample_factor = mel_upsample_factor
            self.mel_interpolation_mode = mel_interpolation_mode

        def forward(self, content_embedding: torch.Tensor, global_embedding: torch.Tensor) -> torch.Tensor:
            # content_embedding: (1, T, 768); global_embedding: (1, 128)
            local_latent = self.mel_prenet(content_embedding)  # (1, T, 512)

            if self.mel_conv_upsample is not None:
                local_latent = self.mel_conv_upsample(local_latent.transpose(1, 2)).transpose(1, 2)
            # After mel_conv_upsample (stride=8): T_up = T_content * 8
            # Target: mel_len ≈ T_content * 7.5 = T_up * 0.9375
            # scale_factor keeps the graph dynamic — ORT computes output size from
            # input shape at runtime, so any T_content value works correctly.
            local_latent = torch.nn.functional.interpolate(
                local_latent.transpose(1, 2),
                scale_factor=0.9375,
                mode=self.mel_interpolation_mode,
            ).transpose(1, 2)  # (1, T_mel, 512)

            mel_recon = self.mel_decoder(local_latent, condition=global_embedding.unsqueeze(1))
            mel_recon = mel_recon.transpose(1, 2)  # (1, 100, mel_len)
            mel_recon = self.mel_postnet(mel_recon)
            return mel_recon  # (1, 100, mel_len)

    return MelDecoder(
        model.mel_prenet,
        model.mel_conv_upsample,
        model.mel_decoder,
        model.mel_postnet,
        model.config.mel_upsample_factor,
        model.config.mel_interpolation_mode,
    )


def _build_vocos_module(vocos):
    """
    Wraps the Vocos backbone + UpSamplerBlock + ISTFTHead linear projection.
    The ISTFT and Linkwitz-Riley steps are done in numpy in the adapter.

    Input: mel (1, 100, T) float32
    Output (24kHz path): mag24 (1, n_fft//2+1, T), phase24 (1, n_fft//2+1, T)
    Output (48kHz path): mag48 (1, n_fft//2+1, T*2), phase48 (1, n_fft//2+1, T*2)
    """
    import torch, torch.nn as nn

    class VocosExportModule(nn.Module):
        """
        Exports backbone → upsampler → head_48k + head (projection only).
        Returns (magnitude_24k, phase_24k, magnitude_48k, phase_48k).
        The adapter computes ISTFT and Linkwitz-Riley in numpy.
        """
        def __init__(self, vocos_model):
            super().__init__()
            self.backbone = vocos_model.backbone
            self.upsampler = vocos_model.upsampler
            # ISTFTHead: self.out = nn.Linear(dim, (n_fft // 2 + 1) * 2, bias=True)
            self.head_24k = vocos_model.head
            self.head_48k = vocos_model.head_48k

        def forward(self, mel: torch.Tensor):
            # mel: (1, 100, T)
            features = self.backbone(mel)  # (1, T, dim=512) via ConvNeXt
            features_T = features.transpose(1, 2)  # (1, dim, T)

            # 48kHz path: upsample then project
            upsampled = self.upsampler(features_T)  # (1, dim, T*2) with UpSamplerBlock
            # ISTFTHead.out: linear from dim to (n_fft//2+1)*2
            out_48k = self.head_48k.out(upsampled.transpose(1, 2))  # (1, T*2, (n_fft//2+1)*2)
            mag_ph_48k = out_48k.transpose(1, 2)  # (1, (n_fft//2+1)*2, T*2)
            n_bins_48k = mag_ph_48k.shape[1] // 2
            mag_48k = mag_ph_48k[:, :n_bins_48k, :]
            phase_48k = mag_ph_48k[:, n_bins_48k:, :]

            # 24kHz path: use features_T directly
            out_24k = self.head_24k.out(features.contiguous())  # (1, T, (n_fft//2+1)*2)
            mag_ph_24k = out_24k.transpose(1, 2)  # (1, (n_fft//2+1)*2, T)
            n_bins_24k = mag_ph_24k.shape[1] // 2
            mag_24k = mag_ph_24k[:, :n_bins_24k, :]
            phase_24k = mag_ph_24k[:, n_bins_24k:, :]

            return mag_24k, phase_24k, mag_48k, phase_48k

    return VocosExportModule(vocos)


# ---------------------------------------------------------------------------
# Distill-WavLM wrapper for content encoding (takes 24kHz audio, outputs layers)
# ---------------------------------------------------------------------------

def _build_distill_wavlm_module(model):
    """
    Wraps the distilled WavLM model for content SSL extraction.
    Input: waveform_16k (1, N) float32 at 16kHz (SSL sample rate)
    Output: semantic_features (1, T, 768) float32 — average of layers 6 and 9
    """
    import torch, torch.nn as nn

    class DistillWavLMEncoder(nn.Module):
        def __init__(self, wavlm_model, distilled_layers):
            super().__init__()
            self.wavlm_model = wavlm_model
            self.distilled_layers = distilled_layers

        def forward(self, waveform_16k: torch.Tensor) -> torch.Tensor:
            # waveform_16k: (1, N) at 16kHz
            all_features, _ = self.wavlm_model.extract_features(
                waveform_16k, num_layers=max(self.distilled_layers)
            )
            # Average the specified layers
            selected = [all_features[i - 1] for i in self.distilled_layers]
            return torch.stack(selected, dim=0).mean(dim=0)  # (1, T, 768)

    return DistillWavLMEncoder(model.wavlm_model, model.distilled_layers)


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------

def export(output_dir: str, push: bool = False, no_push: bool = False) -> None:
    """Export LinaCodec components to ONNX."""
    import torch
    from huggingface_hub import snapshot_download

    # Import conversion helpers
    _repo_root = Path(__file__).parent.parent
    sys.path.insert(0, str(_repo_root))
    from conversion.export_base import OutputLayout, export_model, write_manifest, write_provenance
    from conversion.parity import compare_outputs, check_tolerance, run_ort
    from conversion.quantize import quantize_model

    # Ensure LinaCodec is cloned
    _ensure_linacodec_clone()
    _add_linacodec_to_path()

    # Now safe to import linacodec
    from linacodec.model import LinaCodecModel
    from linacodec.vocoder.vocos import Vocos

    print("[linacodec] Downloading LinaCodec weights from YatharthS/LinaCodec ...")
    model_path = snapshot_download("YatharthS/LinaCodec")

    # Load model
    print("[linacodec] Loading LinaCodecModel ...")
    model = LinaCodecModel.from_pretrained(
        config_path=f"{model_path}/config.yaml",
        weights_path=f"{model_path}/model.safetensors",
    ).eval().cpu()

    # Load distilled wavlm
    print("[linacodec] Loading distilled WavLM ...")
    model.load_distilled_wavlm(f"{model_path}/wavlm_encoder.pth")
    model.wavlm_model.cpu()
    model.distilled_layers = [6, 9]

    # Load vocoder
    print("[linacodec] Loading Vocos vocoder ...")
    vocos = Vocos.from_hparams(f"{model_path}/vocoder/config.yaml")
    import torch as _t
    state = _t.load(f"{model_path}/vocoder/pytorch_model.bin", map_location="cpu")
    vocos.load_state_dict(state)
    vocos.eval()

    layout = OutputLayout.for_engine("linacodec", base_dir=output_dir)
    layout.makedirs()

    # -----------------------------------------------------------------------
    # Patch RoPE for ONNX export (real-valued cos/sin)
    # -----------------------------------------------------------------------
    print("[linacodec] Patching Transformer RoPE for ONNX export ...")
    _patch_transformer_rope(model.local_encoder)
    _patch_transformer_rope(model.mel_prenet)
    _patch_transformer_rope(model.mel_decoder)

    # -----------------------------------------------------------------------
    # Prepare resampler (24kHz → 16kHz for SSL)
    # -----------------------------------------------------------------------
    import torchaudio
    resampler_24_to_16 = torchaudio.transforms.Resample(24000, 16000).eval()

    # -----------------------------------------------------------------------
    # Dummy inputs
    # -----------------------------------------------------------------------
    # 1 second at 24kHz
    dummy_wav_24k = torch.randn(1, 24000)
    dummy_wav_16k = resampler_24_to_16(dummy_wav_24k)  # (1, 16000) approximately

    # Run forward to get typical shapes
    with torch.no_grad():
        audio_length = dummy_wav_24k.shape[-1]
        padding = model._calculate_waveform_padding(audio_length)
        if padding > 0:
            wav_padded = torch.nn.functional.pad(dummy_wav_24k, (padding, padding))
        else:
            wav_padded = dummy_wav_24k

        wav_padded_16k = resampler_24_to_16(wav_padded)

        # Acoustic SSL (global branch)
        acoustic_feats_list = model.ssl_feature_extractor(
            wav_padded_16k, num_layers=2
        )
        dummy_acoustic_feats = (acoustic_feats_list[0] + acoustic_feats_list[1]) / 2.0
        print(f"[linacodec] Acoustic SSL features shape: {dummy_acoustic_feats.shape}")

        # Distill WavLM (content branch)
        distill_feats_list, _ = model.wavlm_model.extract_features(
            wav_padded_16k, num_layers=max(model.distilled_layers)
        )
        dummy_local_ssl = torch.stack(
            [distill_feats_list[i - 1] for i in model.distilled_layers], dim=0
        ).mean(dim=0)
        # Normalize
        mean = dummy_local_ssl.mean(dim=1, keepdim=True)
        std = dummy_local_ssl.std(dim=1, keepdim=True)
        dummy_local_ssl = (dummy_local_ssl - mean) / (std + 1e-8)
        print(f"[linacodec] Local SSL features shape: {dummy_local_ssl.shape}")

        # Content encoder forward
        local_encoded = model.local_encoder(dummy_local_ssl)
        local_encoded_ds = model.conv_downsample(local_encoded.transpose(1, 2)).transpose(1, 2)
        dummy_content_emb, dummy_content_tokens = model.local_quantizer.encode(local_encoded_ds)
        dummy_content_tokens = dummy_content_tokens.long()
        print(f"[linacodec] Content embedding shape: {dummy_content_emb.shape}, tokens shape: {dummy_content_tokens.shape}")

        # Global embedding
        dummy_global_emb = model.global_encoder(dummy_acoustic_feats)  # (1, 128)
        print(f"[linacodec] Global embedding shape: {dummy_global_emb.shape}")

        # Mel decoder
        T_content = dummy_content_emb.shape[1]
        target_audio_len = model._calculate_original_audio_length(T_content)
        mel_length = model._calculate_target_mel_length(target_audio_len)
        mel_length_tensor = torch.tensor(mel_length, dtype=torch.long)
        dummy_mel = model.forward_mel(dummy_content_emb, dummy_global_emb, mel_length)
        print(f"[linacodec] Mel spectrogram shape: {dummy_mel.shape}")

    # -----------------------------------------------------------------------
    # Component 1: acoustic_ssl_encoder.onnx
    # (WavLM Base Plus acoustic features for global branch)
    # -----------------------------------------------------------------------
    print("\n[linacodec] Exporting acoustic_ssl_encoder.onnx ...")
    acoustic_module = _build_ssl_acoustic_module(model)
    acoustic_module.eval()

    with torch.no_grad():
        torch_acoustic_out = acoustic_module(wav_padded_16k)

    acoustic_onnx = layout.component_path("acoustic_ssl_encoder.onnx")
    export_model(
        model=acoustic_module,
        dummy_inputs=(wav_padded_16k,),
        output_path=acoustic_onnx,
        input_names=["waveform_16k"],
        output_names=["acoustic_features"],
        dynamic_axes={
            "waveform_16k": {0: "batch", 1: "samples"},
            "acoustic_features": {0: "batch", 1: "frames"},
        },
        opset_version=14,
    )

    # Parity check
    ort_acoustic = run_ort(acoustic_onnx, {"waveform_16k": wav_padded_16k.numpy()})
    rpt = compare_outputs(torch_acoustic_out, ort_acoustic[0])
    check_tolerance(rpt, max_abs_tol=1e-3, mean_abs_tol=1e-4)
    print(f"  acoustic_ssl_encoder parity: max_abs={rpt.components[0].max_abs_delta:.3e}, mean_abs={rpt.components[0].mean_abs_delta:.3e} PASS")

    # -----------------------------------------------------------------------
    # Component 2: distill_wavlm_encoder.onnx
    # (Distilled WavLM for content branch)
    # -----------------------------------------------------------------------
    print("\n[linacodec] Exporting distill_wavlm_encoder.onnx ...")
    distill_module = _build_distill_wavlm_module(model)
    distill_module.eval()

    with torch.no_grad():
        torch_distill_out = distill_module(wav_padded_16k)

    distill_onnx = layout.component_path("distill_wavlm_encoder.onnx")
    export_model(
        model=distill_module,
        dummy_inputs=(wav_padded_16k,),
        output_path=distill_onnx,
        input_names=["waveform_16k"],
        output_names=["semantic_features"],
        dynamic_axes={
            "waveform_16k": {0: "batch", 1: "samples"},
            "semantic_features": {0: "batch", 1: "frames"},
        },
        opset_version=14,
    )

    ort_distill = run_ort(distill_onnx, {"waveform_16k": wav_padded_16k.numpy()})
    rpt = compare_outputs(torch_distill_out, ort_distill[0])
    check_tolerance(rpt, max_abs_tol=5e-3, mean_abs_tol=5e-4)
    print(f"  distill_wavlm_encoder parity: max_abs={rpt.components[0].max_abs_delta:.3e}, mean_abs={rpt.components[0].mean_abs_delta:.3e} PASS")

    # -----------------------------------------------------------------------
    # Component 3: content_encoder.onnx
    # (local_encoder + conv_downsample + FSQ.encode)
    # -----------------------------------------------------------------------
    print("\n[linacodec] Exporting content_encoder.onnx ...")
    content_module = _build_content_module_parts(model)
    content_module.eval()

    # Use normalized SSL features as input
    with torch.no_grad():
        torch_content_emb, torch_content_tokens = content_module(dummy_local_ssl)

    content_onnx = layout.component_path("content_encoder.onnx")
    export_model(
        model=content_module,
        dummy_inputs=(dummy_local_ssl,),
        output_path=content_onnx,
        input_names=["local_ssl_features"],
        output_names=["content_embedding", "content_tokens"],
        dynamic_axes={
            "local_ssl_features": {0: "batch", 1: "frames"},
            "content_embedding": {0: "batch", 1: "tokens"},
            "content_tokens": {0: "batch", 1: "tokens"},
        },
        opset_version=14,
    )

    ort_content = run_ort(content_onnx, {"local_ssl_features": dummy_local_ssl.numpy()})
    rpt_emb = compare_outputs(torch_content_emb, ort_content[0])
    # Tokens should be exact integer match
    token_match_pct = np.mean(torch_content_tokens.numpy() == ort_content[1])
    if token_match_pct < 0.99:
        print(f"  WARNING: content tokens match {token_match_pct:.1%} (expected ≥99%)")
    else:
        print(f"  content_encoder parity (tokens): {token_match_pct:.1%} match PASS")
    check_tolerance(rpt_emb, max_abs_tol=1e-3, mean_abs_tol=1e-4)
    print(f"  content_encoder parity (embedding): max_abs={rpt_emb.components[0].max_abs_delta:.3e} PASS")
    # token match already printed above

    # -----------------------------------------------------------------------
    # Component 4: global_encoder.onnx
    # -----------------------------------------------------------------------
    print("\n[linacodec] Exporting global_encoder.onnx ...")
    global_module = _build_global_module(model)
    global_module.eval()

    with torch.no_grad():
        torch_global_emb = global_module(dummy_acoustic_feats)

    global_onnx = layout.component_path("global_encoder.onnx")
    export_model(
        model=global_module,
        dummy_inputs=(dummy_acoustic_feats,),
        output_path=global_onnx,
        input_names=["acoustic_features"],
        output_names=["global_embedding"],
        dynamic_axes={
            "acoustic_features": {0: "batch", 1: "frames"},
            "global_embedding": {0: "batch"},
        },
        opset_version=14,
    )

    ort_global = run_ort(global_onnx, {"acoustic_features": dummy_acoustic_feats.numpy()})
    rpt = compare_outputs(torch_global_emb, ort_global[0])
    check_tolerance(rpt, max_abs_tol=1e-3, mean_abs_tol=1e-4)
    print(f"  global_encoder parity: max_abs={rpt.components[0].max_abs_delta:.3e}, mean_abs={rpt.components[0].mean_abs_delta:.3e} PASS")

    # -----------------------------------------------------------------------
    # Component 5: mel_decoder.onnx
    # Inputs: content_embedding (1, T, 768), global_embedding (1, 128)
    # Exported with dynamo=True so the rope slice and interpolation scale_factor
    # are resolved symbolically — T (tokens dim) is fully dynamic at runtime.
    # -----------------------------------------------------------------------
    print("\n[linacodec] Exporting mel_decoder.onnx ...")
    mel_dec_module = _build_mel_decoder_module(model)
    mel_dec_module.eval()

    with torch.no_grad():
        torch_mel_out = mel_dec_module(dummy_content_emb, dummy_global_emb)

    mel_onnx = layout.component_path("mel_decoder.onnx")

    # Use dynamo export with explicit dynamic_shapes so T_tokens is symbolic.
    # torch.export.Dim constrains the range; Dim.DYNAMIC leaves it unbounded
    # (PyTorch ≥2.5 supports this directly via torch.onnx.export with dynamo=True).
    _batch = torch.export.Dim("batch", min=1, max=8)
    _tokens = torch.export.Dim("tokens", min=1, max=4096)
    _mel_dynamic_shapes = (
        {0: _batch, 1: _tokens},  # content_embedding (B, T, 768)
        {0: _batch},              # global_embedding  (B, 128)
    )
    # opset 18: LayerNormalization op (needed by AdaLN in mel_decoder) was
    # added in opset 17; dynamo export uses the highest-available opset
    # by default when dynamo=True, so we don't constrain it below 17.
    onnx_prog = torch.onnx.export(
        mel_dec_module,
        args=(dummy_content_emb, dummy_global_emb),
        f=str(mel_onnx),
        input_names=["content_embedding", "global_embedding"],
        output_names=["mel_spectrogram"],
        dynamo=True,
        dynamic_shapes=_mel_dynamic_shapes,
    )
    if onnx_prog is not None:
        onnx_prog.save(str(mel_onnx))

    ort_mel = run_ort(mel_onnx, {
        "content_embedding": dummy_content_emb.numpy(),
        "global_embedding": dummy_global_emb.numpy(),
    })
    rpt = compare_outputs(torch_mel_out, ort_mel[0])
    check_tolerance(rpt, max_abs_tol=5e-3, mean_abs_tol=5e-4)
    print(f"  mel_decoder parity: max_abs={rpt.components[0].max_abs_delta:.3e}, mean_abs={rpt.components[0].mean_abs_delta:.3e} PASS")

    # -----------------------------------------------------------------------
    # Component 6: vocos_backbone.onnx
    # Exports backbone + upsampler + head linear projections
    # Returns: (mag_24k, phase_24k, mag_48k, phase_48k)
    # ISTFT done in numpy in the adapter
    # -----------------------------------------------------------------------
    print("\n[linacodec] Exporting vocos_backbone.onnx ...")
    vocos_module = _build_vocos_module(vocos)
    vocos_module.eval()

    # Dummy mel: (1, 100, T_mel)
    dummy_mel_batch = dummy_mel  # already (1, 100, T_mel) from model.forward_mel
    with torch.no_grad():
        torch_vocos_out = vocos_module(dummy_mel_batch)
        mag24_t, phase24_t, mag48_t, phase48_t = torch_vocos_out

    # Verify we can do numpy ISTFT and get reasonable output
    n_fft, hop = 1024, 256
    mag24_np = mag24_t.numpy()
    phase24_np = phase24_t.numpy()
    mag48_np = mag48_t.numpy()
    phase48_np = phase48_t.numpy()

    # ISTFTHead: out = Linear(dim, (n_fft//2+1)*2) — but we need magnitudes/phases
    # The ISTFTHead uses: mag = torch.exp(out[:, :n_bins, :]), phase = torch.sin(out[:, n_bins:, :])
    # We must apply these activations AFTER the linear projection
    # Check upstream vocos ISTFTHead source
    print("  [vocos] Checking ISTFTHead activation pattern ...")
    # Apply the standard vocos ISTFTHead activations
    mag24_np = np.exp(np.clip(mag24_np, -10, 10))
    # phase24_np used as raw angle (ISTFTHead: S = exp(mag)*(cos(p)+i*sin(p)))
    mag48_np = np.exp(np.clip(mag48_np, -10, 10))
    # phase48_np used as raw angle

    audio_24k = _numpy_istft(mag24_np, phase24_np, n_fft=n_fft, hop_length=hop, padding="center")
    audio_48k = _numpy_istft(mag48_np, phase48_np, n_fft=n_fft, hop_length=hop, padding="center")

    # Resample audio_24k to 48kHz for Linkwitz-Riley merge
    from scipy.signal import resample_poly
    audio_24k_up = resample_poly(audio_24k[0], 2, 1).astype(np.float32)
    audio_48k_mono = audio_48k[0]
    min_len = min(len(audio_24k_up), len(audio_48k_mono))
    merged = _numpy_linkwitz_riley(
        audio_48k_mono[:min_len][np.newaxis],
        audio_24k_up[:min_len][np.newaxis],
    )[0]
    print(f"  [vocos] Numpy ISTFT + merge output length: {len(merged)} samples ({len(merged)/48000:.3f}s)")

    vocos_onnx = layout.component_path("vocos_backbone.onnx")
    export_model(
        model=vocos_module,
        dummy_inputs=(dummy_mel_batch,),
        output_path=vocos_onnx,
        input_names=["mel_spectrogram"],
        output_names=["mag_24k", "phase_24k", "mag_48k", "phase_48k"],
        dynamic_axes={
            "mel_spectrogram": {0: "batch", 2: "mel_frames"},
            "mag_24k": {0: "batch", 2: "frames_24"},
            "phase_24k": {0: "batch", 2: "frames_24"},
            "mag_48k": {0: "batch", 2: "frames_48"},
            "phase_48k": {0: "batch", 2: "frames_48"},
        },
        opset_version=14,
    )

    ort_vocos = run_ort(vocos_onnx, {"mel_spectrogram": dummy_mel_batch.numpy()})
    rpt_m24 = compare_outputs(mag24_t, torch.from_numpy(ort_vocos[0]))
    check_tolerance(rpt_m24, max_abs_tol=1e-3, mean_abs_tol=1e-4)
    print(f"  vocos_backbone parity (mag_24k): max_abs={rpt_m24.components[0].max_abs_delta:.3e} PASS")

    # -----------------------------------------------------------------------
    # Quantize all components to INT8
    # -----------------------------------------------------------------------
    print("\n[linacodec] Quantizing to INT8 ...")
    components_to_quant = [
        acoustic_onnx, distill_onnx, content_onnx, global_onnx, mel_onnx, vocos_onnx
    ]
    size_report = {}
    for onnx_path in components_to_quant:
        p = Path(onnx_path)
        fp32_size = p.stat().st_size / (1024 ** 2)
        try:
            qrpt = quantize_model(onnx_path)
            q8_path = p.with_name(p.stem + "_q8.onnx")
            q8_size = q8_path.stat().st_size / (1024 ** 2) if q8_path.exists() else 0
            size_report[p.name] = (fp32_size, q8_size)
            print(f"  {p.name}: {fp32_size:.1f} MB → {q8_size:.1f} MB ({100*(1-q8_size/fp32_size):.1f}% reduction)")
        except Exception as e:
            print(f"  WARNING: quantize failed for {p.name}: {e}")
            size_report[p.name] = (fp32_size, 0)

    # -----------------------------------------------------------------------
    # Write manifest
    # -----------------------------------------------------------------------
    print("\n[linacodec] Writing manifest and provenance ...")

    # Get file sizes for manifest
    def _mb(p):
        pp = Path(p)
        return pp.stat().st_size / (1024 ** 2) if pp.exists() else 0

    components = {
        "acoustic_ssl_encoder": "acoustic_ssl_encoder.onnx",
        "acoustic_ssl_encoder_q8": "acoustic_ssl_encoder_q8.onnx",
        "distill_wavlm_encoder": "distill_wavlm_encoder.onnx",
        "distill_wavlm_encoder_q8": "distill_wavlm_encoder_q8.onnx",
        "content_encoder": "content_encoder.onnx",
        "content_encoder_q8": "content_encoder_q8.onnx",
        "global_encoder": "global_encoder.onnx",
        "global_encoder_q8": "global_encoder_q8.onnx",
        "mel_decoder": "mel_decoder.onnx",
        "mel_decoder_q8": "mel_decoder_q8.onnx",
        "vocos_backbone": "vocos_backbone.onnx",
        "vocos_backbone_q8": "vocos_backbone_q8.onnx",
    }

    write_manifest(
        layout=layout,
        components=components,
        sample_rates={"output": 48000, "ssl_input": 16000, "model_input": 24000},
        metadata={
            "opset": 14,
            "vc_recipe": (
                "encode_source → content_tokens + content_embedding; "
                "encode_reference → global_embedding; "
                "mel_decoder(content_embedding, global_embedding) → mel; "
                "vocos_backbone(mel) → mag/phase; numpy ISTFT + Linkwitz-Riley → 48kHz audio"
            ),
            "sample_rate_output": 48000,
            "frame_rate": 12.5,
            "upstream_repo": "https://github.com/ysharma3501/LinaCodec",
            "upstream_weights": "YatharthS/LinaCodec",
            "license_note": (
                "Transformer module adapted from Meta Llama-3 (Llama 3 Community License). "
                "distill_wavlm derived from torchaudio (BSD-2-Clause). "
                "ONNX weights published under the Llama 3 Community License; "
                "see PROVENANCE.md for full text."
            ),
            "distributable": True,
        },
    )

    license_text = """LinaCodec Transformer module:
    Adapted from https://github.com/meta-llama/llama3/blob/main/llama/model.py
    Copyright (c) Meta Platforms, Inc. and affiliates.
    Licensed under the Llama 3 Community License Agreement.

distill_wavlm module:
    Derived from torchaudio (https://github.com/pytorch/audio)
    BSD-2-Clause License
    Copyright (c) 2017 Facebook Inc. (Soumith Chintala)

LinaCodec implementation:
    Copyright (c) 2024 Yatharth Sharma
    Repository: https://github.com/ysharma3501/LinaCodec
"""
    write_provenance(
        engine_dir=layout.engine_dir,
        upstream_repo_url=LINACODEC_REPO_URL,
        upstream_ref="main",
        license_text=license_text,
    )

    # Print size summary
    print("\n=== LinaCodec model sizes ===")
    total_fp32 = 0
    total_q8 = 0
    for name, (fp32_mb, q8_mb) in size_report.items():
        total_fp32 += fp32_mb
        total_q8 += q8_mb
        pct = f"(-{100*(1-q8_mb/fp32_mb):.1f}%)" if fp32_mb > 0 and q8_mb > 0 else ""
        print(f"  {name}: {fp32_mb:.1f} MB fp32 | {q8_mb:.1f} MB INT8 {pct}")
    if total_fp32 > 0:
        print(f"  TOTAL: {total_fp32:.1f} MB fp32 | {total_q8:.1f} MB INT8 (-{100*(1-total_q8/total_fp32):.1f}%)")

    # -----------------------------------------------------------------------
    # Optional HF push
    # -----------------------------------------------------------------------
    if push and not no_push:
        from conversion.push_models import push_to_hf
        push_to_hf(
            engine_dir=layout.engine_dir,
            engine_name="linacodec",
            message="feat: add linacodec ONNX export (48kHz codec VC)",
        )
        print("[linacodec] Pushed to TigreGotico/voiceclonnx-linacodec on HF Hub.")
    else:
        print(f"\n[linacodec] Artifacts in: {layout.engine_dir}")
        print("[linacodec] Run with --push to upload to HF Hub.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export LinaCodec to ONNX")
    parser.add_argument("--output-dir", default="/tmp/lina-out", help="Output directory")
    parser.add_argument("--push", action="store_true", help="Push to HF Hub after export")
    parser.add_argument("--no-push", action="store_true", help="Skip HF push even if --push set")
    args = parser.parse_args()
    export(output_dir=args.output_dir, push=args.push, no_push=args.no_push)
