"""Export Mimi (Kyutai) encoder + decoder to ONNX.

Mimi is the neural audio codec powering Moshi (Kyutai, 2024).  It uses 32
residual-vector-quantizer streams at 12.5 Hz / 24 kHz.  Stream 0 is
semantically distilled from WavLM and carries phonetic content; streams 1–31
carry acoustic/timbre residuals.

Voice-conversion recipe
-----------------------
Encode source and reference with the encoder ONNX.  Keep stream 0 of the
source, take streams 1–31 from the reference; decode the combined codes.  All
stream-swap logic is pure numpy — no ONNX node involved.

Export quirks (transformers 5.5.0)
-----------------------------------
1. ``create_sliding_window_causal_mask`` passes a scalar Tensor as ``q_length``
   to ``sdpa_mask``, which then tries ``q_length.shape[0]`` (IndexError).
   Patch: extract the scalar value when ``q_length`` is a 0-d Tensor.
2. ``find_packed_sequence_indices`` calls ``torch.diff(prepend=…)`` which is
   not supported by the legacy TorchScript ONNX exporter.  Patch: return
   ``None`` (single-sequence — correct for inference).
3. ``MimiEuclideanCodebook.quantize`` calls ``torch.cdist`` with dynamic shapes
   which the legacy exporter cannot lower.  Patch: replace with manual
   ``||h||^2 + ||e||^2 - 2·h·eᵀ`` expansion (numerically equivalent).
4. ``MimiAttention.forward`` is replaced with full bidirectional (no-mask)
   attention.  The upstream code builds a sliding-window causal mask of shape
   ``(B, 1, T, T)`` which the TorchScript tracer bakes as a constant —
   causing a broadcast error for any input with a different T.  Full attention
   is correct for offline (non-streaming) VC where the whole sequence is
   available.  This makes the ONNX graph fully dynamic.
5. Transformer is run with ``use_cache=False`` to disable streaming KV caches
   (incompatible with static tracing).
6. Opset 14 throughout; ``dynamo=False`` forces the TorchScript path.

Usage
-----
::

    pip install "vconnx[convert]"
    python -m conversion.export_mimi --output-dir /tmp/mimi-out [--no-push]

Requires: ``torch``, ``onnx``, ``onnxruntime``, ``transformers>=4.48``,
``huggingface_hub``.  Never imported at vconnx runtime.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Patch transformers BEFORE importing MimiModel to fix tracing issues
# ---------------------------------------------------------------------------

import torch
import transformers
import transformers.masking_utils as _mu
import transformers.models.mimi.modeling_mimi as _mimi_mod

_original_sdpa_mask = _mu.sdpa_mask


def _patched_sdpa_mask(batch_size, q_length, kv_length, q_offset=0, **kwargs):
    """Fix: scalar Tensor q_length → int (happens during TorchScript tracing)."""
    if isinstance(q_length, torch.Tensor) and q_length.ndim == 0:
        q_length = int(q_length.item())
    return _original_sdpa_mask(batch_size, q_length, kv_length, q_offset, **kwargs)


_mu.sdpa_mask = _patched_sdpa_mask
transformers.masking_utils.sdpa_mask = _patched_sdpa_mask

# Disable packed-sequence detection (uses torch.diff which ONNX can't lower)
_mu.find_packed_sequence_indices = lambda position_ids: None
transformers.masking_utils.find_packed_sequence_indices = lambda position_ids: None


class _PatchedMimiEuclideanCodebook(_mimi_mod.MimiEuclideanCodebook):
    """Replace torch.cdist (dynamic-shape cdist not traceable) with manual L2."""

    def quantize(self, hidden_states: torch.Tensor) -> torch.Tensor:  # noqa: D401
        hs = hidden_states.float()      # (N, D)
        embed = self.embed.float()      # (C, D)
        hs_sq = (hs ** 2).sum(dim=-1, keepdim=True)         # (N, 1)
        embed_sq = (embed ** 2).sum(dim=-1, keepdim=True).T  # (1, C)
        dists = hs_sq + embed_sq - 2.0 * (hs @ embed.T)     # (N, C)
        return dists.argmin(dim=-1)


_mimi_mod.MimiEuclideanCodebook = _PatchedMimiEuclideanCodebook


class _NoMaskMimiAttention(_mimi_mod.MimiAttention):
    """Full bidirectional attention — no sliding-window causal mask.

    The upstream MimiTransformerModel creates a sliding-window causal mask
    (shape (B, 1, T, T)) during each forward pass.  When traced by the
    TorchScript ONNX exporter the mask is baked as a constant of size T from
    the dummy input, causing broadcast errors on any other input length.

    For offline (non-streaming) voice conversion the full bidirectional
    attention is equivalent and correct — the full sequence is available at
    encode/decode time.  Removing the mask makes the ONNX model fully dynamic.
    """

    def forward(  # noqa: D401
        self,
        hidden_states: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position=None,
        position_embeddings=None,
        **kwargs,
    ):
        import torch.nn.functional as F

        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = self.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if self.num_key_value_groups > 1:
            key_states = key_states.repeat_interleave(self.num_key_value_groups, dim=1)
            value_states = value_states.repeat_interleave(self.num_key_value_groups, dim=1)

        scale = self.head_dim ** -0.5
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        return attn_output, None   # (attn_output, past_key_values)


from transformers import MimiModel  # noqa: E402 — must import AFTER patches

# ---------------------------------------------------------------------------
# vconnx conversion helpers
# ---------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from conversion.export_base import OutputLayout, export_model, write_manifest, write_provenance  # noqa: E402

_UPSTREAM_REPO = "kyutai/mimi"
_UPSTREAM_REF = "main"
_CC_BY_40 = (
    "Creative Commons Attribution 4.0 International (CC BY 4.0)\n"
    "https://creativecommons.org/licenses/by/4.0/\n\n"
    "This work is licensed under the Creative Commons Attribution 4.0 International\n"
    "License. You are free to share and adapt the material for any purpose, even\n"
    "commercially, under the following terms:\n"
    "  Attribution — You must give appropriate credit to Kyutai and provide a link\n"
    "  to the license. https://huggingface.co/kyutai/mimi\n"
)

# ---------------------------------------------------------------------------
# Wrapper modules
# ---------------------------------------------------------------------------


class _MimiEncoderWrapper(torch.nn.Module):
    """audio (B,1,samples) → codes (B,32,frames) int64."""

    def __init__(self, model: MimiModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        embeddings = self.model.encoder(input_values)
        enc_out = self.model.encoder_transformer(
            embeddings.transpose(1, 2),
            use_cache=False,
            return_dict=False,
        )
        embeddings = enc_out[0].transpose(1, 2)
        embeddings = self.model.downsample(embeddings)
        codes = self.model.quantizer.encode(embeddings)
        return codes.transpose(0, 1)  # (B, Q, T)


class _MimiDecoderWrapper(torch.nn.Module):
    """codes (B,32,frames) int64 → audio (B,1,samples) float32."""

    def __init__(self, model: MimiModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        embeddings = self.model.quantizer.decode(audio_codes)
        embeddings = self.model.upsample(embeddings)
        dec_out = self.model.decoder_transformer(
            embeddings.transpose(1, 2),
            use_cache=False,
            return_dict=False,
        )
        embeddings = dec_out[0].transpose(1, 2)
        return self.model.decoder(embeddings)


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def _patch_model(model: MimiModel) -> None:
    """Apply all ONNX-tracing patches to loaded model instances.

    - cdist → manual L2 for all codebook instances.
    - Sliding-window causal mask → full attention for all attention instances.
    """
    for _name, mod in model.named_modules():
        if type(mod).__name__ == "MimiEuclideanCodebook":
            type(mod).quantize = _PatchedMimiEuclideanCodebook.quantize
        if type(mod).__name__ == "MimiAttention":
            mod.__class__ = _NoMaskMimiAttention


def export(output_dir: str, no_push: bool = False) -> None:
    """Full export pipeline: load → export → parity → quantize → (push)."""
    layout = OutputLayout.for_engine("mimi", base_dir=output_dir)
    layout.makedirs()

    print("[mimi] Loading kyutai/mimi (eager attention)…")
    model = MimiModel.from_pretrained(_UPSTREAM_REPO, attn_implementation="eager").eval()
    _patch_model(model)

    sr = model.config.sampling_rate   # 24000
    num_q = model.config.num_quantizers  # 32
    dummy_audio = torch.randn(1, 1, sr)
    with torch.no_grad():
        dummy_codes = _MimiEncoderWrapper(model)(dummy_audio)

    # ---- Encoder ----
    print("[mimi] Exporting encoder…")
    enc_wrapper = _MimiEncoderWrapper(model)
    enc_path = export_model(
        model=enc_wrapper,
        dummy_inputs=(dummy_audio,),
        output_path=layout.component_path("mimi_encoder.onnx"),
        input_names=["input_values"],
        output_names=["audio_codes"],
        dynamic_axes={
            "input_values": {0: "batch", 2: "samples"},
            "audio_codes": {0: "batch", 2: "frames"},
        },
        opset_version=14,
    )
    print(f"[mimi] Encoder → {enc_path}")

    # ---- Decoder ----
    print("[mimi] Exporting decoder…")
    dec_wrapper = _MimiDecoderWrapper(model)
    dec_path = export_model(
        model=dec_wrapper,
        dummy_inputs=(dummy_codes,),
        output_path=layout.component_path("mimi_decoder.onnx"),
        input_names=["audio_codes"],
        output_names=["audio_values"],
        dynamic_axes={
            "audio_codes": {0: "batch", 2: "frames"},
            "audio_values": {0: "batch", 2: "samples"},
        },
        opset_version=14,
    )
    print(f"[mimi] Decoder → {dec_path}")

    # ---- Parity ----
    print("[mimi] Parity check…")
    import numpy as np
    import onnxruntime as ort
    from conversion.parity import compare_outputs, check_tolerance

    enc_sess = ort.InferenceSession(enc_path, providers=["CPUExecutionProvider"])
    dec_sess = ort.InferenceSession(dec_path, providers=["CPUExecutionProvider"])

    audio_np = dummy_audio.numpy()
    ort_codes = enc_sess.run(None, {"input_values": audio_np})[0]
    ort_audio = dec_sess.run(None, {"audio_codes": ort_codes})[0]

    with torch.no_grad():
        torch_codes = enc_wrapper(dummy_audio).numpy()
        torch_audio = dec_wrapper(dummy_codes).numpy()

    print("[mimi] Encoder exact match:", np.all(ort_codes == torch_codes))
    dec_report = compare_outputs(torch_audio, ort_audio)
    check_tolerance(dec_report)
    print(f"[mimi] Decoder parity: max_abs={dec_report['max_abs']:.2e} mean_abs={dec_report['mean_abs']:.2e} PASS")

    # ---- Quantize ----
    print("[mimi] Quantizing…")
    from conversion.quantize import quantize_model
    quantize_model(enc_path)
    quantize_model(dec_path)

    # ---- Manifest + provenance ----
    write_manifest(
        layout=layout,
        components={
            "encoder": "mimi_encoder.onnx",
            "encoder_q8": "mimi_encoder_q8.onnx",
            "decoder": "mimi_decoder.onnx",
            "decoder_q8": "mimi_decoder_q8.onnx",
        },
        sample_rates={"input": sr, "output": sr},
        metadata={
            "opset": 14,
            "num_quantizers": num_q,
            "num_semantic_quantizers": model.quantizer.num_semantic_quantizers,
            "num_acoustic_quantizers": model.quantizer.num_acoustic_quantizers,
            "frame_rate_hz": model.config.frame_rate,
            "vc_recipe": (
                "source stream 0 (semantic/content) + "
                "reference streams 1-31 (acoustic/timbre) → decode"
            ),
        },
    )

    write_provenance(
        engine_dir=layout.engine_dir,
        upstream_repo_url=f"https://huggingface.co/{_UPSTREAM_REPO}",
        upstream_ref=_UPSTREAM_REF,
        license_text=_CC_BY_40,
    )

    if not no_push:
        print("[mimi] Pushing to TigreGotico/vconnx-mimi…")
        from conversion.push_models import push
        push(str(layout.engine_dir), engine="mimi")

    print("[mimi] Export complete.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export Mimi (Kyutai) to ONNX for vconnx.")
    p.add_argument("--output-dir", default="/tmp/mimi-out", help="Staging directory.")
    p.add_argument("--no-push", action="store_true", help="Skip HF upload.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()
    export(args.output_dir, no_push=args.no_push)
