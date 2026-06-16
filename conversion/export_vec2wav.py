"""Export vec2wav 2.0 components to ONNX.

vec2wav 2.0 (Guo et al., Interspeech 2024) is a VC-native neural vocoder.

Pipeline:
  1. **vq-wav2vec content encoder** — 16 kHz audio → discrete VQ indices
     (groups=2, vocab=320 each) → codebook lookup → (1, L, 512) float vectors.
  2. **WavLM-Large speaker encoder** (layer 6) — target audio → (1, T, 1024)
     continuous features → temporal mean → (1, 1024) speaker embedding.
  3. **CTXVEC2WAV frontend** (Conformer cross-attention decoder) —
     (1, L, 512) content + (1, T, 1024) prompt → (1, L, 184) hidden states.
  4. **BigVGAN vocoder** (conditioned snakebeta) —
     (1, 184, L) hidden + (1, 1024) cond → (1, 1, N) waveform at 24 kHz.

Components exported to ONNX:
  - ``vqwav2vec_encoder.onnx``   — CNN feature extractor (before VQ) → (1, T, 512) continuous features
    NOTE: VQ discretization is pure numpy at inference (codebook look-up).
  - ``wavlm_speaker.onnx``       — WavLM-Large layer-6 for speaker prompt
  - ``vec2wav_frontend.onnx``    — CTXVEC2WAVFrontend (Conformer)
  - ``vec2wav_vocoder.onnx``     — BigVGAN vocoder (no weight-norm)

The codebook is saved as ``vqwav2vec_codebook.npy`` (shape [2, 320, 256]).

License note: code Apache-2.0; HF weights (cantabile-kwok/vec2wav2.0) are
**GPL-3.0**. Derivative ONNX artifacts inherit GPL-3.0 — see PROVENANCE.md.

Upstream:
  - https://github.com/cantabile-kwok/vec2wav2.0  (Apache-2.0 code)
  - https://huggingface.co/cantabile-kwok/vec2wav2.0  (GPL-3.0 weights)
  - https://arxiv.org/abs/2409.01995

Usage::

    python -m conversion.export_vec2wav --output-dir /tmp/vec2wav-out --no-push
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

VEC2WAV_HF_REPO = "cantabile-kwok/vec2wav2.0"
VEC2WAV_UPSTREAM_URL = "https://github.com/cantabile-kwok/vec2wav2.0"
VEC2WAV_UPSTREAM_REF = "main"

# vq-wav2vec (fairseq kmeans checkpoint) — public Facebook AI model
VQWAV2VEC_URL = (
    "https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec_kmeans.pt"
)

# WavLM-Large via HuggingFace (Microsoft/microsoft/wavlm-large) — MIT
WAVLM_HF_REPO = "microsoft/wavlm-large"

VEC2WAV_SR = 24000   # output sample rate

# The vec2wav2.0 model uses VQ-wav2vec as the content token extractor.
# vq-wav2vec has 2 groups × 320 codewords × 256 dimensions → concat → 512 dim.
VQWAV2VEC_GROUPS = 2
VQWAV2VEC_VOCAB = 320
VQWAV2VEC_DIM_PER_GROUP = 256  # 512 / 2
VQWAV2VEC_TOTAL_DIM = VQWAV2VEC_GROUPS * VQWAV2VEC_DIM_PER_GROUP  # 512

GPL3_LICENSE = """\
                    GNU GENERAL PUBLIC LICENSE
                       Version 3, 29 June 2007

Weights source: https://huggingface.co/cantabile-kwok/vec2wav2.0
License declared: GPL-3.0

These ONNX artifacts were converted from the cantabile-kwok/vec2wav2.0
pretrained weights, which are released under the GNU General Public License v3.
Redistribution and use of these artifacts is subject to the GPL-3.0 terms.
See: https://www.gnu.org/licenses/gpl-3.0.html

Code of the vec2wav 2.0 system (GitHub: cantabile-kwok/vec2wav2.0) is
Apache-2.0. The weight files (HF: cantabile-kwok/vec2wav2.0) are GPL-3.0.

Copyright 2024 Yiwei Guo (SJTU X-LANCE Lab).
"""

# vec2wav config matching the released generator.ckpt
VEC2WAV_CONFIG = {
    "sampling_rate": 24000,
    "num_mels": 80,
    "hop_size": 240,
    "win_length": 697,
    "dropout_features": 0.0,
    "prompt_fold_by_2": True,
    "prompt_net_type": "ConvPromptPrenet",
    "frontend_params": {
        "vqvec_channels": 512,
        "prompt_channels": 1024,
        "conformer_params": {
            "attention_dim": 184,
            "attention_heads": 2,
            "linear_units": 1536,
            "num_blocks": 2,
            "dropout_rate": 0.2,
            "positional_dropout_rate": 0.2,
            "attention_dropout_rate": 0.2,
            "normalize_before": True,
            "concat_after": False,
            "positionwise_layer_type": "conv1d",
            "positionwise_conv_kernel_size": 3,
            "macaron_style": True,
            "pos_enc_layer_type": "rel_pos",
            "selfattention_layer_type": "rel_selfattn",
            "activation_type": "swish",
            "use_cnn_module": True,
            "cnn_module_kernel": 31,
        },
    },
    "generator_type": "BigVGAN",
    "generator_params": {
        "in_channels": 184,
        "out_channels": 1,
        "channels": 512,
        "kernel_size": 7,
        "upsample_scales": [8, 5, 3, 2],
        "upsample_kernel_sizes": [16, 10, 6, 4],
        "resblock": "1",
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilations": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "use_additional_convs": True,
        "bias": True,
        "nonlinear_activation": "snakebeta-condition",
        "snake_logscale": True,
        "condition_dim": 1024,
        "use_weight_norm": True,
    },
}


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download(url: str, dest: Path, desc: str) -> Path:
    if dest.exists():
        print(f"[export] {desc} already cached at {dest}")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"[export] Downloading {desc} from {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"[export] Downloaded {dest.stat().st_size / 1024**2:.1f} MB")
    return dest


# ---------------------------------------------------------------------------
# Model wrappers for ONNX export
# ---------------------------------------------------------------------------


def _load_vqwav2vec_checkpoint_standalone(ckpt_path: str):
    """Load vq-wav2vec checkpoint without requiring fairseq to be importable.

    Uses a meta-path finder to stub all fairseq submodules so that torch.load
    can unpickle the checkpoint (which contains fairseq Namespace objects) without
    triggering fairseq's hydra initialisation (broken on Python ≥3.11).

    Returns (model_state_dict, args_namespace).
    """
    import sys
    import types
    import argparse
    import torch

    # Stub every fairseq submodule so pickle can find the classes
    class _AutoStubModule(types.ModuleType):
        def __getattr__(self, name):
            # Return a string for dunder-file so inspect.getsourcefile doesn't fail
            if name == "__file__":
                return "<fairseq-stub>"
            class _Stub:
                def __init__(self, *a, **kw): pass
                def __call__(self, *a, **kw): return self
                def __getattr__(self, n): return _Stub
            return _Stub

    class _FairseqFinder:
        def find_module(self, name, path=None):
            if name.startswith("fairseq"):
                return self
        def load_module(self, name):
            if name in sys.modules:
                return sys.modules[name]
            mod = _AutoStubModule(name)
            mod.__path__ = []
            mod.__package__ = name
            mod.__file__ = f"<fairseq-stub:{name}>"
            sys.modules[name] = mod
            return mod

    # Only inject the finder if fairseq is not already importable
    _finder = _FairseqFinder()
    if "fairseq" not in sys.modules:
        sys.meta_path.insert(0, _finder)

    try:
        torch.serialization.add_safe_globals([argparse.Namespace])
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    finally:
        if _finder in sys.meta_path:
            sys.meta_path.remove(_finder)
        # Remove stub modules from sys.modules so later imports (e.g. torch.onnx)
        # don't find _AutoStubModule instances when inspect iterates sys.modules.
        for key in [k for k in sys.modules if k.startswith("fairseq")]:
            del sys.modules[key]

    return ckpt["model"], ckpt["args"]


def _build_vqwav2vec_cnn_standalone(ckpt_path: str):
    """Build a standalone PyTorch CNN module from vq-wav2vec weights.

    Reconstructs the feature_extractor CNN directly from the checkpoint state dict
    (8 conv layers: (512,10,5), (512,8,4), (512,4,2)×3, (512,1,1)×3) with
    group-norm, without needing any fairseq code at export time.

    Returns (cnn_module, codebook_np) where:
    - cnn_module: nn.Module that takes (1, T) → (1, T', 512)
    - codebook_np: np.ndarray of shape (2, 320, 256) — the VQ codebook
    """
    import torch
    import torch.nn as nn
    import numpy as np

    model_state, args = _load_vqwav2vec_checkpoint_standalone(ckpt_path)

    # ----------------------------------------------------------------
    # Reconstruct the CNN feature extractor
    # The architecture is given by args.conv_feature_layers:
    # [(512, 10, 5), (512, 8, 4), (512, 4, 2), (512, 4, 2), (512, 4, 2),
    #  (512, 1, 1), (512, 1, 1), (512, 1, 1)]
    # Each layer is: Conv1d + GroupNorm (groups=1, affine) [fp32]
    # ----------------------------------------------------------------
    import ast
    conv_cfg = ast.literal_eval(args.conv_feature_layers)  # [(channels, kernel, stride), ...]

    class VQWav2VecCNN(nn.Module):
        """Standalone vq-wav2vec CNN feature extractor (fairseq-independent)."""

        def __init__(self, layers):
            super().__init__()
            self.conv_layers = nn.ModuleList(layers)

        def forward(self, input_values: torch.Tensor) -> torch.Tensor:
            """(1, T) → (1, T', 512)"""
            x = input_values.unsqueeze(1)  # (1, 1, T) → first layer needs mono input
            for layer in self.conv_layers:
                x = layer(x)
            return x.transpose(1, 2)  # (1, 512, T') → (1, T', 512)

    layers = []
    in_ch = 1
    for i, (out_ch, kernel, stride) in enumerate(conv_cfg):
        conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, bias=False)
        # GroupNorm with groups=1 (from fp32_group_norm=True in args)
        gn = nn.GroupNorm(1, out_ch, affine=True)
        # Correct fairseq order: Conv → GroupNorm → GELU (dropout=0 at inference)
        # Original ConvFeatureExtractionModel: block = Conv → Dropout → GN → activation
        # With dropout=0 at inference: Conv → GN → GELU
        layer = nn.Sequential(conv, gn, nn.GELU())
        layers.append(layer)
        in_ch = out_ch

    cnn = VQWav2VecCNN(layers)

    # Load weights from checkpoint
    # Keys in checkpoint: feature_extractor.conv_layers.{i}.0.weight (conv)
    #                     feature_extractor.conv_layers.{i}.2.weight (gn weight)
    #                     feature_extractor.conv_layers.{i}.2.bias   (gn bias)
    # Our Sequential layout: 0=Conv1d, 1=GroupNorm, 2=GELU (no params for GELU)
    # Original layout: 0=Conv1d (idx 0), 1=Dropout (no params), 2=GroupNorm (idx 2)
    cnn_state = {}
    for key, val in model_state.items():
        if key.startswith("feature_extractor.conv_layers."):
            rest = key[len("feature_extractor."):]  # conv_layers.{i}.0.weight
            parts = rest.split(".")
            idx = int(parts[1])
            sub_idx = int(parts[2])
            name = parts[3]  # "weight" or "bias"
            # Map: original 0 (Conv) → our 0; original 2 (GN) → our 1
            if sub_idx == 0:
                new_key = f"conv_layers.{idx}.0.{name}"
            elif sub_idx == 2:
                new_key = f"conv_layers.{idx}.1.{name}"
            else:
                continue
            cnn_state[new_key] = val

    missing, unexpected = cnn.load_state_dict(cnn_state, strict=False)
    if missing:
        print(f"[export] WARNING: missing CNN keys: {missing[:5]}")
    if unexpected:
        print(f"[export] WARNING: unexpected CNN keys: {unexpected[:5]}")

    cnn.eval()
    for p in cnn.parameters():
        p.requires_grad_(False)

    # ----------------------------------------------------------------
    # Extract VQ codebook
    # vector_quantizer.embedding: (320, 1, 256)
    # With combine_groups=True and vq_groups=2, the same codebook is used
    # for both groups. The codebook shape expected by the adapter is (2, 320, 256).
    # ----------------------------------------------------------------
    cb = model_state["vector_quantizer.embedding"]  # (320, 1, 256)
    # Expand to (2, 320, 256) — both groups use the same embeddings
    codebook_np = cb.squeeze(1).numpy()  # (320, 256)
    codebook_np = np.stack([codebook_np, codebook_np], axis=0)  # (2, 320, 256)

    return cnn, codebook_np, model_state


def _build_wavlm_layer6_wrapper():
    """WavLM-Large layer-6 hidden states using HuggingFace transformers API.

    Uses microsoft/wavlm-large from HF hub. Outputs hidden states from layer 6.
    This is the same extraction the vec2wav prompt extractor performs (output_layer=6).
    """
    import torch
    import torch.nn as nn
    from transformers import WavLMModel

    class WavLMLayer6(nn.Module):
        def __init__(self, wavlm: WavLMModel):
            super().__init__()
            self.wavlm = wavlm

        def forward(self, input_values: torch.Tensor) -> torch.Tensor:
            """(1, T) → (1, T', 1024) layer-6 hidden states."""
            out = self.wavlm(
                input_values=input_values,
                output_hidden_states=True,
            )
            # out.hidden_states is a tuple of (batch, frames, 1024), layers 0..24
            # layer 6 = index 6
            return out.hidden_states[6]  # (1, T', 1024)

    return WavLMLayer6


def _build_generator_wrapper(generator):
    """Wrap VEC2WAV2Generator for ONNX export as two separate components."""
    import torch
    import torch.nn as nn

    class FrontendWrapper(nn.Module):
        """CTXVEC2WAV frontend: vqvec + prompt → hidden states."""
        def __init__(self, frontend):
            super().__init__()
            self.frontend = frontend

        def forward(self, vqvec: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            vqvec  : (1, L, 512) float32 — content VQ-vec sequence
            prompt : (1, T, 1024) float32 — speaker WavLM features

            Returns
            -------
            (1, 184, L) float32 — hidden states for vocoder
            """
            h, _, _ = self.frontend(vqvec, prompt)  # (1, L, 184)
            return h.transpose(1, 2)  # (1, 184, L)

    class VocoderWrapper(nn.Module):
        """BigVGAN vocoder: hidden + cond → waveform."""
        def __init__(self, backend):
            super().__init__()
            self.backend = backend

        def forward(self, hidden: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            """
            Parameters
            ----------
            hidden : (1, 184, L) float32 — frontend output
            cond   : (1, 1024) float32 — mean speaker embedding

            Returns
            -------
            (1, 1, N) float32 — waveform at 24 kHz
            """
            return self.backend(hidden, cond)

    return FrontendWrapper(generator.frontend), VocoderWrapper(generator.backend)


# ---------------------------------------------------------------------------
# Main export function
# ---------------------------------------------------------------------------


def export_vec2wav(output_dir: str, cache_dir: Optional[str] = None) -> Path:
    """Export vec2wav 2.0 components to ONNX and return the engine output dir."""
    import torch
    import numpy as np
    import sys

    from conversion.export_base import OutputLayout, export_model, write_manifest, write_provenance
    from conversion.parity import compare_outputs, check_tolerance, run_ort
    from conversion.quantize import quantize_model

    output_dir = Path(output_dir)
    layout = OutputLayout.for_engine("vec2wav", base_dir=output_dir)
    layout.makedirs()

    _cache = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "voiceclonnx" / "vec2wav"
    _cache.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Download pretrained weights
    # ------------------------------------------------------------------
    vqw2v_ckpt = _download(VQWAV2VEC_URL, _cache / "vq-wav2vec_kmeans.pt", "vq-wav2vec kmeans")

    from huggingface_hub import hf_hub_download
    # Download vec2wav generator.ckpt from HF
    gen_ckpt_path = hf_hub_download(
        repo_id=VEC2WAV_HF_REPO,
        filename="generator.ckpt",
        cache_dir=str(_cache / "hf"),
    )
    print(f"[export] Generator checkpoint: {gen_ckpt_path}")

    # ------------------------------------------------------------------
    # 1. Export vq-wav2vec CNN encoder (pre-VQ features)
    # ------------------------------------------------------------------
    print("[export] Loading vq-wav2vec (standalone, no fairseq required) ...")
    cnn_wrapper, codebook_np, model_state = _build_vqwav2vec_cnn_standalone(str(vqw2v_ckpt))

    # Save codebook for numpy VQ at runtime (shape [G, V, D])
    codebook_path = layout.component_path("vqwav2vec_codebook.npy")
    np.save(str(codebook_path), codebook_np)
    print(f"[export] Codebook saved: {codebook_path}  shape={codebook_np.shape}")

    # Save the KmeansVectorQuantizer projection weights for numpy application at runtime.
    # The KmeansVectorQuantizer applies ``self.projection(x)`` (grouped Conv1d + GroupNorm)
    # BEFORE computing L2 distances to the codebook. Without this step the VQ indices are
    # computed on raw CNN features instead of projected features, producing completely wrong
    # token sequences (0% index agreement with upstream fairseq).
    projection_path = layout.component_path("vqwav2vec_projection.npz")
    proj_conv_weight = model_state["vector_quantizer.projection.0.weight"].numpy()  # (512, 256, 1)
    proj_gn_weight = model_state["vector_quantizer.projection.1.weight"].numpy()    # (512,)
    proj_gn_bias = model_state["vector_quantizer.projection.1.bias"].numpy()        # (512,)
    np.savez(
        str(projection_path),
        conv_weight=proj_conv_weight.astype(np.float32),
        gn_weight=proj_gn_weight.astype(np.float32),
        gn_bias=proj_gn_bias.astype(np.float32),
    )
    print(f"[export] Projection weights saved: {projection_path}")

    dummy_audio = torch.zeros(1, 16000)  # 1 second
    cnn_onnx = layout.component_path("vqwav2vec_encoder.onnx")
    print(f"[export] Exporting vq-wav2vec CNN encoder → {cnn_onnx} ...")
    export_model(
        model=cnn_wrapper,
        dummy_inputs=(dummy_audio,),
        output_path=cnn_onnx,
        input_names=["input_values"],
        output_names=["cnn_features"],
        dynamic_axes={
            "input_values": {0: "batch", 1: "time"},
            "cnn_features": {0: "batch", 1: "frames"},
        },
        opset_version=14,
    )
    sz = cnn_onnx.stat().st_size / 1024**2
    print(f"[export] vq-wav2vec CNN ONNX written: {sz:.1f} MB")

    # Parity check
    with torch.no_grad():
        torch_cnn_out = cnn_wrapper(dummy_audio).detach().cpu().numpy()
    ort_cnn_out = run_ort(cnn_onnx, {"input_values": dummy_audio.numpy()})
    cnn_report = compare_outputs(
        [torch_cnn_out], ort_cnn_out, names=["cnn_features"],
        max_abs_tol=1e-3, mean_abs_tol=1e-4,
    )
    print(f"[export] vq-wav2vec CNN parity: {cnn_report.summary()}")
    check_tolerance(cnn_report)
    cnn_report.save(layout.component_path("vqwav2vec_parity_report.json"))

    # Quantize CNN encoder
    cnn_q8_path = layout.component_path("vqwav2vec_encoder_q8.onnx")
    cnn_quant = quantize_model(cnn_onnx, output_path=cnn_q8_path)
    print(cnn_quant.summary())

    del cnn_wrapper

    # ------------------------------------------------------------------
    # 2. Export WavLM-Large layer-6 speaker encoder
    # ------------------------------------------------------------------
    print("[export] Loading WavLM-Large from HF (microsoft/wavlm-large) ...")
    from transformers import WavLMModel

    wavlm_model = WavLMModel.from_pretrained(WAVLM_HF_REPO)
    wavlm_model.eval()
    for p in wavlm_model.parameters():
        p.requires_grad_(False)

    WavLMLayer6Class = _build_wavlm_layer6_wrapper()
    wavlm_wrapper = WavLMLayer6Class(wavlm_model)
    wavlm_wrapper.eval()

    dummy_ref = torch.zeros(1, 24000)  # 1.5 seconds of 16 kHz = 24000 samples
    wavlm_onnx = layout.component_path("wavlm_speaker.onnx")
    print(f"[export] Exporting WavLM speaker encoder → {wavlm_onnx} ...")
    export_model(
        model=wavlm_wrapper,
        dummy_inputs=(dummy_ref,),
        output_path=wavlm_onnx,
        input_names=["input_values"],
        output_names=["hidden_states"],
        dynamic_axes={
            "input_values": {0: "batch", 1: "time"},
            "hidden_states": {0: "batch", 1: "frames"},
        },
        opset_version=14,
    )
    sz = wavlm_onnx.stat().st_size / 1024**2
    print(f"[export] WavLM ONNX written: {sz:.1f} MB")

    # Parity check
    with torch.no_grad():
        torch_wavlm_out = wavlm_wrapper(dummy_ref).detach().cpu().numpy()
    ort_wavlm_out = run_ort(wavlm_onnx, {"input_values": dummy_ref.numpy()})
    wavlm_report = compare_outputs(
        [torch_wavlm_out], ort_wavlm_out, names=["hidden_states"],
        max_abs_tol=1e-3, mean_abs_tol=1e-4,
    )
    print(f"[export] WavLM speaker parity: {wavlm_report.summary()}")
    check_tolerance(wavlm_report)
    wavlm_report.save(layout.component_path("wavlm_parity_report.json"))

    # Quantize WavLM
    wavlm_q8_path = layout.component_path("wavlm_speaker_q8.onnx")
    wavlm_quant = quantize_model(wavlm_onnx, output_path=wavlm_q8_path)
    print(wavlm_quant.summary())

    del wavlm_model, wavlm_wrapper

    # ------------------------------------------------------------------
    # 3. Load vec2wav generator and export frontend + vocoder
    # ------------------------------------------------------------------
    print("[export] Loading vec2wav generator ...")
    sys.path.insert(0, "/tmp/vec2wav2_upstream")
    import vec2wav2.models
    generator = vec2wav2.models.VEC2WAV2Generator(
        vec2wav2.models.CTXVEC2WAVFrontend(
            VEC2WAV_CONFIG["prompt_net_type"],
            VEC2WAV_CONFIG["num_mels"],
            **VEC2WAV_CONFIG["frontend_params"],
        ),
        vec2wav2.models.BigVGAN(**VEC2WAV_CONFIG["generator_params"]),
    )
    state = torch.load(gen_ckpt_path, map_location="cpu")
    generator.load_state_dict(state["model"]["generator"])
    generator.backend.remove_weight_norm()
    generator.eval()
    for p in generator.parameters():
        p.requires_grad_(False)

    frontend_wrapper, vocoder_wrapper = _build_generator_wrapper(generator)
    frontend_wrapper.eval()
    vocoder_wrapper.eval()

    # Dummy inputs: 50 frames content (1 second at 50 Hz), 60 frames prompt
    dummy_vqvec = torch.zeros(1, 50, 512)
    dummy_prompt = torch.zeros(1, 60, 1024)
    dummy_hidden = torch.zeros(1, 184, 50)
    dummy_cond = torch.zeros(1, 1024)

    # Export frontend
    frontend_onnx = layout.component_path("vec2wav_frontend.onnx")
    print(f"[export] Exporting CTXVEC2WAV frontend → {frontend_onnx} ...")
    export_model(
        model=frontend_wrapper,
        dummy_inputs=(dummy_vqvec, dummy_prompt),
        output_path=frontend_onnx,
        input_names=["vqvec", "prompt"],
        output_names=["hidden"],
        dynamic_axes={
            "vqvec": {0: "batch", 1: "content_len"},
            "prompt": {0: "batch", 1: "prompt_len"},
            "hidden": {0: "batch", 2: "content_len"},
        },
        opset_version=14,
    )
    sz = frontend_onnx.stat().st_size / 1024**2
    print(f"[export] Frontend ONNX written: {sz:.1f} MB")

    # Parity check frontend
    with torch.no_grad():
        torch_frontend_out = frontend_wrapper(dummy_vqvec, dummy_prompt).detach().cpu().numpy()
    ort_frontend_out = run_ort(frontend_onnx, {"vqvec": dummy_vqvec.numpy(), "prompt": dummy_prompt.numpy()})
    frontend_report = compare_outputs(
        [torch_frontend_out], ort_frontend_out, names=["hidden"],
        max_abs_tol=1e-3, mean_abs_tol=1e-4,
    )
    print(f"[export] Frontend parity: {frontend_report.summary()}")
    check_tolerance(frontend_report)
    frontend_report.save(layout.component_path("frontend_parity_report.json"))

    # Quantize frontend
    frontend_q8_path = layout.component_path("vec2wav_frontend_q8.onnx")
    frontend_quant = quantize_model(frontend_onnx, output_path=frontend_q8_path)
    print(frontend_quant.summary())

    # Export vocoder
    # BigVGAN uses alias_free_torch (Snake-Beta + sinc-resampling) which requires
    # the dynamo exporter path (torch.export.export) for reliable ONNX tracing.
    # The dynamo path emits external-data files (.onnx + .onnx.data); we merge
    # them into a single self-contained .onnx immediately after export.
    vocoder_onnx = layout.component_path("vec2wav_vocoder.onnx")
    _vocoder_onnx_tmp = vocoder_onnx.with_suffix(".onnx_tmp")
    print(f"[export] Exporting BigVGAN vocoder → {vocoder_onnx} ...")
    torch.onnx.export(
        vocoder_wrapper,
        (dummy_hidden, dummy_cond),
        str(_vocoder_onnx_tmp),
        input_names=["hidden", "cond"],
        output_names=["waveform"],
        dynamic_axes={
            "hidden": {0: "batch", 2: "frames"},
            "waveform": {0: "batch", 2: "samples"},
        },
    )
    # Inline external data back into a single .onnx file
    import onnx
    from onnx.external_data_helper import load_external_data_for_model
    _voc_model = onnx.load(str(_vocoder_onnx_tmp), load_external_data=False)
    load_external_data_for_model(_voc_model, str(_vocoder_onnx_tmp.parent))
    onnx.save(_voc_model, str(vocoder_onnx), save_as_external_data=False)
    # Clean up tmp files
    _vocoder_onnx_tmp.unlink(missing_ok=True)
    _data_file = Path(str(_vocoder_onnx_tmp) + ".data")
    if _data_file.exists():
        _data_file.unlink()
    del _voc_model
    sz = vocoder_onnx.stat().st_size / 1024**2
    print(f"[export] Vocoder ONNX written: {sz:.1f} MB")

    # Parity check vocoder
    with torch.no_grad():
        torch_vocoder_out = vocoder_wrapper(dummy_hidden, dummy_cond).detach().cpu().numpy()
    ort_vocoder_out = run_ort(vocoder_onnx, {"hidden": dummy_hidden.numpy(), "cond": dummy_cond.numpy()})
    vocoder_report = compare_outputs(
        [torch_vocoder_out], ort_vocoder_out, names=["waveform"],
        max_abs_tol=1e-3, mean_abs_tol=1e-4,
    )
    print(f"[export] Vocoder parity: {vocoder_report.summary()}")
    check_tolerance(vocoder_report)
    vocoder_report.save(layout.component_path("vocoder_parity_report.json"))

    # Quantize vocoder
    # The dynamo-exported BigVGAN ONNX has shape annotations that confuse
    # onnxruntime's shape-inference pass; skip INT8 quantization if it fails.
    vocoder_q8_path = layout.component_path("vec2wav_vocoder_q8.onnx")
    try:
        vocoder_quant = quantize_model(vocoder_onnx, output_path=vocoder_q8_path)
        print(vocoder_quant.summary())
    except Exception as _e:
        print(f"[export] Vocoder INT8 quantization skipped ({type(_e).__name__}: {_e})")

    del generator, frontend_wrapper, vocoder_wrapper

    # ------------------------------------------------------------------
    # 4. Manifest + provenance
    # ------------------------------------------------------------------
    write_manifest(
        layout=layout,
        components={
            "vqwav2vec_encoder": "vqwav2vec_encoder.onnx",
            "vqwav2vec_encoder_q8": "vqwav2vec_encoder_q8.onnx",
            "vqwav2vec_codebook": "vqwav2vec_codebook.npy",
            "vqwav2vec_projection": "vqwav2vec_projection.npz",
            "wavlm_speaker": "wavlm_speaker.onnx",
            "wavlm_speaker_q8": "wavlm_speaker_q8.onnx",
            "vec2wav_frontend": "vec2wav_frontend.onnx",
            "vec2wav_frontend_q8": "vec2wav_frontend_q8.onnx",
            "vec2wav_vocoder": "vec2wav_vocoder.onnx",
            "vec2wav_vocoder_q8": "vec2wav_vocoder_q8.onnx",
        },
        sample_rates={"input": 16000, "output": VEC2WAV_SR},
        metadata={
            "opset": 14,
            "vqwav2vec_groups": VQWAV2VEC_GROUPS,
            "vqwav2vec_vocab_per_group": VQWAV2VEC_VOCAB,
            "vqwav2vec_dim_per_group": VQWAV2VEC_DIM_PER_GROUP,
            "wavlm_layer": 6,
            "conformer_attention_dim": 184,
            "bigvgan_upsample_product": 240,
            "weights_license": "GPL-3.0",
            "code_license": "Apache-2.0",
        },
    )
    write_provenance(
        engine_dir=layout.engine_dir,
        upstream_repo_url=VEC2WAV_UPSTREAM_URL,
        upstream_ref=VEC2WAV_UPSTREAM_REF,
        license_text=GPL3_LICENSE,
        extra={
            "weights_hf_repo": VEC2WAV_HF_REPO,
            "weights_license": "GPL-3.0",
            "code_license": "Apache-2.0",
            "vqwav2vec_source": VQWAV2VEC_URL,
            "wavlm_source": f"https://huggingface.co/{WAVLM_HF_REPO}",
            "paper": "https://arxiv.org/abs/2409.01995",
        },
    )

    print(f"\n[export] vec2wav 2.0 export complete → {layout.engine_dir}")
    return layout.engine_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export vec2wav 2.0 (vq-wav2vec + WavLM + CTXVEC2WAV + BigVGAN) to ONNX."
    )
    p.add_argument("--output-dir", default="/tmp/vec2wav-out", help="Staging output directory.")
    p.add_argument("--cache-dir", default=None, help="Cache directory for downloaded checkpoints.")
    p.add_argument("--no-push", action="store_true", help="Skip HF upload.")
    p.add_argument("--dry-run-push", action="store_true", help="Dry-run HF upload.")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    engine_dir = export_vec2wav(output_dir=args.output_dir, cache_dir=args.cache_dir)

    if not args.no_push:
        from conversion.push_models import push_engine
        push_engine(
            engine_dir=engine_dir,
            engine_name="vec2wav",
            dry_run=args.dry_run_push,
            commit_message="export: add vec2wav 2.0 ONNX artifacts (vq-wav2vec + WavLM + CTXVEC2WAV + BigVGAN)",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
