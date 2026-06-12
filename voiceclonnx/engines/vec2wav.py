"""vec2wav 2.0 adapter for voiceclonnx.

vec2wav 2.0 (Guo et al., Interspeech 2024) is a speech discrete-token vocoder
purpose-built for voice conversion.

Pipeline:
  1. **vq-wav2vec CNN encoder** — 16 kHz source audio → pre-VQ CNN features
     → numpy VQ discretisation (codebook lookup) → (1, L, 512) VQ-vectors.
  2. **WavLM-Large speaker encoder** (layer 6) — 16 kHz reference audio →
     (1, T, 1024) continuous features → temporal mean → (1, 1024) speaker
     embedding.
  3. **CTXVEC2WAV frontend** (Conformer cross-attention) — (1, L, 512) content
     + (1, T, 1024) prompt → (1, 184, L) hidden states.
  4. **BigVGAN vocoder** (conditioned Snake-Beta activation) — (1, 184, L)
     hidden + (1, 1024) speaker mean → (1, 1, N) waveform at 24 kHz.

VQ discretisation step (pure numpy, no ONNX):
  The vq-wav2vec model uses 2 codebook groups. After CNN encoding, each frame
  is quantised to the nearest codeword per group. The indices are then looked
  up in the saved codebook (``vqwav2vec_codebook.npy``, shape [2, 320, 256])
  and concatenated to produce 512-dim VQ-vectors.

License note:
  Code Apache-2.0 (github.com/cantabile-kwok/vec2wav2.0).
  Pretrained weights (huggingface.co/cantabile-kwok/vec2wav2.0) are **GPL-3.0**.
  ONNX artifacts in TigreGotico/voiceclonnx-vec2wav inherit this GPL-3.0 term.

ONNX artifacts: ``TigreGotico/voiceclonnx-vec2wav``.

Requires: ``pip install voiceclonnx``
  -> onnxruntime, numpy, soundfile, huggingface_hub

References
----------
- https://github.com/cantabile-kwok/vec2wav2.0
- https://arxiv.org/abs/2409.01995
- https://huggingface.co/cantabile-kwok/vec2wav2.0
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from voiceclonnx.engines.base import EngineEntry, VoiceClonerBase, register_engine

# vec2wav 2.0 outputs 24 kHz audio
_VEC2WAV_SR = 24000

# Input sample rate for all encoders
_INPUT_SR = 16000

# HF repo housing the exported ONNX artifacts
_HF_REPO_ID = "TigreGotico/voiceclonnx-vec2wav"

# File names within the HF repo
_CNN_FP32 = "vqwav2vec_encoder.onnx"
_CNN_INT8 = "vqwav2vec_encoder_q8.onnx"
_CODEBOOK = "vqwav2vec_codebook.npy"
_PROJECTION = "vqwav2vec_projection.npz"
_WAVLM_FP32 = "wavlm_speaker.onnx"
_WAVLM_INT8 = "wavlm_speaker_q8.onnx"
_FRONTEND_FP32 = "vec2wav_frontend.onnx"
_FRONTEND_INT8 = "vec2wav_frontend_q8.onnx"
_VOCODER_FP32 = "vec2wav_vocoder.onnx"
_VOCODER_INT8 = "vec2wav_vocoder_q8.onnx"

# vq-wav2vec codebook dimensions
_VQWAV2VEC_GROUPS = 2
_VQWAV2VEC_VOCAB = 320
_VQWAV2VEC_DIM_PER_GROUP = 256


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------


def _load_wav(path: str, target_sr: int = _INPUT_SR) -> np.ndarray:
    """Load a WAV file, resample to *target_sr*, return float32 mono array."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        n_out = int(len(audio) * target_sr / sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n_out),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)

    return audio


def _save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    """Write *audio* (float32) as 16-bit PCM WAV to *path*."""
    import soundfile as sf

    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    sf.write(str(path), audio_int16, sr, subtype="PCM_16")


# ---------------------------------------------------------------------------
# Pure-numpy VQ discretisation and codebook lookup
# ---------------------------------------------------------------------------


def _grouped_conv1d(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Grouped Conv1d with kernel_size=1 and no bias.

    Implements ``nn.Conv1d(C, C, kernel_size=1, groups=G, bias=False)`` in numpy.

    Parameters
    ----------
    x      : (L, C) float32 — input features.
    weight : (C, C//G, 1) float32 — Conv1d weight; out_ch=C, in_ch=C//G, k=1.

    Returns
    -------
    (L, C) float32
    """
    C_out, C_in_per_group, _ = weight.shape  # (512, 256, 1)
    G = C_out // C_in_per_group  # = 2
    L = x.shape[0]
    out = np.empty((L, C_out), dtype=np.float32)
    for g in range(G):
        x_g = x[:, g * C_in_per_group: (g + 1) * C_in_per_group]   # (L, 256)
        w_g = weight[g * C_in_per_group: (g + 1) * C_in_per_group, :, 0]  # (256, 256)
        # Grouped conv1d k=1: y = x @ w^T (each output channel is a dot product)
        # weight[g*256:(g+1)*256] has shape (256, 256, 1) — 256 output channels, 256 input
        # Reindex: out_ch_in_group=256, in_ch=256 → w_g: (256, 256)
        w_g = weight[g * (C_out // G): (g + 1) * (C_out // G), :, 0]  # (256, 256)
        out[:, g * (C_out // G): (g + 1) * (C_out // G)] = x_g @ w_g.T  # (L, 256)
    return out


def _group_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, groups: int = 2,
                eps: float = 1e-5) -> np.ndarray:
    """GroupNorm applied to (L, C) features.

    Replicates ``nn.GroupNorm(groups, C)`` applied to a (batch=1, C, T=L) BCT
    tensor.  PyTorch GroupNorm normalises each group over *both* the
    C_per_group channels AND the spatial dimension T simultaneously — so the
    mean and variance are scalar per (batch, group), not per time step.

    Parameters
    ----------
    x      : (L, C) float32 — input, interpreted as a single BCT slice (C, L)
             transposed for convenience.
    weight : (C,) float32 — per-channel scale (gamma)
    bias   : (C,) float32 — per-channel shift (beta)
    groups : number of normalisation groups
    """
    x = x.astype(np.float32)
    L, C = x.shape
    C_per_group = C // groups
    out = np.empty_like(x)
    for g in range(groups):
        sl = slice(g * C_per_group, (g + 1) * C_per_group)
        xg = x[:, sl]  # (L, C_per_group)
        # Normalise across all L*C_per_group elements (matches PyTorch GroupNorm
        # which pools over the spatial+channel dims of each group jointly).
        mean = xg.mean()
        var = xg.var()
        out[:, sl] = (xg - mean) / np.sqrt(var + eps)
    return out * weight + bias


def _apply_vq_projection(
    cnn_features: np.ndarray,
    proj_conv_weight: np.ndarray,
    proj_gn_weight: np.ndarray,
    proj_gn_bias: np.ndarray,
) -> np.ndarray:
    """Apply the KmeansVectorQuantizer projection (grouped conv + GroupNorm).

    The fairseq KmeansVectorQuantizer applies ``self.projection(x)`` before
    computing L2 distances to the codebook entries.  Skipping this step yields
    completely wrong token indices (0% agreement with upstream).

    Parameters
    ----------
    cnn_features    : (L, 512) float32 — raw CNN output (BTC, single batch).
    proj_conv_weight: (512, 256, 1) float32 — grouped Conv1d weight.
    proj_gn_weight  : (512,) float32 — GroupNorm scale.
    proj_gn_bias    : (512,) float32 — GroupNorm bias.

    Returns
    -------
    (L, 512) float32 — projected features ready for codebook argmin.
    """
    ze = _grouped_conv1d(cnn_features, proj_conv_weight)
    ze = _group_norm(ze, proj_gn_weight, proj_gn_bias, groups=2)
    return ze


def _vq_encode(
    cnn_features: np.ndarray,
    codebook: np.ndarray,
    proj_conv_weight: np.ndarray | None = None,
    proj_gn_weight: np.ndarray | None = None,
    proj_gn_bias: np.ndarray | None = None,
) -> np.ndarray:
    """Project → quantise CNN features → return nearest codebook entries.

    The upstream KmeansVectorQuantizer first passes CNN features through a
    grouped Conv1d + GroupNorm projection before computing L2 distances to the
    codebook.  The adapter must replicate this step to produce correct token
    indices.

    Parameters
    ----------
    cnn_features    : (L, 512) float32 — raw CNN output.
    codebook        : (G, V, D) float32 — G=2, V=320, D=256.
    proj_conv_weight: (512, 256, 1) float32 — grouped Conv1d weight, or None
                      to skip projection (unit tests / synthetic data).
    proj_gn_weight  : (512,) float32 — GroupNorm scale, or None.
    proj_gn_bias    : (512,) float32 — GroupNorm bias, or None.

    Returns
    -------
    np.ndarray  (L, 512) float32 — VQ-vectors (codebook entries, concatenated
    across groups), matching upstream ``zq`` output.
    """
    # Apply projection before codebook lookup (matches upstream forward()).
    # Projection weights are None only in unit-test / synthetic-data paths.
    if proj_conv_weight is not None:
        ze = _apply_vq_projection(cnn_features, proj_conv_weight, proj_gn_weight, proj_gn_bias)
    else:
        ze = cnn_features

    G, V, D = codebook.shape  # G=2, V=320, D=256
    # Split projected features into G groups of D dims
    groups = np.split(ze, G, axis=1)  # list of (L, D)
    vqvecs = []
    for g in range(G):
        feat_g = groups[g]  # (L, D)
        cb_g = codebook[g]  # (V, D)
        # L2 nearest-neighbour search against codebook
        dists = (
            np.sum(feat_g ** 2, axis=1, keepdims=True)    # (L, 1)
            + np.sum(cb_g ** 2, axis=1)                    # (V,)
            - 2.0 * (feat_g @ cb_g.T)                      # (L, V)
        )
        idx = np.argmin(dists, axis=1)  # (L,)
        vqvecs.append(cb_g[idx])        # (L, D)

    return np.concatenate(vqvecs, axis=1).astype(np.float32)  # (L, 512)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class Vec2WavAdapter(VoiceClonerBase):
    """Voice-cloning adapter backed by vec2wav 2.0 ONNX models.

    Parameters
    ----------
    quantized:
        Use INT8 quantized ONNX models (default ``False``).
    **cfg:
        Additional keyword arguments stored but not used.
    """

    _sample_rate = _VEC2WAV_SR

    def __init__(
        self,
        quantized: bool = False,
        **cfg,
    ):
        super().__init__(**cfg)
        self._quantized = quantized
        self._cnn_sess = None
        self._wavlm_sess = None
        self._frontend_sess = None
        self._vocoder_sess = None
        self._codebook: Optional[np.ndarray] = None
        self._proj_conv_weight: Optional[np.ndarray] = None
        self._proj_gn_weight: Optional[np.ndarray] = None
        self._proj_gn_bias: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_models(self) -> None:
        if self._cnn_sess is not None:
            return

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for engine='vec2wav'. "
                "Install it with: pip install voiceclonnx"
            ) from exc

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("huggingface_hub is required.") from exc

        cnn_file = _CNN_INT8 if self._quantized else _CNN_FP32
        wavlm_file = _WAVLM_INT8 if self._quantized else _WAVLM_FP32
        frontend_file = _FRONTEND_INT8 if self._quantized else _FRONTEND_FP32
        # The BigVGAN vocoder uses alias_free_torch ops (Snake-Beta + sinc resamplers)
        # that prevent INT8 shape-inference during quantization; vocoder is fp32-only.
        vocoder_file = _VOCODER_FP32

        cnn_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=cnn_file)
        codebook_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=_CODEBOOK)
        projection_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=_PROJECTION)
        wavlm_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=wavlm_file)
        frontend_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=frontend_file)
        vocoder_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=vocoder_file)

        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = os.cpu_count() or 4
        sess_opts.intra_op_num_threads = os.cpu_count() or 4
        providers = ["CPUExecutionProvider"]

        self._cnn_sess = ort.InferenceSession(
            cnn_path, sess_options=sess_opts, providers=providers
        )
        self._wavlm_sess = ort.InferenceSession(
            wavlm_path, sess_options=sess_opts, providers=providers
        )
        self._frontend_sess = ort.InferenceSession(
            frontend_path, sess_options=sess_opts, providers=providers
        )
        self._vocoder_sess = ort.InferenceSession(
            vocoder_path, sess_options=sess_opts, providers=providers
        )
        self._codebook = np.load(codebook_path)  # (G, V, D)
        proj = np.load(projection_path)
        self._proj_conv_weight = proj["conv_weight"].astype(np.float32)   # (512, 256, 1)
        self._proj_gn_weight = proj["gn_weight"].astype(np.float32)       # (512,)
        self._proj_gn_bias = proj["gn_bias"].astype(np.float32)           # (512,)

    # ------------------------------------------------------------------
    # Content encoding
    # ------------------------------------------------------------------

    def _encode_content(self, audio: np.ndarray) -> np.ndarray:
        """vq-wav2vec CNN → projection → VQ quantisation → (1, L, 512) VQ-vectors."""
        self._ensure_models()
        inp = audio[np.newaxis, :].astype(np.float32)  # (1, T)
        cnn_out = self._cnn_sess.run(None, {"input_values": inp})
        # cnn_out[0]: (1, L, 512) — raw CNN features before VQ projection
        cnn_feats = cnn_out[0][0]  # (L, 512)
        vqvec = _vq_encode(
            cnn_feats,
            self._codebook,
            self._proj_conv_weight,
            self._proj_gn_weight,
            self._proj_gn_bias,
        )  # (L, 512)
        return vqvec[np.newaxis, :, :]  # (1, L, 512)

    # ------------------------------------------------------------------
    # Speaker encoding
    # ------------------------------------------------------------------

    def _encode_speaker(self, audio: np.ndarray) -> np.ndarray:
        """WavLM layer-6 → temporal mean → (1, 1024) speaker embedding."""
        self._ensure_models()
        inp = audio[np.newaxis, :].astype(np.float32)  # (1, T)
        out = self._wavlm_sess.run(None, {"input_values": inp})
        # out[0]: (1, T', 1024) → mean over T' → (1, 1024)
        return out[0].mean(axis=1)  # (1, 1024)

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def _synthesize(self, vqvec: np.ndarray, prompt: np.ndarray, cond: np.ndarray) -> np.ndarray:
        """Frontend + vocoder → (samples,) float32 waveform.

        Parameters
        ----------
        vqvec  : (1, L, 512) float32 — content VQ-vectors.
        prompt : (1, T, 1024) float32 — speaker WavLM features.
        cond   : (1, 1024) float32 — mean speaker embedding.
        """
        self._ensure_models()
        # Frontend: (1, L, 512) + (1, T, 1024) → (1, 184, L)
        hidden = self._frontend_sess.run(
            None, {"vqvec": vqvec.astype(np.float32), "prompt": prompt.astype(np.float32)}
        )[0]  # (1, 184, L)

        # Vocoder: (1, 184, L) + (1, 1024) → (1, 1, N)
        wav = self._vocoder_sess.run(
            None, {"hidden": hidden, "cond": cond.astype(np.float32)}
        )[0]  # (1, 1, N)
        return wav[0, 0]  # (N,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clone_voice(
        self,
        audio: str,
        reference_voice: str,
        out_path: str,
    ) -> str:
        """Convert *audio* to sound like *reference_voice* using vec2wav 2.0.

        Parameters
        ----------
        audio:
            Path to source WAV (any sample rate; resampled to 16 kHz).
        reference_voice:
            Path to reference speaker WAV (any sample rate; resampled to 16 kHz).
        out_path:
            Destination path for the 16-bit 24 kHz output WAV.

        Returns
        -------
        str
            Absolute path to the written output file.
        """
        src_wav = _load_wav(str(audio), target_sr=_INPUT_SR)
        ref_wav = _load_wav(str(reference_voice), target_sr=_INPUT_SR)

        # Content: vq-wav2vec CNN + VQ → (1, L, 512) VQ-vectors
        vqvec = self._encode_content(src_wav)  # (1, L, 512)

        # Speaker: WavLM layer-6 features (kept as sequence for Conformer cross-attn)
        self._ensure_models()
        ref_inp = ref_wav[np.newaxis, :].astype(np.float32)
        wavlm_out = self._wavlm_sess.run(None, {"input_values": ref_inp})
        prompt = wavlm_out[0]  # (1, T', 1024)
        cond = prompt.mean(axis=1)  # (1, 1024) — temporal mean for BigVGAN conditioning

        waveform = self._synthesize(vqvec, prompt, cond)

        out_path = str(Path(out_path).resolve())
        _save_wav(out_path, waveform, sr=_VEC2WAV_SR)
        return out_path


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_engine(
    EngineEntry(
        alias="vec2wav",
        adapter_class=Vec2WavAdapter,
        description=(
            "vec2wav 2.0: vq-wav2vec content tokens + WavLM-Large layer-6 speaker "
            "features + CTXVEC2WAV Conformer frontend + BigVGAN vocoder. "
            "Any-to-any VC at 24 kHz. "
            "ONNX artifacts from TigreGotico/voiceclonnx-vec2wav. "
            "Weights GPL-3.0 (cantabile-kwok/vec2wav2.0); code Apache-2.0. "
            "(Guo et al., Interspeech 2024)"
        ),
        extras="",
        onnx_native=True,
    )
)
