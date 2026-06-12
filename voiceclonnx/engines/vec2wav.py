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


def _vq_encode(cnn_features: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Quantise CNN features to nearest codebook entries.

    Parameters
    ----------
    cnn_features : (L, 512) float32 — pre-VQ CNN features.
    codebook     : (G, V, D) float32 — G=2, V=320, D=256.

    Returns
    -------
    np.ndarray  (L, 512) float32 — VQ-vectors (codebook entries, concatenated
    across groups), equivalent to what idx2vec returns in the original code.
    """
    G, V, D = codebook.shape  # G=2, V=320, D=256
    L = cnn_features.shape[0]

    # Split the 512-dim feature into G groups of D dims
    groups = np.split(cnn_features, G, axis=1)  # list of (L, D)
    vqvecs = []
    for g in range(G):
        feat_g = groups[g]  # (L, D)
        cb_g = codebook[g]  # (V, D)
        # L2 nearest-neighbour search
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

    # ------------------------------------------------------------------
    # Content encoding
    # ------------------------------------------------------------------

    def _encode_content(self, audio: np.ndarray) -> np.ndarray:
        """vq-wav2vec CNN → VQ discretisation → (1, L, 512) float32 VQ-vectors."""
        self._ensure_models()
        inp = audio[np.newaxis, :].astype(np.float32)  # (1, T)
        cnn_out = self._cnn_sess.run(None, {"input_values": inp})
        # cnn_out[0]: (1, L, 512)
        cnn_feats = cnn_out[0][0]  # (L, 512)
        vqvec = _vq_encode(cnn_feats, self._codebook)  # (L, 512)
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
