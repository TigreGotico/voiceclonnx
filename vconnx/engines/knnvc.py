"""kNN-VC adapter for vconnx.

kNN-VC (Baas et al., Interspeech 2023) is a zero-shot any-to-any voice
conversion system.  At inference it uses:

1. **WavLM-Large encoder** (layer 6) — converts 16 kHz audio to 1024-dim
   feature frames at 50 Hz.
2. **k-nearest-neighbour matching** (pure numpy) — replaces each source
   feature frame with the nearest reference frame(s).
3. **HiFi-GAN vocoder** — converts matched features back to waveform.

All neural components run via onnxruntime.  The kNN step is pure numpy —
no ONNX, no torch at inference.

Requires: ``pip install vconnx[knnvc]``
  → onnxruntime, numpy, soundfile (already in core deps except soundfile)

References
----------
- https://github.com/bshall/knn-vc
- https://arxiv.org/abs/2305.18975
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

from vconnx.engines.base import EngineEntry, VoiceClonerBase, register_engine

# kNN-VC outputs 16 kHz audio (same rate as WavLM input)
_KNNVC_SR = 16000

# HF repo housing the exported ONNX artifacts
_HF_REPO_ID = "TigreGotico/vconnx-knn-vc"

# Paths within the HF repo
_WAVLM_FP32 = "wavlm_layer6.onnx"
_WAVLM_INT8 = "wavlm_layer6_q8.onnx"
_HIFIGAN_FP32 = "hifigan_knnvc.onnx"
_HIFIGAN_INT8 = "hifigan_knnvc_q8.onnx"
_CONFIG = "knn-vc/config.json"


# ---------------------------------------------------------------------------
# Audio I/O helpers (soundfile)
# ---------------------------------------------------------------------------


def _load_wav(path: str, target_sr: int = 16000) -> np.ndarray:
    """Load a WAV file, resample to *target_sr*, return float32 mono array."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    # Mix down to mono if stereo
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        # Simple linear resampling via numpy (low quality but dependency-free)
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
# kNN matching (pure numpy)
# ---------------------------------------------------------------------------


def _knn_match(
    source_features: np.ndarray,
    reference_features: np.ndarray,
    k: int = 4,
) -> np.ndarray:
    """Replace each source feature frame with the mean of its k nearest
    reference neighbours (L2 distance).

    Parameters
    ----------
    source_features:
        (T_src, D) float32 — source WavLM features.
    reference_features:
        (T_ref, D) float32 — reference speaker WavLM features.
    k:
        Number of neighbours to average.

    Returns
    -------
    np.ndarray
        (T_src, D) float32 — matched features.
    """
    # Compute squared L2 distances: (T_src, T_ref)
    # ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b^T
    src_sq = np.sum(source_features ** 2, axis=1, keepdims=True)   # (T_src, 1)
    ref_sq = np.sum(reference_features ** 2, axis=1, keepdims=True)  # (T_ref, 1)
    dists = src_sq + ref_sq.T - 2.0 * (source_features @ reference_features.T)

    # Clip negative floats from numerical noise
    dists = np.clip(dists, 0.0, None)

    # Clamp k to valid range; argpartition requires kth < axis-length
    n_ref = reference_features.shape[0]
    k = min(k, n_ref)
    if k == n_ref:
        # Requesting all reference frames — just sort all
        nn_idx = np.argsort(dists, axis=1)[:, :k]
    else:
        # argpartition(kth) must be < axis-size; kth=k-1 gives k smallest
        nn_idx = np.argpartition(dists, k - 1, axis=1)[:, :k]

    # Average the k nearest reference frames
    matched = reference_features[nn_idx].mean(axis=1)  # (T_src, D)
    return matched.astype(np.float32)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class KNNVCAdapter(VoiceClonerBase):
    """Voice-cloning adapter backed by kNN-VC ONNX models.

    Parameters
    ----------
    quantized:
        Use INT8 quantized ONNX models (default ``False`` — fp32 for
        better quality; use ``True`` for faster CPU inference).
    k:
        Number of nearest neighbours for the matching step (default ``4``).
    **cfg:
        Additional keyword arguments stored but not used.
    """

    _sample_rate = _KNNVC_SR

    def __init__(
        self,
        quantized: bool = False,
        k: int = 4,
        **cfg,
    ):
        super().__init__(**cfg)
        self._quantized = quantized
        self._k = k
        self._wavlm_sess = None
        self._hifigan_sess = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_models(self) -> None:
        if self._wavlm_sess is not None:
            return

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for engine='knnvc'. "
                "Install it with: pip install vconnx[knnvc]"
            ) from exc

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("huggingface_hub is required.") from exc

        wavlm_file = _WAVLM_INT8 if self._quantized else _WAVLM_FP32
        hifigan_file = _HIFIGAN_INT8 if self._quantized else _HIFIGAN_FP32

        wavlm_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=wavlm_file)
        hifigan_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=hifigan_file)

        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = os.cpu_count() or 4
        sess_opts.intra_op_num_threads = os.cpu_count() or 4

        providers = ["CPUExecutionProvider"]

        self._wavlm_sess = ort.InferenceSession(
            wavlm_path, sess_options=sess_opts, providers=providers
        )
        self._hifigan_sess = ort.InferenceSession(
            hifigan_path, sess_options=sess_opts, providers=providers
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Run WavLM encoder and return (T, 1024) feature array."""
        self._ensure_models()
        # Input shape: (1, T)
        inp = audio[np.newaxis, :].astype(np.float32)
        out = self._wavlm_sess.run(None, {"input_values": inp})
        # Output: (1, frames, 1024) → (frames, 1024)
        return out[0][0]

    # ------------------------------------------------------------------
    # Vocoder
    # ------------------------------------------------------------------

    def _vocode(self, features: np.ndarray) -> np.ndarray:
        """Run HiFi-GAN and return (samples,) float32 waveform."""
        self._ensure_models()
        # Input shape: (1, 1024, T) — transpose from (T, 1024)
        inp = features.T[np.newaxis, :, :].astype(np.float32)
        out = self._hifigan_sess.run(None, {"features": inp})
        # Output: (1, 1, samples) → (samples,)
        return out[0][0, 0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clone_voice(
        self,
        audio: str,
        reference_voice: str,
        out_path: str,
    ) -> str:
        """Convert *audio* to sound like *reference_voice* using kNN-VC.

        Parameters
        ----------
        audio:
            Path to the source WAV file.
        reference_voice:
            Path to the reference speaker WAV file.
        out_path:
            Destination path for the 16-bit 16 kHz output WAV.

        Returns
        -------
        str
            Absolute path to the written output file.
        """
        # Load both audio files at 16 kHz
        src_wav = _load_wav(str(audio), target_sr=_KNNVC_SR)
        ref_wav = _load_wav(str(reference_voice), target_sr=_KNNVC_SR)

        # Extract WavLM features
        src_feats = self._extract_features(src_wav)    # (T_src, 1024)
        ref_feats = self._extract_features(ref_wav)    # (T_ref, 1024)

        # kNN matching (pure numpy)
        matched_feats = _knn_match(src_feats, ref_feats, k=self._k)  # (T_src, 1024)

        # Vocode
        waveform = self._vocode(matched_feats)  # (samples,)

        # Write output
        out_path = str(Path(out_path).resolve())
        _save_wav(out_path, waveform, sr=_KNNVC_SR)
        return out_path


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_engine(
    EngineEntry(
        alias="knnvc",
        adapter_class=KNNVCAdapter,
        description=(
            "kNN-VC: WavLM-Large layer-6 encoder + k-nearest-neighbour matching "
            "(pure numpy) + HiFi-GAN vocoder. Zero-shot any-to-any VC at 16 kHz. "
            "ONNX artifacts from TigreGotico/vconnx-knn-vc. "
            "(Baas et al., Interspeech 2023, MIT license)"
        ),
        extras="knnvc",
        onnx_native=True,
    )
)
