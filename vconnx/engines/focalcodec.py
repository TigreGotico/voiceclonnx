"""FocalCodec adapter for vconnx.

FocalCodec (Della Libera et al., NeurIPS 2025) is a single-codebook binary
speech codec using focal modulation networks.  At inference, voice conversion
uses a **kNN feature-space swap** on continuous pre-quantisation features —
the same approach as kNN-VC but with a different encoder (WavLM inside
FocalCodec) and decoder (Vocos ISTFT vocoder):

1. **WavLM encoder** — converts 16 kHz audio to (T, 1024) feature frames at
   50 Hz.  Exported to ONNX (``focalcodec_encoder.onnx``).
2. **kNN matching** (pure numpy, cosine distance, k=4) — replaces each source
   feature frame with the weighted mean of its k nearest reference frames.
3. **Vocos backbone + linear proj** — maps matched (T, 1024) features to STFT
   coefficients (T, n_fft+2).  Exported to ONNX (``focalcodec_vocoder.onnx``).
4. **numpy ISTFT** — pure numpy overlap-add from STFT coefficients to waveform.
   Not in ONNX because ``nn.functional.fold`` with dynamic output_size cannot
   be traced; parity vs torch ≤1.3e-5 max abs error.

Runtime requirements: ``onnxruntime``, ``numpy``, ``soundfile``.

Requires: ``pip install vconnx``
  -> onnxruntime, numpy, soundfile, huggingface_hub

References
----------
- https://github.com/lucadellalib/focalcodec
- https://arxiv.org/abs/2502.04465
- https://huggingface.co/TigreGotico/vconnx-focalcodec
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

from vconnx.engines.base import EngineEntry, VoiceClonerBase, register_engine

# FocalCodec operates at 16 kHz; Vocos hop=320 → 50 Hz feature rate
_FC_SR = 16000

# HF repo housing the exported ONNX artifacts
_HF_REPO_ID = "TigreGotico/vconnx-focalcodec"

# Paths within the HF repo
_ENC_FP32 = "focalcodec_encoder.onnx"
_ENC_INT8 = "focalcodec_encoder_q8.onnx"
_VOC_FP32 = "focalcodec_vocoder.onnx"
_VOC_INT8 = "focalcodec_vocoder_q8.onnx"
_CONFIG = "config.json"

# Vocos ISTFT parameters (hard-coded to match lucadellalib/focalcodec_50hz)
_N_FFT = 1024
_HOP_LENGTH = 320
_WIN_LENGTH = 1024
_NON_CAUSAL_PAD = (_WIN_LENGTH - _HOP_LENGTH) // 2  # 352


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------


def _load_wav(path: str, target_sr: int = _FC_SR) -> np.ndarray:
    """Load *path* as float32 mono, resample to *target_sr*."""
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
    """Write *audio* float32 as 16-bit PCM WAV to *path*."""
    import soundfile as sf

    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    sf.write(str(path), audio_int16, sr, subtype="PCM_16")


# ---------------------------------------------------------------------------
# kNN matching — cosine distance, pure numpy
# ---------------------------------------------------------------------------


def _cosine_knn_match(
    source_features: np.ndarray,
    reference_features: np.ndarray,
    k: int = 4,
) -> np.ndarray:
    """Replace each source frame with the weighted mean of its k cosine-nearest
    reference frames.

    FocalCodec uses cosine distance (unlike kNN-VC which uses L2).  The weighted
    mean uses equal weights (mean of k frames) matching the upstream implementation
    in ``FocalCodec.feats_to_sig``.

    Parameters
    ----------
    source_features:
        (T_src, D) float32 — source encoder features.
    reference_features:
        (T_ref, D) float32 — reference speaker encoder features.
    k:
        Number of nearest neighbours (default 4).

    Returns
    -------
    np.ndarray
        (T_src, D) float32 — matched features.
    """
    # L2-normalise for cosine similarity
    src_norm = source_features / (np.linalg.norm(source_features, axis=1, keepdims=True) + 1e-8)
    ref_norm = reference_features / (np.linalg.norm(reference_features, axis=1, keepdims=True) + 1e-8)

    # Cosine similarity → cosine distance
    sim = src_norm @ ref_norm.T          # (T_src, T_ref)
    dist = 1.0 - sim                     # (T_src, T_ref)

    n_ref = reference_features.shape[0]
    k_eff = min(k, n_ref)

    if k_eff == n_ref:
        nn_idx = np.argsort(dist, axis=1)[:, :k_eff]
    else:
        nn_idx = np.argpartition(dist, k_eff - 1, axis=1)[:, :k_eff]

    matched = reference_features[nn_idx].mean(axis=1)   # (T_src, D)
    return matched.astype(np.float32)


# ---------------------------------------------------------------------------
# numpy ISTFT
# ---------------------------------------------------------------------------


def _numpy_istft(
    stft_coeffs: np.ndarray,
    n_fft: int = _N_FFT,
    hop_length: int = _HOP_LENGTH,
    win_length: int = _WIN_LENGTH,
) -> np.ndarray:
    """Vocos-compatible numpy ISTFT (non-causal, Hann window OLA).

    Parameters
    ----------
    stft_coeffs:
        (B, T, n_fft+2) — concatenated mag_logits and phase columns.
    n_fft, hop_length, win_length:
        STFT parameters matching the Vocos head configuration.

    Returns
    -------
    np.ndarray
        (B, samples) float32 waveform.
    """
    B, T, _ = stft_coeffs.shape
    half = n_fft // 2 + 1                     # 513 for n_fft=1024

    mag_logits = stft_coeffs[:, :, :half]     # (B, T, 513)
    phase = stft_coeffs[:, :, half:]           # (B, T, 513)

    mag = np.exp(mag_logits)
    mag = np.clip(mag, 0.0, 1e2)

    # Complex spectrum — (B, T, 513) → permute to (B, 513, T) for irfft
    stft_complex = (mag * np.cos(phase) + 1j * mag * np.sin(phase)).astype(np.complex64)
    stft_complex = stft_complex.transpose(0, 2, 1)   # (B, 513, T)

    # Inverse FFT: (B, n_fft, T)
    ifft = np.fft.irfft(stft_complex, n=n_fft, axis=1).astype(np.float32)

    # Hann window  (numpy hanning(N+1)[:-1] ≡ torch.hann_window(N) to 2e-7)
    window = np.hanning(win_length + 1)[:-1].astype(np.float32)
    window_sq = window ** 2
    non_causal_pad = (win_length - hop_length) // 2

    output_size = (T - 1) * hop_length + win_length

    results = []
    for b in range(B):
        out_b = np.zeros(output_size, dtype=np.float32)
        env_b = np.zeros(output_size, dtype=np.float32)
        for t in range(T):
            start = t * hop_length
            out_b[start : start + win_length] += ifft[b, :, t] * window
            env_b[start : start + win_length] += window_sq
        crop = out_b[non_causal_pad : output_size - non_causal_pad]
        env_crop = env_b[non_causal_pad : output_size - non_causal_pad]
        results.append((crop / env_crop).astype(np.float32))

    return np.stack(results, axis=0)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class FocalCodecAdapter(VoiceClonerBase):
    """Voice-cloning adapter backed by FocalCodec ONNX models.

    Pipeline:
    1. Encode source and reference audio with the WavLM encoder ONNX model.
    2. kNN cosine-distance matching in the 1024-dim feature space (pure numpy).
    3. Decode matched features with the Vocos vocoder ONNX model + numpy ISTFT.

    Parameters
    ----------
    quantized:
        Use INT8 quantized ONNX models (default ``False``).
        **Not recommended for production** — INT8 degrades WER significantly
        (benchmark: 31% int8 vs 15% fp32).  Prefer fp32.
        See demo/QUANTS.md for the full comparison.
    k:
        Number of nearest neighbours for the kNN matching step (default 4).
    **cfg:
        Additional keyword arguments stored but not used at runtime.
    """

    _sample_rate = _FC_SR

    def __init__(
        self,
        quantized: bool = False,
        k: int = 4,
        **cfg,
    ):
        super().__init__(**cfg)
        self._quantized = quantized
        self._k = k
        self._enc_sess = None
        self._voc_sess = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_models(self) -> None:
        if self._enc_sess is not None:
            return

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for engine='focalcodec'. "
                "Install it with: pip install vconnx[focalcodec]"
            ) from exc

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("huggingface_hub is required.") from exc

        enc_file = _ENC_INT8 if self._quantized else _ENC_FP32
        voc_file = _VOC_INT8 if self._quantized else _VOC_FP32

        enc_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=enc_file)
        voc_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=voc_file)

        sess_opts = ort.SessionOptions()
        n_threads = os.cpu_count() or 4
        sess_opts.inter_op_num_threads = n_threads
        sess_opts.intra_op_num_threads = n_threads

        providers = ["CPUExecutionProvider"]

        self._enc_sess = ort.InferenceSession(enc_path, sess_options=sess_opts, providers=providers)
        self._voc_sess = ort.InferenceSession(voc_path, sess_options=sess_opts, providers=providers)

    # ------------------------------------------------------------------
    # Encode: audio → (T, 1024) features
    # ------------------------------------------------------------------

    def _encode(self, audio: np.ndarray) -> np.ndarray:
        """Run WavLM encoder and return (T, 1024) float32 features."""
        self._ensure_models()
        inp = audio[np.newaxis, :].astype(np.float32)  # (1, N)
        out = self._enc_sess.run(None, {"sig": inp})
        return out[0][0]  # (T, 1024)

    # ------------------------------------------------------------------
    # Decode: (T, 1024) → waveform
    # ------------------------------------------------------------------

    def _decode(self, features: np.ndarray) -> np.ndarray:
        """Run Vocos backbone + numpy ISTFT and return (samples,) waveform."""
        self._ensure_models()
        inp = features[np.newaxis, :, :].astype(np.float32)  # (1, T, 1024)
        stft_coeffs = self._voc_sess.run(None, {"feats": inp})[0]  # (1, T, n_fft+2)
        wav = _numpy_istft(stft_coeffs)   # (1, samples)
        return wav[0]  # (samples,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clone_voice(
        self,
        audio: str,
        reference_voice: str,
        out_path: str,
    ) -> str:
        """Convert *audio* to sound like *reference_voice* using FocalCodec kNN-VC.

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
        src_wav = _load_wav(str(audio), target_sr=_FC_SR)
        ref_wav = _load_wav(str(reference_voice), target_sr=_FC_SR)

        # Encode
        src_feats = self._encode(src_wav)    # (T_src, 1024)
        ref_feats = self._encode(ref_wav)    # (T_ref, 1024)

        # kNN cosine matching (pure numpy)
        matched = _cosine_knn_match(src_feats, ref_feats, k=self._k)  # (T_src, 1024)

        # Decode
        waveform = self._decode(matched)     # (samples,)

        out_path = str(Path(out_path).resolve())
        _save_wav(out_path, waveform, sr=_FC_SR)
        return out_path


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_engine(
    EngineEntry(
        alias="focalcodec",
        adapter_class=FocalCodecAdapter,
        description=(
            "FocalCodec: WavLM encoder + kNN cosine matching (pure numpy) + "
            "Vocos ISTFT decoder.  Zero-shot any-to-any VC at 16 kHz. "
            "ONNX artifacts from TigreGotico/vconnx-focalcodec. "
            "(Della Libera et al., NeurIPS 2025, Apache-2.0)"
        ),
        extras="",
        onnx_native=True,
    )
)
