"""TriAAN-VC adapter for vconnx.

TriAAN-VC (winddori2002 et al., ICASSP 2023) — "Triple Adaptive Attention
Normalization for Any-to-Any Voice Conversion."

Three-stage inference pipeline (all ONNX, no torch at runtime):

1. **CPC encoder** — 5-layer strided Conv1d + 2-layer GRU; maps 16 kHz
   waveform to 256-dim feature frames at 100 Hz (downsampled by 160×).
2. **TriAAN-VC main model** — LF0Encoder + SpeakerEncoder + ContentEncoder +
   TriAAN decoder + PostNet; takes ``(src_cpc, src_lf0, trg_cpc)`` and
   produces an 80-bin mel spectrogram.
3. **ParallelWaveGAN vocoder** — fully convolutional conditional upsampling
   network; converts the mel spectrogram back to 16 kHz waveform.

F0 (fundamental frequency / pitch) extraction uses a lightweight pure-numpy
WORLD-style autocorrelation estimator — no torch, no pyworld required.

Requires: ``pip install vconnx[triaan]``
  → onnxruntime, numpy, soundfile, librosa

References
----------
- https://github.com/winddori2002/TriAAN-VC
- https://arxiv.org/abs/2303.09057
- https://ieeexplore.ieee.org/document/10096642/
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from vconnx.engines.base import EngineEntry, VoiceClonerBase, register_engine

# TriAAN-VC operates at 16 kHz throughout
_TRIAAN_SR = 16000
_N_MELS = 80
_HOP_LENGTH = 160   # samples; matches CPC downsampling factor
_N_FFT = 400
_WIN_LENGTH = 400
_CPC_HIDDEN = 256

# HF repo housing the exported ONNX artifacts
_HF_REPO_ID = "TigreGotico/vconnx-triaan-vc"

# Component filenames
_CPC_FP32 = "cpc_encoder.onnx"
_CPC_INT8 = "cpc_encoder_q8.onnx"
_TRIAAN_FP32 = "triaan_vc.onnx"
_TRIAAN_INT8 = "triaan_vc_q8.onnx"
_PWG_FP32 = "pwg_vocoder.onnx"
_PWG_INT8 = "pwg_vocoder_q8.onnx"


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------


def _load_wav(path: str, target_sr: int = _TRIAAN_SR) -> np.ndarray:
    """Load a WAV file, resample to *target_sr*, return float32 mono array."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if sr != target_sr:
        # Librosa for quality resampling
        try:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        except ImportError:
            n_out = int(len(audio) * target_sr / sr)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, n_out),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)

    # Normalise peak amplitude
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.99
    return audio.astype(np.float32)


def _save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf

    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    sf.write(str(path), audio_int16, sr, subtype="PCM_16")


# ---------------------------------------------------------------------------
# F0 extraction (pure numpy — no pyworld dependency)
# ---------------------------------------------------------------------------


def _extract_log_f0(audio: np.ndarray, sr: int = _TRIAAN_SR,
                    hop_length: int = _HOP_LENGTH,
                    f0_min: float = 80.0, f0_max: float = 800.0) -> np.ndarray:
    """Estimate log-F0 contour using autocorrelation.

    Returns a float32 array of shape (n_frames,) where 0.0 marks unvoiced
    frames and log(f0) marks voiced frames — matching the TriAAN-VC convention.
    """
    n_frames = len(audio) // hop_length
    lf0 = np.zeros(n_frames, dtype=np.float32)

    min_lag = int(sr / f0_max)
    max_lag = int(sr / f0_min)
    frame_len = max_lag * 2 + 1  # analysis window

    for i in range(n_frames):
        start = i * hop_length
        frame = audio[start: start + frame_len]
        if len(frame) < frame_len:
            frame = np.pad(frame, (0, frame_len - len(frame)))

        # Normalize frame
        frame = frame - frame.mean()
        energy = np.dot(frame, frame)
        if energy < 1e-8:
            continue  # unvoiced / silence

        # Autocorrelation via direct summation over candidate lags
        best_lag = -1
        best_corr = -1.0
        for lag in range(min_lag, max_lag + 1):
            corr = np.dot(frame[:-lag], frame[lag:]) / (energy + 1e-8)
            if corr > best_corr:
                best_corr = corr
                best_lag = lag

        # Voiced if correlation is strong enough
        if best_corr > 0.3 and best_lag > 0:
            f0 = sr / best_lag
            lf0[i] = np.log(f0 + 1e-8)

    # Normalise voiced frames: subtract mean, divide by std
    voiced = lf0 != 0
    if voiced.sum() > 1:
        mu = lf0[voiced].mean()
        sigma = lf0[voiced].std() + 1e-8
        lf0[voiced] = (lf0[voiced] - mu) / sigma
    return lf0


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class TriAANVCAdapter(VoiceClonerBase):
    """Voice-cloning adapter backed by TriAAN-VC ONNX models.

    Parameters
    ----------
    quantized:
        Use INT8 quantized ONNX models (default ``False``).
    **cfg:
        Additional keyword arguments stored but not used.

    Architecture
    ------------
    CPC encoder → (src_cpc, trg_cpc) + log-F0 → TriAAN-VC → mel → PWG → waveform

    All three stages run via onnxruntime; no torch at inference time.
    Sample rate is 16 kHz throughout.
    """

    _sample_rate = _TRIAAN_SR

    def __init__(self, quantized: bool = False, **cfg):
        super().__init__(**cfg)
        self._quantized = quantized
        self._cpc_sess = None
        self._triaan_sess = None
        self._pwg_sess = None
        self._pwg_context_trim: int = 4  # refined after model load (aux_context_window=2 → trims 4 frames)

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_models(self) -> None:
        if self._cpc_sess is not None:
            return

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for engine='triaan'. "
                "Install it with: pip install vconnx[triaan]"
            ) from exc

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("huggingface_hub is required.") from exc

        cpc_file = _CPC_INT8 if self._quantized else _CPC_FP32
        triaan_file = _TRIAAN_INT8 if self._quantized else _TRIAAN_FP32
        pwg_file = _PWG_INT8 if self._quantized else _PWG_FP32

        cpc_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=cpc_file)
        triaan_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=triaan_file)
        pwg_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=pwg_file)

        sess_opts = ort.SessionOptions()
        n_threads = os.cpu_count() or 4
        sess_opts.inter_op_num_threads = n_threads
        sess_opts.intra_op_num_threads = n_threads
        providers = ["CPUExecutionProvider"]

        self._cpc_sess = ort.InferenceSession(
            cpc_path, sess_options=sess_opts, providers=providers
        )
        self._triaan_sess = ort.InferenceSession(
            triaan_path, sess_options=sess_opts, providers=providers
        )
        self._pwg_sess = ort.InferenceSession(
            pwg_path, sess_options=sess_opts, providers=providers
        )

        # The ParallelWaveGAN vocoder uses ConvInUpsampleNetwork with
        # aux_context_window=2, which strips 2*2=4 context frames from each end.
        # Actual audio length = (T_mel - 4) * HOP_LENGTH.
        # We detect this at load time with a small probe to be robust to config changes.
        _probe_T = 20  # must be > 4 (context_window trim)
        _probe_mel = np.zeros((1, _N_MELS, _probe_T), dtype=np.float32)
        _probe_T_audio_nominal = (_probe_T - 4) * _HOP_LENGTH
        _probe_noise = np.zeros((1, 1, _probe_T_audio_nominal), dtype=np.float32)
        try:
            _probe_out = self._pwg_sess.run(None, {"noise": _probe_noise, "mel": _probe_mel})
            # Effective per-frame samples (for frames after context removal)
            self._pwg_audio_len_fn: int = _probe_out[0].shape[-1]  # store reference
            self._pwg_context_trim: int = _probe_T - (_probe_out[0].shape[-1] // _HOP_LENGTH)
        except Exception:
            self._pwg_context_trim = 4  # default for aux_context_window=2

    # ------------------------------------------------------------------
    # CPC feature extraction
    # ------------------------------------------------------------------

    def _extract_cpc(self, audio: np.ndarray) -> np.ndarray:
        """Run CPC encoder on audio.

        Parameters
        ----------
        audio : (T,) float32

        Returns
        -------
        features : (1, n_frames, 256) float32
            Caller is responsible for transposing to (1, 256, n_frames) if needed.
        """
        self._ensure_models()
        # Input: (1, 1, T)
        inp = audio[np.newaxis, np.newaxis, :].astype(np.float32)
        out = self._cpc_sess.run(None, {"audio": inp})
        # Output: (1, n_frames, 256) — time-last expected by TriAAN, transpose in clone_voice
        return out[0]

    # ------------------------------------------------------------------
    # TriAAN-VC conversion
    # ------------------------------------------------------------------

    def _convert(
        self,
        src_cpc: np.ndarray,
        src_lf0: np.ndarray,
        trg_cpc: np.ndarray,
    ) -> np.ndarray:
        """Run TriAAN-VC decoder.

        Parameters
        ----------
        src_cpc : (1, 256, T_s)
        src_lf0 : (1, T_s)
        trg_cpc : (1, 256, T_t)

        Returns
        -------
        mel : (1, 80, T_s) float32 mel spectrogram
        """
        self._ensure_models()
        out = self._triaan_sess.run(
            None,
            {
                "src_cpc": src_cpc.astype(np.float32),
                "src_lf0": src_lf0.astype(np.float32),
                "trg_cpc": trg_cpc.astype(np.float32),
            },
        )
        return out[0]  # (1, 80, T_s)

    # ------------------------------------------------------------------
    # ParallelWaveGAN vocoding
    # ------------------------------------------------------------------

    def _vocode(self, mel: np.ndarray) -> np.ndarray:
        """Run ParallelWaveGAN vocoder.

        Parameters
        ----------
        mel : (1, 80, T_mel) float32

        Returns
        -------
        waveform : (T_audio,) float32
        """
        self._ensure_models()
        T_mel = mel.shape[2]
        T_audio = (T_mel - self._pwg_context_trim) * _HOP_LENGTH
        noise = np.random.randn(1, 1, T_audio).astype(np.float32)
        out = self._pwg_sess.run(None, {"noise": noise, "mel": mel})
        return out[0][0, 0]  # (T_audio,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clone_voice(
        self,
        audio: str,
        reference_voice: str,
        out_path: str,
    ) -> str:
        """Convert *audio* to sound like *reference_voice* using TriAAN-VC.

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
        src_wav = _load_wav(str(audio), target_sr=_TRIAAN_SR)
        ref_wav = _load_wav(str(reference_voice), target_sr=_TRIAAN_SR)

        # Extract CPC features; CPC ONNX returns (1, T, 256), TriAAN expects (1, 256, T)
        # CPC ONNX returns (1, T, 256); TriAAN ONNX expects channels-first (1, 256, T)
        src_cpc = self._extract_cpc(src_wav).transpose(0, 2, 1)   # (1, 256, T_s)
        trg_cpc = self._extract_cpc(ref_wav).transpose(0, 2, 1)   # (1, 256, T_t)

        # Extract log-F0 for source (pure numpy)
        src_lf0 = _extract_log_f0(src_wav)     # (T_s_frames,)
        T_cpc = src_cpc.shape[2]  # shape is (1, 256, T) after transpose

        # Align lf0 length to CPC frames (may differ slightly due to truncation)
        if len(src_lf0) < T_cpc:
            src_lf0 = np.pad(src_lf0, (0, T_cpc - len(src_lf0)))
        else:
            src_lf0 = src_lf0[:T_cpc]
        src_lf0 = src_lf0[np.newaxis, :]      # (1, T_s)

        # TriAAN-VC conversion
        mel = self._convert(src_cpc, src_lf0, trg_cpc)  # (1, 80, T_s)

        # Vocode mel → waveform
        waveform = self._vocode(mel)  # (T_audio,)

        # Write output
        out_path = str(Path(out_path).resolve())
        _save_wav(out_path, waveform, sr=_TRIAAN_SR)
        return out_path


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_engine(
    EngineEntry(
        alias="triaan",
        adapter_class=TriAANVCAdapter,
        description=(
            "TriAAN-VC: CPC encoder + Triple Adaptive Attention Normalization decoder "
            "+ ParallelWaveGAN vocoder. Zero-shot any-to-any VC at 16 kHz. "
            "ONNX artifacts from TigreGotico/vconnx-triaan-vc. "
            "(winddori2002 et al., ICASSP 2023, MIT license)"
        ),
        extras="triaan",
        onnx_native=True,
    )
)
