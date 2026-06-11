"""OpenVoice v2 tone-color converter adapter for vconnx.

OpenVoice v2 (myshell-ai/OpenVoice, MIT license) is a zero-shot voice-cloning
library.  The **tone-color converter** sub-component is a standalone
audio-to-audio VC path: it transplants the speaker timbre from a reference
utterance onto a source utterance without requiring TTS text-encoders.

Architecture (ONNX inference path):
  1. **Reference encoder** (``tone_ref_encoder.onnx``) — mel-spectrogram → 256-dim
     tone-color embedding.  Run on both source and reference audio.
  2. **Converter** (``tone_converter.onnx``) — (source_mel, src_tone, tgt_tone) →
     converted_mel.  A flow-based AdaIN conditioned network.
  3. **Griffin-Lim vocoder** (pure numpy/scipy, no ONNX) — mel → waveform.
     A lightweight alternative to a neural vocoder; optional HiFi-GAN can be
     plugged in later as a separate component.

All neural components run via onnxruntime.  The mel extraction and Griffin-Lim
vocoder are pure numpy — no torch at inference.

Requires: ``pip install vconnx[openvoice]``
  → onnxruntime, numpy, soundfile

References
----------
- https://github.com/myshell-ai/OpenVoice
- https://huggingface.co/myshell-ai/OpenVoiceV2
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from vconnx.engines.base import EngineEntry, VoiceClonerBase, register_engine

# OpenVoice v2 operates at 22050 Hz
_OV2_SR = 22050

# HF repo housing the exported ONNX artifacts
_HF_REPO_ID = "TigreGotico/vconnx-models"

# Paths within the HF repo
_REF_ENC_FP32 = "openvoice-v2/tone_ref_encoder.onnx"
_REF_ENC_INT8 = "openvoice-v2/tone_ref_encoder_q8.onnx"
_CONVERTER_FP32 = "openvoice-v2/tone_converter.onnx"
_CONVERTER_INT8 = "openvoice-v2/tone_converter_q8.onnx"
_CONFIG = "openvoice-v2/config.json"

# Mel-spectrogram parameters matching OpenVoice v2 training config
_N_MELS = 80
_SAMPLE_RATE = _OV2_SR
_HOP_LENGTH = 256
_WIN_LENGTH = 1024
_N_FFT = 1024
_F_MIN = 0.0
_F_MAX = 8000.0


# ---------------------------------------------------------------------------
# Mel-spectrogram extraction (pure numpy / scipy)
# ---------------------------------------------------------------------------


def _stft(audio: np.ndarray, n_fft: int, hop_length: int, win_length: int) -> np.ndarray:
    """Compute magnitude STFT spectrogram using numpy.

    Returns
    -------
    np.ndarray
        Magnitude spectrogram of shape (n_fft//2+1, frames), float32.
    """
    window = np.hanning(win_length).astype(np.float32)
    # Pad to centre first frame
    pad = n_fft // 2
    audio = np.pad(audio, pad, mode="reflect")

    n_frames = 1 + (len(audio) - n_fft) // hop_length
    frames = np.stack(
        [audio[i * hop_length: i * hop_length + n_fft] for i in range(n_frames)],
        axis=0,
    )  # (n_frames, n_fft)

    # Zero-pad window to n_fft if win_length < n_fft
    if win_length < n_fft:
        pad_len = (n_fft - win_length) // 2
        window = np.pad(window, (pad_len, n_fft - win_length - pad_len))

    windowed = frames * window[np.newaxis, :]
    spec = np.fft.rfft(windowed, n=n_fft, axis=1)  # (n_frames, n_fft//2+1)
    return np.abs(spec).T.astype(np.float32)  # (bins, frames)


def _mel_filterbank(sr: int, n_fft: int, n_mels: int, f_min: float, f_max: float) -> np.ndarray:
    """Build a mel filterbank matrix (n_mels, n_fft//2+1), float32."""
    def _hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    n_bins = n_fft // 2 + 1
    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)

    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        lo, ctr, hi = bin_points[m - 1], bin_points[m], bin_points[m + 1]
        for k in range(lo, ctr):
            if ctr > lo:
                fb[m - 1, k] = (k - lo) / (ctr - lo)
        for k in range(ctr, hi):
            if hi > ctr:
                fb[m - 1, k] = (hi - k) / (hi - ctr)
    return fb


def _compute_mel(
    audio: np.ndarray,
    sr: int = _SAMPLE_RATE,
    n_mels: int = _N_MELS,
    n_fft: int = _N_FFT,
    hop_length: int = _HOP_LENGTH,
    win_length: int = _WIN_LENGTH,
    f_min: float = _F_MIN,
    f_max: float = _F_MAX,
) -> np.ndarray:
    """Compute log-mel spectrogram from a float32 mono waveform.

    Returns
    -------
    np.ndarray
        Shape (n_mels, frames), float32 log-mel.
    """
    mag = _stft(audio, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    fb = _mel_filterbank(sr, n_fft, n_mels, f_min, f_max)
    mel = np.maximum(fb @ mag, 1e-10)
    return np.log(mel).astype(np.float32)


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------


def _load_wav(path: str, target_sr: int = _OV2_SR) -> np.ndarray:
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
    """Write *audio* (float32) as 16-bit PCM WAV."""
    import soundfile as sf

    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    sf.write(str(path), audio_int16, sr, subtype="PCM_16")


# ---------------------------------------------------------------------------
# Griffin-Lim vocoder (mel → waveform, pure numpy)
# ---------------------------------------------------------------------------


def _griffin_lim(
    mel: np.ndarray,
    sr: int = _SAMPLE_RATE,
    n_fft: int = _N_FFT,
    hop_length: int = _HOP_LENGTH,
    win_length: int = _WIN_LENGTH,
    n_iter: int = 32,
    f_min: float = _F_MIN,
    f_max: float = _F_MAX,
) -> np.ndarray:
    """Reconstruct waveform from a log-mel spectrogram via Griffin-Lim.

    Parameters
    ----------
    mel:
        (n_mels, T) log-mel spectrogram.

    Returns
    -------
    np.ndarray
        (samples,) float32 waveform.
    """
    # Invert log-mel to linear mel
    mel_lin = np.exp(mel).astype(np.float32)

    # Invert mel filterbank: pseudo-inverse projection back to STFT bins
    fb = _mel_filterbank(sr, n_fft, mel.shape[0], f_min, f_max)
    # fb shape: (n_mels, n_bins); pseudo-inverse: (n_bins, n_mels)
    fb_pinv = np.linalg.pinv(fb).astype(np.float32)
    spec_mag = np.maximum(fb_pinv @ mel_lin, 0.0)  # (n_bins, T)

    n_frames = spec_mag.shape[1]
    n_bins = n_fft // 2 + 1
    window = np.hanning(win_length).astype(np.float32)
    if win_length < n_fft:
        pad_len = (n_fft - win_length) // 2
        window = np.pad(window, (pad_len, n_fft - win_length - pad_len))

    # Initialise with random phases
    angles = np.exp(2j * np.pi * np.random.uniform(size=(n_bins, n_frames))).astype(np.complex64)

    for _ in range(n_iter):
        # Build complex spectrum from magnitude + current angles
        spec_complex = spec_mag * angles  # (n_bins, T)
        # iSTFT: synthesise waveform from each frame
        n_out = n_fft + hop_length * (n_frames - 1)
        audio = np.zeros(n_out, dtype=np.float32)
        win_sq = np.zeros(n_out, dtype=np.float32)

        for i in range(n_frames):
            frame_complex = spec_complex[:, i]
            # Real part of irfft
            frame = np.fft.irfft(frame_complex, n=n_fft).real.astype(np.float32)
            start = i * hop_length
            audio[start: start + n_fft] += frame * window
            win_sq[start: start + n_fft] += window ** 2

        # Normalise by synthesis window
        win_sq = np.maximum(win_sq, 1e-8)
        audio /= win_sq

        # Re-analyse to update phases
        # Trim to avoid padding artifact
        audio_trim = audio[n_fft // 2: n_fft // 2 + n_frames * hop_length]
        audio_padded = np.pad(audio_trim, n_fft // 2, mode="reflect")
        frames = np.stack(
            [audio_padded[i * hop_length: i * hop_length + n_fft] for i in range(n_frames)],
            axis=0,
        )
        windowed = frames * window[np.newaxis, :]
        spec_new = np.fft.rfft(windowed, n=n_fft, axis=1).T.astype(np.complex64)
        mag_new = np.abs(spec_new)
        angles = np.where(mag_new > 1e-10, spec_new / mag_new, angles)

    # Final synthesis from last iteration's angles
    spec_final = spec_mag * angles
    n_out = n_fft + hop_length * (n_frames - 1)
    audio_out = np.zeros(n_out, dtype=np.float32)
    win_sq_out = np.zeros(n_out, dtype=np.float32)
    for i in range(n_frames):
        frame = np.fft.irfft(spec_final[:, i], n=n_fft).real.astype(np.float32)
        start = i * hop_length
        audio_out[start: start + n_fft] += frame * window
        win_sq_out[start: start + n_fft] += window ** 2
    win_sq_out = np.maximum(win_sq_out, 1e-8)
    audio_out /= win_sq_out

    # Trim to approximate original length
    trim = n_fft // 2
    audio_out = audio_out[trim: trim + n_frames * hop_length]
    return audio_out.astype(np.float32)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class OpenVoiceV2Adapter(VoiceClonerBase):
    """Voice-cloning adapter backed by OpenVoice v2 ONNX models.

    Parameters
    ----------
    quantized:
        Use INT8 quantized ONNX models (default ``False``).
    gl_iters:
        Griffin-Lim iterations for vocoding (default 32).  More iterations
        improve quality at the cost of latency.
    **cfg:
        Additional keyword arguments stored but not used.
    """

    _sample_rate = _OV2_SR

    def __init__(
        self,
        quantized: bool = False,
        gl_iters: int = 32,
        **cfg,
    ):
        super().__init__(**cfg)
        self._quantized = quantized
        self._gl_iters = gl_iters
        self._ref_enc_sess = None
        self._converter_sess = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_models(self) -> None:
        if self._ref_enc_sess is not None:
            return

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for engine='openvoice'. "
                "Install it with: pip install vconnx[openvoice]"
            ) from exc

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("huggingface_hub is required.") from exc

        ref_file = _REF_ENC_INT8 if self._quantized else _REF_ENC_FP32
        conv_file = _CONVERTER_INT8 if self._quantized else _CONVERTER_FP32

        ref_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=ref_file)
        conv_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=conv_file)

        sess_opts = ort.SessionOptions()
        n_threads = os.cpu_count() or 4
        sess_opts.inter_op_num_threads = n_threads
        sess_opts.intra_op_num_threads = n_threads

        providers = ["CPUExecutionProvider"]

        self._ref_enc_sess = ort.InferenceSession(
            ref_path, sess_options=sess_opts, providers=providers
        )
        self._converter_sess = ort.InferenceSession(
            conv_path, sess_options=sess_opts, providers=providers
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_tone_embedding(self, audio: np.ndarray) -> np.ndarray:
        """Compute the 256-dim tone-color embedding for *audio*.

        Parameters
        ----------
        audio:
            Float32 mono waveform at :data:`_OV2_SR`.

        Returns
        -------
        np.ndarray
            Shape (1, 256) tone-color embedding.
        """
        self._ensure_models()
        mel = _compute_mel(audio)              # (n_mels, T)
        mel_batch = mel[np.newaxis, :, :]      # (1, n_mels, T)
        out = self._ref_enc_sess.run(None, {"mel": mel_batch})
        return out[0]  # (1, 256)

    def _convert(
        self,
        src_mel: np.ndarray,
        src_tone: np.ndarray,
        tgt_tone: np.ndarray,
    ) -> np.ndarray:
        """Run the converter ONNX model.

        Parameters
        ----------
        src_mel : (1, n_mels, T) float32
        src_tone : (1, 256) float32
        tgt_tone : (1, 256) float32

        Returns
        -------
        np.ndarray
            Converted mel (1, n_mels, T) float32.
        """
        self._ensure_models()
        out = self._converter_sess.run(
            None,
            {
                "mel": src_mel,
                "src_tone": src_tone,
                "tgt_tone": tgt_tone,
            },
        )
        return out[0]  # (1, n_mels, T)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clone_voice(
        self,
        audio: str,
        reference_voice: str,
        out_path: str,
    ) -> str:
        """Convert *audio* to sound like *reference_voice* using OpenVoice v2.

        Parameters
        ----------
        audio:
            Path to the source WAV file.
        reference_voice:
            Path to the reference speaker WAV file.
        out_path:
            Destination path for the 16-bit 22050 Hz output WAV.

        Returns
        -------
        str
            Absolute path to the written output file.
        """
        src_wav = _load_wav(str(audio), target_sr=_OV2_SR)
        ref_wav = _load_wav(str(reference_voice), target_sr=_OV2_SR)

        # Extract tone-color embeddings from both utterances
        src_tone = self._extract_tone_embedding(src_wav)   # (1, 256)
        tgt_tone = self._extract_tone_embedding(ref_wav)   # (1, 256)

        # Compute source mel
        src_mel = _compute_mel(src_wav)[np.newaxis, :, :]  # (1, n_mels, T)

        # Run converter
        converted_mel = self._convert(src_mel, src_tone, tgt_tone)  # (1, n_mels, T)

        # Vocode mel → waveform via Griffin-Lim
        waveform = _griffin_lim(
            converted_mel[0],  # (n_mels, T)
            sr=_OV2_SR,
            n_iter=self._gl_iters,
        )

        out_path = str(Path(out_path).resolve())
        _save_wav(out_path, waveform, sr=_OV2_SR)
        return out_path


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_engine(
    EngineEntry(
        alias="openvoice",
        adapter_class=OpenVoiceV2Adapter,
        description=(
            "OpenVoice v2 tone-color converter: reference encoder (mel → 256-dim "
            "tone-color embedding) + flow-based converter (AdaIN conditioned) + "
            "Griffin-Lim vocoder.  Zero-shot any-to-any VC at 22050 Hz. "
            "ONNX artifacts from TigreGotico/vconnx-models. "
            "(myshell-ai/OpenVoice, MIT license)"
        ),
        extras="openvoice",
        onnx_native=True,
    )
)
