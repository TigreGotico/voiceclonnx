"""OpenVoice v2 tone-color converter adapter for voiceclonnx.

OpenVoice v2 (myshell-ai/OpenVoice, MIT license) is a zero-shot voice-cloning
library.  The **tone-color converter** sub-component is a standalone
audio-to-audio VC path: it transplants the speaker timbre from a reference
utterance onto a source utterance without requiring TTS text-encoders.

Architecture (ONNX inference path):
  1. **Reference encoder** (``tone_ref_encoder.onnx``) -- linear spectrogram
     ``(B, T, 513)`` -> 256-dim tone-color embedding.  Run on both source and
     reference audio.
  2. **Converter** (``tone_converter.onnx``) -- ``(spec[B,513,T], spec_lengths,
     src_g[B,256,1], tgt_g[B,256,1])`` -> raw waveform ``(B, 1, samples)``.
     A full VITS-style flow decoder with HiFi-GAN vocoder inside.  No separate
     vocoder step needed.

Both components are exported from the upstream ``SynthesizerTrn`` (myshell-ai/
OpenVoice) with strict state-dict loading -- no reconstruction.

Preprocessing (pure numpy, matches upstream ``spectrogram_torch``):
  - Hann window STFT, n_fft=1024, hop=256, win=1024
  - Padding: (n_fft - hop) // 2 = 384 samples on each side (reflect)
  - Magnitude: ``sqrt(Re^2 + Im^2 + 1e-6)`` (NOT log-compressed)
  - Shape passed to ref_enc: ``(1, T, 513)``
  - Shape passed to converter: ``(1, 513, T)``

All neural components run via onnxruntime.  The STFT is pure numpy.

Requires: ``pip install voiceclonnx``
  -> onnxruntime, numpy, soundfile, huggingface_hub

References
----------
- https://github.com/myshell-ai/OpenVoice
- https://huggingface.co/myshell-ai/OpenVoiceV2
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from voiceclonnx.engines.base import EngineEntry, VoiceClonerBase, register_engine

# OpenVoice v2 operates at 22050 Hz
_OV2_SR = 22050

# HF repo housing the exported ONNX artifacts
_HF_REPO_ID = "TigreGotico/voiceclonnx-openvoice-v2"

# Paths within the HF repo
_REF_ENC_FP32 = "tone_ref_encoder.onnx"
_REF_ENC_INT8 = "tone_ref_encoder_q8.onnx"
_CONVERTER_FP32 = "tone_converter.onnx"
_CONVERTER_INT8 = "tone_converter_q8.onnx"

# STFT parameters matching upstream mel_processing.spectrogram_torch
_N_FFT = 1024
_HOP_LENGTH = 256
_WIN_LENGTH = 1024
_SPEC_CHANNELS = 513  # n_fft // 2 + 1

# Tone-color embedding dimension
_TONE_DIM = 256


# ---------------------------------------------------------------------------
# Linear spectrogram extraction (pure numpy, matches upstream spectrogram_torch)
# ---------------------------------------------------------------------------


def _compute_linear_spec(
    audio: np.ndarray,
    n_fft: int = _N_FFT,
    hop_length: int = _HOP_LENGTH,
    win_length: int = _WIN_LENGTH,
) -> np.ndarray:
    """Compute linear magnitude spectrogram matching upstream ``spectrogram_torch``.

    Upstream applies padding ``(n_fft - hop_size) / 2`` on each side (reflect),
    then ``sqrt(Re^2 + Im^2 + 1e-6)``.

    Parameters
    ----------
    audio:
        Float32 mono waveform.

    Returns
    -------
    np.ndarray
        Shape ``(spec_channels, T)`` float32, where
        ``spec_channels = n_fft // 2 + 1 = 513``.
    """
    pad = (n_fft - hop_length) // 2  # 384
    audio_padded = np.pad(audio, pad, mode="reflect")

    window = np.hanning(win_length).astype(np.float32)

    n_frames = 1 + (len(audio_padded) - n_fft) // hop_length
    frames = np.stack(
        [audio_padded[i * hop_length: i * hop_length + n_fft] for i in range(n_frames)],
        axis=0,
    )  # (n_frames, n_fft)

    windowed = frames * window[np.newaxis, :]
    spec_cplx = np.fft.rfft(windowed, n=n_fft, axis=1)  # (n_frames, n_fft//2+1)
    mag = np.sqrt(spec_cplx.real ** 2 + spec_cplx.imag ** 2 + 1e-6)
    return mag.T.astype(np.float32)  # (513, n_frames)


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
# Adapter
# ---------------------------------------------------------------------------


class OpenVoiceV2Adapter(VoiceClonerBase):
    """Voice-cloning adapter backed by OpenVoice v2 ONNX models.

    The reference encoder and voice-conversion models are exported from the
    upstream ``SynthesizerTrn`` (myshell-ai/OpenVoice) with strict state-dict
    loading.  The converter outputs raw waveform directly -- no separate vocoder.

    Parameters
    ----------
    quantized:
        Use INT8 quantized ONNX models (default ``False``).
    **cfg:
        Additional keyword arguments stored but not used.
    """

    _sample_rate = _OV2_SR

    def __init__(
        self,
        quantized: bool = False,
        **cfg,
    ):
        super().__init__(**cfg)
        self._quantized = quantized
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
                "Install it with: pip install voiceclonnx[openvoice]"
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
            Shape ``(1, 256)`` tone-color embedding.
        """
        self._ensure_models()
        spec = _compute_linear_spec(audio)           # (513, T)
        spec_t = spec.T[np.newaxis, :, :]            # (1, T, 513) for ref_enc
        out = self._ref_enc_sess.run(None, {"spec": spec_t})
        return out[0]  # (1, 256)

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

        # Extract tone-color embeddings
        src_tone = self._extract_tone_embedding(src_wav)   # (1, 256)
        tgt_tone = self._extract_tone_embedding(ref_wav)   # (1, 256)

        # Compute source linear spectrogram for the converter
        src_spec = _compute_linear_spec(src_wav)           # (513, T)
        src_spec_batch = src_spec[np.newaxis, :, :]        # (1, 513, T)
        T = src_spec.shape[1]

        # Add trailing dim for tone embeddings: (1, 256, 1)
        src_g = src_tone[:, :, np.newaxis]                 # (1, 256, 1)
        tgt_g = tgt_tone[:, :, np.newaxis]                 # (1, 256, 1)

        # Run converter -> raw waveform (1, 1, samples)
        self._ensure_models()
        ort_out = self._converter_sess.run(
            None,
            {
                "spec": src_spec_batch,
                "spec_lengths": np.array([T], dtype=np.int64),
                "src_g": src_g,
                "tgt_g": tgt_g,
            },
        )
        waveform = ort_out[0][0, 0]  # (samples,) float32

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
            "OpenVoice v2 tone-color converter: reference encoder (linear spec -> "
            "256-dim tone-color embedding) + VITS-style flow with HiFi-GAN vocoder. "
            "Zero-shot any-to-any VC at 22050 Hz. "
            "ONNX artifacts from TigreGotico/voiceclonnx-openvoice-v2. "
            "(myshell-ai/OpenVoice, MIT license)"
        ),
        extras="",
        onnx_native=True,
    )
)
