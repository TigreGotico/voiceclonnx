"""FreeVC adapter for voiceclonnx.

FreeVC (Qian et al., ICASSP 2023) is a zero-shot any-to-any voice conversion
system.  At inference it uses:

1. **WavLM-Large encoder** (full final hidden states) — converts 16 kHz audio
   to 1024-dim feature frames at 50 Hz.  Note: this is the transformer's
   complete final output, unlike kNN-VC which extracts only layer 6.
2. **Speaker encoder** (GE2E LSTM) — extracts a 256-dim d-vector from a
   log-mel spectrogram of the reference audio.
3. **VITS decoder** (SynthesizerTrn) — maps content features + d-vector to a
   16 kHz waveform via a normalising flow + HiFi-GAN generator.

All components run via onnxruntime.  Zero torch dependency at inference.

ONNX artifacts: ``TigreGotico/voiceclonnx-freevc`` (MIT license).

WavLM artifact note:
  ``wavlm_freevc.onnx`` is a separate export from the kNN-VC
  ``wavlm_layer6.onnx`` in ``TigreGotico/voiceclonnx-knn-vc``.  They both use
  WavLM-Large but extract different outputs and are NOT interchangeable.

Chunking note:
  WavLM-Large and the VITS decoder both produce degraded output on sequences
  longer than ~5 seconds when run as ONNX models.  ``clone_voice`` therefore
  processes source audio in overlapping chunks (``_CHUNK_SECONDS``) and
  crossfades the waveform segments back together.

Requires: ``pip install voiceclonnx``
  -> onnxruntime, numpy, soundfile, huggingface_hub

Log-mel note:
  The GE2E speaker encoder expects a 40-bin log-mel spectrogram.  This is
  computed with a pure-numpy HTK-scale mel filterbank + Hann-window STFT so
  that librosa is NOT required at inference.

References
----------
- https://github.com/OlaWod/FreeVC
- https://arxiv.org/abs/2210.15418
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from voiceclonnx.engines.base import EngineEntry, VoiceClonerBase, register_engine

# FreeVC outputs 16 kHz audio
_FREEVC_SR = 16000

# Mel-spectrogram parameters for the speaker encoder (GE2E)
_MEL_N_CHANNELS = 40
_MEL_WINDOW_MS = 25       # window length in milliseconds
_MEL_STEP_MS = 10         # hop length in milliseconds
_MEL_FMIN = 40.0          # minimum frequency for mel filterbank
_MEL_FMAX = 8000.0        # maximum frequency for mel filterbank

# Chunked inference parameters.
# WavLM-Large and the VITS decoder both degrade on sequences longer than ~3 s
# when run as ONNX models.  2-second chunks with 0.25 s crossfade overlap give
# ≤ 10% WER in practice; larger chunks (3–5 s) score significantly worse.
_CHUNK_SECONDS = 2.0      # maximum chunk length fed to WavLM + decoder
_OVERLAP_SECONDS = 0.25   # crossfade overlap between adjacent chunks

# HF repo housing the exported ONNX artifacts
_HF_REPO_ID = "TigreGotico/voiceclonnx-freevc"

# File names within the HF repo
_WAVLM_FP32 = "wavlm_freevc.onnx"
_WAVLM_INT8 = "wavlm_freevc_q8.onnx"
_SPK_FP32 = "speaker_encoder.onnx"
_SPK_INT8 = "speaker_encoder_q8.onnx"
_DECODER_FP32 = "freevc_decoder.onnx"
_DECODER_INT8 = "freevc_decoder_q8.onnx"


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------


def _load_wav(path: str, target_sr: int = _FREEVC_SR) -> np.ndarray:
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
# Log-mel spectrogram for speaker encoder
# ---------------------------------------------------------------------------


def _mel_filterbank(
    n_fft: int,
    n_mels: int,
    sr: int,
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Build an HTK-scale mel filterbank matrix of shape (n_mels, n_fft//2+1)."""

    def _hz2mel(hz: float) -> float:
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def _mel2hz(mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    n_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0.0, sr / 2.0, n_bins)

    mel_min = _hz2mel(fmin)
    mel_max = _hz2mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    freq_points = np.array([_mel2hz(m) for m in mel_points])

    fb = np.zeros((n_mels, n_bins), dtype=np.float64)
    for m in range(n_mels):
        lo, center, hi = freq_points[m], freq_points[m + 1], freq_points[m + 2]
        up = (fft_freqs >= lo) & (fft_freqs <= center)
        down = (fft_freqs > center) & (fft_freqs <= hi)
        fb[m, up] = (fft_freqs[up] - lo) / (center - lo + 1e-10)
        fb[m, down] = (hi - fft_freqs[down]) / (hi - center + 1e-10)
    return fb.astype(np.float32)


def _compute_log_mel(
    audio: np.ndarray,
    sr: int = _FREEVC_SR,
    n_mels: int = _MEL_N_CHANNELS,
    window_ms: float = _MEL_WINDOW_MS,
    step_ms: float = _MEL_STEP_MS,
    fmin: float = _MEL_FMIN,
    fmax: float = _MEL_FMAX,
) -> np.ndarray:
    """Compute log-mel spectrogram compatible with the GE2E speaker encoder.

    Pure-numpy implementation (Hann-window STFT + HTK mel filterbank).
    Output matches librosa.feature.melspectrogram + librosa.power_to_db
    to within the precision of the mel filterbank discretization.

    Returns
    -------
    np.ndarray
        (n_frames, n_mels) float32 — log-mel spectrogram.
    """
    n_fft = int(sr * window_ms / 1000)
    hop_length = int(sr * step_ms / 1000)

    window = np.hanning(n_fft).astype(np.float32)
    pad = n_fft // 2
    audio_padded = np.pad(audio, (pad, pad), mode="reflect")

    n_frames = 1 + (len(audio_padded) - n_fft) // hop_length
    frames = np.stack(
        [audio_padded[i * hop_length: i * hop_length + n_fft] for i in range(n_frames)],
        axis=0,
    )  # (n_frames, n_fft)

    windowed = frames * window[np.newaxis, :]
    spec = np.fft.rfft(windowed, n=n_fft, axis=1)  # (n_frames, n_fft//2+1)
    power = (spec.real ** 2 + spec.imag ** 2).astype(np.float32)  # power spectrogram

    fb = _mel_filterbank(n_fft, n_mels, sr, fmin, fmax)  # (n_mels, n_fft//2+1)
    mel = (fb @ power.T).T  # (n_frames, n_mels)

    # power_to_db: 10 * log10(S / ref) where ref = S.max()
    ref = mel.max() if mel.max() > 0 else 1.0
    log_mel = (10.0 * np.log10(np.maximum(mel, 1e-10) / ref)).astype(np.float32)
    return log_mel  # (n_frames, n_mels)


# ---------------------------------------------------------------------------
# Crossfade helper
# ---------------------------------------------------------------------------


def _crossfade_segments(
    segments: list,
    overlap_samples: int,
) -> np.ndarray:
    """Blend a list of (source_offset, waveform) pairs into one waveform.

    Adjacent segments overlap by *overlap_samples*.  The overlap region is
    blended with a linear crossfade so that the seam is inaudible.

    Parameters
    ----------
    segments:
        List of ``(src_offset, wav)`` tuples in chronological order.
        *src_offset* is the position in the original source audio (samples)
        from which *wav* was generated.
    overlap_samples:
        Number of samples that consecutive decoded segments share.

    Returns
    -------
    np.ndarray
        (N,) float32 blended waveform.
    """
    wavs = [s.astype(np.float32) for _, s in segments]
    n_segs = len(wavs)

    if n_segs == 0:
        return np.zeros(0, dtype=np.float32)
    if n_segs == 1:
        return wavs[0]

    fade_out = np.linspace(1.0, 0.0, overlap_samples, dtype=np.float32)
    fade_in  = np.linspace(0.0, 1.0, overlap_samples, dtype=np.float32)

    # Total length: each consecutive pair loses `overlap_samples`
    total = sum(len(w) for w in wavs) - overlap_samples * (n_segs - 1)
    out = np.zeros(total, dtype=np.float32)

    write_pos = 0
    prev_tail: Optional[np.ndarray] = None  # tail of previous segment for blending

    for i, seg in enumerate(wavs):
        n = len(seg)
        is_last = (i == n_segs - 1)

        if i == 0:
            # Write the body of the first segment (all but the tail overlap)
            body_len = n - overlap_samples
            out[write_pos : write_pos + body_len] = seg[:body_len]
            write_pos += body_len
            prev_tail = seg[body_len:]          # (overlap_samples,)
        else:
            # Crossfade: blend prev_tail with the head of this segment
            head = seg[:overlap_samples]
            blend = fade_out * prev_tail + fade_in * head
            out[write_pos : write_pos + overlap_samples] = blend
            write_pos += overlap_samples

            if is_last:
                # Write the rest of the last segment
                rest = seg[overlap_samples:]
                out[write_pos : write_pos + len(rest)] = rest
                write_pos += len(rest)
            else:
                # Write body between the head overlap and the tail overlap
                body = seg[overlap_samples : n - overlap_samples]
                out[write_pos : write_pos + len(body)] = body
                write_pos += len(body)
                prev_tail = seg[n - overlap_samples:]

    return out


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class FreeVCAdapter(VoiceClonerBase):
    """Voice-cloning adapter backed by FreeVC ONNX models.

    Parameters
    ----------
    quantized:
        Use INT8 quantized ONNX models (default ``False``).
        **Not recommended for production** — INT8 degrades WER significantly
        (benchmark: 62% int8 vs 12% fp32).  Prefer fp32.
        See demo/QUANTS.md for the full comparison.
    **cfg:
        Additional keyword arguments stored but not used.
    """

    _sample_rate = _FREEVC_SR

    def __init__(
        self,
        quantized: bool = False,
        **cfg,
    ):
        super().__init__(**cfg)
        self._quantized = quantized
        self._wavlm_sess = None
        self._spk_sess = None
        self._decoder_sess = None

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
                "onnxruntime is required for engine='freevc'. "
                "Install it with: pip install voiceclonnx[freevc]"
            ) from exc

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("huggingface_hub is required.") from exc

        wavlm_file = _WAVLM_INT8 if self._quantized else _WAVLM_FP32
        spk_file = _SPK_INT8 if self._quantized else _SPK_FP32
        decoder_file = _DECODER_INT8 if self._quantized else _DECODER_FP32

        wavlm_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=wavlm_file)
        spk_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=spk_file)
        decoder_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=decoder_file)

        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = os.cpu_count() or 4
        sess_opts.intra_op_num_threads = os.cpu_count() or 4

        providers = ["CPUExecutionProvider"]

        self._wavlm_sess = ort.InferenceSession(
            wavlm_path, sess_options=sess_opts, providers=providers
        )
        self._spk_sess = ort.InferenceSession(
            spk_path, sess_options=sess_opts, providers=providers
        )
        self._decoder_sess = ort.InferenceSession(
            decoder_path, sess_options=sess_opts, providers=providers
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_content(self, audio: np.ndarray) -> np.ndarray:
        """Run WavLM encoder and return (T, 1024) content feature array."""
        self._ensure_models()
        inp = audio[np.newaxis, :].astype(np.float32)
        out = self._wavlm_sess.run(None, {"input_values": inp})
        # Output: (1, frames, 1024) → (1024, frames) for decoder input
        return out[0][0]  # (frames, 1024)

    def _extract_speaker(self, audio: np.ndarray) -> np.ndarray:
        """Run speaker encoder and return (256,) d-vector."""
        self._ensure_models()
        log_mel = _compute_log_mel(audio)  # (n_frames, 40)
        inp = log_mel[np.newaxis, :, :].astype(np.float32)  # (1, n_frames, 40)
        out = self._spk_sess.run(None, {"mels": inp})
        return out[0][0]  # (256,)

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------

    def _decode(self, content_features: np.ndarray, speaker_emb: np.ndarray) -> np.ndarray:
        """Run VITS decoder and return (samples,) float32 waveform.

        Parameters
        ----------
        content_features:
            (frames, 1024) — WavLM features from source audio.
        speaker_emb:
            (256,) — d-vector from reference audio.
        """
        self._ensure_models()
        # Decoder expects c=(1, 1024, frames), g=(1, 256)
        c = content_features.T[np.newaxis, :, :].astype(np.float32)  # (1, 1024, T)
        g = speaker_emb[np.newaxis, :].astype(np.float32)             # (1, 256)
        out = self._decoder_sess.run(None, {"c": c, "g": g})
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
        """Convert *audio* to sound like *reference_voice* using FreeVC.

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
        src_wav = _load_wav(str(audio), target_sr=_FREEVC_SR)
        ref_wav = _load_wav(str(reference_voice), target_sr=_FREEVC_SR)

        # Extract speaker embedding from reference (uses the full reference clip)
        speaker_emb = self._extract_speaker(ref_wav)      # (256,)

        # Process source in chunks to avoid WavLM + VITS decoder degradation
        # on long sequences.  Chunks overlap by _OVERLAP_SECONDS; the overlap
        # region is blended with a linear crossfade to hide the seam.
        chunk_samples = int(_CHUNK_SECONDS * _FREEVC_SR)
        overlap_samples = int(_OVERLAP_SECONDS * _FREEVC_SR)
        step_samples = chunk_samples - overlap_samples

        n_src = len(src_wav)
        if n_src <= chunk_samples:
            # Short audio — single pass, no chunking needed
            content_feats = self._extract_content(src_wav)
            waveform = self._decode(content_feats, speaker_emb)
        else:
            segments: list = []
            pos = 0
            while pos < n_src:
                end = min(pos + chunk_samples, n_src)
                chunk = src_wav[pos:end]
                content = self._extract_content(chunk)   # (T_c, 1024)
                seg_wav = self._decode(content, speaker_emb)  # (S_c,)
                segments.append((pos, seg_wav))
                if end == n_src:
                    break
                pos += step_samples

            # Crossfade-blend overlapping segments into one waveform
            waveform = _crossfade_segments(segments, overlap_samples)

        out_path = str(Path(out_path).resolve())
        _save_wav(out_path, waveform, sr=_FREEVC_SR)
        return out_path


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_engine(
    EngineEntry(
        alias="freevc",
        adapter_class=FreeVCAdapter,
        description=(
            "FreeVC: WavLM-Large full encoder + GE2E speaker encoder "
            "+ VITS decoder. Zero-shot any-to-any VC at 16 kHz. "
            "ONNX artifacts from TigreGotico/voiceclonnx-freevc. "
            "(Qian et al., ICASSP 2023, MIT license)"
        ),
        extras="",
        onnx_native=True,
    )
)
