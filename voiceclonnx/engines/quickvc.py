"""QuickVC adapter for voiceclonnx.

QuickVC (quickvc/QuickVC-VoiceConversion, MIT) is an any-to-many voice-conversion
system built on HuBERT-soft content features and a VITS-style normalising-flow
decoder with a Multistream-iSTFT generator head.  It is notably fast on CPU
(~0.2× RTF) because the final synthesis uses a tiny ISTFT (n_fft=16, hop=4)
over 4 subbands rather than a full-resolution upsampling network.

Pipeline:

1. **HuBERT-soft content encoder** — converts 16 kHz audio to (B, 256, T)
   content features at 50 Hz.  Exported to ONNX (``quickvc_content_encoder.onnx``).
2. **Speaker encoder (LSTM)** — converts an 80-channel log-mel spectrogram of the
   reference audio to a 256-dim d-vector.  Exported to ONNX
   (``quickvc_speaker_encoder.onnx``).
3. **Posterior encoder + normalising flow + Multistream-iSTFT pre-ISTFT** — maps
   (content features, speaker d-vector) to per-subband (magnitude, phase) STFT
   coefficients.  Exported to ONNX (``quickvc_decoder.onnx``).
4. **numpy Multistream-iSTFT** — pure-numpy center-mode OLA over 4 subbands.
   Not in ONNX because ``torch.istft`` cannot be traced; parity vs torch = 0.0
   max abs error (exactly reproducible from numpy).
5. **Postnet** — learned 1-D subband mixing (``updown_filter`` + 1-D conv).
   Exported to ONNX (``quickvc_postnet.onnx``).

Runtime requirements: ``onnxruntime``, ``numpy``, ``soundfile``.

Requires: ``pip install voiceclonnx``
  -> onnxruntime, numpy, soundfile, huggingface_hub

References
----------
- https://github.com/quickvc/QuickVC-VoiceConversion  (MIT)
- https://github.com/bshall/hubert  (MIT)
- https://huggingface.co/TigreGotico/voiceclonnx-quickvc
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from voiceclonnx.engines.base import EngineEntry, VoiceClonerBase, register_engine

# QuickVC operates at 16 kHz
_QVC_SR = 16000

# HF repo housing the exported ONNX artifacts
_HF_REPO_ID = "TigreGotico/voiceclonnx-quickvc"

# File names within the HF repo
_ENC_FP32 = "quickvc_content_encoder.onnx"
_ENC_INT8 = "quickvc_content_encoder_q8.onnx"
_SPK_FP32 = "quickvc_speaker_encoder.onnx"
_SPK_INT8 = "quickvc_speaker_encoder_q8.onnx"
_DEC_FP32 = "quickvc_decoder.onnx"
_DEC_INT8 = "quickvc_decoder_q8.onnx"
_POST_FP32 = "quickvc_postnet.onnx"
_POST_INT8 = "quickvc_postnet_q8.onnx"
_CONFIG = "config.json"

# Multistream-iSTFT parameters (hard-coded from configs/quickvc.json)
_GEN_ISTFT_N_FFT = 16
_GEN_ISTFT_HOP_SIZE = 4
_SUBBANDS = 4

# HuBERT-soft content encoder: fixed 1-second context window in the ONNX graph.
# PyTorch MHA batch_first=True reshapes freeze T in the traced graph; audio is
# split into non-overlapping 1-second chunks and features are concatenated.
_CHUNK_SAMPLES = _QVC_SR  # 16000 samples = 1 second

# Log-mel parameters for the speaker encoder
_MEL_N_FFT = 1280
_MEL_HOP_LENGTH = 320
_MEL_WIN_LENGTH = 1280
_MEL_N_MELS = 80
_MEL_FMIN = 0.0
_MEL_FMAX = None   # no upper cutoff


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------


def _load_wav(path: str, target_sr: int = _QVC_SR) -> np.ndarray:
    """Load *path* as float32 mono, resampled to *target_sr*."""
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
    """Write *audio* float32 as 16-bit PCM WAV."""
    import soundfile as sf

    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    sf.write(str(path), audio_int16, sr, subtype="PCM_16")


# ---------------------------------------------------------------------------
# Log-mel spectrogram (pure numpy, HTK-scale, no librosa at runtime)
# ---------------------------------------------------------------------------


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    """HTK mel scale."""
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_filterbank(
    sr: int,
    n_fft: int,
    n_mels: int,
    fmin: float,
    fmax: Optional[float],
) -> np.ndarray:
    """Return (n_mels, n_fft//2+1) HTK-scale mel filterbank weights."""
    if fmax is None:
        fmax = sr / 2.0
    n_freqs = n_fft // 2 + 1
    freqs = np.linspace(0.0, sr / 2.0, n_freqs)
    mel_min = _hz_to_mel(np.array([fmin]))[0]
    mel_max = _hz_to_mel(np.array([fmax]))[0]
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    # Convert mel centre points back to Hz (for lookup in freqs)
    hz_pts = 700.0 * (10.0 ** (mel_pts / 2595.0) - 1.0)

    fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
    for m in range(n_mels):
        lo, ctr, hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        for k, f in enumerate(freqs):
            if lo <= f <= ctr:
                fb[m, k] = (f - lo) / (ctr - lo + 1e-8)
            elif ctr < f <= hi:
                fb[m, k] = (hi - f) / (hi - ctr + 1e-8)
    return fb


# Cache filterbank so it is built only once per process
_mel_fb_cache: Optional[np.ndarray] = None


def _get_mel_fb() -> np.ndarray:
    global _mel_fb_cache
    if _mel_fb_cache is None:
        _mel_fb_cache = _mel_filterbank(
            sr=_QVC_SR,
            n_fft=_MEL_N_FFT,
            n_mels=_MEL_N_MELS,
            fmin=_MEL_FMIN,
            fmax=_MEL_FMAX,
        )
    return _mel_fb_cache


def _log_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """Compute log-mel spectrogram matching QuickVC's mel_spectrogram_torch.

    Parameters
    ----------
    audio:
        (N,) float32 waveform at 16 kHz.

    Returns
    -------
    np.ndarray
        (1, n_mels, T) float32 log-mel spectrogram, where T = len(audio) // hop.
    """
    n_fft = _MEL_N_FFT
    hop = _MEL_HOP_LENGTH
    win = _MEL_WIN_LENGTH

    # Reflect-pad to match mel_spectrogram_torch (center=False, pad=(n_fft-hop)//2)
    pad = (n_fft - hop) // 2
    audio_padded = np.pad(audio, (pad, pad), mode="reflect")

    # Hann window
    window = np.hanning(win + 1)[:-1].astype(np.float32)

    # STFT frame extraction
    n_frames = (len(audio_padded) - n_fft) // hop + 1
    frames = np.stack(
        [audio_padded[i * hop : i * hop + n_fft] for i in range(n_frames)],
        axis=0,
    )  # (T, n_fft)
    frames = frames * window[np.newaxis, :]

    # FFT → magnitude
    spec = np.abs(np.fft.rfft(frames, n=n_fft, axis=1)).astype(np.float32)  # (T, half)

    # Apply mel filterbank
    fb = _get_mel_fb()  # (n_mels, half)
    mel = spec @ fb.T   # (T, n_mels)

    # Log-magnitude (spectral_normalize_torch: log(clamp(x, 1e-5)))
    mel = np.log(np.maximum(mel, 1e-5))

    return mel[np.newaxis, :, :]  # (1, T, n_mels) — match enc_spk input


# ---------------------------------------------------------------------------
# numpy Multistream-iSTFT
# ---------------------------------------------------------------------------


def _numpy_ms_istft(
    spec_phase: np.ndarray,
    n_fft: int = _GEN_ISTFT_N_FFT,
    hop: int = _GEN_ISTFT_HOP_SIZE,
    subbands: int = _SUBBANDS,
) -> np.ndarray:
    """Pure-numpy Multistream-iSTFT matching TorchSTFT.inverse (center=True OLA).

    Parameters
    ----------
    spec_phase:
        (B*subbands, 2, half, T_dec) where half = n_fft//2+1,
        dim-1 index 0 = magnitude, 1 = phase (radians).
    n_fft, hop, subbands:
        iSTFT parameters from the QuickVC config.

    Returns
    -------
    np.ndarray
        (B, subbands, T_audio) float32 per-subband waveforms, where
        T_audio = (T_dec - 1) * hop.
    """
    BS, two, half, T_dec = spec_phase.shape
    assert two == 2
    B = BS // subbands
    win_length = n_fft

    window = np.hanning(win_length + 1)[:-1].astype(np.float32)
    out_len = (T_dec - 1) * hop
    raw_len = (T_dec - 1) * hop + n_fft
    pad = n_fft // 2

    results = []
    for b in range(BS):
        mag = spec_phase[b, 0, :, :]    # (half, T_dec)
        phase = spec_phase[b, 1, :, :]  # (half, T_dec)

        stft_c = (mag * np.exp(1j * phase)).astype(np.complex64)  # (half, T_dec)
        frames = np.fft.irfft(stft_c.T, n=n_fft, axis=1)          # (T_dec, n_fft)
        frames = frames * window[np.newaxis, :]

        out_b = np.zeros(raw_len, dtype=np.float32)
        win_sq = np.zeros(raw_len, dtype=np.float32)
        for t in range(T_dec):
            start = t * hop
            out_b[start : start + win_length] += frames[t]
            win_sq[start : start + win_length] += window ** 2

        out_b /= np.maximum(win_sq, 1e-8)
        results.append(out_b[pad : pad + out_len])

    # Reshape from (B*subbands, T_audio) to (B, subbands, T_audio)
    all_subs = np.stack(results, axis=0)           # (B*subbands, T_audio)
    return all_subs.reshape(B, subbands, -1)       # (B, subbands, T_audio)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class QuickVCAdapter(VoiceClonerBase):
    """Voice-cloning adapter backed by QuickVC ONNX models.

    Pipeline:
    1. Encode source audio with the HuBERT-soft content encoder.
    2. Compute log-mel spectrogram of the reference audio; encode with the
       speaker encoder to get a 256-dim d-vector.
    3. Run the VITS decoder (posterior encoder + normalising flow +
       Multistream-iSTFT pre-ISTFT generator) to produce subband STFT coefficients.
    4. Apply pure-numpy Multistream-iSTFT to reconstruct per-subband waveforms.
    5. Apply postnet (subband upsampling + mixing conv via ONNX) to produce
       the final mono waveform.

    Parameters
    ----------
    quantized:
        Use INT8 quantized ONNX models (default ``False``).
    **cfg:
        Additional keyword arguments stored but not used at runtime.
    """

    _sample_rate = _QVC_SR

    def __init__(self, quantized: bool = False, **cfg):
        super().__init__(**cfg)
        self._quantized = quantized
        self._enc_sess = None
        self._spk_sess = None
        self._dec_sess = None
        self._post_sess = None

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
                "onnxruntime is required for engine='quickvc'. "
                "Install it with: pip install voiceclonnx"
            ) from exc

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("huggingface_hub is required.") from exc

        enc_file = _ENC_INT8 if self._quantized else _ENC_FP32
        spk_file = _SPK_INT8 if self._quantized else _SPK_FP32
        dec_file = _DEC_INT8 if self._quantized else _DEC_FP32
        post_file = _POST_INT8 if self._quantized else _POST_FP32

        enc_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=enc_file)
        spk_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=spk_file)
        dec_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=dec_file)
        post_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=post_file)

        sess_opts = ort.SessionOptions()
        n_threads = os.cpu_count() or 4
        sess_opts.inter_op_num_threads = n_threads
        sess_opts.intra_op_num_threads = n_threads
        providers = ["CPUExecutionProvider"]

        self._enc_sess = ort.InferenceSession(enc_path, sess_options=sess_opts, providers=providers)
        self._spk_sess = ort.InferenceSession(spk_path, sess_options=sess_opts, providers=providers)
        self._dec_sess = ort.InferenceSession(dec_path, sess_options=sess_opts, providers=providers)
        self._post_sess = ort.InferenceSession(post_path, sess_options=sess_opts, providers=providers)

    # ------------------------------------------------------------------
    # Encode content: audio → (1, 256, T) features
    # ------------------------------------------------------------------

    def _encode_content(self, audio: np.ndarray) -> np.ndarray:
        """Run HuBERT-soft content encoder in 1-second chunks: → (1, 256, T).

        The ONNX graph has a fixed 1-second (16000-sample) context window due to
        PyTorch MHA attention reshapes that cannot be made fully dynamic.  Audio
        longer than 16000 samples is split into non-overlapping 1-second chunks,
        each encoded independently, and the feature frames are concatenated.
        """
        self._ensure_models()
        chunk_size = _QVC_SR  # 16000 samples = 1 second
        all_feats = []
        for start in range(0, len(audio), chunk_size):
            chunk = audio[start : start + chunk_size]
            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            inp = chunk[np.newaxis, np.newaxis, :].astype(np.float32)  # (1, 1, 16000)
            feats = self._enc_sess.run(None, {"wav": inp})[0]           # (1, 256, 50)
            all_feats.append(feats)
        return np.concatenate(all_feats, axis=2)  # (1, 256, T)

    # ------------------------------------------------------------------
    # Encode speaker: mel → (1, 256, 1) d-vector
    # ------------------------------------------------------------------

    def _encode_speaker(self, audio: np.ndarray) -> np.ndarray:
        """Compute log-mel and run speaker encoder: → (1, 256, 1)."""
        self._ensure_models()
        mel = _log_mel_spectrogram(audio)    # (1, T, 80)
        return self._spk_sess.run(None, {"mel": mel})[0]  # (1, 256, 1)

    # ------------------------------------------------------------------
    # Decode: content + speaker → (B, subbands, T_audio)
    # ------------------------------------------------------------------

    def _decode(
        self,
        c: np.ndarray,
        g: np.ndarray,
    ) -> np.ndarray:
        """Run decoder → numpy MS-iSTFT → postnet → (1,) waveform."""
        self._ensure_models()
        T = c.shape[2]
        c_lengths = np.array([T], dtype=np.int64)

        # Decoder: spec_phase (B*sub, 2, half, T_dec)
        spec_phase = self._dec_sess.run(
            None,
            {"c": c, "c_lengths": c_lengths, "g": g},
        )[0]

        # numpy Multistream-iSTFT → (B, subbands, T_audio)
        y_mb = _numpy_ms_istft(spec_phase)  # (1, 4, T_audio)

        # Postnet: (1, 4, T_audio) → (1, 1, T_out)
        y_hat = self._post_sess.run(None, {"y_mb_hat": y_mb})[0]  # (1, 1, T_out)
        return y_hat[0, 0]  # (T_out,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clone_voice(
        self,
        audio: str,
        reference_voice: str,
        out_path: str,
    ) -> str:
        """Convert *audio* to sound like *reference_voice* using QuickVC.

        Parameters
        ----------
        audio:
            Path to the source WAV file (16 kHz mono preferred).
        reference_voice:
            Path to the reference speaker WAV file.
        out_path:
            Destination path for the 16-bit 16 kHz output WAV.

        Returns
        -------
        str
            Absolute path to the written output file.
        """
        src_wav = _load_wav(str(audio), target_sr=_QVC_SR)
        ref_wav = _load_wav(str(reference_voice), target_sr=_QVC_SR)

        # Encode content features from source
        c = self._encode_content(src_wav)      # (1, 256, T_src)

        # Encode speaker d-vector from reference mel
        g = self._encode_speaker(ref_wav)      # (1, 256, 1)

        # Decode to waveform
        waveform = self._decode(c, g)          # (T_out,)

        out_path = str(Path(out_path).resolve())
        _save_wav(out_path, waveform, sr=_QVC_SR)
        return out_path


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_engine(
    EngineEntry(
        alias="quickvc",
        adapter_class=QuickVCAdapter,
        description=(
            "QuickVC: HuBERT-soft content encoder + VITS normalising-flow decoder "
            "with Multistream-iSTFT generator.  Any-to-many VC at 16 kHz.  "
            "Notably fast on CPU (~0.2× RTF) due to tiny subband iSTFT (n_fft=16). "
            "ONNX artifacts from TigreGotico/voiceclonnx-quickvc.  (MIT)"
        ),
        extras="",
        onnx_native=True,
    )
)
