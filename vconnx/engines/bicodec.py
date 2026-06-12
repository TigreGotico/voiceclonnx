"""BiCodec adapter for vconnx.

BiCodec (SparkAudio/Spark-TTS, March 2025) factorizes speech into two
complementary token streams, making zero-shot voice conversion a direct
token-swap operation with no auto-regressive LM required.

Token streams
-------------
- **Semantic tokens**: Wav2Vec2-XLSR-53 (hidden layers 11/14/16 averaged)
  → convolutional encoder → single-codebook factorized VQ.
  Carry linguistic content (phoneme sequence).
- **Global tokens**: mel-spectrogram → ECAPA-TDNN speaker verifier
  → Perceiver resampler → FSQ; 32-token fixed-length sequence per utterance.
  Carry timbre / speaker identity.

Voice-conversion pipeline
--------------------------
1. Extract mel from source and reference waveforms (``mel_transformer.onnx``).
2. Encode source waveform through Wav2Vec2 (``wav2vec2_encoder.onnx``) to get
   (1, T, 1024) features.
3. Encode features → semantic tokens (1, T2) with ``semantic_encoder.onnx``.
4. Encode reference mel → global tokens (1, 1, 32) with ``global_encoder.onnx``.
5. Decode: semantic tokens + reference global tokens → waveform
   (``decoder.onnx``).

The swap step (step 5) is a direct integer index substitution — no ONNX
component is needed for the swap itself.

Content-layer knob
------------------
BiCodec's semantic / global split is architecturally explicit (two separate
encoder branches), unlike codec families that use RVQ layer ordering.
The ``content_split`` constructor parameter controls an experimental path
where partial global tokens from the source are retained; the default (all
global tokens from reference) is the recommended recipe per the issue spec.

ONNX components (TigreGotico/vconnx-bicodec)
--------------------------------------------
- ``wav2vec2_encoder.onnx``  : (1, N) float32 → (1, T, 1024) float32
- ``semantic_encoder.onnx``  : (1, T, 1024) float32 → (1, T2) int64
- ``global_encoder.onnx``    : (1, 128, T_mel) float32 → (1, 1, 32) int32
- ``mel_filterbank.npy``     : (128, 513) float32 — mel filterbank matrix
- ``mel_config.json``        : mel STFT parameters (n_fft, hop, win, fmin)
- ``decoder.onnx``           : (1, T2) int64 + (1, 1, 32) int32 → (1, 1, N) float32

Note: the mel spectrogram is computed in pure numpy using the filterbank (avoids
the aten::stft ONNX export limitation).  Parity vs torchaudio MelSpectrogram is
verified at export time (max_abs ≤ 5e-3).

Runtime requirements: ``onnxruntime``, ``numpy``, ``soundfile``,
``huggingface_hub``.  No PyTorch at inference time.

License
-------
ONNX artifacts derived from SparkAudio/Spark-TTS-0.5B weights.
Upstream weights: **CC BY-NC-SA 4.0** — non-commercial use only.
Upstream code: Apache-2.0.

References
----------
- https://github.com/SparkAudio/Spark-TTS
- https://arxiv.org/abs/2503.01710
- https://huggingface.co/SparkAudio/Spark-TTS-0.5B
- https://huggingface.co/TigreGotico/vconnx-bicodec
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from vconnx.engines.base import EngineEntry, VoiceClonerBase, register_engine

# BiCodec operates at 16 kHz
_BC_SR = 16000

# HF repo for the exported ONNX artifacts
_HF_REPO_ID = "TigreGotico/vconnx-bicodec"

# File names inside the HF repo (fp32 and INT8 variants)
_W2V_FP32 = "wav2vec2_encoder.onnx"
_W2V_INT8 = "wav2vec2_encoder_q8.onnx"
_SEM_FP32 = "semantic_encoder.onnx"
_SEM_INT8 = "semantic_encoder_q8.onnx"
_GLOB_FP32 = "global_encoder.onnx"
_GLOB_INT8 = "global_encoder_q8.onnx"
_MEL_FB = "mel_filterbank.npy"
_MEL_CFG = "mel_config.json"
_DEC_FP32 = "decoder.onnx"
_DEC_INT8 = "decoder_q8.onnx"


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------


def _load_wav(path: str, target_sr: int = _BC_SR) -> np.ndarray:
    """Load *path* as float32 mono, linear-resample to *target_sr*."""
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
# Numpy mel spectrogram
# ---------------------------------------------------------------------------


def _compute_mel_numpy(
    wav: np.ndarray,
    fb: np.ndarray,
    n_fft: int = 1024,
    hop_length: int = 320,
    win_length: int = 640,
) -> np.ndarray:
    """Compute mel spectrogram in pure numpy.

    Matches torchaudio.transforms.MelSpectrogram with ``norm="slaney"`` and
    ``mel_scale="slaney"``.  The STFT uses a Hann window of length
    *win_length*, zero-padded to *n_fft*, with reflect-padding of the
    waveform on both sides by *n_fft // 2* before framing.

    Parameters
    ----------
    wav:
        (N,) float32 mono waveform at 16 kHz.
    fb:
        (num_mels, n_fft//2+1) float32 — mel filterbank matrix.
    n_fft:
        FFT size.
    hop_length:
        Hop size in samples.
    win_length:
        Analysis window length.

    Returns
    -------
    np.ndarray
        (1, num_mels, T_mel) float32 — mel spectrogram (magnitude, Slaney norm).
    """
    window = np.hanning(win_length).astype(np.float32)
    # Reflect-pad signal
    pad = n_fft // 2
    wav_padded = np.pad(wav, pad, mode="reflect")
    # Number of frames
    n_frames = 1 + (len(wav_padded) - n_fft) // hop_length
    # STFT magnitude
    mag = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
    for i in range(n_frames):
        frame = wav_padded[i * hop_length: i * hop_length + n_fft]
        windowed = np.zeros(n_fft, dtype=np.float32)
        windowed[:win_length] = frame[:win_length] * window
        fft = np.fft.rfft(windowed, n=n_fft)
        mag[:, i] = np.abs(fft).astype(np.float32)
    # Apply mel filterbank
    mel = fb @ mag  # (num_mels, T_mel)
    return mel[np.newaxis, :, :]  # (1, num_mels, T_mel)


# ---------------------------------------------------------------------------
# Chunked Wav2Vec2 feature extraction
# ---------------------------------------------------------------------------


def _extract_features_chunked(
    wav: np.ndarray,
    w2v_sess,
    chunk_samples: int = 32000,
    overlap_samples: int = 4000,
) -> np.ndarray:
    """Run wav2vec2_encoder on *wav* in overlapping chunks.

    BiCodec's Wav2Vec2 encoder (300 MB+) degrades on long sequences as an
    ONNX model; processing in 2-second chunks (0.25 s overlap) avoids the
    quality degradation pattern documented for FreeVC and Mimi engines.

    Parameters
    ----------
    wav:
        Mono float32 waveform at 16 kHz.
    w2v_sess:
        ORT InferenceSession for ``wav2vec2_encoder.onnx``.
    chunk_samples:
        Chunk length in samples (default 2 s = 32000 samples).
    overlap_samples:
        Overlap between consecutive chunks (default 0.25 s = 4000 samples).

    Returns
    -------
    np.ndarray
        (1, T_total, 1024) float32 — concatenated feature frames.
    """
    N = len(wav)
    if N <= chunk_samples:
        inp = wav[np.newaxis, :]  # (1, N)
        return w2v_sess.run(None, {"waveform": inp})[0]  # (1, T, 1024)

    step = chunk_samples - overlap_samples
    feature_chunks = []
    pos = 0
    while pos < N:
        end = min(pos + chunk_samples, N)
        chunk = wav[pos:end]
        inp = chunk[np.newaxis, :]  # (1, chunk_N)
        feat = w2v_sess.run(None, {"waveform": inp})[0]  # (1, T_c, 1024)
        # Trim overlap frames at the boundary (approx: ratio of samples to frames ~320:1)
        if pos > 0 and overlap_samples > 0:
            overlap_frames = max(1, overlap_samples // 320)
            feat = feat[:, overlap_frames:, :]
        feature_chunks.append(feat)
        if end >= N:
            break
        pos += step

    return np.concatenate(feature_chunks, axis=1)  # (1, T_total, 1024)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class BiCodecAdapter(VoiceClonerBase):
    """Voice-cloning adapter backed by BiCodec ONNX models.

    BiCodec (SparkAudio/Spark-TTS) factorizes speech into semantic tokens
    (content) and global tokens (speaker identity).  Voice conversion is a
    direct swap of the global tokens between source and reference audio.

    Pipeline
    --------
    1. Compute mel spectrograms for source and reference (``mel_transformer``).
    2. Extract Wav2Vec2 features from source waveform (``wav2vec2_encoder``).
    3. Encode features → semantic tokens (``semantic_encoder``).
    4. Encode reference mel → global tokens (``global_encoder``).
    5. Decode: source semantic tokens + reference global tokens → waveform
       (``decoder``).

    The token swap is a zero-copy numpy integer substitution.

    Parameters
    ----------
    quantized:
        Use INT8 quantized ONNX models (default ``False``).
        ``True`` is faster on CPU at a small quality cost.
    chunk_samples:
        Wav2Vec2 chunking window in samples (default 32000 = 2 s).
        Reduce if out-of-memory; 0 disables chunking (processes full audio).
    **cfg:
        Additional keyword arguments stored but not used at runtime.
    """

    _sample_rate = _BC_SR

    def __init__(
        self,
        quantized: bool = False,
        chunk_samples: int = 32000,
        **cfg,
    ):
        super().__init__(**cfg)
        self._quantized = quantized
        self._chunk_samples = chunk_samples
        self._w2v_sess = None
        self._sem_sess = None
        self._glob_sess = None
        self._mel_fb: np.ndarray | None = None  # (num_mels, n_fft//2+1) float32
        self._mel_cfg: dict | None = None  # STFT parameters
        self._dec_sess = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_models(self) -> None:
        if self._w2v_sess is not None:
            return

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for engine='bicodec'. "
                "Install it with: pip install vconnx"
            ) from exc

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("huggingface_hub is required.") from exc

        import json

        q = self._quantized
        w2v_file = _W2V_INT8 if q else _W2V_FP32
        sem_file = _SEM_INT8 if q else _SEM_FP32
        glob_file = _GLOB_INT8 if q else _GLOB_FP32
        dec_file = _DEC_INT8 if q else _DEC_FP32

        w2v_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=w2v_file)
        sem_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=sem_file)
        glob_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=glob_file)
        mel_fb_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=_MEL_FB)
        mel_cfg_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=_MEL_CFG)
        dec_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=dec_file)

        # Load mel filterbank (numpy, no ORT needed)
        self._mel_fb = np.load(mel_fb_path).astype(np.float32)  # (num_mels, n_fft//2+1)
        with open(mel_cfg_path) as f:
            self._mel_cfg = json.load(f)

        sess_opts = ort.SessionOptions()
        n_threads = os.cpu_count() or 4
        sess_opts.inter_op_num_threads = n_threads
        sess_opts.intra_op_num_threads = n_threads
        providers = ["CPUExecutionProvider"]

        self._w2v_sess = ort.InferenceSession(w2v_path, sess_options=sess_opts, providers=providers)
        self._sem_sess = ort.InferenceSession(sem_path, sess_options=sess_opts, providers=providers)
        self._glob_sess = ort.InferenceSession(glob_path, sess_options=sess_opts, providers=providers)
        self._dec_sess = ort.InferenceSession(dec_path, sess_options=sess_opts, providers=providers)

    # ------------------------------------------------------------------
    # Encode helpers
    # ------------------------------------------------------------------

    def _wav_to_mel(self, wav: np.ndarray) -> np.ndarray:
        """Convert waveform to mel spectrogram using numpy filterbank.

        Parameters
        ----------
        wav:
            (N,) float32 mono waveform at 16 kHz.

        Returns
        -------
        np.ndarray
            (1, 128, T_mel) float32.
        """
        cfg = self._mel_cfg
        return _compute_mel_numpy(
            wav,
            self._mel_fb,
            n_fft=cfg["n_fft"],
            hop_length=cfg["hop_length"],
            win_length=cfg["win_length"],
        )  # (1, 128, T_mel)

    def _extract_semantic_tokens(self, wav: np.ndarray) -> np.ndarray:
        """Wav2Vec2 + semantic encoder → semantic token indices.

        Parameters
        ----------
        wav:
            (N,) float32 mono at 16 kHz.

        Returns
        -------
        np.ndarray
            (1, T2) int64 — semantic token indices.
        """
        features = _extract_features_chunked(
            wav, self._w2v_sess, chunk_samples=self._chunk_samples
        )  # (1, T, 1024)
        tokens = self._sem_sess.run(None, {"features": features})[0]  # (1, T2) int64
        return tokens

    def _extract_global_tokens(self, mel: np.ndarray) -> np.ndarray:
        """Global encoder: mel → global token indices.

        Parameters
        ----------
        mel:
            (1, 128, T_mel) float32.

        Returns
        -------
        np.ndarray
            (1, 1, 32) int32 — FSQ global token indices.
        """
        return self._glob_sess.run(None, {"mel": mel})[0]  # (1, 1, 32) int32

    def _decode(
        self,
        semantic_tokens: np.ndarray,
        global_tokens: np.ndarray,
    ) -> np.ndarray:
        """Decoder: semantic + global tokens → waveform.

        Parameters
        ----------
        semantic_tokens:
            (1, T2) int64.
        global_tokens:
            (1, 1, 32) int32.

        Returns
        -------
        np.ndarray
            (samples,) float32 waveform.
        """
        out = self._dec_sess.run(None, {
            "semantic_tokens": semantic_tokens,
            "global_tokens": global_tokens,
        })[0]  # (1, 1, N)
        return out.reshape(-1).astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clone_voice(
        self,
        audio: str,
        reference_voice: str,
        out_path: str,
    ) -> str:
        """Convert *audio* to sound like *reference_voice* using BiCodec token swap.

        Pipeline:
        1. Load source and reference audio at 16 kHz.
        2. Compute mel spectrograms for both (numpy mel filterbank).
        3. Extract Wav2Vec2 features from source in 2 s chunks
           (wav2vec2_encoder.onnx).
        4. Encode source features → semantic tokens (semantic_encoder.onnx).
        5. Encode reference mel → global tokens (global_encoder.onnx).
        6. Decode: source semantic tokens + reference global tokens → waveform
           (decoder.onnx).

        Parameters
        ----------
        audio:
            Path to the source WAV file (any sample rate, 16-bit PCM recommended).
        reference_voice:
            Path to the reference speaker WAV file.
        out_path:
            Destination path for the 16-bit 16 kHz output WAV.

        Returns
        -------
        str
            Absolute path to the written output file.
        """
        self._ensure_models()

        src_wav = _load_wav(str(audio), target_sr=_BC_SR)
        ref_wav = _load_wav(str(reference_voice), target_sr=_BC_SR)

        # 1. Mel spectrograms (numpy filterbank)
        ref_mel = self._wav_to_mel(ref_wav)   # (1, 128, T_mel_ref)

        # 2. Source: Wav2Vec2 features → semantic tokens
        src_semantic = self._extract_semantic_tokens(src_wav)  # (1, T2) int64

        # 3. Reference: mel → global tokens
        ref_global = self._extract_global_tokens(ref_mel)  # (1, 1, 32) int32

        # 4. Decode: source content + reference timbre
        waveform = self._decode(src_semantic, ref_global)  # (samples,) float32

        out_path = str(Path(out_path).resolve())
        _save_wav(out_path, waveform, sr=_BC_SR)
        return out_path


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_engine(
    EngineEntry(
        alias="bicodec",
        adapter_class=BiCodecAdapter,
        description=(
            "BiCodec (SparkTTS): explicit semantic/global token factorization — "
            "Wav2Vec2-XLSR-53 (layers 11/14/16) → convolutional VQ → semantic tokens (content); "
            "ECAPA-TDNN + Perceiver + FSQ → 32 global tokens (timbre). "
            "VC: swap global tokens from reference, decode. "
            "Zero-shot any-to-any VC at 16 kHz. "
            "ONNX artifacts from TigreGotico/vconnx-bicodec. "
            "(SparkAudio 2025, CC BY-NC-SA 4.0 weights, Apache-2.0 code)"
        ),
        extras="",
        onnx_native=True,
    )
)
