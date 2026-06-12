"""FACodec (NaturalSpeech 3) adapter for voiceclonnx.

FACodec (Amphion / Microsoft Research, ICML 2024) disentangles speech waveforms
into factorised subspaces — content, prosody, timbre, acoustic detail — and
reconstructs high-quality speech from them.  Voice conversion is zero-shot:
encode source, encode reference, swap the timbre embedding, decode.

Voice-conversion recipe (V2 path)
-----------------------------------
1. Load source and reference audio at 16 kHz.
2. Run the convolutional encoder → ``(1, 256, T)`` continuous features.
3. Compute the prosody mel from source (20 mel bins, hop=200, numpy).
4. Run the quantizer → ``(6, 1, T)`` integer VQ token ids.
5. Extract timbre embedding from reference encoder features via the
   TransformerEncoder + mean-pool → ``(1, 256)``.
6. VC decode: embed source VQ ids (prosody+content only, no residual) →
   inject reference timbre via AdaIN-style conditioning → decoder → wav.

The prosody mel is computed in pure numpy using a fixed mel filter bank
(n_fft=1024, hop=200, win=800, n_mels=80, sr=16000, fmin=0, fmax=8000,
log-compressed, first 20 bins) — no ONNX component needed.

ONNX components (export: conversion/export_facodec.py)
-------------------------------------------------------
- ``facodec_encoder.onnx``   : wav (1,1,N) float32   → enc_feats (1,256,T)
- ``facodec_timbre.onnx``    : enc_feats (1,256,T)   → spk_embs (1,256)
- ``facodec_quantize.onnx``  : (enc_feats, mel_20)   → vq_ids (6,1,T) int64
- ``facodec_decoder.onnx``   : (vq_ids, spk_embs)   → wav (1,1,N)

Runtime requirements: ``onnxruntime``, ``numpy``, ``soundfile``.

References
----------
- https://huggingface.co/amphion/naturalspeech3_facodec
- https://github.com/open-mmlab/Amphion
- https://arxiv.org/abs/2403.03100
- https://huggingface.co/TigreGotico/voiceclonnx-facodec
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from voiceclonnx.engines.base import EngineEntry, VoiceClonerBase, register_engine

# FACodec operates at 16 kHz, hop=200 samples
_FA_SR = 16000
_FA_HOP = 200
_FA_N_FFT = 1024
_FA_WIN = 800
_FA_N_MELS = 80
_FA_FMIN = 0.0
_FA_FMAX = 8000.0

# HF repo housing the exported ONNX artifacts
_HF_REPO_ID = "TigreGotico/voiceclonnx-facodec"

_ENC_FP32 = "facodec_encoder.onnx"
_ENC_INT8 = "facodec_encoder_q8.onnx"
_TIMBRE_FP32 = "facodec_timbre.onnx"
_TIMBRE_INT8 = "facodec_timbre_q8.onnx"
_QUANT_FP32 = "facodec_quantize.onnx"
_QUANT_INT8 = "facodec_quantize_q8.onnx"
_DEC_FP32 = "facodec_decoder.onnx"
_DEC_INT8 = "facodec_decoder_q8.onnx"


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------


def _load_wav(path: str, target_sr: int = _FA_SR) -> np.ndarray:
    """Load WAV as float32 mono, linear-resample to target_sr."""
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
    """Write float32 as 16-bit PCM WAV."""
    import soundfile as sf

    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    sf.write(str(path), audio_int16, sr, subtype="PCM_16")


# ---------------------------------------------------------------------------
# Prosody mel computation (pure numpy — mirrors FACodecEncoderV2.get_prosody_feature)
# ---------------------------------------------------------------------------

_MEL_FILTERBANK: "np.ndarray | None" = None


def _get_mel_filterbank() -> np.ndarray:
    """Return (80, n_fft//2+1) mel filter bank, cached globally."""
    global _MEL_FILTERBANK
    if _MEL_FILTERBANK is not None:
        return _MEL_FILTERBANK

    # Build HTK mel filter bank identical to librosa defaults
    n_fft = _FA_N_FFT
    sr = _FA_SR
    n_mels = _FA_N_MELS
    fmin = _FA_FMIN
    fmax = _FA_FMAX

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bin_pts = np.floor((n_fft + 1) * hz_pts / sr).astype(int)
    n_bins = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_bins), dtype=np.float32)
    for m in range(1, n_mels + 1):
        f_l, f_c, f_r = bin_pts[m - 1], bin_pts[m], bin_pts[m + 1]
        for k in range(f_l, f_c):
            if f_c != f_l:
                fb[m - 1, k] = (k - f_l) / (f_c - f_l)
        for k in range(f_c, f_r):
            if f_r != f_c:
                fb[m - 1, k] = (f_r - k) / (f_r - f_c)

    # L2-normalise per mel filter (slaney norm) — matches librosa default
    enorm = 2.0 / (hz_pts[2 : n_mels + 2] - hz_pts[:n_mels])
    fb *= enorm[:, np.newaxis]

    _MEL_FILTERBANK = fb
    return fb


def _compute_prosody_mel(audio: np.ndarray) -> np.ndarray:
    """Compute 20-bin prosody mel matching FACodecEncoderV2.get_prosody_feature.

    Parameters
    ----------
    audio:
        (N,) float32 mono at 16 kHz.

    Returns
    -------
    np.ndarray
        (1, 20, T) float32 — first 20 mel bins, log-compressed, batch dim prepended.
    """
    # Reflect-pad: torch MelSpectrogram pads (n_fft - hop_size) // 2 on each side
    pad = (_FA_N_FFT - _FA_HOP) // 2
    audio_padded = np.pad(audio, (pad, pad), mode="reflect")

    # Hann window (win_size=800, zero-padded to n_fft=1024 for each frame)
    window = np.hanning(_FA_WIN).astype(np.float32)

    # STFT (magnitude) — extract win_size samples per hop, window, zero-pad to n_fft
    n_frames = 1 + (len(audio_padded) - _FA_WIN) // _FA_HOP
    n_bins = _FA_N_FFT // 2 + 1
    spec = np.zeros((n_bins, n_frames), dtype=np.float32)
    for i in range(n_frames):
        frame = audio_padded[i * _FA_HOP : i * _FA_HOP + _FA_WIN]
        if len(frame) < _FA_WIN:
            frame = np.pad(frame, (0, _FA_WIN - len(frame)))
        windowed = frame * window
        fft = np.fft.rfft(windowed, n=_FA_N_FFT)
        spec[:, i] = np.abs(fft).astype(np.float32)

    # Mel filterbank + log compression
    fb = _get_mel_filterbank()  # (80, n_bins)
    mel = fb @ spec  # (80, T)
    mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))  # log compression

    # Return first 20 bins with batch dim: (1, 20, T)
    return mel[np.newaxis, :20, :].astype(np.float32)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class FACodecAdapter(VoiceClonerBase):
    """Voice-cloning adapter backed by FACodec V2 ONNX models.

    Pipeline:
    1. Encode source and reference at 16 kHz through the convolutional encoder.
    2. Compute prosody mel (20 bins, numpy) from source.
    3. Quantize source encoder features + prosody mel → VQ token ids (6, 1, T).
    4. Extract reference timbre embedding from reference encoder features.
    5. Decode: embed source VQ ids (prosody+content, no residual) → apply
       reference timbre via AdaIN → convolutional decoder → wav.

    Parameters
    ----------
    quantized:
        Use INT8 models when ``True`` (default ``False`` — fp32 for quality).
    **cfg:
        Additional keyword arguments stored but unused at runtime.
    """

    _sample_rate = _FA_SR

    def __init__(self, quantized: bool = False, **cfg):
        super().__init__(**cfg)
        self._quantized = quantized
        self._enc_sess = None
        self._timbre_sess = None
        self._quant_sess = None
        self._dec_sess = None

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
                "onnxruntime is required for engine='facodec'. "
                "Install it with: pip install voiceclonnx"
            ) from exc

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("huggingface_hub is required.") from exc

        enc_f = _ENC_INT8 if self._quantized else _ENC_FP32
        tim_f = _TIMBRE_INT8 if self._quantized else _TIMBRE_FP32
        qnt_f = _QUANT_INT8 if self._quantized else _QUANT_FP32
        dec_f = _DEC_INT8 if self._quantized else _DEC_FP32

        enc_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=enc_f)
        tim_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=tim_f)
        qnt_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=qnt_f)
        dec_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=dec_f)

        sess_opts = ort.SessionOptions()
        n_threads = os.cpu_count() or 4
        sess_opts.inter_op_num_threads = n_threads
        sess_opts.intra_op_num_threads = n_threads
        providers = ["CPUExecutionProvider"]

        self._enc_sess = ort.InferenceSession(enc_path, sess_options=sess_opts, providers=providers)
        self._timbre_sess = ort.InferenceSession(tim_path, sess_options=sess_opts, providers=providers)
        self._quant_sess = ort.InferenceSession(qnt_path, sess_options=sess_opts, providers=providers)
        self._dec_sess = ort.InferenceSession(dec_path, sess_options=sess_opts, providers=providers)

    # ------------------------------------------------------------------
    # Encode: audio → continuous features (1, 256, T)
    # ------------------------------------------------------------------

    def _encode(self, audio: np.ndarray) -> np.ndarray:
        inp = audio[np.newaxis, np.newaxis, :].astype(np.float32)  # (1,1,N)
        out = self._enc_sess.run(None, {"wav": inp})
        return out[0]  # (1, 256, T)

    # ------------------------------------------------------------------
    # Timbre: enc_feats → spk_embs (1, 256)
    # ------------------------------------------------------------------

    def _get_timbre(self, enc_feats: np.ndarray) -> np.ndarray:
        out = self._timbre_sess.run(None, {"enc_feats": enc_feats})
        return out[0]  # (1, 256)

    # ------------------------------------------------------------------
    # Quantize: (enc_feats, mel_20) → vq_ids (6, 1, T)
    # ------------------------------------------------------------------

    def _quantize(self, enc_feats: np.ndarray, mel_20: np.ndarray) -> np.ndarray:
        out = self._quant_sess.run(None, {"enc_feats": enc_feats, "mel_20": mel_20})
        return out[0]  # (6, 1, T) int64

    # ------------------------------------------------------------------
    # Decode: (vq_ids, spk_embs) → wav (samples,)
    # ------------------------------------------------------------------

    def _decode(self, vq_ids: np.ndarray, spk_embs: np.ndarray) -> np.ndarray:
        out = self._dec_sess.run(None, {"vq_ids": vq_ids, "spk_embs": spk_embs})
        wav = out[0]  # (1, 1, N)
        return wav.reshape(-1).astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clone_voice(
        self,
        audio: str,
        reference_voice: str,
        out_path: str,
    ) -> str:
        """Convert *audio* to sound like *reference_voice* using FACodec V2 timbre swap.

        Pipeline:
        1. Encode source and reference → continuous features (1, 256, T).
        2. Compute prosody mel from source (20 mel bins, numpy).
        3. Quantize source → VQ token ids (6, 1, T).
        4. Extract reference timbre embedding (1, 256).
        5. Decode: embed source VQ ids (prosody+content) + reference timbre → wav.

        Parameters
        ----------
        audio:
            Path to source WAV file.
        reference_voice:
            Path to reference speaker WAV file.
        out_path:
            Destination path for 16-bit 16 kHz output WAV.

        Returns
        -------
        str
            Absolute path to written output file.
        """
        self._ensure_models()

        src_wav = _load_wav(str(audio), target_sr=_FA_SR)
        ref_wav = _load_wav(str(reference_voice), target_sr=_FA_SR)

        # 1. Encode
        src_feats = self._encode(src_wav)   # (1, 256, T_src)
        ref_feats = self._encode(ref_wav)   # (1, 256, T_ref)

        # 2. Prosody mel (numpy) — align T to encoder output to handle off-by-one differences
        src_mel = _compute_prosody_mel(src_wav)   # (1, 20, T_mel)
        T_enc = src_feats.shape[2]
        T_mel = src_mel.shape[2]
        if T_mel > T_enc:
            src_mel = src_mel[:, :, :T_enc]
        elif T_mel < T_enc:
            # Pad with log(1e-5) (silence value)
            pad_val = np.log(1e-5)
            src_mel = np.pad(src_mel, ((0, 0), (0, 0), (0, T_enc - T_mel)), constant_values=pad_val)

        # 3. Quantize source
        src_vq_ids = self._quantize(src_feats, src_mel)  # (6, 1, T_src)

        # 4. Reference timbre
        ref_spk_embs = self._get_timbre(ref_feats)       # (1, 256)

        # 5. Decode source content + reference timbre
        waveform = self._decode(src_vq_ids, ref_spk_embs)  # (N,)

        out_path = str(Path(out_path).resolve())
        _save_wav(out_path, waveform, sr=_FA_SR)
        return out_path


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_engine(
    EngineEntry(
        alias="facodec",
        adapter_class=FACodecAdapter,
        description=(
            "FACodec (NaturalSpeech 3): factorised codec VC — convolutional encoder, "
            "disentangled VQ (prosody/content/timbre/acoustic), "
            "zero-shot timbre-swap via TransformerEncoder timbre extractor + AdaIN decoder. "
            "16 kHz. ONNX artifacts from TigreGotico/voiceclonnx-facodec. "
            "(Ju et al., ICML 2024, Apache-2.0 weights)"
        ),
        extras="",
        onnx_native=True,
    )
)
