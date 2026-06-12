"""SpeechTokenizer adapter for voiceclonnx.

SpeechTokenizer (ACL 2024, Apache-2.0) is a hierarchical RVQ speech codec
with 8 quantizers at 50 Hz.  Quantizer 1 (RVQ-1) is semantically distilled
via HuBERT and captures linguistic content; quantizers 2-8 carry speaker
timbre.

Voice-conversion recipe implemented here
-----------------------------------------
1. Encode source audio → continuous features (1, 1024, T_src) with encoder.onnx.
2. Encode reference audio → continuous features (1, 1024, T_ref).
3. RVQ-quantize both using the numpy codebook lookup (codebooks.npy):
   - source_codes: (8, T_src) int64 — nearest-neighbour in each codebook layer
   - ref_codes:    (8, T_ref) int64
4. Swap: mixed_codes = [source_codes[0], ref_codes[1..7]]
   Reference rows are truncated or tiled to match T_src.
5. RVQ-decode mixed_codes → mixed quantized features (1, 1024, T_src).
6. Decode mixed features → waveform with decoder.onnx.

ONNX components (export: conversion/export_speechtokenizer.py)
---------------------------------------------------------------
- ``encoder.onnx``  : waveform (1, 1, N) float32 → features (1, 1024, T) float32
- ``decoder.onnx``  : features (1, 1024, T) float32 → waveform (1, 1, N) float32
- ``codebooks.npy`` : (8, 1024, 1024) float32 — VQ codebook embeddings

The RVQ quantize/dequantize step is pure numpy using the codebook weights —
no ONNX component is required for this step.

Why continuous-feature split?
------------------------------
SpeechTokenizer's convolutional encoder uses custom asymmetric padding
(``SConv1d``) where the extra-padding amount is computed from the input
length.  This produces a fixed-length output when traced naively with
TorchScript-based ONNX export.  Splitting at the continuous-feature boundary
(before the RVQ quantizer) sidesteps this by exporting only the convolutional
stack, where the padding is an integer constant computed from the kernel
parameters — making the output shape fully dynamic.

Runtime requirements: ``onnxruntime``, ``numpy``, ``soundfile``.

References
----------
- https://github.com/ZhangXInFD/SpeechTokenizer
- https://arxiv.org/abs/2308.16692
- https://huggingface.co/fnlp/SpeechTokenizer
- https://huggingface.co/TigreGotico/voiceclonnx-speechtokenizer
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from voiceclonnx.engines.base import EngineEntry, VoiceClonerBase, register_engine

# SpeechTokenizer operates at 16 kHz, 50 Hz token rate (320-sample hop)
_ST_SR = 16000

# HF repo housing the exported ONNX artifacts
_HF_REPO_ID = "TigreGotico/voiceclonnx-speechtokenizer"

# File names inside the HF repo
_ENC_FP32 = "encoder.onnx"
_ENC_INT8 = "encoder_q8.onnx"
_DEC_FP32 = "decoder.onnx"
_DEC_INT8 = "decoder_q8.onnx"
_CODEBOOKS = "codebooks.npy"

# Number of RVQ quantizers
_N_QUANTIZERS = 8


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------


def _load_wav(path: str, target_sr: int = _ST_SR) -> np.ndarray:
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
# Numpy RVQ quantize / dequantize
# ---------------------------------------------------------------------------


def _rvq_encode(features_bdt: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
    """Nearest-neighbour RVQ encode using numpy.

    Parameters
    ----------
    features_bdt:
        (1, D, T) float32 — continuous encoder features.
    codebooks:
        (Q, codebook_size, D) float32 — VQ codebook embeddings.

    Returns
    -------
    np.ndarray
        (Q, T) int64 — codebook indices per layer.
    """
    feats = features_bdt[0].T  # (T, D)
    residual = feats.copy()
    codes_list = []
    for cb in codebooks:
        # cb: (codebook_size, D)
        dist = -(
            (residual ** 2).sum(1, keepdims=True)
            - 2 * residual @ cb.T
            + (cb ** 2).sum(1, keepdims=True).T
        )
        idx = dist.argmax(axis=1)  # (T,)
        codes_list.append(idx)
        residual -= cb[idx]
    return np.stack(codes_list)  # (Q, T)


def _rvq_decode(codes: np.ndarray, codebooks: np.ndarray) -> np.ndarray:
    """Reconstruct quantized features from RVQ indices.

    Parameters
    ----------
    codes:
        (Q, T) int64 — codebook indices per layer.
    codebooks:
        (Q, codebook_size, D) float32 — VQ codebook embeddings.

    Returns
    -------
    np.ndarray
        (1, D, T) float32 — summed quantized features.
    """
    T = codes.shape[1]
    D = codebooks.shape[2]
    quant = np.zeros((T, D), dtype=np.float32)
    for idx, cb in zip(codes, codebooks):
        quant += cb[idx]
    return quant.T[np.newaxis, :, :]  # (1, D, T)


def _swap_rvq_tokens(
    src_codes: np.ndarray,
    ref_codes: np.ndarray,
    content_layers: int = 2,
) -> np.ndarray:
    """Swap timbre layers from *ref_codes* into *src_codes*.

    Parameters
    ----------
    src_codes:
        (Q, T_src) int64 — RVQ codes from source audio.
    ref_codes:
        (Q, T_ref) int64 — RVQ codes from reference audio.
    content_layers:
        Number of leading RVQ layers treated as content (default 2 —
        measured intelligibility/timbre tradeoff: 1 layer → strong timbre
        but reference-dependent WER (58% on one demo voice); 2 → 12% WER;
        3 → 0% WER with weaker timbre transfer).
        Layers [0 … content_layers-1] are kept from source.
        Layers [content_layers … Q-1] are taken from reference.

    Returns
    -------
    np.ndarray
        (Q, T_src) int64 — mixed codes.
    """
    Q = src_codes.shape[0]
    T_src = src_codes.shape[1]
    T_ref = ref_codes.shape[1]

    mixed = src_codes.copy()

    if T_ref == 0 or T_src == 0:
        return mixed

    for q in range(content_layers, Q):
        if T_ref >= T_src:
            mixed[q] = ref_codes[q, :T_src]
        else:
            repeats = (T_src + T_ref - 1) // T_ref
            tiled = np.tile(ref_codes[q], repeats)
            mixed[q] = tiled[:T_src]

    return mixed


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class SpeechTokenizerAdapter(VoiceClonerBase):
    """Voice-cloning adapter backed by SpeechTokenizer ONNX models.

    Pipeline:
    1. Encode source and reference audio with the encoder ONNX model to get
       (1, 1024, T) continuous features.
    2. RVQ-encode features to (8, T) integer code sequences (pure numpy,
       nearest-neighbour in each of the 8 codebooks).
    3. Swap: keep RVQ-1 codes from source (linguistic content) and replace
       RVQ-2..8 codes with those from the reference (timbre/speaker).
       Reference codes are looped or truncated to match the source length.
    4. RVQ-decode the mixed codes back to (1, 1024, T) quantized features
       (pure numpy, codebook lookup + sum).
    5. Decode quantized features to waveform with the decoder ONNX model.

    Parameters
    ----------
    quantized:
        Accepted for API uniformity but **not supported** — the INT8
        exports in TigreGotico/voiceclonnx-speechtokenizer use a different
        interface (codes-in/waveform-out) that is incompatible with the
        continuous-feature pipeline this adapter implements.  Passing
        ``quantized=True`` raises ``NotImplementedError``.
        See demo/QUANTS.md for the full int8 status.
    content_layers:
        Number of leading RVQ layers used as content proxy (default 1).
        The paper reports RVQ-1 as the semantic layer; raising this to 2
        modestly improves timbre transfer at some cost to intelligibility.
    **cfg:
        Additional keyword arguments stored but not used at runtime.
    """

    _sample_rate = _ST_SR

    def __init__(
        self,
        quantized: bool = False,
        content_layers: int = 2,
        **cfg,
    ):
        super().__init__(**cfg)
        if quantized:
            raise NotImplementedError(
                "speechtokenizer quantized=True is not supported: the INT8 exports "
                "in TigreGotico/voiceclonnx-speechtokenizer use a codes-in/waveform-out "
                "interface incompatible with the continuous-feature pipeline. "
                "Use quantized=False (fp32) instead."
            )
        self._quantized = quantized
        self._content_layers = content_layers
        self._enc_sess = None
        self._dec_sess = None
        self._codebooks: "np.ndarray | None" = None

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
                "onnxruntime is required for engine='speechtokenizer'. "
                "Install it with: pip install voiceclonnx"
            ) from exc

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("huggingface_hub is required.") from exc

        enc_file = _ENC_INT8 if self._quantized else _ENC_FP32
        dec_file = _DEC_INT8 if self._quantized else _DEC_FP32

        enc_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=enc_file)
        dec_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=dec_file)
        cb_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=_CODEBOOKS)

        sess_opts = ort.SessionOptions()
        n_threads = os.cpu_count() or 4
        sess_opts.inter_op_num_threads = n_threads
        sess_opts.intra_op_num_threads = n_threads

        providers = ["CPUExecutionProvider"]

        self._enc_sess = ort.InferenceSession(enc_path, sess_options=sess_opts, providers=providers)
        self._dec_sess = ort.InferenceSession(dec_path, sess_options=sess_opts, providers=providers)
        self._codebooks = np.load(cb_path)  # (Q, codebook_size, D)

    # ------------------------------------------------------------------
    # Encode: audio → continuous features
    # ------------------------------------------------------------------

    def _encode_feats(self, audio: np.ndarray) -> np.ndarray:
        """Run convolutional encoder; return (1, 1024, T) float32 features."""
        self._ensure_models()
        inp = audio[np.newaxis, np.newaxis, :].astype(np.float32)  # (1, 1, N)
        out = self._enc_sess.run(None, {"audio": inp})
        return out[0]  # (1, 1024, T)

    # ------------------------------------------------------------------
    # Decode: quantized features → audio
    # ------------------------------------------------------------------

    def _decode_feats(self, features: np.ndarray) -> np.ndarray:
        """Run convolutional decoder; return (samples,) float32 waveform."""
        self._ensure_models()
        # features: (1, 1024, T)
        out = self._dec_sess.run(None, {"features": features})
        wav = out[0]  # (1, 1, N) or (1, N)
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
        """Convert *audio* to sound like *reference_voice* using SpeechTokenizer RVQ swap.

        Pipeline:
        1. Encode source + reference → continuous features.
        2. RVQ-encode both (numpy nearest-neighbour in 8 codebooks).
        3. Swap: source RVQ-1 tokens (content) + reference RVQ-2..8 tokens (timbre).
        4. RVQ-decode mixed codes → mixed quantized features.
        5. Decode features → waveform.

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
        self._ensure_models()
        src_wav = _load_wav(str(audio), target_sr=_ST_SR)
        ref_wav = _load_wav(str(reference_voice), target_sr=_ST_SR)

        # 1. Encode continuous features
        src_feats = self._encode_feats(src_wav)    # (1, 1024, T_src)
        ref_feats = self._encode_feats(ref_wav)    # (1, 1024, T_ref)

        # 2. RVQ encode (pure numpy)
        src_codes = _rvq_encode(src_feats, self._codebooks)  # (Q, T_src)
        ref_codes = _rvq_encode(ref_feats, self._codebooks)  # (Q, T_ref)

        # 3. Token swap: source content (layer 0) + reference timbre (layers 1-7)
        mixed_codes = _swap_rvq_tokens(
            src_codes, ref_codes, content_layers=self._content_layers
        )  # (Q, T_src)

        # 4. RVQ decode mixed codes → quantized features (pure numpy)
        mixed_feats = _rvq_decode(mixed_codes, self._codebooks)  # (1, 1024, T_src)

        # 5. Decode to waveform
        waveform = self._decode_feats(mixed_feats)  # (samples,)

        out_path = str(Path(out_path).resolve())
        _save_wav(out_path, waveform, sr=_ST_SR)
        return out_path


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_engine(
    EngineEntry(
        alias="speechtokenizer",
        adapter_class=SpeechTokenizerAdapter,
        description=(
            "SpeechTokenizer: hierarchical RVQ-8 codec — convolutional encoder, "
            "numpy nearest-neighbour VQ (8 codebooks), RVQ-1 source content + "
            "RVQ-2..8 reference timbre swap, convolutional decoder. "
            "Zero-shot any-to-any VC at 16 kHz. "
            "ONNX artifacts from TigreGotico/voiceclonnx-speechtokenizer. "
            "(Zhang et al., ACL 2024, Apache-2.0)"
        ),
        extras="",
        onnx_native=True,
    )
)
