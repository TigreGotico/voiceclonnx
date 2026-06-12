"""Mimi (Kyutai) adapter for voiceclonnx.

Mimi is the neural audio codec powering Moshi (Kyutai, 2024).  It produces
32 residual-vector-quantizer (RVQ) code streams at 12.5 Hz / 24 kHz.

Voice-conversion pipeline
--------------------------
Stream 0 is semantically distilled from WavLM and carries high-level speaker
style and prosodic characteristics.  Streams 1–31 carry the detailed acoustic
sequence including the phonetic content (who is speaking AND what is said).

Empirically verified VC recipe:

1. Encode source audio → 32 code streams (B, 32, T_src).
2. Encode reference audio → 32 code streams (B, 32, T_ref).
3. Build mixed codes: **stream 0 from reference** (speaker style),
   **streams 1–31 from source** (phonetic/acoustic content).
   Length-adaptation via nearest-frame lookup when T_src ≠ T_ref.
4. Decode mixed codes → converted waveform.

This preserves source intelligibility while injecting reference prosodic style.

Runtime requirements: ``onnxruntime``, ``numpy``, ``soundfile``.

Requires: ``pip install voiceclonnx``
  -> onnxruntime, numpy, soundfile, huggingface_hub

References
----------
- https://huggingface.co/kyutai/mimi
- https://github.com/kyutai-labs/moshi
- https://kyutai.org/codec-explainer
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

from voiceclonnx.engines.base import EngineEntry, VoiceClonerBase, register_engine

# Mimi operates at 24 kHz; RVQ frame rate = 12.5 Hz
_MIMI_SR = 24000
_FRAME_RATE = 12.5
_NUM_QUANTIZERS = 32
_NUM_SEMANTIC = 1    # stream 0 — WavLM-distilled content
_NUM_ACOUSTIC = 31   # streams 1–31 — timbre/texture

# HF repo housing the exported ONNX artifacts
_HF_REPO_ID = "TigreGotico/voiceclonnx-mimi"

_ENC_FP32 = "mimi_encoder.onnx"
_ENC_INT8 = "mimi_encoder_q8.onnx"
_DEC_FP32 = "mimi_decoder.onnx"
_DEC_INT8 = "mimi_decoder_q8.onnx"


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------


def _load_wav(path: str, target_sr: int = _MIMI_SR) -> np.ndarray:
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
# Stream-swap: numpy RVQ VC
# ---------------------------------------------------------------------------


def _swap_streams(
    src_codes: np.ndarray,
    ref_codes: np.ndarray,
    n_semantic: int = _NUM_SEMANTIC,
) -> np.ndarray:
    """Construct mixed codes for voice conversion.

    Empirically verified recipe (see docs/engines/mimi.md):
    - Streams 1-31 (acoustic, from **source**) carry the phonetic sequence
      that determines the spoken content.
    - Stream 0 (WavLM-semantic, from **reference**) carries higher-level
      speaker style and prosodic characteristics.

    The output therefore sounds like the source content with reference style,
    which is the desired VC behaviour.

    When the reference has a different temporal length, stream 0 is adapted
    by nearest-neighbour index lookup so the output has the source frame count.

    Parameters
    ----------
    src_codes:
        (B, Q, T_src) int64 — encoded source audio.
    ref_codes:
        (B, Q, T_ref) int64 — encoded reference audio.
    n_semantic:
        Number of leading streams treated as semantic style (default 1).

    Returns
    -------
    np.ndarray
        (B, Q, T_src) int64 — mixed code tensor ready for decoding.
    """
    B, Q, T_src = src_codes.shape
    T_ref = ref_codes.shape[2]

    mixed = src_codes.copy()   # start with source (preserves acoustic streams 1-31)

    if T_src != T_ref:
        # Nearest-frame adaptation for stream 0
        src_idx = np.arange(T_src)
        ref_idx = np.round(src_idx * (T_ref - 1) / max(T_src - 1, 1)).astype(np.int64)
        ref_idx = np.clip(ref_idx, 0, T_ref - 1)
        mixed[:, :n_semantic, :] = ref_codes[:, :n_semantic, :][:, :, ref_idx]
    else:
        mixed[:, :n_semantic, :] = ref_codes[:, :n_semantic, :]

    return mixed


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class MimiAdapter(VoiceClonerBase):
    """Voice-cloning adapter backed by Mimi (Kyutai) ONNX models.

    Pipeline:
    1. Encode source and reference audio with the Mimi encoder ONNX.
    2. Swap streams: keep stream 0 (semantic/content) from source, use
       streams 1–31 (acoustic/timbre) from reference — pure numpy.
    3. Decode the mixed code tensor with the Mimi decoder ONNX.

    Parameters
    ----------
    quantized:
        Use INT8 quantized ONNX models (default ``False`` — fp32 for best
        quality; ``True`` for faster CPU inference).
    **cfg:
        Additional keyword arguments stored but unused at runtime.
    """

    _sample_rate = _MIMI_SR

    def __init__(self, quantized: bool = False, **cfg) -> None:
        super().__init__(**cfg)
        self._quantized = quantized
        self._enc_sess = None
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
                "onnxruntime is required for engine='mimi'. "
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

        sess_opts = ort.SessionOptions()
        n_threads = os.cpu_count() or 4
        sess_opts.inter_op_num_threads = n_threads
        sess_opts.intra_op_num_threads = n_threads

        providers = ["CPUExecutionProvider"]
        self._enc_sess = ort.InferenceSession(enc_path, sess_options=sess_opts, providers=providers)
        self._dec_sess = ort.InferenceSession(dec_path, sess_options=sess_opts, providers=providers)

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def _encode(self, audio: np.ndarray) -> np.ndarray:
        """Run encoder and return (1, 32, T) int64 codes."""
        self._ensure_models()
        inp = audio[np.newaxis, np.newaxis, :].astype(np.float32)  # (1, 1, N)
        codes = self._enc_sess.run(None, {"input_values": inp})[0]
        return codes  # (1, 32, T)

    def _decode(self, codes: np.ndarray) -> np.ndarray:
        """Run decoder and return (samples,) float32 waveform."""
        self._ensure_models()
        audio = self._dec_sess.run(None, {"audio_codes": codes})[0]
        return audio[0, 0]  # (samples,)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clone_voice(
        self,
        audio: str,
        reference_voice: str,
        out_path: str,
    ) -> str:
        """Convert *audio* to sound like *reference_voice* using Mimi RVQ stream swap.

        Recipe: stream 0 (style) from reference; streams 1-31 (content) from source.

        Parameters
        ----------
        audio:
            Path to the source WAV file.
        reference_voice:
            Path to the reference speaker WAV file.
        out_path:
            Destination path for the 16-bit 24 kHz output WAV.

        Returns
        -------
        str
            Absolute path to the written output file.
        """
        src_wav = _load_wav(str(audio), target_sr=_MIMI_SR)
        ref_wav = _load_wav(str(reference_voice), target_sr=_MIMI_SR)

        # Encode both
        src_codes = self._encode(src_wav)  # (1, 32, T_src)
        ref_codes = self._encode(ref_wav)  # (1, 32, T_ref)

        # RVQ stream swap: semantic from source, acoustic from reference
        mixed_codes = _swap_streams(src_codes, ref_codes)  # (1, 32, T_src)

        # Decode
        waveform = self._decode(mixed_codes)  # (samples,)

        out_path = str(Path(out_path).resolve())
        _save_wav(out_path, waveform, sr=_MIMI_SR)
        return out_path


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_engine(
    EngineEntry(
        alias="mimi",
        adapter_class=MimiAdapter,
        description=(
            "Mimi: Kyutai RVQ codec VC — stream-0 (WavLM-semantic) from source, "
            "streams 1–31 (acoustic/timbre) from reference (pure numpy swap). "
            "ONNX artifacts from TigreGotico/voiceclonnx-mimi. "
            "24 kHz, 12.5 Hz frame rate, 32 code streams. "
            "(Kyutai, 2024, CC BY 4.0)"
        ),
        extras="",
        onnx_native=True,
    )
)
