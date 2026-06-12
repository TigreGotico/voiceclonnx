"""Chatterbox AR codec-LM voice-conversion adapter for vconnx.

Ported directly from TigreGotico/chatterbox-onnx (TigreGotico IP).
Only the voice-conversion (VC) path is implemented; TTS is intentionally
omitted (audio-to-audio only, no text-encoder, no tokenizer required).

VC pipeline
-----------
1. **Speech encoder** (``speech_encoder.onnx``) — 24 kHz audio ->
   (cond_emb, prompt_token, ref_x_vector, prompt_feat).
   Run on *both* source and target audio.
2. **Conditional decoder** (``conditional_decoder.onnx``) — (speech_tokens,
   speaker_embeddings, speaker_features) -> (1, T) waveform at 24 kHz.

The LLM (language_model / embed_tokens) is NOT needed for VC: the source
audio's prompt tokens are concatenated directly with the target prompt tokens
and fed straight to the decoder — bypassing the generation loop entirely.
No tokenizer.json is downloaded or parsed.

Models: ``onnx-community/chatterbox-onnx`` (HF, Apache-2.0).
Output sample rate: 24 kHz.
Core deps: onnxruntime, numpy, soundfile, huggingface_hub.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from vconnx.engines.base import VoiceClonerBase, EngineEntry, register_engine

# Chatterbox outputs 24 kHz audio
_CHATTERBOX_SR = 24000

# HF model repo hosting the ONNX files
_HF_MODEL_ID = "onnx-community/chatterbox-onnx"

# ONNX filenames within the ``onnx/`` subdirectory of the repo
_SPEECH_ENC_ONNX = "onnx/speech_encoder.onnx"
_SPEECH_ENC_DATA = "onnx/speech_encoder.onnx_data"
_COND_DEC_ONNX = "onnx/conditional_decoder.onnx"
_COND_DEC_DATA = "onnx/conditional_decoder.onnx_data"


# ---------------------------------------------------------------------------
# Audio I/O helpers (soundfile + numpy; no librosa)
# ---------------------------------------------------------------------------


def _load_wav(path: str, target_sr: int = _CHATTERBOX_SR) -> np.ndarray:
    """Load *path* as float32 mono, resampling to *target_sr* if needed.

    Uses linear interpolation for resampling — sufficient quality for the
    speech-encoder input (band-limited speech signal).
    """
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        n_out = int(round(len(audio) * target_sr / sr))
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n_out),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
    return audio


def _save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    """Write *audio* (float32) as a 16-bit PCM WAV to *path*."""
    import soundfile as sf

    audio_clipped = np.clip(audio, -1.0, 1.0)
    sf.write(str(path), audio_clipped, sr, subtype="PCM_16")


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class ChatterboxAdapter(VoiceClonerBase):
    """Voice-conversion adapter backed by Chatterbox ONNX (VC path only).

    Uses two ONNX sessions:
    - ``speech_encoder`` -- encodes audio to speaker conditioning tensors.
    - ``conditional_decoder`` -- synthesizes waveform from speech tokens +
      speaker embeddings.

    The LLM / tokenizer components are skipped entirely: the VC path
    concatenates the source prompt tokens with the target prompt tokens and
    feeds them directly to the decoder (same approach as the upstream
    ``voice_convert`` implementation in chatterbox-onnx).

    Parameters
    ----------
    exaggeration:
        Voice-exaggeration scalar (default ``0.6``).  Stored for API
        compatibility; the conditional decoder does not expose this as a
        direct ONNX input — it is encoded into the speaker embeddings at
        training time.
    **cfg:
        Additional keyword arguments stored but not forwarded.
    """

    _sample_rate = _CHATTERBOX_SR

    def __init__(
        self,
        exaggeration: float = 0.6,
        **cfg,
    ):
        super().__init__(**cfg)
        self._exaggeration = exaggeration
        self._speech_enc_sess = None
        self._cond_dec_sess = None

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _ensure_models(self) -> None:
        if self._speech_enc_sess is not None:
            return

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for engine='chatterbox'. "
                "Install it with: pip install vconnx"
            ) from exc

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("huggingface_hub is required.") from exc

        # Download ONNX files and their external-data companions
        enc_path = hf_hub_download(repo_id=_HF_MODEL_ID, filename=_SPEECH_ENC_ONNX)
        hf_hub_download(repo_id=_HF_MODEL_ID, filename=_SPEECH_ENC_DATA)
        dec_path = hf_hub_download(repo_id=_HF_MODEL_ID, filename=_COND_DEC_ONNX)
        hf_hub_download(repo_id=_HF_MODEL_ID, filename=_COND_DEC_DATA)

        sess_opts = ort.SessionOptions()
        n = os.cpu_count() or 4
        sess_opts.inter_op_num_threads = n
        sess_opts.intra_op_num_threads = n
        providers = ["CPUExecutionProvider"]

        self._speech_enc_sess = ort.InferenceSession(
            enc_path, sess_options=sess_opts, providers=providers
        )
        self._cond_dec_sess = ort.InferenceSession(
            dec_path, sess_options=sess_opts, providers=providers
        )

    # ------------------------------------------------------------------
    # Speaker encoding
    # ------------------------------------------------------------------

    def _embed_speaker(self, audio: np.ndarray):
        """Run the speech encoder on *audio* (float32, 24 kHz).

        Returns
        -------
        tuple
            ``(cond_emb, prompt_token, ref_x_vector, prompt_feat)`` --
            the four tensors produced by the speech encoder.
        """
        self._ensure_models()
        inp = audio[np.newaxis, :].astype(np.float32)  # (1, T)
        cond_emb, prompt_token, ref_x_vector, prompt_feat = self._speech_enc_sess.run(
            None, {"audio_values": inp}
        )
        return cond_emb, prompt_token, ref_x_vector, prompt_feat

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clone_voice(
        self,
        audio: str,
        reference_voice: str,
        out_path: str,
    ) -> str:
        """Convert *audio* to sound like *reference_voice* using Chatterbox VC.

        The voice-conversion path bypasses the LLM generation loop:
        - Source audio and target audio are each embedded by the speech encoder.
        - The target prompt tokens are prepended to the source prompt tokens.
        - The concatenated tokens + target speaker embeddings are fed directly
          to the conditional decoder.

        Parameters
        ----------
        audio:
            Path to the source WAV file (any sample rate).
        reference_voice:
            Path to the reference speaker WAV file.
        out_path:
            Destination path for the 16-bit 24 kHz output WAV.

        Returns
        -------
        str
            Absolute path to the written output WAV.
        """
        self._ensure_models()

        # Load both audio clips at 24 kHz (speech encoder input rate)
        src_wav = _load_wav(str(audio), target_sr=_CHATTERBOX_SR)
        ref_wav = _load_wav(str(reference_voice), target_sr=_CHATTERBOX_SR)

        # Encode target (reference) speaker — speaker conditioning
        tgt_cond_emb, tgt_prompt_token, tgt_x_vector, tgt_prompt_feat = self._embed_speaker(ref_wav)

        # Encode source audio — we only need the prompt tokens
        _src_cond_emb, src_tokens, _src_x, _src_feat = self._embed_speaker(src_wav)

        # Concatenate: target prompt tokens + source speech tokens (VC shortcut)
        speech_tokens = np.concatenate([tgt_prompt_token, src_tokens], axis=1)

        # Run the conditional decoder
        wav = self._cond_dec_sess.run(
            None,
            {
                "speech_tokens": speech_tokens,
                "speaker_embeddings": tgt_x_vector,
                "speaker_features": tgt_prompt_feat,
            },
        )[0]
        wav = np.squeeze(wav, axis=0)  # (1, T) -> (T,)

        out_path = str(Path(out_path).resolve())
        _save_wav(out_path, wav, sr=_CHATTERBOX_SR)
        return out_path


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_engine(
    EngineEntry(
        alias="chatterbox",
        adapter_class=ChatterboxAdapter,
        description=(
            "Chatterbox AR codec-LM (Resemble AI). ONNX export via "
            "onnx-community/chatterbox-onnx (HF). Voice conversion at 24 kHz. "
            "VC path only — no tokenizer, no LLM generation loop."
        ),
        extras="",
        onnx_native=True,
    )
)
