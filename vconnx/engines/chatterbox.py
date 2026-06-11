"""Chatterbox-ONNX adapter for vconnx.

Requires: ``pip install vconnx[chatterbox]`` (pulls in ``chatterbox_onnx``).

The adapter is a thin shim over :class:`chatterbox_onnx.ChatterboxOnnx`.
Only the voice-conversion path is used; TTS capabilities of chatterbox_onnx
are intentionally not exposed here (audio-to-audio only).
"""

from __future__ import annotations

from vconnx.engines.base import VoiceClonerBase, EngineEntry, register_engine

# Chatterbox outputs 24 kHz audio
_CHATTERBOX_SR = 24000


class ChatterboxAdapter(VoiceClonerBase):
    """Voice-cloning adapter backed by ``chatterbox_onnx.ChatterboxOnnx``.

    Parameters
    ----------
    quantized:
        Use the Q4-quantized language model (default ``True``).
        Set to ``False`` for full-precision (much larger, slower on CPU).
    exaggeration:
        Voice exaggeration factor forwarded to Chatterbox (default ``0.6``).
    max_new_tokens:
        Maximum speech tokens to generate per call (default ``512``).
    **cfg:
        Additional keyword arguments stored but not forwarded.
    """

    _sample_rate = _CHATTERBOX_SR

    def __init__(
        self,
        quantized: bool = True,
        exaggeration: float = 0.6,
        max_new_tokens: int = 512,
        **cfg,
    ):
        super().__init__(**cfg)
        self._quantized = quantized
        self._exaggeration = exaggeration
        self._max_new_tokens = max_new_tokens
        self._model = None  # lazy-loaded on first call

    def _ensure_model(self):
        if self._model is None:
            try:
                from chatterbox_onnx import ChatterboxOnnx
            except ImportError as exc:
                raise ImportError(
                    "chatterbox_onnx is required for engine='chatterbox'. "
                    "Install it with: pip install vconnx[chatterbox]"
                ) from exc
            self._model = ChatterboxOnnx(quantized=self._quantized)

    def clone_voice(
        self,
        audio: str,
        reference_voice: str,
        out_path: str,
    ) -> str:
        """Convert *audio* to sound like *reference_voice* using Chatterbox VC."""
        self._ensure_model()
        return self._model.voice_convert(
            source_audio_path=audio,
            target_voice_path=reference_voice,
            output_file_name=out_path,
            exaggeration=self._exaggeration,
            max_new_tokens=self._max_new_tokens,
            apply_watermark=False,
        )


# Register entry
register_engine(
    EngineEntry(
        alias="chatterbox",
        adapter_class=ChatterboxAdapter,
        description=(
            "Chatterbox AR codec-LM (Resemble AI). ONNX export via "
            "onnx-community/chatterbox-onnx (HF). Voice conversion at 24 kHz."
        ),
        extras="chatterbox",
        onnx_native=True,
    )
)
