"""VoiceCloner — the public facade for voiceclonnx.

All engines are accessed through this class.  Internally it delegates to
the per-engine :class:`~voiceclonnx.engines.base.VoiceClonerBase` adapter
resolved from :data:`~voiceclonnx.engines.base.ENGINE_REGISTRY`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from voiceclonnx.engines.base import VoiceClonerBase, get_engine

DEFAULT_ENGINE = "chatterbox"


class VoiceCloner:
    """Unified facade for ONNX voice-cloning engines.

    Parameters
    ----------
    engine:
        Engine alias from the registry (default ``"chatterbox"``).
    **cfg:
        Engine-specific keyword arguments forwarded to the adapter
        constructor (e.g. ``quantized=True``, ``exaggeration=0.6``).

    Examples
    --------
    ::

        cloner = VoiceCloner(engine="chatterbox", exaggeration=0.5)
        out = cloner.clone_voice("src.wav", "ref.wav", "out.wav")
        print(cloner.sample_rate)  # 24000
    """

    def __init__(self, engine: str = DEFAULT_ENGINE, **cfg):
        entry = get_engine(engine)
        self._adapter: VoiceClonerBase = entry.adapter_class(**cfg)
        self._engine_alias = engine

    def clone_voice(
        self,
        audio: str,
        reference_voice: str,
        out_path: Optional[str] = None,
    ) -> str:
        """Convert *audio* to sound like *reference_voice*.

        Parameters
        ----------
        audio:
            Path to the source WAV file.  Any sample rate; 16-bit PCM
            recommended for best compatibility.
        reference_voice:
            Path to a short (~5–30 s) reference WAV providing the target
            speaker identity.
        out_path:
            Destination path for the converted WAV (16-bit, engine SR).
            Defaults to ``audio`` with ``_converted`` suffix in the same dir.

        Returns
        -------
        str
            Path to the written output file.
        """
        if out_path is None:
            p = Path(audio)
            out_path = str(p.with_stem(p.stem + "_converted"))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        return self._adapter.clone_voice(audio, reference_voice, out_path)

    @property
    def sample_rate(self) -> int:
        """Output sample rate in Hz for the active engine."""
        return self._adapter.sample_rate

    @property
    def engine(self) -> str:
        """Active engine alias."""
        return self._engine_alias
