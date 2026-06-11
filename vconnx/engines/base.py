"""Base adapter contract and engine registry for vconnx.

Adding a new engine
-------------------
1. Subclass :class:`VoiceClonerBase` and implement :meth:`clone_voice`.
2. Create an :class:`EngineEntry` describing the engine.
3. Call :func:`register_engine` (or add to ``ENGINE_REGISTRY`` directly).
4. Expose the adapter via an optional extras group in *pyproject.toml*.

The adapter is intentionally thin — all heavy lifting (model download,
inference session management) belongs in the engine package itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Type


# ---------------------------------------------------------------------------
# Public data model
# ---------------------------------------------------------------------------


@dataclass
class EngineEntry:
    """Registry entry describing one voice-cloning engine adapter."""

    alias: str
    adapter_class: "Type[VoiceClonerBase]"
    description: str = ""
    #: pip extras key (e.g. "chatterbox" → ``pip install vconnx[chatterbox]``)
    extras: str = ""
    #: ONNX models are available / can be exported — True for all registered engines
    onnx_native: bool = True


ENGINE_REGISTRY: Dict[str, EngineEntry] = {}


def register_engine(entry: EngineEntry) -> EngineEntry:
    """Register an engine entry; returns *entry* for decorator-style use."""
    ENGINE_REGISTRY[entry.alias] = entry
    return entry


def get_engine(alias: str) -> EngineEntry:
    """Retrieve a registry entry; raises *KeyError* with a helpful message."""
    if alias not in ENGINE_REGISTRY:
        known = ", ".join(sorted(ENGINE_REGISTRY))
        raise KeyError(
            f"Unknown engine {alias!r}. Known engines: {known or '(none registered)'}"
        )
    return ENGINE_REGISTRY[alias]


# ---------------------------------------------------------------------------
# Adapter base class
# ---------------------------------------------------------------------------


class VoiceClonerBase:
    """Abstract base for per-engine voice-cloning adapters.

    Subclasses **must** implement :meth:`clone_voice`.

    Parameters
    ----------
    **cfg:
        Engine-specific keyword arguments forwarded to the underlying model.
    """

    #: Override in subclass to expose the true output sample rate.
    _sample_rate: int = 16000

    def __init__(self, **cfg):
        self._cfg = cfg

    def clone_voice(
        self,
        audio: str,
        reference_voice: str,
        out_path: str,
    ) -> str:
        """Convert *audio* to sound like *reference_voice*.

        Parameters
        ----------
        audio:
            Path to the source WAV file (any sample rate; 16-bit PCM recommended).
        reference_voice:
            Path to a reference WAV file providing the target speaker identity.
        out_path:
            Destination path for the converted 16-bit WAV.

        Returns
        -------
        str
            Absolute path to the written output file (same as *out_path*).
        """
        raise NotImplementedError

    @property
    def sample_rate(self) -> int:
        """Output sample rate in Hz."""
        return self._sample_rate
