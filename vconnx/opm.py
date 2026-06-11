"""OPM voice-clone plugin bundled with vconnx.

Entry-point group: ``opm.vc``
Entry-point name:  ``ovos-vc-plugin-chatterbox-onnx``

This module is import-guarded — vconnx itself does not depend on
ovos-plugin-manager.  The plugin activates automatically once OPM ships
the ``VoiceClonePlugin`` base class (``opm.vc`` family).

Until OPM ships that family, this file is inert for all callers that
don't import it explicitly.
"""

from __future__ import annotations

try:
    from ovos_plugin_manager.templates.voice_clone import VoiceClonePlugin  # type: ignore
    _HAS_OPM = True
except ImportError:
    _HAS_OPM = False

if _HAS_OPM:
    from vconnx import VoiceCloner

    class ChatterboxVCPlugin(VoiceClonePlugin):
        """OPM voice-clone plugin wrapping the Chatterbox engine.

        Configuration keys (all optional):
            exaggeration (float): Voice exaggeration factor (default 0.6).
            max_new_tokens (int): Max speech tokens per call (default 512).
            quantized (bool): Use Q4-quantized LM (default True).
        """

        def __init__(self, config=None):
            super().__init__(config=config)
            self._cloner = VoiceCloner(
                engine="chatterbox",
                quantized=self.config.get("quantized", True),
                exaggeration=self.config.get("exaggeration", 0.6),
                max_new_tokens=self.config.get("max_new_tokens", 512),
            )

        def clone_voice(self, audio: str, reference_voice: str, out_path: str) -> str:
            return self._cloner.clone_voice(audio, reference_voice, out_path)

        @property
        def sample_rate(self) -> int:
            return self._cloner.sample_rate

else:
    # Stub so that the entry-point can be imported without OPM installed.
    class ChatterboxVCPlugin:  # type: ignore[no-redef]
        """Placeholder — install ovos-plugin-manager to activate."""

        def __init__(self, *args, **kwargs):
            raise ImportError(
                "ovos-plugin-manager is required for the OPM plugin. "
                "Use VoiceCloner directly if you do not need OPM integration."
            )
