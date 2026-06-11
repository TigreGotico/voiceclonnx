"""vconnx — pure-ONNX multi-engine voice-cloning library.

**Audio-to-audio only.**  vconnx converts the voice in an existing audio
file to sound like a reference speaker.  Text-driven synthesis belongs to
TTS engines; see the README scope section.

Runtime dependencies: onnxruntime, numpy, huggingface_hub (no torch).

Engine registry
---------------
Engines are discovered by importing their adapter module.  The built-in
engines ship alongside this package and are auto-imported here.

``"chatterbox"`` (default):
    Chatterbox AR codec-LM ONNX export (onnx-community/chatterbox-onnx).
    Requires: ``pip install vconnx[chatterbox]``

``"triaan"``:
    TriAAN-VC: CPC encoder + Triple Adaptive Attention Normalization decoder
    + ParallelWaveGAN vocoder. Zero-shot any-to-any VC at 16 kHz.
    Requires: ``pip install vconnx[triaan]``

Usage
-----
::

    from vconnx import VoiceCloner

    cloner = VoiceCloner(engine="chatterbox")
    out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
    print(cloner.sample_rate)   # 24000
"""

from vconnx.cloner import VoiceCloner
from vconnx.engines.base import (
    ENGINE_REGISTRY,
    EngineEntry,
    VoiceClonerBase,
    get_engine,
    register_engine,
)

# Auto-import built-in engine adapters so they self-register
import vconnx.engines.chatterbox  # noqa: F401
import vconnx.engines.freevc  # noqa: F401
import vconnx.engines.knnvc  # noqa: F401
import vconnx.engines.openvoice  # noqa: F401
import vconnx.engines.rvc  # noqa: F401
import vconnx.engines.triaan  # noqa: F401

__all__ = [
    "VoiceCloner",
    "ENGINE_REGISTRY",
    "EngineEntry",
    "VoiceClonerBase",
    "get_engine",
    "register_engine",
]
