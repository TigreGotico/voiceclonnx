"""vconnx — pure-ONNX multi-engine voice-cloning library.

**Audio-to-audio only.**  vconnx converts the voice in an existing audio
file to sound like a reference speaker.  Text-driven synthesis belongs to
TTS engines; see the README scope section.

Runtime dependencies: onnxruntime, numpy, soundfile, huggingface_hub (no torch,
no librosa). A single ``pip install vconnx`` enables every engine — no per-engine
extras required. ONNX models are downloaded on first use from Hugging Face Hub.

Engine registry
---------------
Engines are discovered by importing their adapter module.  The built-in
engines ship alongside this package and are auto-imported here.

``"chatterbox"``:
    Chatterbox AR codec-LM — VC path only (no tokenizer, no LLM generation).
    Models: onnx-community/chatterbox-onnx.  Output: 24 kHz.

``"focalcodec"``:
    FocalCodec: WavLM encoder + kNN cosine matching (pure numpy) + Vocos
    ISTFT decoder. Zero-shot any-to-any VC at 16 kHz. Apache-2.0.

``"freevc"``:
    FreeVC: WavLM-Large + GE2E speaker encoder + VITS decoder.
    Zero-shot any-to-any VC at 16 kHz. MIT.

``"knnvc"``:
    kNN-VC: WavLM-Large layer-6 + L2-kNN matching (pure numpy) + HiFi-GAN.
    Zero-shot any-to-any VC at 16 kHz. MIT.

``"openvoice"``:
    OpenVoice v2: tone-color reference encoder + VITS-style converter.
    Zero-shot any-to-any VC at 22 kHz. MIT.

``"rvc"``:
    RVC: ContentVec + RMVPE F0 + VITS synthesizer. Any-to-ONE.
    reference_voice = path to an RVC .onnx model. MIT.

    24 kHz, 12.5 Hz frame rate, 32 code streams. CC BY 4.0.
    Apache-2.0.
    Mimi: Kyutai RVQ codec VC — stream-0 (WavLM-semantic) from source,
    SpeechTokenizer: hierarchical RVQ-8 codec — RVQ-1 (HuBERT-distilled)
    carries content, RVQ-2..8 carry timbre.  VC: source RVQ-1 tokens +
    reference RVQ-2..8 tokens → decode.  Zero-shot any-to-any VC at 16 kHz.
    streams 1–31 (acoustic/timbre) from reference (pure numpy stream swap).
``"mimi"``:
``"speechtokenizer"``:

``"triaan"``:
    TriAAN-VC: CPC encoder + Triple Adaptive Attention Normalization decoder
    + ParallelWaveGAN vocoder. Zero-shot any-to-any VC at 16 kHz. MIT.

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
import vconnx.engines.focalcodec  # noqa: F401
import vconnx.engines.freevc  # noqa: F401
import vconnx.engines.knnvc  # noqa: F401
import vconnx.engines.openvoice  # noqa: F401
import vconnx.engines.rvc  # noqa: F401
import vconnx.engines.mimi  # noqa: F401
import vconnx.engines.speechtokenizer  # noqa: F401
import vconnx.engines.triaan  # noqa: F401
import vconnx.engines.facodec  # noqa: F401

__all__ = [
    "VoiceCloner",
    "ENGINE_REGISTRY",
    "EngineEntry",
    "VoiceClonerBase",
    "get_engine",
    "register_engine",
]
