"""Tests for VoiceCloner facade using a mock engine."""

import os
import pytest

from voiceclonnx import VoiceCloner
from voiceclonnx.engines.base import ENGINE_REGISTRY, EngineEntry, VoiceClonerBase, register_engine


# ---------------------------------------------------------------------------
# Fixture: mock engine registered for the duration of tests
# ---------------------------------------------------------------------------

class MockAdapter(VoiceClonerBase):
    """In-memory mock that writes an empty file."""

    _sample_rate = 16000

    def clone_voice(self, audio, reference_voice, out_path):
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(b"RIFF")  # minimal placeholder
        return out_path


@pytest.fixture(autouse=True)
def mock_engine():
    entry = EngineEntry(
        alias="_mock",
        adapter_class=MockAdapter,
        description="mock engine for tests",
        extras="",
        onnx_native=False,
    )
    register_engine(entry)
    yield
    ENGINE_REGISTRY.pop("_mock", None)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_voice_cloner_default_engine():
    """VoiceCloner resolves the correct adapter class."""
    import voiceclonnx  # noqa: F401
    cloner = VoiceCloner(engine="_mock")
    assert cloner.engine == "_mock"
    assert cloner.sample_rate == 16000


def test_clone_voice_returns_path(tmp_path):
    cloner = VoiceCloner(engine="_mock")
    src = tmp_path / "src.wav"
    ref = tmp_path / "ref.wav"
    out = tmp_path / "out.wav"
    src.write_bytes(b"\x00" * 44)
    ref.write_bytes(b"\x00" * 44)

    result = cloner.clone_voice(str(src), str(ref), str(out))
    assert result == str(out)
    assert out.exists()


def test_clone_voice_default_out_path(tmp_path):
    """When out_path is None, an _converted suffix path is returned."""
    cloner = VoiceCloner(engine="_mock")
    src = tmp_path / "speech.wav"
    ref = tmp_path / "ref.wav"
    src.write_bytes(b"\x00" * 44)
    ref.write_bytes(b"\x00" * 44)

    result = cloner.clone_voice(str(src), str(ref))
    assert "speech_converted" in result
    assert result.endswith(".wav")


def test_unknown_engine_raises():
    with pytest.raises(KeyError, match="no_such_engine"):
        VoiceCloner(engine="no_such_engine")


def test_no_tts_surface():
    """VoiceCloner must not expose tts_clone or supports_tts."""
    cloner = VoiceCloner(engine="_mock")
    assert not hasattr(cloner, "tts_clone"), "tts_clone must not exist on VoiceCloner"
    assert not hasattr(cloner, "supports_tts"), "supports_tts must not exist on VoiceCloner"
