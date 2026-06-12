"""Tests for the engine registry and base adapter."""

import pytest
from voiceclonnx.engines.base import (
    ENGINE_REGISTRY,
    EngineEntry,
    VoiceClonerBase,
    get_engine,
    register_engine,
)


def test_registry_has_chatterbox():
    """Chatterbox engine must be registered on import."""
    import voiceclonnx  # triggers auto-import of adapters  # noqa: F401
    assert "chatterbox" in ENGINE_REGISTRY


def test_get_engine_returns_entry():
    import voiceclonnx  # noqa: F401
    entry = get_engine("chatterbox")
    assert entry.alias == "chatterbox"
    assert entry.onnx_native is True
    assert entry.extras == ""


def test_get_engine_unknown_raises():
    with pytest.raises(KeyError, match="unknown_xyz"):
        get_engine("unknown_xyz")


def test_register_engine_roundtrip():
    """Custom engine can be registered and retrieved."""

    class DummyAdapter(VoiceClonerBase):
        _sample_rate = 22050

        def clone_voice(self, audio, reference_voice, out_path):
            return out_path

    entry = EngineEntry(
        alias="_test_dummy",
        adapter_class=DummyAdapter,
        description="test only",
        extras="",
        onnx_native=False,
    )
    register_engine(entry)
    assert get_engine("_test_dummy") is entry
    # cleanup
    del ENGINE_REGISTRY["_test_dummy"]


def test_base_adapter_sample_rate():
    class MyAdapter(VoiceClonerBase):
        _sample_rate = 44100

    adapter = MyAdapter()
    assert adapter.sample_rate == 44100


def test_base_adapter_clone_voice_not_implemented():
    adapter = VoiceClonerBase()
    with pytest.raises(NotImplementedError):
        adapter.clone_voice("a.wav", "b.wav", "c.wav")
