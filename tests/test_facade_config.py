"""Facade config-plumbing tests.

Verifies that engine kwargs passed to VoiceCloner reach the adapter
constructor and are stored/accessible, without touching any real engine
or ONNX sessions.
"""

from __future__ import annotations

import pytest

from voiceclonnx import VoiceCloner
from voiceclonnx.engines.base import ENGINE_REGISTRY, EngineEntry, VoiceClonerBase, register_engine


# ---------------------------------------------------------------------------
# Spy adapter that records its constructor kwargs
# ---------------------------------------------------------------------------


class SpyAdapter(VoiceClonerBase):
    """Records all kwargs passed to __init__ for introspection in tests."""

    _sample_rate = 22050
    _instances: list = []

    def __init__(self, **cfg):
        super().__init__(**cfg)
        SpyAdapter._instances.append(self)
        # Explicitly capture known kwargs
        self.received_cfg = dict(cfg)

    def clone_voice(self, audio, reference_voice, out_path):
        with open(out_path, "wb") as f:
            f.write(b"RIFF")
        return out_path


@pytest.fixture(autouse=True)
def spy_engine():
    SpyAdapter._instances.clear()
    entry = EngineEntry(
        alias="_spy",
        adapter_class=SpyAdapter,
        description="spy adapter for config tests",
        extras="",
        onnx_native=False,
    )
    register_engine(entry)
    yield
    ENGINE_REGISTRY.pop("_spy", None)
    SpyAdapter._instances.clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_no_extra_kwargs_stored_in_cfg():
    """With no extra kwargs, _cfg is empty."""
    VoiceCloner(engine="_spy")
    adapter = SpyAdapter._instances[-1]
    assert adapter.received_cfg == {}


def test_single_kwarg_reaches_adapter():
    """A single engine kwarg is forwarded to the adapter constructor."""
    VoiceCloner(engine="_spy", quantized=True)
    adapter = SpyAdapter._instances[-1]
    assert adapter.received_cfg.get("quantized") is True


def test_multiple_kwargs_reach_adapter():
    """Multiple engine kwargs are all forwarded."""
    VoiceCloner(engine="_spy", quantized=False, k=8, some_flag="hello")
    adapter = SpyAdapter._instances[-1]
    assert adapter.received_cfg["quantized"] is False
    assert adapter.received_cfg["k"] == 8
    assert adapter.received_cfg["some_flag"] == "hello"


def test_engine_kwarg_not_forwarded():
    """The 'engine' key itself is not forwarded as a cfg kwarg."""
    VoiceCloner(engine="_spy", exaggeration=0.7)
    adapter = SpyAdapter._instances[-1]
    assert "engine" not in adapter.received_cfg


def test_sample_rate_from_adapter():
    """sample_rate property reads from the adapter class attribute."""
    cloner = VoiceCloner(engine="_spy")
    assert cloner.sample_rate == 22050


def test_engine_property():
    """engine property returns the alias passed to VoiceCloner."""
    cloner = VoiceCloner(engine="_spy")
    assert cloner.engine == "_spy"


def test_clone_voice_delegates_to_adapter(tmp_path):
    """clone_voice result comes from the adapter's clone_voice method."""
    cloner = VoiceCloner(engine="_spy")
    src = tmp_path / "s.wav"
    ref = tmp_path / "r.wav"
    out = tmp_path / "o.wav"
    src.write_bytes(b"\x00" * 44)
    ref.write_bytes(b"\x00" * 44)

    result = cloner.clone_voice(str(src), str(ref), str(out))
    assert result == str(out)
    assert out.exists()


def test_clone_voice_default_out_suffix(tmp_path):
    """When out_path is omitted, result has '_converted' in name."""
    cloner = VoiceCloner(engine="_spy")
    src = tmp_path / "speech.wav"
    ref = tmp_path / "ref.wav"
    src.write_bytes(b"\x00" * 44)
    ref.write_bytes(b"\x00" * 44)

    result = cloner.clone_voice(str(src), str(ref))
    assert "speech_converted" in result
    assert result.endswith(".wav")


def test_adapter_replaced_per_instance():
    """Each VoiceCloner instantiation creates a fresh adapter."""
    VoiceCloner(engine="_spy", x=1)
    VoiceCloner(engine="_spy", x=2)
    assert len(SpyAdapter._instances) == 2
    assert SpyAdapter._instances[0].received_cfg["x"] == 1
    assert SpyAdapter._instances[1].received_cfg["x"] == 2
