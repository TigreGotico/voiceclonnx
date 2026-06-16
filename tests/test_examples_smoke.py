"""Examples smoke tests.

Runs examples/basic_clone.py with a monkeypatched engine so CI needs no
model downloads, no edge-tts, and no HF Hub access.

Strategy:
- Monkeypatch VoiceCloner so that engine="knnvc" resolves to _SmokeAdapter
  (writes a tiny placeholder WAV) instead of the real knnvc adapter.
- Patch _synth_wav so it writes a tiny WAV directly (no edge-tts).
- Import and call main() from the example module.
- Verify the reported output path exists and has the _converted suffix.
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Smoke WAV writer (no soundfile dependency)
# ---------------------------------------------------------------------------


def _write_smoke_wav(path: str, sr: int = 16000, duration_s: float = 0.5) -> None:
    """Write a tiny sine-wave WAV without soundfile."""
    n = int(sr * duration_s)
    t = np.linspace(0, duration_s, n, endpoint=False)
    data = (np.sin(2 * np.pi * 440 * t) * 16383).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())


# ---------------------------------------------------------------------------
# Smoke adapter — no ONNX, no HF, no model download
# ---------------------------------------------------------------------------


class _SmokeKNNVCAdapter:
    """Drop-in for KNNVCAdapter that just copies a sine-wave placeholder."""

    sample_rate: int = 16000

    def __init__(self, **_cfg):
        pass

    def clone_voice(self, audio: str, reference_voice: str, out_path: Optional[str] = None) -> str:
        if out_path is None:
            p = Path(audio)
            out_path = str(p.with_stem(p.stem + "_converted"))
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        _write_smoke_wav(out_path, sr=16000, duration_s=0.5)
        return out_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_basic_clone_smoke(tmp_path, monkeypatch):
    """basic_clone.py main() runs end-to-end with mocked synth and engine."""
    # 1. Ensure examples/ is importable
    examples_dir = Path(__file__).parent.parent / "examples"
    if str(examples_dir) not in sys.path:
        sys.path.insert(0, str(examples_dir))

    # 2. Pre-write the WAV files so _synth_wav is never called for real
    demo_dir = tmp_path / "voiceclonnx_demo"
    demo_dir.mkdir()
    src = demo_dir / "source.wav"
    ref = demo_dir / "reference.wav"
    _write_smoke_wav(str(src))
    _write_smoke_wav(str(ref))

    # 3. Import the example module freshly
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "basic_clone",
        examples_dir / "basic_clone.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # 4. Patch _synth_wav so it just writes the pre-made files (no-op since files exist)
    def _fake_synth(text, voice, path):
        _write_smoke_wav(str(path))

    monkeypatch.setattr(mod, "_synth_wav", _fake_synth)

    # 5. Patch tempfile.gettempdir to point to tmp_path
    import tempfile
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    # 6. Patch VoiceCloner to use _SmokeKNNVCAdapter


    class _SmokeCloner:
        def __init__(self, engine="knnvc", **cfg):
            self._adapter = _SmokeKNNVCAdapter(**cfg)
            self._engine = engine

        @property
        def sample_rate(self):
            return self._adapter.sample_rate

        @property
        def engine(self):
            return self._engine

        def clone_voice(self, audio, reference_voice, out_path=None):
            return self._adapter.clone_voice(audio, reference_voice, out_path)

    # Patch VoiceCloner at the voiceclonnx module level (it's imported inside main())
    import voiceclonnx
    import voiceclonnx.cloner
    monkeypatch.setattr(voiceclonnx, "VoiceCloner", _SmokeCloner)
    monkeypatch.setattr(voiceclonnx.cloner, "VoiceCloner", _SmokeCloner)

    # 7. Run and capture output
    import io
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stdout", captured)

    mod.main()

    output = captured.getvalue()
    assert "Converted:" in output
    assert "source_converted.wav" in output
    assert "Engine sample rate: 16000" in output
    assert "Done." in output


def test_basic_clone_output_file_exists(tmp_path, monkeypatch):
    """basic_clone.py produces a WAV file that can be opened with wave."""
    examples_dir = Path(__file__).parent.parent / "examples"

    import importlib.util
    spec = importlib.util.spec_from_file_location("basic_clone2", examples_dir / "basic_clone.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def _fake_synth(text, voice, path):
        _write_smoke_wav(str(path))

    monkeypatch.setattr(mod, "_synth_wav", _fake_synth)

    import tempfile
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path))

    class _SmokeCloner:
        def __init__(self, engine="knnvc", **cfg):
            self._adapter = _SmokeKNNVCAdapter(**cfg)

        @property
        def sample_rate(self):
            return self._adapter.sample_rate

        @property
        def engine(self):
            return "knnvc"

        def clone_voice(self, audio, reference_voice, out_path=None):
            return self._adapter.clone_voice(audio, reference_voice, out_path)

    import voiceclonnx
    import voiceclonnx.cloner
    monkeypatch.setattr(voiceclonnx, "VoiceCloner", _SmokeCloner)
    monkeypatch.setattr(voiceclonnx.cloner, "VoiceCloner", _SmokeCloner)

    import io
    monkeypatch.setattr(sys, "stdout", io.StringIO())

    mod.main()

    converted = tmp_path / "voiceclonnx_demo" / "source_converted.wav"
    assert converted.exists(), f"Expected {converted} to exist"

    with wave.open(str(converted), "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getnframes() > 0


def test_wav_info_helper(tmp_path):
    """_wav_info returns a string with sr and samples."""
    examples_dir = Path(__file__).parent.parent / "examples"

    import importlib.util
    spec = importlib.util.spec_from_file_location("basic_clone3", examples_dir / "basic_clone.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    wav_path = tmp_path / "test.wav"
    _write_smoke_wav(str(wav_path), sr=16000, duration_s=1.0)

    info = mod._wav_info(wav_path)
    assert "sr=16000" in info
    assert "samples=" in info
