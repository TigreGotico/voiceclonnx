"""Tests for the FreeVC adapter — vconnx/engines/freevc.py.

Structure
---------
- Registry wiring (no model loading)
- Log-mel spectrogram helper (real numpy computation)
- Mock-session contract tests (full adapter pipeline with stubbed ORT)
- Pure-numpy pieces tested on synthetic data
- skipif-gated e2e: real conversion using edge-tts generated audio
"""

from __future__ import annotations

import os
import wave
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_wav(path: str, duration_s: float = 1.0, sr: int = 16000) -> str:
    """Write a short sine-wave WAV for testing."""
    n = int(duration_s * sr)
    t = np.linspace(0, duration_s, n, endpoint=False)
    wave_data = (np.sin(2 * np.pi * 440 * t) * 32767 * 0.5).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(wave_data.tobytes())
    return path


# ---------------------------------------------------------------------------
# 1. Registry wiring
# ---------------------------------------------------------------------------


def test_freevc_registered():
    """freevc engine must appear in ENGINE_REGISTRY after importing vconnx."""
    import vconnx.engines.freevc  # noqa: F401 — trigger registration
    from vconnx.engines.base import ENGINE_REGISTRY

    assert "freevc" in ENGINE_REGISTRY


def test_freevc_entry_metadata():
    import vconnx.engines.freevc  # noqa: F401
    from vconnx.engines.base import get_engine

    entry = get_engine("freevc")
    assert entry.alias == "freevc"
    assert entry.onnx_native is True
    assert entry.extras == ""
    assert entry.adapter_class.__name__ == "FreeVCAdapter"


def test_freevc_sample_rate():
    from vconnx.engines.freevc import FreeVCAdapter

    adapter = FreeVCAdapter()
    assert adapter.sample_rate == 16000


def test_freevc_quantized_flag():
    from vconnx.engines.freevc import FreeVCAdapter

    a = FreeVCAdapter(quantized=True)
    assert a._quantized is True
    b = FreeVCAdapter(quantized=False)
    assert b._quantized is False


# ---------------------------------------------------------------------------
# 2. Log-mel spectrogram (real computation, no mocks)
# ---------------------------------------------------------------------------


def test_log_mel_shape():
    """Log-mel output shape should be (n_frames, 40)."""
    pass  # no longer requires librosa; pure numpy
    from vconnx.engines.freevc import _compute_log_mel

    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)  # 1 second silence
    mel = _compute_log_mel(audio, sr=sr)

    assert mel.ndim == 2
    assert mel.shape[1] == 40
    assert mel.dtype == np.float32


def test_log_mel_nonzero_for_signal():
    """Non-silent audio should produce non-uniform mel values."""
    pass  # no longer requires librosa; pure numpy
    from vconnx.engines.freevc import _compute_log_mel

    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    mel = _compute_log_mel(audio, sr=sr)

    # Should not be all the same value
    assert mel.std() > 0.0


# ---------------------------------------------------------------------------
# 3. Adapter contract with mocked ORT sessions
# ---------------------------------------------------------------------------


class _MockWavLMSession:
    """Fake WavLM ORT session — returns (1, T//320, 1024) features."""

    def run(self, output_names, inputs):
        n_samples = inputs["input_values"].shape[1]
        n_frames = max(1, n_samples // 320)
        rng = np.random.default_rng(0)
        return [rng.random((1, n_frames, 1024), dtype=np.float32)]


class _MockSpeakerSession:
    """Fake speaker encoder ORT session — returns (1, 256) embedding."""

    def run(self, output_names, inputs):
        rng = np.random.default_rng(1)
        emb = rng.random((1, 256), dtype=np.float32)
        # L2-normalise to mimic real encoder
        emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        return [emb]


class _MockDecoderSession:
    """Fake VITS decoder ORT session — returns (1, 1, frames*320) samples."""

    def run(self, output_names, inputs):
        n_frames = inputs["c"].shape[2]
        n_samples = n_frames * 320
        rng = np.random.default_rng(2)
        return [rng.random((1, 1, n_samples), dtype=np.float32) * 0.1]


def test_adapter_clone_voice_mock(tmp_path):
    """Adapter pipeline completes with mocked ORT sessions."""
    pass  # no longer requires librosa; pure numpy
    from vconnx.engines.freevc import FreeVCAdapter

    src_wav = _make_wav(str(tmp_path / "src.wav"), duration_s=1.0)
    ref_wav = _make_wav(str(tmp_path / "ref.wav"), duration_s=2.0)
    out_wav = str(tmp_path / "out.wav")

    adapter = FreeVCAdapter()
    adapter._wavlm_sess = _MockWavLMSession()
    adapter._spk_sess = _MockSpeakerSession()
    adapter._decoder_sess = _MockDecoderSession()

    result = adapter.clone_voice(src_wav, ref_wav, out_wav)
    assert result == str(Path(out_wav).resolve())
    assert Path(out_wav).exists()

    with wave.open(out_wav, "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getsampwidth() == 2
        assert wf.getnframes() > 0


def test_adapter_output_is_16khz(tmp_path):
    """Output WAV must be 16 kHz regardless of input sample rate."""
    pass  # no longer requires librosa; pure numpy
    from vconnx.engines.freevc import FreeVCAdapter

    src_path = str(tmp_path / "src44.wav")
    n = 44100
    with wave.open(src_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(np.zeros(n, dtype=np.int16).tobytes())

    ref_wav = _make_wav(str(tmp_path / "ref.wav"))
    out_wav = str(tmp_path / "out.wav")

    adapter = FreeVCAdapter()
    adapter._wavlm_sess = _MockWavLMSession()
    adapter._spk_sess = _MockSpeakerSession()
    adapter._decoder_sess = _MockDecoderSession()

    adapter.clone_voice(src_path, ref_wav, out_wav)

    with wave.open(out_wav, "rb") as wf:
        assert wf.getframerate() == 16000


def test_adapter_lazy_load_raises_without_onnxruntime(tmp_path, monkeypatch):
    """Missing onnxruntime raises ImportError with a helpful message."""
    import builtins
    from vconnx.engines.freevc import FreeVCAdapter

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("onnxruntime not installed")
        return real_import(name, *args, **kwargs)

    adapter = FreeVCAdapter()
    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="onnxruntime"):
            adapter._ensure_models()


def test_decode_input_shapes(tmp_path):
    """Decoder receives correct tensor shapes: c=(1,1024,T), g=(1,256)."""
    pass  # no longer requires librosa; pure numpy
    from vconnx.engines.freevc import FreeVCAdapter

    captured = {}

    class _InspectDecoder:
        def run(self, output_names, inputs):
            captured["c_shape"] = inputs["c"].shape
            captured["g_shape"] = inputs["g"].shape
            n_frames = inputs["c"].shape[2]
            return [np.zeros((1, 1, n_frames * 320), dtype=np.float32)]

    src_wav = _make_wav(str(tmp_path / "src.wav"))
    ref_wav = _make_wav(str(tmp_path / "ref.wav"))
    out_wav = str(tmp_path / "out.wav")

    adapter = FreeVCAdapter()
    adapter._wavlm_sess = _MockWavLMSession()
    adapter._spk_sess = _MockSpeakerSession()
    adapter._decoder_sess = _InspectDecoder()

    adapter.clone_voice(src_wav, ref_wav, out_wav)

    c_shape = captured["c_shape"]
    g_shape = captured["g_shape"]
    assert c_shape[0] == 1
    assert c_shape[1] == 1024
    assert c_shape[2] > 0
    assert g_shape == (1, 256)


# ---------------------------------------------------------------------------
# 4. E2E test — real model, real audio (opt-in via VCONNX_E2E=1)
# ---------------------------------------------------------------------------

_SKIP_E2E = not os.environ.get("VCONNX_E2E", "")
_E2E_REASON = (
    "E2E freevc test downloads >1 GB of public models; set VCONNX_E2E=1 to run."
)


@pytest.mark.skipif(_SKIP_E2E, reason=_E2E_REASON)
def test_e2e_freevc_clone_edge_tts_voices(tmp_path):
    """Real end-to-end: convert an edge-tts source to a second edge-tts reference.

    Validates:
    - Output is a valid 16-bit 16 kHz WAV.
    - Duration is in a reasonable range (0.5 s – 15 s).
    - File size > 0 bytes.
    """
    import asyncio
    import subprocess
    import soundfile as sf

    try:
        import edge_tts
    except ImportError:
        pytest.skip("edge-tts not installed")

    from vconnx.engines.freevc import FreeVCAdapter

    async def _synth(text: str, voice: str, out: str):
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(out)

    src_mp3 = str(tmp_path / "src.mp3")
    ref_mp3 = str(tmp_path / "ref.mp3")
    src_wav = str(tmp_path / "src.wav")
    ref_wav = str(tmp_path / "ref.wav")
    out_wav = str(tmp_path / "out_freevc.wav")

    asyncio.run(_synth("Hello, this is a test of voice conversion.", "en-US-GuyNeural", src_mp3))
    asyncio.run(_synth("The quick brown fox jumps over the lazy dog.", "en-US-AriaNeural", ref_mp3))

    try:
        data, sr = sf.read(src_mp3, dtype="float32")
        sf.write(src_wav, data, sr)
        data, sr = sf.read(ref_mp3, dtype="float32")
        sf.write(ref_wav, data, sr)
    except Exception:
        subprocess.run(["ffmpeg", "-y", "-i", src_mp3, src_wav], check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-i", ref_mp3, ref_wav], check=True, capture_output=True)

    adapter = FreeVCAdapter(quantized=False)
    result = adapter.clone_voice(src_wav, ref_wav, out_wav)

    assert Path(result).exists(), f"Output not found: {result}"
    size_bytes = Path(result).stat().st_size
    assert size_bytes > 0, "Output file is empty"

    with wave.open(result, "rb") as wf:
        sr_out = wf.getframerate()
        n_frames = wf.getnframes()
        sampwidth = wf.getsampwidth()
        duration_s = n_frames / sr_out

    assert sr_out == 16000, f"Expected 16000 Hz, got {sr_out}"
    assert sampwidth == 2, f"Expected 16-bit PCM, got {sampwidth * 8}-bit"
    assert 0.5 <= duration_s <= 15.0, f"Suspicious output duration: {duration_s:.2f} s"

    print(
        f"\n[e2e freevc] output={result}  "
        f"duration={duration_s:.2f}s  sr={sr_out}  "
        f"size={size_bytes // 1024} KiB"
    )
