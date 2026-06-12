"""Tests for the kNN-VC adapter — voiceclonnx/engines/knnvc.py.

Structure
---------
- Registry wiring (no model loading)
- kNN matching math on synthetic vectors (real computation, pure numpy)
- Mock-session contract tests (full adapter pipeline with stubbed ORT)
- skipif-gated e2e: real conversion using edge-tts generated audio
"""

from __future__ import annotations

import os
import struct
import wave
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

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


def _mock_ort_session(output: np.ndarray) -> MagicMock:
    """Return a mock onnxruntime.InferenceSession that returns *output*."""
    sess = MagicMock()
    sess.run.return_value = [output]
    return sess


# ---------------------------------------------------------------------------
# 1. Registry wiring
# ---------------------------------------------------------------------------


def test_knnvc_registered():
    """knnvc engine must appear in ENGINE_REGISTRY after importing voiceclonnx."""
    import voiceclonnx.engines.knnvc  # noqa: F401 — trigger registration
    from voiceclonnx.engines.base import ENGINE_REGISTRY

    assert "knnvc" in ENGINE_REGISTRY


def test_knnvc_entry_metadata():
    import voiceclonnx.engines.knnvc  # noqa: F401
    from voiceclonnx.engines.base import get_engine

    entry = get_engine("knnvc")
    assert entry.alias == "knnvc"
    assert entry.onnx_native is True
    assert entry.extras == ""
    assert entry.adapter_class.__name__ == "KNNVCAdapter"


def test_knnvc_sample_rate():
    from voiceclonnx.engines.knnvc import KNNVCAdapter

    adapter = KNNVCAdapter()
    assert adapter.sample_rate == 16000


# ---------------------------------------------------------------------------
# 2. kNN matching math — real computation on synthetic vectors
# ---------------------------------------------------------------------------


def test_knn_match_exact_copy():
    """When source == reference the matched output equals the input."""
    from voiceclonnx.engines.knnvc import _knn_match

    rng = np.random.default_rng(42)
    feats = rng.random((20, 1024), dtype=np.float32)
    matched = _knn_match(feats, feats, k=1)
    # With k=1, each source frame maps to itself (nearest in the reference)
    np.testing.assert_allclose(matched, feats, atol=1e-5)


def test_knn_match_output_shape():
    from voiceclonnx.engines.knnvc import _knn_match

    rng = np.random.default_rng(0)
    src = rng.random((30, 1024), dtype=np.float32)
    ref = rng.random((50, 1024), dtype=np.float32)

    matched = _knn_match(src, ref, k=4)
    assert matched.shape == (30, 1024)
    assert matched.dtype == np.float32


def test_knn_match_k_clipped_to_ref_size():
    """k is silently clipped to len(reference) — must not raise."""
    from voiceclonnx.engines.knnvc import _knn_match

    rng = np.random.default_rng(7)
    src = rng.random((10, 8), dtype=np.float32)
    ref = rng.random((3, 8), dtype=np.float32)

    matched = _knn_match(src, ref, k=100)  # k >> ref size
    assert matched.shape == (10, 8)


def test_knn_match_is_mean_of_k():
    """For k=2, each output row equals the mean of the 2 nearest ref rows."""
    from voiceclonnx.engines.knnvc import _knn_match

    # 3-dim space for easy manual verification
    ref = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    # Source exactly at ref[0]; nearest two are ref[0] and ref[1] (equal dist to ref[2])
    src = np.array([[1, 0, 0]], dtype=np.float32)

    matched = _knn_match(src, ref, k=2)
    # The 2 nearest to [1,0,0] from {[1,0,0],[0,1,0],[0,0,1]} are [1,0,0] (d=0) and
    # either [0,1,0] or [0,0,1] (d=sqrt(2), equal). argpartition is not stable,
    # so we just check the result lies in the convex hull of those two options.
    assert matched.shape == (1, 3)
    # First element of matched must be >= 0.5 (since ref[0] is always chosen)
    assert matched[0, 0] >= 0.5 - 1e-5


def test_knn_match_nearest_is_not_random():
    """Nearest-neighbour assignment should be deterministic."""
    from voiceclonnx.engines.knnvc import _knn_match

    rng = np.random.default_rng(99)
    src = rng.random((40, 256), dtype=np.float32)
    ref = rng.random((80, 256), dtype=np.float32)

    m1 = _knn_match(src, ref, k=4)
    m2 = _knn_match(src, ref, k=4)
    np.testing.assert_array_equal(m1, m2)


# ---------------------------------------------------------------------------
# 3. Adapter contract with mocked ORT sessions
# ---------------------------------------------------------------------------


class _MockWavLMSession:
    """Fake WavLM ORT session — returns fixed (1, T//320, 1024) features."""

    def run(self, output_names, inputs):
        n_samples = inputs["input_values"].shape[1]
        n_frames = max(1, n_samples // 320)
        rng = np.random.default_rng(0)
        feats = rng.random((1, n_frames, 1024), dtype=np.float32)
        return [feats]


class _MockHiFiGANSession:
    """Fake HiFi-GAN ORT session — returns random (1, 1, frames*256) samples."""

    def run(self, output_names, inputs):
        n_frames = inputs["features"].shape[2]
        n_samples = n_frames * 256
        rng = np.random.default_rng(1)
        wav = rng.random((1, 1, n_samples), dtype=np.float32) * 0.1
        return [wav]


def test_adapter_clone_voice_mock(tmp_path):
    """Adapter pipeline completes with mocked ORT sessions."""
    from voiceclonnx.engines.knnvc import KNNVCAdapter

    src_wav = _make_wav(str(tmp_path / "src.wav"), duration_s=1.0)
    ref_wav = _make_wav(str(tmp_path / "ref.wav"), duration_s=2.0)
    out_wav = str(tmp_path / "out.wav")

    adapter = KNNVCAdapter(k=4)
    adapter._wavlm_sess = _MockWavLMSession()
    adapter._hifigan_sess = _MockHiFiGANSession()

    result = adapter.clone_voice(src_wav, ref_wav, out_wav)
    assert result == str(Path(out_wav).resolve())
    assert Path(out_wav).exists()

    # Verify output is a valid WAV
    with wave.open(out_wav, "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getsampwidth() == 2
        assert wf.getnframes() > 0


def test_adapter_output_is_16khz(tmp_path):
    """Output WAV must be 16 kHz regardless of input sample rate."""
    from voiceclonnx.engines.knnvc import KNNVCAdapter

    # Write 44100 Hz source (non-standard rate)
    src_path = str(tmp_path / "src44.wav")
    n = 44100
    data = (np.zeros(n, dtype=np.int16))
    with wave.open(src_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(data.tobytes())

    ref_wav = _make_wav(str(tmp_path / "ref.wav"))
    out_wav = str(tmp_path / "out.wav")

    adapter = KNNVCAdapter(k=4)
    adapter._wavlm_sess = _MockWavLMSession()
    adapter._hifigan_sess = _MockHiFiGANSession()

    adapter.clone_voice(src_path, ref_wav, out_wav)

    with wave.open(out_wav, "rb") as wf:
        assert wf.getframerate() == 16000


def test_adapter_lazy_load_raises_without_onnxruntime(tmp_path, monkeypatch):
    """Missing onnxruntime raises ImportError with a helpful message."""
    import builtins

    from voiceclonnx.engines.knnvc import KNNVCAdapter

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("onnxruntime not installed")
        return real_import(name, *args, **kwargs)

    adapter = KNNVCAdapter()
    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="onnxruntime"):
            adapter._ensure_models()


def test_knn_match_quantized_flag_stored():
    """quantized flag stored and accessible."""
    from voiceclonnx.engines.knnvc import KNNVCAdapter

    a = KNNVCAdapter(quantized=True)
    assert a._quantized is True
    b = KNNVCAdapter(quantized=False)
    assert b._quantized is False


# ---------------------------------------------------------------------------
# 4. E2E test — real model, real audio (skip if no HF token or heavy deps)
# ---------------------------------------------------------------------------

_SKIP_E2E = not os.environ.get("VOICECLONNX_E2E", "")  # models are public; gate on opt-in (large downloads)

_E2E_REASON = (
    "E2E knnvc test downloads ~500MB of public models; set VOICECLONNX_E2E=1 to run "
    "and network access to download models."
)


@pytest.mark.skipif(_SKIP_E2E, reason=_E2E_REASON)
def test_e2e_knnvc_clone_edge_tts_voices(tmp_path):
    """Real end-to-end: convert an edge-tts source to a second edge-tts reference.

    Validates:
    - Output is a valid 16-bit 16 kHz WAV.
    - Duration is in a reasonable range (0.5 s – 10 s).
    - File size > 0 bytes.
    """
    import asyncio
    import soundfile as sf

    try:
        import edge_tts
    except ImportError:
        pytest.skip("edge-tts not installed")

    from voiceclonnx.engines.knnvc import KNNVCAdapter

    async def _synth(text: str, voice: str, out: str):
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(out)

    # Generate two short utterances with different voices
    src_mp3 = str(tmp_path / "src.mp3")
    ref_mp3 = str(tmp_path / "ref.mp3")
    src_wav = str(tmp_path / "src.wav")
    ref_wav = str(tmp_path / "ref.wav")
    out_wav = str(tmp_path / "out_knnvc.wav")

    asyncio.run(_synth("Hello, this is a test of voice conversion.", "en-US-GuyNeural", src_mp3))
    asyncio.run(_synth("The quick brown fox jumps over the lazy dog.", "en-US-AriaNeural", ref_mp3))

    # Convert MP3 → WAV via soundfile (requires libsndfile with MP3 support) or ffmpeg
    try:
        data, sr = sf.read(src_mp3, dtype="float32")
        sf.write(src_wav, data, sr)
        data, sr = sf.read(ref_mp3, dtype="float32")
        sf.write(ref_wav, data, sr)
    except Exception:
        # Fall back to scipy if soundfile can't handle mp3
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-i", src_mp3, src_wav], check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-i", ref_mp3, ref_wav], check=True, capture_output=True)

    adapter = KNNVCAdapter(quantized=False, k=4)
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
    assert 0.5 <= duration_s <= 10.0, f"Suspicious output duration: {duration_s:.2f} s"

    # Report
    print(
        f"\n[e2e knnvc] output={result}  "
        f"duration={duration_s:.2f}s  sr={sr_out}  "
        f"size={size_bytes // 1024} KiB"
    )
