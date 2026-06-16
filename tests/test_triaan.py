"""Tests for the TriAAN-VC adapter — voiceclonnx/engines/triaan.py.

Structure
---------
- Registry wiring (no model loading)
- F0 extraction math on synthetic signals (real numpy DSP)
- Mock-session contract tests (full pipeline with stubbed ORT)
- VOICECLONNX_E2E-gated real conversion test
"""

from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SR = 16000


def _make_wav(path: str, duration_s: float = 1.0, sr: int = _SR,
              freq: float = 220.0) -> str:
    """Write a pure-tone sine WAV for testing."""
    n = int(duration_s * sr)
    t = np.linspace(0, duration_s, n, endpoint=False)
    data = (np.sin(2 * np.pi * freq * t) * 32767 * 0.5).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())
    return path


class _MockCPCSess:
    """Fake CPC ORT session — returns (1, T//160, 256) features."""

    def run(self, output_names, inputs):
        n_samples = inputs["audio"].shape[2]
        n_frames = max(1, n_samples // 160)
        rng = np.random.default_rng(0)
        return [rng.random((1, n_frames, 256), dtype=np.float32)]


class _MockTriAANSess:
    """Fake TriAAN ORT session — returns (1, 80, T_s) mel.

    src_cpc arrives as (1, 256, T) after the clone_voice transpose.
    """

    def run(self, output_names, inputs):
        T = inputs["src_cpc"].shape[2]  # (1, 256, T) — channels first
        rng = np.random.default_rng(1)
        return [rng.random((1, 80, T), dtype=np.float32)]


class _MockPWGSess:
    """Fake PWG ORT session — returns (1, 1, T_mel*160) waveform."""

    def run(self, output_names, inputs):
        T_mel = inputs["mel"].shape[2]
        T_audio = T_mel * 160
        rng = np.random.default_rng(2)
        wav = rng.random((1, 1, T_audio), dtype=np.float32) * 0.1
        return [wav]


def _make_mock_adapter(**kwargs):
    """Return a TriAANVCAdapter with all sessions and mel_stats stubbed out."""
    from voiceclonnx.engines.triaan import TriAANVCAdapter

    adapter = TriAANVCAdapter(**kwargs)
    adapter._cpc_sess = _MockCPCSess()
    adapter._triaan_sess = _MockTriAANSess()
    adapter._pwg_sess = _MockPWGSess()
    # Identity denorm stats so _convert is a no-op under the mock
    adapter._mel_mean = np.zeros((80, 1), dtype=np.float32)
    adapter._mel_std = np.ones((80, 1), dtype=np.float32)
    return adapter


# ---------------------------------------------------------------------------
# 1. Registry wiring
# ---------------------------------------------------------------------------


def test_triaan_registered():
    """triaan engine must appear in ENGINE_REGISTRY after importing voiceclonnx."""
    import voiceclonnx.engines.triaan  # noqa: F401 — trigger registration
    from voiceclonnx.engines.base import ENGINE_REGISTRY

    assert "triaan" in ENGINE_REGISTRY


def test_triaan_entry_metadata():
    import voiceclonnx.engines.triaan  # noqa: F401
    from voiceclonnx.engines.base import get_engine

    entry = get_engine("triaan")
    assert entry.alias == "triaan"
    assert entry.onnx_native is True
    assert entry.extras == ""
    assert entry.adapter_class.__name__ == "TriAANVCAdapter"


def test_triaan_sample_rate():
    from voiceclonnx.engines.triaan import TriAANVCAdapter

    adapter = TriAANVCAdapter()
    assert adapter.sample_rate == 16000


def test_triaan_quantized_flag():
    from voiceclonnx.engines.triaan import TriAANVCAdapter

    a = TriAANVCAdapter(quantized=True)
    assert a._quantized is True
    b = TriAANVCAdapter(quantized=False)
    assert b._quantized is False


# ---------------------------------------------------------------------------
# 2. F0 extraction (real numpy DSP)
# ---------------------------------------------------------------------------


def test_lf0_extraction_sine_is_voiced():
    """A clean sine tone should produce mostly voiced (non-zero) frames."""
    from voiceclonnx.engines.triaan import _extract_log_f0

    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    # 220 Hz sine — within voiced range
    audio = np.sin(2 * np.pi * 220 * t).astype(np.float32)
    lf0 = _extract_log_f0(audio, sr=sr, hop_length=160)

    assert lf0.ndim == 1
    assert len(lf0) == sr // 160
    # Most frames should be voiced
    voiced_ratio = (lf0 != 0).mean()
    assert voiced_ratio > 0.5, f"Expected most frames voiced, got {voiced_ratio:.2f}"


def test_lf0_extraction_silence_is_unvoiced():
    """Silence produces all-zero lf0."""
    from voiceclonnx.engines.triaan import _extract_log_f0

    audio = np.zeros(16000, dtype=np.float32)
    lf0 = _extract_log_f0(audio, sr=16000, hop_length=160)

    assert np.all(lf0 == 0), "Silence should yield unvoiced (0) lf0 frames"


def test_lf0_output_shape():
    from voiceclonnx.engines.triaan import _extract_log_f0

    audio = np.random.randn(16000).astype(np.float32) * 0.1
    lf0 = _extract_log_f0(audio, sr=16000, hop_length=160)
    assert lf0.shape == (100,)
    assert lf0.dtype == np.float32


def test_lf0_deterministic():
    from voiceclonnx.engines.triaan import _extract_log_f0

    audio = np.sin(2 * np.pi * 150 * np.linspace(0, 1, 16000)).astype(np.float32)
    assert np.array_equal(
        _extract_log_f0(audio),
        _extract_log_f0(audio),
    )


# ---------------------------------------------------------------------------
# 3. Adapter contract with mocked ORT sessions
# ---------------------------------------------------------------------------


def test_adapter_clone_voice_mock(tmp_path):
    """Adapter pipeline completes with mocked ORT sessions."""
    src = _make_wav(str(tmp_path / "src.wav"), duration_s=1.0)
    ref = _make_wav(str(tmp_path / "ref.wav"), duration_s=1.5, freq=330.0)
    out = str(tmp_path / "out.wav")

    adapter = _make_mock_adapter()

    result = adapter.clone_voice(src, ref, out)
    assert result == str(Path(out).resolve())
    assert Path(out).exists()

    with wave.open(out, "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getsampwidth() == 2
        assert wf.getnframes() > 0


def test_adapter_output_is_16khz(tmp_path):
    """Output WAV must be 16 kHz regardless of input sample rate."""

    # Write a 44100 Hz source
    src_path = str(tmp_path / "src44.wav")
    n = 44100
    with wave.open(src_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(np.zeros(n, dtype=np.int16).tobytes())

    ref = _make_wav(str(tmp_path / "ref.wav"))
    out = str(tmp_path / "out.wav")

    adapter = _make_mock_adapter()

    adapter.clone_voice(src_path, ref, out)

    with wave.open(out, "rb") as wf:
        assert wf.getframerate() == 16000


def test_adapter_lazy_load_raises_without_onnxruntime(tmp_path, monkeypatch):
    """Missing onnxruntime raises ImportError with a helpful message."""
    import builtins

    from voiceclonnx.engines.triaan import TriAANVCAdapter

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("onnxruntime not installed")
        return real_import(name, *args, **kwargs)

    adapter = TriAANVCAdapter()
    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="onnxruntime"):
            adapter._ensure_models()


def test_adapter_lf0_alignment(tmp_path):
    """lf0 length is aligned to CPC frame count — pipeline does not crash on
    mismatch between acoustic F0 frame count and CPC frame count."""

    # Use a short clip; alignment handles truncation/padding
    src = _make_wav(str(tmp_path / "src.wav"), duration_s=0.5)
    ref = _make_wav(str(tmp_path / "ref.wav"), duration_s=0.7)
    out = str(tmp_path / "out.wav")

    adapter = _make_mock_adapter()

    # Should not raise
    adapter.clone_voice(src, ref, out)
    assert Path(out).exists()


def test_adapter_stereo_source_mixed_down(tmp_path):
    """Stereo source is mixed to mono before processing."""
    import soundfile as sf

    # Write stereo WAV
    src_path = str(tmp_path / "stereo.wav")
    data = np.zeros((8000, 2), dtype=np.float32)
    sf.write(src_path, data, 16000)

    ref = _make_wav(str(tmp_path / "ref.wav"))
    out = str(tmp_path / "out.wav")

    adapter = _make_mock_adapter()

    adapter.clone_voice(src_path, ref, out)
    assert Path(out).exists()


# ---------------------------------------------------------------------------
# 4. E2E test — real models (opt-in, gated on VOICECLONNX_E2E)
# ---------------------------------------------------------------------------

_SKIP_E2E = not os.environ.get("VOICECLONNX_E2E", "")
_E2E_REASON = (
    "E2E triaan test downloads ~XXX MB of public models; set VOICECLONNX_E2E=1 to run."
)


@pytest.mark.skipif(_SKIP_E2E, reason=_E2E_REASON)
def test_e2e_triaan_clone_edge_tts_voices(tmp_path):
    """Real end-to-end: CPC encoder + TriAAN decoder + PWG vocoder.

    Downloads models from TigreGotico/voiceclonnx-triaan-vc (public HF repo).
    Validates:
    - Output is a valid 16-bit 16 kHz WAV.
    - Duration is in a reasonable range (0.5 s – 10 s).
    - File size > 0 bytes.
    """
    import asyncio
    import subprocess

    try:
        import edge_tts
    except ImportError:
        pytest.skip("edge-tts not installed")

    from voiceclonnx.engines.triaan import TriAANVCAdapter

    async def _synth(text: str, voice: str, out: str):
        await edge_tts.Communicate(text, voice).save(out)

    src_mp3 = str(tmp_path / "src.mp3")
    ref_mp3 = str(tmp_path / "ref.mp3")
    src_wav = str(tmp_path / "src.wav")
    ref_wav = str(tmp_path / "ref.wav")
    out_wav = str(tmp_path / "out_triaan.wav")

    asyncio.run(_synth("Hello, this is a test of TriAAN-VC voice conversion.", "en-US-GuyNeural", src_mp3))
    asyncio.run(_synth("The quick brown fox jumps over the lazy dog.", "en-US-AriaNeural", ref_mp3))

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", src_mp3,
                    "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", src_wav], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", ref_mp3,
                    "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", ref_wav], check=True)

    adapter = TriAANVCAdapter(quantized=False)
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

    print(
        f"\n[e2e triaan] output={result}  "
        f"duration={duration_s:.2f}s  sr={sr_out}  "
        f"size={size_bytes // 1024} KiB"
    )
