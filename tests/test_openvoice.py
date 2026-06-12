"""Tests for the OpenVoice v2 adapter -- vconnx/engines/openvoice.py.

Structure
---------
- Registry wiring (no model loading)
- Linear spectrogram helper -- real computation, pure numpy
- Mock-session contract tests (full adapter pipeline with stubbed ORT)
- skipif-gated e2e: real conversion using the real ONNX models
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


def _make_wav(path: str, duration_s: float = 1.0, sr: int = 22050) -> str:
    """Write a short sine-wave WAV at *sr* Hz for testing."""
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


def test_openvoice_registered():
    import vconnx.engines.openvoice  # noqa: F401 -- trigger registration
    from vconnx.engines.base import ENGINE_REGISTRY

    assert "openvoice" in ENGINE_REGISTRY


def test_openvoice_entry_metadata():
    import vconnx.engines.openvoice  # noqa: F401
    from vconnx.engines.base import get_engine

    entry = get_engine("openvoice")
    assert entry.alias == "openvoice"
    assert entry.onnx_native is True
    assert entry.extras == ""
    assert entry.adapter_class.__name__ == "OpenVoiceV2Adapter"


def test_openvoice_sample_rate():
    from vconnx.engines.openvoice import OpenVoiceV2Adapter

    adapter = OpenVoiceV2Adapter()
    assert adapter.sample_rate == 22050


def test_openvoice_quantized_flag_stored():
    from vconnx.engines.openvoice import OpenVoiceV2Adapter

    a = OpenVoiceV2Adapter(quantized=True)
    assert a._quantized is True
    b = OpenVoiceV2Adapter(quantized=False)
    assert b._quantized is False


# ---------------------------------------------------------------------------
# 2. Linear spectrogram helper -- real computation on synthetic signals
# ---------------------------------------------------------------------------


def test_compute_linear_spec_shape():
    from vconnx.engines.openvoice import _compute_linear_spec

    audio = np.zeros(22050, dtype=np.float32)
    spec = _compute_linear_spec(audio)
    assert spec.shape[0] == 513, f"Expected 513 freq bins, got {spec.shape[0]}"
    assert spec.shape[1] > 0
    assert spec.dtype == np.float32


def test_compute_linear_spec_positive():
    """Magnitude spectrogram must be positive (sqrt of sum of squares + eps)."""
    from vconnx.engines.openvoice import _compute_linear_spec

    rng = np.random.default_rng(42)
    audio = rng.uniform(-0.5, 0.5, 22050).astype(np.float32)
    spec = _compute_linear_spec(audio)
    assert (spec > 0).all(), "Linear spec has non-positive values"


def test_compute_linear_spec_finite():
    from vconnx.engines.openvoice import _compute_linear_spec

    audio = np.zeros(22050, dtype=np.float32)
    spec = _compute_linear_spec(audio)
    assert np.isfinite(spec).all()


def test_compute_linear_spec_sine_energy():
    """Sine wave should concentrate energy at the tone frequency bin."""
    from vconnx.engines.openvoice import _compute_linear_spec

    sr = 22050
    freq = 440.0
    n_fft = 1024
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = np.sin(2 * np.pi * freq * t).astype(np.float32)

    spec = _compute_linear_spec(audio)  # (513, T)
    energy_per_bin = spec.sum(axis=1)
    peak_bin = int(np.argmax(energy_per_bin))
    expected_bin = int(round(freq * n_fft / sr))
    assert abs(peak_bin - expected_bin) <= 2, (
        f"Peak energy at bin {peak_bin}, expected ~{expected_bin} for {freq} Hz"
    )


def test_compute_linear_spec_different_signals():
    from vconnx.engines.openvoice import _compute_linear_spec

    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    silence = np.zeros(sr, dtype=np.float32)
    assert not np.allclose(_compute_linear_spec(sine), _compute_linear_spec(silence))


# ---------------------------------------------------------------------------
# 3. Adapter contract with mocked ORT sessions
# ---------------------------------------------------------------------------


class _MockRefEncSession:
    """Fake reference encoder ORT session -- returns (1, 256) tone embedding."""

    def run(self, output_names, inputs):
        rng = np.random.default_rng(42)
        tone = rng.uniform(-0.5, 0.5, (1, 256)).astype(np.float32)
        return [tone]


class _MockConverterSession:
    """Fake converter ORT session -- returns (1, 1, samples) waveform."""

    def run(self, output_names, inputs):
        spec = inputs["spec"]  # (1, 513, T)
        T = spec.shape[2]
        # Upscale factor: hop_length = 256; HiFi-GAN upsamples 256x
        n_samples = T * 256
        rng = np.random.default_rng(7)
        waveform = rng.uniform(-0.01, 0.01, (1, 1, n_samples)).astype(np.float32)
        return [waveform]


def test_adapter_clone_voice_mock(tmp_path):
    """Adapter pipeline completes with mocked ORT sessions."""
    from vconnx.engines.openvoice import OpenVoiceV2Adapter

    src_wav = _make_wav(str(tmp_path / "src.wav"), duration_s=0.5, sr=22050)
    ref_wav = _make_wav(str(tmp_path / "ref.wav"), duration_s=0.5, sr=22050)
    out_wav = str(tmp_path / "out.wav")

    adapter = OpenVoiceV2Adapter()
    adapter._ref_enc_sess = _MockRefEncSession()
    adapter._converter_sess = _MockConverterSession()

    result = adapter.clone_voice(src_wav, ref_wav, out_wav)
    assert result == str(Path(out_wav).resolve())
    assert Path(out_wav).exists()

    with wave.open(out_wav, "rb") as wf:
        assert wf.getframerate() == 22050
        assert wf.getsampwidth() == 2
        assert wf.getnframes() > 0


def test_adapter_output_is_22050hz(tmp_path):
    """Output WAV must be 22050 Hz regardless of input sample rate."""
    from vconnx.engines.openvoice import OpenVoiceV2Adapter

    src_path = str(tmp_path / "src16k.wav")
    n = 16000
    data = np.zeros(n, dtype=np.int16)
    with wave.open(src_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(data.tobytes())

    ref_wav = _make_wav(str(tmp_path / "ref.wav"), sr=22050)
    out_wav = str(tmp_path / "out.wav")

    adapter = OpenVoiceV2Adapter()
    adapter._ref_enc_sess = _MockRefEncSession()
    adapter._converter_sess = _MockConverterSession()

    adapter.clone_voice(src_path, ref_wav, out_wav)

    with wave.open(out_wav, "rb") as wf:
        assert wf.getframerate() == 22050


def test_adapter_lazy_load_raises_without_onnxruntime(tmp_path, monkeypatch):
    """Missing onnxruntime raises ImportError with a helpful message."""
    import builtins
    from vconnx.engines.openvoice import OpenVoiceV2Adapter

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("onnxruntime not installed")
        return real_import(name, *args, **kwargs)

    adapter = OpenVoiceV2Adapter()
    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="onnxruntime"):
            adapter._ensure_models()


def test_adapter_extract_tone_embedding_shape():
    """_extract_tone_embedding returns (1, 256) array."""
    from vconnx.engines.openvoice import OpenVoiceV2Adapter

    adapter = OpenVoiceV2Adapter()
    adapter._ref_enc_sess = _MockRefEncSession()
    adapter._converter_sess = _MockConverterSession()

    sr = 22050
    audio = np.zeros(sr, dtype=np.float32)
    emb = adapter._extract_tone_embedding(audio)
    assert emb.shape == (1, 256)
    assert emb.dtype == np.float32


def test_adapter_spec_shape_to_ref_enc():
    """Spec passed to ref_enc session has shape (1, T, 513)."""
    from vconnx.engines.openvoice import OpenVoiceV2Adapter, _SPEC_CHANNELS

    received = {}

    class _SpySession:
        def run(self, output_names, inputs):
            received["spec"] = inputs["spec"].shape
            return [np.zeros((1, 256), dtype=np.float32)]

    adapter = OpenVoiceV2Adapter()
    adapter._ref_enc_sess = _SpySession()
    adapter._converter_sess = _MockConverterSession()

    audio = np.zeros(22050, dtype=np.float32)
    adapter._extract_tone_embedding(audio)

    shape = received["spec"]
    assert shape[0] == 1, "Batch dim must be 1"
    assert shape[2] == _SPEC_CHANNELS, f"Expected {_SPEC_CHANNELS} freq bins, got {shape[2]}"


def test_adapter_spec_shape_to_converter():
    """Spec passed to converter session has shape (1, 513, T)."""
    from vconnx.engines.openvoice import OpenVoiceV2Adapter, _SPEC_CHANNELS

    received = {}

    class _SpyConverterSession:
        def run(self, output_names, inputs):
            received["spec"] = inputs["spec"].shape
            T = inputs["spec"].shape[2]
            return [np.zeros((1, 1, T * 256), dtype=np.float32)]

    adapter = OpenVoiceV2Adapter()
    adapter._ref_enc_sess = _MockRefEncSession()
    adapter._converter_sess = _SpyConverterSession()

    audio = np.zeros(22050, dtype=np.float32)
    import wave as _wave, tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        path = f.name
    _make_wav(path, duration_s=0.5)
    import tempfile, os
    out = path + "_out.wav"
    adapter.clone_voice(path, path, out)

    shape = received["spec"]
    assert shape[0] == 1
    assert shape[1] == _SPEC_CHANNELS


def test_adapter_different_tones_for_different_references():
    """Different audio inputs must produce different tone embeddings."""
    from vconnx.engines.openvoice import OpenVoiceV2Adapter

    class _InputDependentSession:
        def run(self, output_names, inputs):
            spec = inputs["spec"]  # (1, T, 513)
            # scalar mean -> broadcast to (1, 256)
            tone = np.full((1, 256), float(np.mean(spec)), dtype=np.float32)
            return [tone]

    adapter = OpenVoiceV2Adapter()
    adapter._ref_enc_sess = _InputDependentSession()
    adapter._converter_sess = _MockConverterSession()

    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio_a = np.sin(2 * np.pi * 200 * t).astype(np.float32)
    audio_b = np.sin(2 * np.pi * 1000 * t).astype(np.float32)

    emb_a = adapter._extract_tone_embedding(audio_a)
    emb_b = adapter._extract_tone_embedding(audio_b)
    assert not np.allclose(emb_a, emb_b), "Different inputs produced identical tone embeddings"


# ---------------------------------------------------------------------------
# 4. E2E test -- real models downloaded from HF (opt-in)
# ---------------------------------------------------------------------------


_SKIP_E2E = not os.environ.get("VCONNX_E2E", "")

_E2E_REASON = (
    "E2E openvoice test downloads public models; set VCONNX_E2E=1 to run "
    "and network access to download models."
)


@pytest.mark.skipif(_SKIP_E2E, reason=_E2E_REASON)
def test_e2e_openvoice_clone_edge_tts_voices(tmp_path):
    """Real end-to-end: convert an edge-tts source to a second edge-tts reference.

    Validates:
    - Output is a valid 16-bit 22050 Hz WAV.
    - Duration is in a reasonable range (0.5 s - 15 s).
    - File size > 0 bytes.
    - Output waveform has nonzero energy (not silence / white noise).
    """
    import asyncio
    import soundfile as sf

    try:
        import edge_tts
    except ImportError:
        pytest.skip("edge-tts not installed")

    from vconnx.engines.openvoice import OpenVoiceV2Adapter

    async def _synth(text: str, voice: str, out: str):
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(out)

    src_mp3 = str(tmp_path / "src.mp3")
    ref_mp3 = str(tmp_path / "ref.mp3")
    src_wav = str(tmp_path / "src.wav")
    ref_wav = str(tmp_path / "ref.wav")
    out_wav = str(tmp_path / "out_openvoice.wav")

    asyncio.run(_synth("Hello, this is a test of voice conversion.", "en-US-GuyNeural", src_mp3))
    asyncio.run(_synth("The quick brown fox jumps over the lazy dog.", "en-US-AriaNeural", ref_mp3))

    try:
        data, sr = sf.read(src_mp3, dtype="float32")
        sf.write(src_wav, data, sr)
        data, sr = sf.read(ref_mp3, dtype="float32")
        sf.write(ref_wav, data, sr)
    except Exception:
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-i", src_mp3, src_wav], check=True, capture_output=True)
        subprocess.run(["ffmpeg", "-y", "-i", ref_mp3, ref_wav], check=True, capture_output=True)

    adapter = OpenVoiceV2Adapter(quantized=False)
    result = adapter.clone_voice(src_wav, ref_wav, out_wav)

    assert Path(result).exists(), f"Output not found: {result}"
    size_bytes = Path(result).stat().st_size
    assert size_bytes > 0, "Output file is empty"

    with wave.open(result, "rb") as wf:
        sr_out = wf.getframerate()
        n_frames = wf.getnframes()
        sampwidth = wf.getsampwidth()
        duration_s = n_frames / sr_out

    assert sr_out == 22050, f"Expected 22050 Hz, got {sr_out}"
    assert sampwidth == 2, f"Expected 16-bit PCM, got {sampwidth * 8}-bit"
    assert 0.5 <= duration_s <= 15.0, f"Suspicious output duration: {duration_s:.2f} s"

    # Sanity: nonzero energy
    audio_data, _ = sf.read(result, dtype="float32")
    rms = float(np.sqrt(np.mean(audio_data ** 2)))
    assert rms > 1e-4, f"Output RMS too low ({rms:.2e}): likely silence or failed conversion"

    print(
        f"\n[e2e openvoice] output={result}  "
        f"duration={duration_s:.2f}s  sr={sr_out}  "
        f"size={size_bytes // 1024} KiB  rms={rms:.4f}"
    )
