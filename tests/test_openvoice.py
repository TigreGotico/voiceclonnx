"""Tests for the OpenVoice v2 adapter — vconnx/engines/openvoice.py.

Structure
---------
- Registry wiring (no model loading)
- Mel-spectrogram / STFT helpers — real computation, pure numpy
- Griffin-Lim vocoder — output shape and energy sanity checks
- Mock-session contract tests (full adapter pipeline with stubbed ORT)
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
    import vconnx.engines.openvoice  # noqa: F401 — trigger registration
    from vconnx.engines.base import ENGINE_REGISTRY

    assert "openvoice" in ENGINE_REGISTRY


def test_openvoice_entry_metadata():
    import vconnx.engines.openvoice  # noqa: F401
    from vconnx.engines.base import get_engine

    entry = get_engine("openvoice")
    assert entry.alias == "openvoice"
    assert entry.onnx_native is True
    assert entry.extras == "openvoice"
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
# 2. Mel / STFT helpers — real computation on synthetic signals
# ---------------------------------------------------------------------------


def test_stft_output_shape():
    from vconnx.engines.openvoice import _stft

    audio = np.zeros(22050, dtype=np.float32)
    spec = _stft(audio, n_fft=1024, hop_length=256, win_length=1024)
    # Expected bins: n_fft//2+1 = 513
    assert spec.shape[0] == 513
    # Frames: roughly sr/hop
    assert spec.shape[1] > 0
    assert spec.dtype == np.float32


def test_stft_sine_energy():
    """STFT of a pure tone should have energy concentrated at the tone frequency."""
    from vconnx.engines.openvoice import _stft

    sr = 22050
    freq = 440.0
    n = sr  # 1 second
    t = np.linspace(0, 1.0, n, endpoint=False)
    audio = np.sin(2 * np.pi * freq * t).astype(np.float32)

    spec = _stft(audio, n_fft=1024, hop_length=256, win_length=1024)
    # Find bin with most energy
    energy_per_bin = spec.sum(axis=1)
    peak_bin = int(np.argmax(energy_per_bin))
    # Expected bin for 440 Hz: 440 / (sr/n_fft) = 440 * 1024 / 22050 ≈ 20
    expected_bin = int(round(freq * 1024 / sr))
    assert abs(peak_bin - expected_bin) <= 2, (
        f"Peak energy at bin {peak_bin}, expected ~{expected_bin} for {freq} Hz"
    )


def test_mel_filterbank_shape():
    from vconnx.engines.openvoice import _mel_filterbank

    fb = _mel_filterbank(sr=22050, n_fft=1024, n_mels=80, f_min=0.0, f_max=8000.0)
    assert fb.shape == (80, 513)
    assert fb.dtype == np.float32
    # All values non-negative
    assert (fb >= 0).all()


def test_mel_filterbank_sums_to_positive():
    """Each mel filter must have at least one positive weight."""
    from vconnx.engines.openvoice import _mel_filterbank

    fb = _mel_filterbank(sr=22050, n_fft=1024, n_mels=80, f_min=0.0, f_max=8000.0)
    row_sums = fb.sum(axis=1)
    assert (row_sums > 0).all(), "Some mel filters have zero weight"


def test_compute_mel_shape():
    from vconnx.engines.openvoice import _compute_mel

    audio = np.zeros(22050, dtype=np.float32)
    mel = _compute_mel(audio)
    assert mel.shape[0] == 80
    assert mel.shape[1] > 0
    assert mel.dtype == np.float32


def test_compute_mel_finite_values():
    """Log-mel of silence should produce finite values (not -inf)."""
    from vconnx.engines.openvoice import _compute_mel

    audio = np.zeros(22050, dtype=np.float32)
    mel = _compute_mel(audio)
    assert np.isfinite(mel).all(), "Mel spectrogram contains non-finite values"


def test_compute_mel_sine_different_from_silence():
    """A sine wave should produce a different mel from silence."""
    from vconnx.engines.openvoice import _compute_mel

    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    silence = np.zeros(sr, dtype=np.float32)

    mel_sine = _compute_mel(sine)
    mel_silence = _compute_mel(silence)
    assert not np.allclose(mel_sine, mel_silence), "Sine and silence give identical mel"


# ---------------------------------------------------------------------------
# 3. Griffin-Lim vocoder — shape and energy checks
# ---------------------------------------------------------------------------


def test_griffin_lim_output_shape():
    from vconnx.engines.openvoice import _griffin_lim

    mel = np.zeros((80, 100), dtype=np.float32)
    audio = _griffin_lim(mel, sr=22050, n_fft=1024, hop_length=256, win_length=1024, n_iter=2)
    assert audio.ndim == 1
    assert len(audio) > 0
    assert audio.dtype == np.float32


def test_griffin_lim_finite():
    from vconnx.engines.openvoice import _griffin_lim

    rng = np.random.default_rng(42)
    mel = rng.uniform(-5, 2, (80, 50)).astype(np.float32)
    audio = _griffin_lim(mel, sr=22050, n_fft=1024, hop_length=256, win_length=1024, n_iter=4)
    assert np.isfinite(audio).all(), "Griffin-Lim produced non-finite audio"


def test_griffin_lim_nonzero_for_nonzero_mel():
    from vconnx.engines.openvoice import _griffin_lim

    rng = np.random.default_rng(7)
    mel = rng.uniform(-3, 0, (80, 60)).astype(np.float32)  # realistic log-mel range
    audio = _griffin_lim(mel, sr=22050, n_fft=1024, hop_length=256, win_length=1024, n_iter=4)
    assert np.any(audio != 0.0), "Griffin-Lim produced all-zero output"


# ---------------------------------------------------------------------------
# 4. Adapter contract with mocked ORT sessions
# ---------------------------------------------------------------------------


class _MockRefEncSession:
    """Fake reference encoder ORT session — returns (1, 256) tone embedding."""

    def run(self, output_names, inputs):
        rng = np.random.default_rng(42)
        tone = rng.uniform(-0.5, 0.5, (1, 256)).astype(np.float32)
        return [tone]


class _MockConverterSession:
    """Fake converter ORT session — returns mel with same shape as input."""

    def run(self, output_names, inputs):
        mel = inputs["mel"]
        # Return slightly modified mel to simulate conversion
        rng = np.random.default_rng(0)
        return [mel + rng.uniform(-0.1, 0.1, mel.shape).astype(np.float32)]


def test_adapter_clone_voice_mock(tmp_path):
    """Adapter pipeline completes with mocked ORT sessions."""
    from vconnx.engines.openvoice import OpenVoiceV2Adapter

    src_wav = _make_wav(str(tmp_path / "src.wav"), duration_s=0.5, sr=22050)
    ref_wav = _make_wav(str(tmp_path / "ref.wav"), duration_s=0.5, sr=22050)
    out_wav = str(tmp_path / "out.wav")

    adapter = OpenVoiceV2Adapter(gl_iters=2)
    adapter._ref_enc_sess = _MockRefEncSession()
    adapter._converter_sess = _MockConverterSession()

    result = adapter.clone_voice(src_wav, ref_wav, out_wav)
    assert result == str(Path(out_wav).resolve())
    assert Path(out_wav).exists()

    # Verify output is a valid WAV at 22050 Hz
    with wave.open(out_wav, "rb") as wf:
        assert wf.getframerate() == 22050
        assert wf.getsampwidth() == 2
        assert wf.getnframes() > 0


def test_adapter_output_is_22050hz(tmp_path):
    """Output WAV must be 22050 Hz regardless of input sample rate."""
    from vconnx.engines.openvoice import OpenVoiceV2Adapter
    import struct

    # Write 16000 Hz source
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

    adapter = OpenVoiceV2Adapter(gl_iters=2)
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


def test_adapter_extract_tone_embedding_shape(tmp_path):
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


def test_adapter_different_tones_for_different_references(tmp_path):
    """Different audio inputs must produce different tone embeddings."""
    from vconnx.engines.openvoice import OpenVoiceV2Adapter

    # Use a session that actually depends on the input
    class _InputDependentSession:
        def run(self, output_names, inputs):
            mel = inputs["mel"]
            # Return mean of mel as a pseudo-embedding
            tone = mel.mean(axis=(1, 2), keepdims=True).repeat(256, axis=2).reshape(1, 256)
            return [tone.astype(np.float32)]

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
# 5. E2E test — real model, real audio (skip unless HF_TOKEN set)
# ---------------------------------------------------------------------------


_SKIP_E2E = not os.environ.get("VCONNX_E2E", "")  # models are public; gate on opt-in (large downloads)

_E2E_REASON = (
    "E2E openvoice test downloads public models; set VCONNX_E2E=1 to run "
    "and network access to download models."
)


@pytest.mark.skipif(_SKIP_E2E, reason=_E2E_REASON)
def test_e2e_openvoice_clone_edge_tts_voices(tmp_path):
    """Real end-to-end: convert an edge-tts source to a second edge-tts reference.

    Validates:
    - Output is a valid 16-bit 22050 Hz WAV.
    - Duration is in a reasonable range (0.5 s – 15 s).
    - File size > 0 bytes.
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

    adapter = OpenVoiceV2Adapter(quantized=False, gl_iters=32)
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

    print(
        f"\n[e2e openvoice] output={result}  "
        f"duration={duration_s:.2f}s  sr={sr_out}  "
        f"size={size_bytes // 1024} KiB"
    )
