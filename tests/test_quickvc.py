"""Tests for the QuickVC adapter — voiceclonnx/engines/quickvc.py.

Structure
---------
- Registry wiring (no model loading)
- Log-mel spectrogram shape and value sanity
- numpy Multistream-iSTFT parity against known synthetic STFT inputs
- Mock-session contract tests (full adapter pipeline with stubbed ORT sessions)
- E2E test (VOICECLONNX_E2E=1 to opt in)
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


def test_quickvc_registered():
    """quickvc engine must appear in ENGINE_REGISTRY after importing the module."""
    import voiceclonnx.engines.quickvc  # noqa: F401
    from voiceclonnx.engines.base import ENGINE_REGISTRY

    assert "quickvc" in ENGINE_REGISTRY


def test_quickvc_entry_metadata():
    import voiceclonnx.engines.quickvc  # noqa: F401
    from voiceclonnx.engines.base import get_engine

    entry = get_engine("quickvc")
    assert entry.alias == "quickvc"
    assert entry.onnx_native is True
    assert entry.adapter_class.__name__ == "QuickVCAdapter"


def test_quickvc_sample_rate():
    from voiceclonnx.engines.quickvc import QuickVCAdapter

    adapter = QuickVCAdapter()
    assert adapter.sample_rate == 16000


def test_quickvc_registered_via_package_import():
    """Importing voiceclonnx package auto-registers quickvc."""
    import voiceclonnx  # noqa: F401
    from voiceclonnx.engines.base import ENGINE_REGISTRY

    assert "quickvc" in ENGINE_REGISTRY


# ---------------------------------------------------------------------------
# 2. Log-mel spectrogram
# ---------------------------------------------------------------------------


def test_logmel_shape():
    """_log_mel_spectrogram returns (1, T, 80) shape for 1 s of audio."""
    from voiceclonnx.engines.quickvc import _log_mel_spectrogram

    audio = np.zeros(16000, dtype=np.float32)
    mel = _log_mel_spectrogram(audio)
    assert mel.ndim == 3
    assert mel.shape[0] == 1    # batch=1
    assert mel.shape[2] == 80   # n_mels
    assert mel.shape[1] > 0     # time frames


def test_logmel_dtype():
    from voiceclonnx.engines.quickvc import _log_mel_spectrogram

    audio = np.random.default_rng(0).random(16000).astype(np.float32)
    mel = _log_mel_spectrogram(audio)
    assert mel.dtype == np.float32


def test_logmel_silence_values():
    """Silence should give log(1e-5) everywhere (spectral_normalize floor)."""
    from voiceclonnx.engines.quickvc import _log_mel_spectrogram

    silence = np.zeros(16000, dtype=np.float32)
    mel = _log_mel_spectrogram(silence)
    expected = np.log(1e-5)
    np.testing.assert_allclose(mel, expected, atol=1e-4)


def test_logmel_sine_not_flat():
    """A 440 Hz sine should produce energy concentrated around the 440 Hz mel bin."""
    from voiceclonnx.engines.quickvc import _log_mel_spectrogram

    t = np.linspace(0, 1, 16000, endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32) * 0.5
    mel = _log_mel_spectrogram(audio)
    # mean energy across mel bins should not be uniform
    per_bin = mel[0].mean(axis=0)  # (80,) mean over time
    assert per_bin.std() > 0.1, "Expected non-uniform mel energy for a sine tone"


# ---------------------------------------------------------------------------
# 3. numpy Multistream-iSTFT
# ---------------------------------------------------------------------------


def test_ms_istft_output_shape():
    """_numpy_ms_istft returns (B, subbands, T_audio) with T_audio=(T_dec-1)*hop."""
    from voiceclonnx.engines.quickvc import _numpy_ms_istft

    B, subbands, half, T_dec = 1, 4, 9, 100
    rng = np.random.default_rng(0)
    # mag > 0, phase bounded
    mag = np.abs(rng.random((B * subbands, half, T_dec), dtype=np.float32)) * 0.5
    phase = rng.random((B * subbands, half, T_dec), dtype=np.float32) * 0.2
    spec_phase = np.stack([mag, phase], axis=1)  # (B*sub, 2, half, T_dec)

    out = _numpy_ms_istft(spec_phase, n_fft=16, hop=4, subbands=subbands)
    assert out.shape == (B, subbands, (T_dec - 1) * 4)
    assert out.dtype == np.float32


def test_ms_istft_zeros_produce_zeros():
    """Zero spec_phase → zero output."""
    from voiceclonnx.engines.quickvc import _numpy_ms_istft

    B, subbands, half, T_dec = 1, 4, 9, 50
    spec_phase = np.zeros((B * subbands, 2, half, T_dec), dtype=np.float32)
    out = _numpy_ms_istft(spec_phase, n_fft=16, hop=4, subbands=subbands)
    np.testing.assert_allclose(out, 0.0, atol=1e-6)


def test_ms_istft_parity_torch():
    """numpy MS-iSTFT matches torch.istft to machine precision (max_abs=0)."""
    try:
        import torch
    except ImportError:
        pytest.skip("torch not available for parity test")

    from voiceclonnx.engines.quickvc import _numpy_ms_istft

    n_fft, hop = 16, 4
    B, subbands, half, T_dec = 1, 4, n_fft // 2 + 1, 50

    rng = np.random.default_rng(42)
    mag = np.abs(rng.random((B * subbands, half, T_dec), dtype=np.float32)) * 0.5 + 0.1
    phase = rng.random((B * subbands, half, T_dec), dtype=np.float32) * np.pi * 0.5

    spec_phase = np.stack([mag, phase], axis=1)  # (B*sub, 2, half, T_dec)
    np_out = _numpy_ms_istft(spec_phase, n_fft=n_fft, hop=hop, subbands=subbands)

    # Compute per-subband torch.istft for comparison
    window = torch.from_numpy(np.hanning(n_fft + 1)[:-1].astype(np.float32))
    torch_results = []
    for sb in range(B * subbands):
        t_mag = torch.from_numpy(mag[sb])    # (half, T_dec)
        t_phase = torch.from_numpy(phase[sb])
        stft_c = t_mag * torch.exp(1j * t_phase)
        out_sb = torch.istft(stft_c, n_fft=n_fft, hop_length=hop, win_length=n_fft, window=window)
        torch_results.append(out_sb.detach().numpy())

    torch_out = np.array(torch_results).reshape(B, subbands, -1)
    np.testing.assert_allclose(np_out, torch_out, atol=1e-5)


# ---------------------------------------------------------------------------
# 4. Mock-session contract tests
# ---------------------------------------------------------------------------


class _MockContentEncoder:
    """Fake content encoder ORT session: (1, 1, N) → (1, 256, T)."""

    def run(self, output_names, inputs):
        n_samples = inputs["wav"].shape[2]
        T = n_samples // 320  # 50 Hz feature rate
        rng = np.random.default_rng(0)
        return [rng.random((1, 256, max(1, T)), dtype=np.float32)]


class _MockSpeakerEncoder:
    """Fake speaker encoder ORT session: (1, T, 80) → (1, 256, 1)."""

    def run(self, output_names, inputs):
        rng = np.random.default_rng(1)
        dvec = rng.random((1, 256, 1), dtype=np.float32)
        dvec /= np.linalg.norm(dvec) + 1e-8
        return [dvec]


class _MockDecoder:
    """Fake decoder ORT session: → (B*sub, 2, half, T_dec)."""

    def run(self, output_names, inputs):
        T = inputs["c"].shape[2]
        T_dec = max(4, T * 20)  # upsampled by 20×
        n_fft, subbands = 16, 4
        half = n_fft // 2 + 1
        rng = np.random.default_rng(2)
        # Low mag so output doesn't saturate
        mag = np.abs(rng.random((subbands, half, T_dec), dtype=np.float32)) * 0.01
        phase = rng.random((subbands, half, T_dec), dtype=np.float32) * 0.1
        sp = np.stack([mag, phase], axis=1)  # (sub, 2, half, T_dec)
        return [sp]


class _MockPostnet:
    """Fake postnet ORT session: (1, 4, T_audio) → (1, 1, T_out)."""

    def run(self, output_names, inputs):
        T_audio = inputs["y_mb_hat"].shape[2]
        rng = np.random.default_rng(3)
        T_out = T_audio * 4  # subband upsampling factor
        return [rng.random((1, 1, T_out), dtype=np.float32) * 0.01]


def test_adapter_clone_voice_mock(tmp_path):
    """Adapter pipeline completes with mocked ORT sessions."""
    from voiceclonnx.engines.quickvc import QuickVCAdapter

    src_wav = _make_wav(str(tmp_path / "src.wav"), duration_s=1.0)
    ref_wav = _make_wav(str(tmp_path / "ref.wav"), duration_s=2.0)
    out_wav = str(tmp_path / "out.wav")

    adapter = QuickVCAdapter()
    adapter._enc_sess = _MockContentEncoder()
    adapter._spk_sess = _MockSpeakerEncoder()
    adapter._dec_sess = _MockDecoder()
    adapter._post_sess = _MockPostnet()

    result = adapter.clone_voice(src_wav, ref_wav, out_wav)
    assert result == str(Path(out_wav).resolve())
    assert Path(out_wav).exists()

    with wave.open(out_wav, "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getsampwidth() == 2
        assert wf.getnframes() > 0


def test_adapter_output_is_16khz(tmp_path):
    """Output WAV is always 16 kHz regardless of input rate."""
    from voiceclonnx.engines.quickvc import QuickVCAdapter

    # Write a 44.1 kHz dummy source
    src_path = str(tmp_path / "src44.wav")
    data = np.zeros(44100, dtype=np.int16)
    with wave.open(src_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(data.tobytes())

    ref_wav = _make_wav(str(tmp_path / "ref.wav"))
    out_wav = str(tmp_path / "out.wav")

    adapter = QuickVCAdapter()
    adapter._enc_sess = _MockContentEncoder()
    adapter._spk_sess = _MockSpeakerEncoder()
    adapter._dec_sess = _MockDecoder()
    adapter._post_sess = _MockPostnet()

    adapter.clone_voice(src_path, ref_wav, out_wav)

    with wave.open(out_wav, "rb") as wf:
        assert wf.getframerate() == 16000


def test_adapter_lazy_load_raises_without_onnxruntime(tmp_path, monkeypatch):
    """Missing onnxruntime raises ImportError with a helpful message."""
    import builtins

    from voiceclonnx.engines.quickvc import QuickVCAdapter

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("onnxruntime not installed")
        return real_import(name, *args, **kwargs)

    adapter = QuickVCAdapter()
    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="onnxruntime"):
            adapter._ensure_models()


def test_adapter_quantized_flag_stored():
    from voiceclonnx.engines.quickvc import QuickVCAdapter

    a = QuickVCAdapter(quantized=True)
    assert a._quantized is True
    b = QuickVCAdapter(quantized=False)
    assert b._quantized is False


# ---------------------------------------------------------------------------
# 5. E2E test
# ---------------------------------------------------------------------------

_SKIP_E2E = not os.environ.get("VOICECLONNX_E2E", "")

_E2E_REASON = (
    "E2E quickvc test downloads ~500 MB of public models; "
    "set VOICECLONNX_E2E=1 to run."
)


@pytest.mark.skipif(_SKIP_E2E, reason=_E2E_REASON)
def test_e2e_quickvc_clone_edge_tts_voices(tmp_path):
    """Real end-to-end: convert edge-tts source to edge-tts reference.

    Validates:
    - Output is a valid 16-bit 16 kHz WAV.
    - Duration is in a reasonable range (0.5 s – 10 s).
    - File size > 0 bytes.
    """
    import asyncio

    try:
        import edge_tts
    except ImportError:
        pytest.skip("edge-tts not installed")

    from voiceclonnx.engines.quickvc import QuickVCAdapter

    async def _synth(text: str, voice: str, out: str):
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(out)

    src_mp3 = str(tmp_path / "src.mp3")
    ref_mp3 = str(tmp_path / "ref.mp3")
    src_wav_path = str(tmp_path / "src.wav")
    ref_wav_path = str(tmp_path / "ref.wav")
    out_wav = str(tmp_path / "out_quickvc.wav")

    asyncio.run(_synth("Hello, this is a test of voice conversion.", "en-US-GuyNeural", src_mp3))
    asyncio.run(_synth("The quick brown fox jumps over the lazy dog.", "en-US-AriaNeural", ref_mp3))

    import subprocess
    subprocess.run(["ffmpeg", "-y", "-i", src_mp3, src_wav_path], check=True, capture_output=True)
    subprocess.run(["ffmpeg", "-y", "-i", ref_mp3, ref_wav_path], check=True, capture_output=True)

    adapter = QuickVCAdapter(quantized=False)
    result = adapter.clone_voice(src_wav_path, ref_wav_path, out_wav)

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
        f"\n[e2e quickvc] output={result}  "
        f"duration={duration_s:.2f}s  sr={sr_out}  "
        f"size={size_bytes // 1024} KiB"
    )
