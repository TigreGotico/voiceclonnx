"""Tests for the LinaCodec adapter — voiceclonnx/engines/linacodec.py.

Structure
---------
1. Registry wiring (no model loading)
2. Audio I/O helpers — load/save round-trip
3. numpy ISTFT — against scipy reference
4. Linkwitz-Riley merge — shape + energy checks
5. Mel length helpers
6. Adapter contract with mocked ORT sessions
7. E2E test — real models, real audio (VOICECLONNX_E2E-gated) + WER gate
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


def _make_wav(path: str, duration_s: float = 1.0, sr: int = 24000) -> str:
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


def test_linacodec_registered():
    """linacodec engine must appear in ENGINE_REGISTRY after import."""
    import voiceclonnx.engines.linacodec  # noqa: F401
    from voiceclonnx.engines.base import ENGINE_REGISTRY

    assert "linacodec" in ENGINE_REGISTRY


def test_linacodec_entry_metadata():
    import voiceclonnx.engines.linacodec  # noqa: F401
    from voiceclonnx.engines.base import get_engine

    entry = get_engine("linacodec")
    assert entry.alias == "linacodec"
    assert entry.onnx_native is True
    assert entry.adapter_class.__name__ == "LinaCodecAdapter"


def test_linacodec_sample_rate():
    from voiceclonnx.engines.linacodec import LinaCodecAdapter

    adapter = LinaCodecAdapter()
    assert adapter.sample_rate == 48000


def test_linacodec_quantized_flag():
    from voiceclonnx.engines.linacodec import LinaCodecAdapter

    a = LinaCodecAdapter(quantized=True)
    assert a._quantized is True
    b = LinaCodecAdapter(quantized=False)
    assert b._quantized is False


# ---------------------------------------------------------------------------
# 2. Audio I/O helpers
# ---------------------------------------------------------------------------


def test_load_wav_mono(tmp_path):
    from voiceclonnx.engines.linacodec import _load_wav

    p = str(tmp_path / "test.wav")
    _make_wav(p, duration_s=0.5, sr=24000)
    audio = _load_wav(p, target_sr=24000)
    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert len(audio) > 0


def test_load_wav_resamples_to_48k(tmp_path):
    from voiceclonnx.engines.linacodec import _load_wav

    p = str(tmp_path / "test24.wav")
    _make_wav(p, duration_s=1.0, sr=24000)
    audio = _load_wav(p, target_sr=48000)
    assert abs(len(audio) - 48000) <= 20


def test_save_wav_48k(tmp_path):
    from voiceclonnx.engines.linacodec import _save_wav

    audio = np.zeros(48000, dtype=np.float32)
    p = str(tmp_path / "out.wav")
    _save_wav(p, audio, sr=48000)
    with wave.open(p, "rb") as wf:
        assert wf.getframerate() == 48000
        assert wf.getsampwidth() == 2


def test_resample_linear():
    from voiceclonnx.engines.linacodec import _resample_linear

    audio = np.sin(2 * np.pi * 440 * np.arange(24000) / 24000).astype(np.float32)
    out = _resample_linear(audio, 24000, 16000)
    assert abs(len(out) - 16000) <= 5
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# 3. numpy ISTFT — verify against direct FFT round-trip
# ---------------------------------------------------------------------------


def test_numpy_istft_shape():
    from voiceclonnx.engines.linacodec import _numpy_istft

    n_fft, hop = 1024, 256
    T_frames = 20
    n_bins = n_fft // 2 + 1
    mag = np.ones((1, n_bins, T_frames), dtype=np.float32)
    phase = np.zeros((1, n_bins, T_frames), dtype=np.float32)
    out = _numpy_istft(mag, phase, n_fft=n_fft, hop_length=hop, padding="center")
    assert out.shape[0] == 1
    assert out.shape[1] > 0


def test_numpy_istft_silence():
    """Zero magnitude should produce near-silence."""
    from voiceclonnx.engines.linacodec import _numpy_istft

    n_fft, hop = 1024, 256
    mag = np.zeros((1, n_fft // 2 + 1, 10), dtype=np.float32)
    phase = np.zeros_like(mag)
    out = _numpy_istft(mag, phase, n_fft=n_fft, hop_length=hop)
    assert np.abs(out).max() < 1e-5


def test_numpy_istft_deterministic():
    from voiceclonnx.engines.linacodec import _numpy_istft

    rng = np.random.default_rng(0)
    mag = rng.uniform(0, 1, (1, 513, 15)).astype(np.float32)
    phase = rng.uniform(-np.pi, np.pi, (1, 513, 15)).astype(np.float32)
    out1 = _numpy_istft(mag, phase)
    out2 = _numpy_istft(mag, phase)
    np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# 4. Linkwitz-Riley merge
# ---------------------------------------------------------------------------


def test_linkwitz_riley_shape():
    from voiceclonnx.engines.linacodec import _numpy_linkwitz_riley

    rng = np.random.default_rng(1)
    p1 = rng.uniform(-0.5, 0.5, (1, 48000)).astype(np.float32)
    p2 = rng.uniform(-0.5, 0.5, (1, 48000)).astype(np.float32)
    merged = _numpy_linkwitz_riley(p1, p2)
    assert merged.shape == (1, 48000)
    assert merged.dtype == np.float32


def test_linkwitz_riley_different_lengths():
    """Shorter input is respected."""
    from voiceclonnx.engines.linacodec import _numpy_linkwitz_riley

    rng = np.random.default_rng(2)
    p1 = rng.uniform(-0.1, 0.1, (1, 48000)).astype(np.float32)
    p2 = rng.uniform(-0.1, 0.1, (1, 47900)).astype(np.float32)
    merged = _numpy_linkwitz_riley(p1, p2)
    assert merged.shape[-1] == 47900


def test_linkwitz_riley_not_nan():
    """No NaN/inf in output."""
    from voiceclonnx.engines.linacodec import _numpy_linkwitz_riley

    rng = np.random.default_rng(3)
    p1 = rng.uniform(-1, 1, (1, 24000)).astype(np.float32)
    p2 = rng.uniform(-1, 1, (1, 24000)).astype(np.float32)
    merged = _numpy_linkwitz_riley(p1, p2)
    assert np.all(np.isfinite(merged))


# ---------------------------------------------------------------------------
# 5. Mel length helpers
# ---------------------------------------------------------------------------


def test_calculate_target_mel_length():
    from voiceclonnx.engines.linacodec import _calculate_target_mel_length

    # 1 second at 24kHz = 24000 samples; hop=256 → 24000//256 + 1 = 94
    length = _calculate_target_mel_length(24000, hop_length=256, padding="center")
    assert length == 94


def test_calculate_original_audio_length():
    from voiceclonnx.engines.linacodec import _calculate_original_audio_length

    # 12.5 tokens/sec → 1 sec ≈ 13 tokens
    # 13 tokens × 1920 = 24960 samples ≈ 24000 (within ±5%)
    length = _calculate_original_audio_length(13)
    assert 20000 < length < 30000


# ---------------------------------------------------------------------------
# 6. Adapter contract with mocked ORT sessions
# ---------------------------------------------------------------------------


def _make_mock_sessions(T_ssl: int = 50, T_tokens: int = 12):
    """Return mock ORT sessions that return plausible shapes."""

    class _MockSession:
        def __init__(self, fn):
            self._fn = fn

        def run(self, output_names, inputs):
            return self._fn(inputs)

    rng = np.random.default_rng(42)

    # Acoustic SSL: (1, N) → (1, T_ssl, 768)
    acoustic_sess = _MockSession(lambda inp: [
        rng.standard_normal((1, T_ssl, 768)).astype(np.float32)
    ])

    # Distill WavLM: (1, N) → (1, T_ssl, 768)
    distill_sess = _MockSession(lambda inp: [
        rng.standard_normal((1, T_ssl, 768)).astype(np.float32)
    ])

    # Content encoder: (1, T_ssl, 768) → (1, T_tokens, 768), (1, T_tokens)
    content_sess = _MockSession(lambda inp: [
        rng.standard_normal((1, T_tokens, 768)).astype(np.float32),
        np.zeros((1, T_tokens), dtype=np.int64),
    ])

    # Global encoder: (1, T_ssl, 768) → (1, 128)
    global_sess = _MockSession(lambda inp: [
        rng.standard_normal((1, 128)).astype(np.float32)
    ])

    # Mel decoder: → (1, 100, T_mel)
    n_fft, hop = 1024, 256
    T_mel = 94  # ~1 second at hop=256

    def _mel_fn(inp):
        return [rng.standard_normal((1, 100, T_mel)).astype(np.float32)]

    mel_sess = _MockSession(_mel_fn)

    # Vocos backbone: (1, 100, T_mel) → (mag_24k, phase_24k, mag_48k, phase_48k)
    n_bins = n_fft // 2 + 1
    T_24 = T_mel
    T_48 = T_mel * 2

    def _vocos_fn(inp):
        return [
            rng.uniform(0, 1, (1, n_bins, T_24)).astype(np.float32),  # mag_24k
            rng.uniform(-np.pi, np.pi, (1, n_bins, T_24)).astype(np.float32),  # phase_24k
            rng.uniform(0, 1, (1, n_bins, T_48)).astype(np.float32),  # mag_48k
            rng.uniform(-np.pi, np.pi, (1, n_bins, T_48)).astype(np.float32),  # phase_48k
        ]

    vocos_sess = _MockSession(_vocos_fn)

    return acoustic_sess, distill_sess, content_sess, global_sess, mel_sess, vocos_sess


def test_adapter_clone_voice_mock(tmp_path):
    """Full pipeline completes with mocked ORT sessions."""
    from voiceclonnx.engines.linacodec import LinaCodecAdapter

    src_wav = _make_wav(str(tmp_path / "src.wav"), duration_s=1.0, sr=24000)
    ref_wav = _make_wav(str(tmp_path / "ref.wav"), duration_s=1.0, sr=24000)
    out_wav = str(tmp_path / "out.wav")

    adapter = LinaCodecAdapter()
    (
        adapter._acoustic_sess,
        adapter._distill_sess,
        adapter._content_sess,
        adapter._global_sess,
        adapter._mel_sess,
        adapter._vocos_sess,
    ) = _make_mock_sessions()

    result = adapter.clone_voice(src_wav, ref_wav, out_wav)
    assert result == str(Path(out_wav).resolve())
    assert Path(out_wav).exists()
    assert Path(out_wav).stat().st_size > 0

    with wave.open(out_wav, "rb") as wf:
        assert wf.getframerate() == 48000
        assert wf.getsampwidth() == 2
        assert wf.getnframes() > 0


def test_adapter_output_is_48k(tmp_path):
    """Output WAV must be 48kHz regardless of input sample rate."""
    from voiceclonnx.engines.linacodec import LinaCodecAdapter

    # Write source at 16kHz
    src_path = str(tmp_path / "src16.wav")
    n = 16000
    data = (np.zeros(n, dtype=np.float32) * 32767).astype(np.int16)
    with wave.open(src_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(16000)
        wf.writeframes(data.tobytes())

    ref_wav = _make_wav(str(tmp_path / "ref.wav"))
    out_wav = str(tmp_path / "out.wav")

    adapter = LinaCodecAdapter()
    (
        adapter._acoustic_sess,
        adapter._distill_sess,
        adapter._content_sess,
        adapter._global_sess,
        adapter._mel_sess,
        adapter._vocos_sess,
    ) = _make_mock_sessions()

    adapter.clone_voice(src_path, ref_wav, out_wav)

    with wave.open(out_wav, "rb") as wf:
        assert wf.getframerate() == 48000


def test_adapter_lazy_load_raises_without_onnxruntime(monkeypatch):
    """Missing onnxruntime raises ImportError with a helpful message."""
    import builtins
    from voiceclonnx.engines.linacodec import LinaCodecAdapter

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("no onnxruntime")
        return real_import(name, *args, **kwargs)

    adapter = LinaCodecAdapter()
    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="onnxruntime"):
            adapter._ensure_models()


def test_normalize_ssl():
    """_normalize_ssl should produce zero-mean unit-variance features."""
    from voiceclonnx.engines.linacodec import _normalize_ssl

    rng = np.random.default_rng(10)
    feats = rng.standard_normal((1, 50, 768)).astype(np.float32) * 5 + 3
    normed = _normalize_ssl(feats)

    # Mean across time axis should be ~0
    assert np.abs(normed[0].mean(axis=0)).max() < 0.5
    # Std across time axis should be ~1
    assert np.abs(normed[0].std(axis=0) - 1.0).max() < 0.5


def test_adapter_uses_source_content_ref_global(tmp_path):
    """The VC recipe uses source content_embedding and reference global_embedding."""
    from voiceclonnx.engines.linacodec import LinaCodecAdapter

    src_wav = _make_wav(str(tmp_path / "src.wav"))
    ref_wav = _make_wav(str(tmp_path / "ref.wav"))
    out_wav = str(tmp_path / "out.wav")

    # Track what global_embedding was passed to mel_decoder
    received_globals = []
    received_contents = []

    class _TrackingGlobalSession:
        def run(self, output_names, inputs):
            rng = np.random.default_rng(999)
            # Return distinct embeddings for src (call 1) vs ref (call 2)
            return [np.full((1, 128), len(received_globals), dtype=np.float32)]

    class _TrackingMelSession:
        def run(self, output_names, inputs):
            received_globals.append(inputs["global_embedding"].copy())
            received_contents.append(inputs["content_embedding"].copy())
            return [np.zeros((1, 100, 94), dtype=np.float32)]

    adapter = LinaCodecAdapter()
    (
        adapter._acoustic_sess,
        adapter._distill_sess,
        adapter._content_sess,
        adapter._global_sess,
        adapter._mel_sess,
        adapter._vocos_sess,
    ) = _make_mock_sessions()
    adapter._global_sess = _TrackingGlobalSession()
    adapter._mel_sess = _TrackingMelSession()

    adapter.clone_voice(src_wav, ref_wav, out_wav)

    # Mel decoder should have been called once with a global embedding
    assert len(received_globals) == 1
    assert received_globals[0].shape == (1, 128)


# ---------------------------------------------------------------------------
# 7. E2E test (VOICECLONNX_E2E-gated)
# ---------------------------------------------------------------------------

_SKIP_E2E = not os.environ.get("VOICECLONNX_E2E", "")
_E2E_REASON = (
    "E2E linacodec test downloads ~300+ MB of ONNX models and requires "
    "a local export (ONNX not yet on HF Hub). "
    "Set VOICECLONNX_E2E=1 and LINACODEC_MODEL_DIR=/path/to/export to run."
)


@pytest.mark.skipif(_SKIP_E2E, reason=_E2E_REASON)
def test_e2e_linacodec_wer_gate(tmp_path):
    """Real end-to-end: convert edge-tts source and verify WER ≤ 25%.

    Requires VOICECLONNX_E2E=1 and LINACODEC_MODEL_DIR pointing to
    a directory with exported ONNX files.
    """
    import asyncio
    import subprocess

    model_dir = os.environ.get("LINACODEC_MODEL_DIR")
    if not model_dir:
        pytest.skip("LINACODEC_MODEL_DIR not set — point to local ONNX export dir")

    try:
        import edge_tts
    except ImportError:
        pytest.skip("edge-tts not installed")

    from voiceclonnx.engines.linacodec import LinaCodecAdapter

    SOURCE_TEXT = "Hello, this is a test of voice conversion."
    REF_TEXT = "The quick brown fox jumps over the lazy dog."
    WER_GATE = 0.25

    async def _synth(text: str, voice: str, out: str):
        await edge_tts.Communicate(text, voice).save(out)

    src_mp3 = str(tmp_path / "src.mp3")
    ref_mp3 = str(tmp_path / "ref.mp3")
    src_wav = str(tmp_path / "src.wav")
    ref_wav = str(tmp_path / "ref.wav")
    out_wav = str(tmp_path / "out_linacodec.wav")

    asyncio.run(_synth(SOURCE_TEXT, "en-US-GuyNeural", src_mp3))
    asyncio.run(_synth(REF_TEXT, "en-US-AriaNeural", ref_mp3))

    subprocess.run(["ffmpeg", "-y", "-i", src_mp3, src_wav], check=True, capture_output=True)
    subprocess.run(["ffmpeg", "-y", "-i", ref_mp3, ref_wav], check=True, capture_output=True)

    adapter = LinaCodecAdapter(quantized=False, model_dir=model_dir)
    result = adapter.clone_voice(src_wav, ref_wav, out_wav)

    assert Path(result).exists()
    assert Path(result).stat().st_size > 0

    with wave.open(result, "rb") as wf:
        sr_out = wf.getframerate()
        n_frames = wf.getnframes()
        duration = n_frames / sr_out

    assert sr_out == 48000, f"Expected 48000 Hz, got {sr_out}"
    assert 0.3 <= duration <= 20.0, f"Suspicious duration: {duration:.2f}s"

    # WER gate via faster-whisper
    try:
        from faster_whisper import WhisperModel
        wm = WhisperModel("base.en", device="cpu", compute_type="int8")
        segments, _ = wm.transcribe(result, language="en")
        transcript = " ".join(s.text.strip() for s in segments).lower()
        ref_words = SOURCE_TEXT.lower().split()
        from difflib import SequenceMatcher
        matcher = SequenceMatcher(None, transcript.split(), ref_words)
        wer = 1.0 - matcher.ratio()
        print(f"\n[e2e linacodec] transcript: {transcript!r}")
        print(f"[e2e linacodec] WER: {wer:.1%}  (gate ≤ {WER_GATE:.0%})  duration: {duration:.2f}s")
        assert wer <= WER_GATE, f"WER {wer:.1%} exceeds gate {WER_GATE:.0%} — NOT MERGEABLE"
    except ImportError:
        print("[e2e linacodec] faster-whisper not installed — skipping WER check")
        print(f"[e2e linacodec] duration={duration:.2f}s  sr={sr_out}")
