"""Tests for the vec2wav 2.0 adapter — voiceclonnx/engines/vec2wav.py.

Structure
---------
- Registry wiring (no model loading)
- Pure-numpy VQ encode logic on synthetic data
- Mock-session contract tests (full adapter pipeline with stubbed ORT)
- E2E test gated on VOICECLONNX_E2E=1 (real models, edge-tts, WER gate ≤25%)
"""

from __future__ import annotations

import os
import wave
from pathlib import Path
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
    sess = MagicMock()
    sess.run.return_value = [output]
    return sess


# ---------------------------------------------------------------------------
# 1. Registry wiring
# ---------------------------------------------------------------------------


def test_vec2wav_registered():
    """vec2wav engine must appear in ENGINE_REGISTRY after importing the module."""
    import voiceclonnx.engines.vec2wav  # noqa: F401 — trigger registration
    from voiceclonnx.engines.base import ENGINE_REGISTRY

    assert "vec2wav" in ENGINE_REGISTRY


def test_vec2wav_entry_metadata():
    import voiceclonnx.engines.vec2wav  # noqa: F401
    from voiceclonnx.engines.base import get_engine

    entry = get_engine("vec2wav")
    assert entry.alias == "vec2wav"
    assert entry.onnx_native is True
    assert entry.extras == ""
    assert entry.adapter_class.__name__ == "Vec2WavAdapter"


def test_vec2wav_sample_rate():
    from voiceclonnx.engines.vec2wav import Vec2WavAdapter

    adapter = Vec2WavAdapter()
    assert adapter.sample_rate == 24000


def test_vec2wav_quantized_flag():
    from voiceclonnx.engines.vec2wav import Vec2WavAdapter

    a = Vec2WavAdapter(quantized=True)
    assert a._quantized is True
    b = Vec2WavAdapter(quantized=False)
    assert b._quantized is False


def test_vec2wav_auto_imported_via_package():
    """Auto-import in voiceclonnx/__init__.py must register vec2wav."""
    import voiceclonnx  # noqa: F401
    from voiceclonnx.engines.base import ENGINE_REGISTRY

    assert "vec2wav" in ENGINE_REGISTRY


# ---------------------------------------------------------------------------
# 2. Pure-numpy VQ encode — real computation on synthetic data
# ---------------------------------------------------------------------------


def test_vq_encode_output_shape():
    from voiceclonnx.engines.vec2wav import _vq_encode

    rng = np.random.default_rng(0)
    L = 40
    cnn_feats = rng.random((L, 512), dtype=np.float32)
    codebook = rng.random((2, 320, 256), dtype=np.float32)

    out = _vq_encode(cnn_feats, codebook)
    assert out.shape == (L, 512)
    assert out.dtype == np.float32


def test_vq_encode_selects_nearest():
    """Each group's output must be one of the actual codebook rows."""
    from voiceclonnx.engines.vec2wav import _vq_encode

    rng = np.random.default_rng(42)
    L = 5
    G, V, D = 2, 10, 8  # small codebook for easy verification
    codebook = rng.random((G, V, D), dtype=np.float32)
    cnn_feats = rng.random((L, G * D), dtype=np.float32)

    out = _vq_encode(cnn_feats, codebook)

    # Each half of the output vector must match one of the codebook rows exactly
    for l in range(L):
        for g in range(G):
            out_vec = out[l, g * D: (g + 1) * D]
            # Must be in the codebook
            dists = np.sum((codebook[g] - out_vec) ** 2, axis=1)
            assert np.min(dists) < 1e-10, f"frame {l} group {g} not matched to any codebook entry"


def test_vq_encode_is_deterministic():
    from voiceclonnx.engines.vec2wav import _vq_encode

    rng = np.random.default_rng(7)
    cnn_feats = rng.random((30, 512), dtype=np.float32)
    codebook = rng.random((2, 320, 256), dtype=np.float32)

    out1 = _vq_encode(cnn_feats, codebook)
    out2 = _vq_encode(cnn_feats, codebook)
    np.testing.assert_array_equal(out1, out2)


def test_vq_encode_identical_feature_picks_nearest():
    """All frames identical → all outputs identical."""
    from voiceclonnx.engines.vec2wav import _vq_encode

    rng = np.random.default_rng(1)
    single = rng.random((1, 512), dtype=np.float32)
    cnn_feats = np.tile(single, (20, 1))  # all frames identical
    codebook = rng.random((2, 320, 256), dtype=np.float32)

    out = _vq_encode(cnn_feats, codebook)
    # All rows must be identical
    for i in range(1, 20):
        np.testing.assert_array_equal(out[0], out[i])


# ---------------------------------------------------------------------------
# 3. Mock-session contract tests — full pipeline
# ---------------------------------------------------------------------------


def test_vec2wav_clone_voice_mock(tmp_path):
    """Adapter's clone_voice must write a non-empty WAV with the correct sample rate."""
    from voiceclonnx.engines.vec2wav import Vec2WavAdapter

    src_path = _make_wav(str(tmp_path / "src.wav"), duration_s=0.5, sr=16000)
    ref_path = _make_wav(str(tmp_path / "ref.wav"), duration_s=0.5, sr=16000)
    out_path = str(tmp_path / "out.wav")

    rng = np.random.default_rng(0)
    L = 25  # ~0.5 s at 50 Hz
    T_ref = 25
    N = 24000 * 1  # 1 second of audio at 24 kHz

    # Mock session outputs
    cnn_out = rng.random((1, L, 512), dtype=np.float32)
    wavlm_out = rng.random((1, T_ref, 1024), dtype=np.float32)
    frontend_out = rng.random((1, 184, L), dtype=np.float32)
    vocoder_out = rng.random((1, 1, N), dtype=np.float32) * 0.1  # small amplitude

    cnn_sess = _mock_ort_session(cnn_out)
    wavlm_sess = _mock_ort_session(wavlm_out)
    frontend_sess = _mock_ort_session(frontend_out)
    vocoder_sess = _mock_ort_session(vocoder_out)
    codebook = rng.random((2, 320, 256), dtype=np.float32)

    adapter = Vec2WavAdapter(quantized=False)
    # Inject mock sessions directly (bypasses HF download)
    adapter._cnn_sess = cnn_sess
    adapter._wavlm_sess = wavlm_sess
    adapter._frontend_sess = frontend_sess
    adapter._vocoder_sess = vocoder_sess
    adapter._codebook = codebook

    result = adapter.clone_voice(src_path, ref_path, out_path)

    assert result == out_path, "clone_voice must return out_path"
    assert Path(out_path).exists(), "output file must exist"

    with wave.open(out_path, "rb") as wf:
        assert wf.getframerate() == 24000
        assert wf.getnframes() > 0


def test_vec2wav_clone_voice_calls_all_sessions(tmp_path):
    """All four ONNX sessions must be called during clone_voice."""
    from voiceclonnx.engines.vec2wav import Vec2WavAdapter

    src_path = _make_wav(str(tmp_path / "src.wav"), duration_s=0.3)
    ref_path = _make_wav(str(tmp_path / "ref.wav"), duration_s=0.3)
    out_path = str(tmp_path / "out.wav")

    rng = np.random.default_rng(5)
    L, T_ref, N = 15, 15, 12000

    cnn_sess = _mock_ort_session(rng.random((1, L, 512), dtype=np.float32))
    wavlm_sess = _mock_ort_session(rng.random((1, T_ref, 1024), dtype=np.float32))
    frontend_sess = _mock_ort_session(rng.random((1, 184, L), dtype=np.float32))
    vocoder_sess = _mock_ort_session(rng.random((1, 1, N), dtype=np.float32) * 0.05)
    codebook = rng.random((2, 320, 256), dtype=np.float32)

    adapter = Vec2WavAdapter()
    adapter._cnn_sess = cnn_sess
    adapter._wavlm_sess = wavlm_sess
    adapter._frontend_sess = frontend_sess
    adapter._vocoder_sess = vocoder_sess
    adapter._codebook = codebook

    adapter.clone_voice(src_path, ref_path, out_path)

    cnn_sess.run.assert_called_once()
    # WavLM is called once (speaker encoding only — content uses CNN not WavLM)
    wavlm_sess.run.assert_called_once()
    frontend_sess.run.assert_called_once()
    vocoder_sess.run.assert_called_once()


def test_vec2wav_output_clipped(tmp_path):
    """Output waveform values must be clamped to [-1, 1] (int16 writing)."""
    from voiceclonnx.engines.vec2wav import Vec2WavAdapter

    src_path = _make_wav(str(tmp_path / "src.wav"), duration_s=0.2)
    ref_path = _make_wav(str(tmp_path / "ref.wav"), duration_s=0.2)
    out_path = str(tmp_path / "out.wav")

    rng = np.random.default_rng(99)
    L, T_ref, N = 10, 10, 4800
    # Deliberately large vocoder output to test clipping
    vocoder_out = rng.random((1, 1, N), dtype=np.float32) * 10.0

    adapter = Vec2WavAdapter()
    adapter._cnn_sess = _mock_ort_session(rng.random((1, L, 512), dtype=np.float32))
    adapter._wavlm_sess = _mock_ort_session(rng.random((1, T_ref, 1024), dtype=np.float32))
    adapter._frontend_sess = _mock_ort_session(rng.random((1, 184, L), dtype=np.float32))
    adapter._vocoder_sess = _mock_ort_session(vocoder_out)
    adapter._codebook = rng.random((2, 320, 256), dtype=np.float32)

    adapter.clone_voice(src_path, ref_path, out_path)

    import soundfile as sf
    audio, sr = sf.read(out_path, dtype="float32")
    assert np.all(np.abs(audio) <= 1.0 + 1e-4), "audio should be clamped to [-1, 1]"


def test_vec2wav_resamples_input(tmp_path):
    """Input WAV at non-16 kHz must be silently resampled (no crash)."""
    from voiceclonnx.engines.vec2wav import Vec2WavAdapter

    # Write a 22 kHz WAV — adapter must resample to 16 kHz internally
    src_path = _make_wav(str(tmp_path / "src_22k.wav"), duration_s=0.3, sr=22050)
    ref_path = _make_wav(str(tmp_path / "ref_22k.wav"), duration_s=0.3, sr=22050)
    out_path = str(tmp_path / "out.wav")

    rng = np.random.default_rng(3)
    L, T_ref, N = 10, 10, 4800

    adapter = Vec2WavAdapter()
    adapter._cnn_sess = _mock_ort_session(rng.random((1, L, 512), dtype=np.float32))
    adapter._wavlm_sess = _mock_ort_session(rng.random((1, T_ref, 1024), dtype=np.float32))
    adapter._frontend_sess = _mock_ort_session(rng.random((1, 184, L), dtype=np.float32))
    adapter._vocoder_sess = _mock_ort_session(rng.random((1, 1, N), dtype=np.float32) * 0.1)
    adapter._codebook = rng.random((2, 320, 256), dtype=np.float32)

    # Should not raise
    adapter.clone_voice(src_path, ref_path, out_path)
    assert Path(out_path).exists()


# ---------------------------------------------------------------------------
# 4. E2E test — gated on VOICECLONNX_E2E=1
# ---------------------------------------------------------------------------

_SKIP_E2E = not os.environ.get("VOICECLONNX_E2E", "")
_E2E_REASON = (
    "E2E vec2wav test downloads large ONNX models and requires edge-tts + faster-whisper; "
    "set VOICECLONNX_E2E=1 to run."
)


def _synth_wav_edge(text: str, voice: str, out_wav: str) -> None:
    """Generate 16 kHz WAV via edge-tts + ffmpeg."""
    import asyncio
    import subprocess
    import edge_tts

    mp3_path = out_wav.replace(".wav", ".mp3")

    async def _run():
        await edge_tts.Communicate(text, voice).save(mp3_path)

    asyncio.run(_run())
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", mp3_path,
         "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", out_wav],
        check=True,
    )
    Path(mp3_path).unlink(missing_ok=True)


def _wer(reference: str, hypothesis: str) -> float:
    """Very simple token-level WER."""
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens:
        return 0.0
    # Levenshtein distance
    n, m = len(ref_tokens), len(hyp_tokens)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        new_dp = [i] + [0] * m
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                new_dp[j] = dp[j - 1]
            else:
                new_dp[j] = 1 + min(dp[j], new_dp[j - 1], dp[j - 1])
        dp = new_dp
    return dp[m] / n


@pytest.mark.skipif(_SKIP_E2E, reason=_E2E_REASON)
@pytest.mark.timeout(600)
def test_vec2wav_e2e_wer(tmp_path):
    """Full pipeline: real models + edge-tts audio; WER must be ≤ 25%."""
    pytest.importorskip("edge_tts", reason="edge-tts not installed")
    faster_whisper = pytest.importorskip("faster_whisper", reason="faster-whisper not installed")

    SRC_TEXT = "Hello, this is a voice cloning test using voiceclonnx."
    REF_TEXT = "The quick brown fox jumps over the lazy dog."
    SRC_VOICE = "en-US-AriaNeural"
    REF_VOICE = "en-GB-SoniaNeural"

    src_path = str(tmp_path / "source.wav")
    ref_path = str(tmp_path / "reference.wav")
    out_path = str(tmp_path / "converted.wav")

    _synth_wav_edge(SRC_TEXT, SRC_VOICE, src_path)
    _synth_wav_edge(REF_TEXT, REF_VOICE, ref_path)

    from voiceclonnx import VoiceCloner
    cloner = VoiceCloner(engine="vec2wav")
    result = cloner.clone_voice(src_path, ref_path, out_path)

    assert result == out_path
    assert Path(out_path).exists()
    assert Path(out_path).stat().st_size > 1000

    with wave.open(out_path, "rb") as wf:
        assert wf.getframerate() == 24000

    # WER check with faster-whisper base.en
    model = faster_whisper.WhisperModel("base.en", device="cpu", compute_type="int8")
    segs, _ = model.transcribe(out_path)
    hypothesis = " ".join(seg.text.strip() for seg in segs)
    print(f"\n[e2e] transcription: {hypothesis!r}")
    print(f"[e2e] reference:      {SRC_TEXT!r}")

    wer_val = _wer(SRC_TEXT, hypothesis)
    print(f"[e2e] WER: {wer_val:.1%}")
    assert wer_val <= 0.25, (
        f"WER {wer_val:.1%} exceeds 25% gate. "
        f"Reference: {SRC_TEXT!r}  Hypothesis: {hypothesis!r}"
    )
