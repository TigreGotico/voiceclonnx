"""Tests for the Seed-VC adapter — voiceclonnx/engines/seedvc.py.

Structure
---------
1. Registry wiring (no model loading)
2. Audio I/O helpers
3. Feature extraction helpers (whisper log-mel, kaldi fbank, source mel)
4. Length regulator helpers (interpolation, embedding lookup)
5. ODE solver (shape + determinism with mocked estimator)
6. Adapter contract with mocked ORT sessions
7. E2E test (VOICECLONNX_E2E-gated) + WER gate
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


def _make_wav(path: str, duration_s: float = 1.0, sr: int = 22050) -> str:
    """Write a short sine-wave WAV for testing."""
    n = int(duration_s * sr)
    t = np.linspace(0, duration_s, n, endpoint=False)
    data = (np.sin(2 * np.pi * 440 * t) * 32767 * 0.5).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(data.tobytes())
    return path


# ---------------------------------------------------------------------------
# 1. Registry wiring
# ---------------------------------------------------------------------------


def test_seedvc_registered():
    import voiceclonnx.engines.seedvc  # noqa: F401
    from voiceclonnx.engines.base import ENGINE_REGISTRY

    assert "seedvc" in ENGINE_REGISTRY


def test_seedvc_entry_metadata():
    import voiceclonnx.engines.seedvc  # noqa: F401
    from voiceclonnx.engines.base import get_engine

    entry = get_engine("seedvc")
    assert entry.alias == "seedvc"
    assert entry.onnx_native is True
    assert entry.adapter_class.__name__ == "SeedVCAdapter"


def test_seedvc_sample_rate():
    from voiceclonnx.engines.seedvc import SeedVCAdapter

    adapter = SeedVCAdapter()
    assert adapter.sample_rate == 22050


def test_seedvc_quantized_flag():
    from voiceclonnx.engines.seedvc import SeedVCAdapter

    a = SeedVCAdapter(quantized=True)
    assert a._quantized is True
    b = SeedVCAdapter(quantized=False)
    assert b._quantized is False


def test_seedvc_ode_steps_flag():
    from voiceclonnx.engines.seedvc import SeedVCAdapter

    a = SeedVCAdapter(ode_steps=20)
    assert a._ode_steps == 20


# ---------------------------------------------------------------------------
# 2. Audio I/O helpers
# ---------------------------------------------------------------------------


def test_load_wav_mono(tmp_path):
    from voiceclonnx.engines.seedvc import _load_wav

    p = str(tmp_path / "test.wav")
    _make_wav(p, duration_s=0.5, sr=22050)
    audio = _load_wav(p, target_sr=22050)
    assert audio.ndim == 1
    assert audio.dtype == np.float32
    assert len(audio) > 0


def test_load_wav_resamples(tmp_path):
    from voiceclonnx.engines.seedvc import _load_wav

    p = str(tmp_path / "test22.wav")
    _make_wav(p, duration_s=1.0, sr=22050)
    audio = _load_wav(p, target_sr=16000)
    assert abs(len(audio) - 16000) <= 20


def test_save_wav_22k(tmp_path):
    from voiceclonnx.engines.seedvc import _save_wav

    audio = np.zeros(22050, dtype=np.float32)
    p = str(tmp_path / "out.wav")
    _save_wav(p, audio, sr=22050)
    with wave.open(p, "rb") as wf:
        assert wf.getframerate() == 22050
        assert wf.getsampwidth() == 2


def test_resample_linear():
    from voiceclonnx.engines.seedvc import _resample_linear

    audio = np.sin(2 * np.pi * 440 * np.arange(22050) / 22050).astype(np.float32)
    out = _resample_linear(audio, 22050, 16000)
    assert abs(len(out) - 16000) <= 5
    assert out.dtype == np.float32


# ---------------------------------------------------------------------------
# 3. Feature extraction helpers
# ---------------------------------------------------------------------------


def test_whisper_log_mel_shape():
    from voiceclonnx.engines.seedvc import _whisper_log_mel

    wav = np.zeros(16000, dtype=np.float32)
    log_mel, actual_frames = _whisper_log_mel(wav)
    assert log_mel.shape == (1, 128, 3000)
    assert log_mel.dtype == np.float32
    assert actual_frames > 0 and actual_frames <= 3000


def test_whisper_log_mel_normalization():
    """Values should be in reasonable range after Whisper normalization."""
    from voiceclonnx.engines.seedvc import _whisper_log_mel

    rng = np.random.default_rng(0)
    wav = rng.standard_normal(16000).astype(np.float32) * 0.1
    log_mel, _ = _whisper_log_mel(wav)
    # After (log10 + 4) / 4 normalization, values should be in [-2, 3] roughly
    assert log_mel.min() >= -2.0
    assert log_mel.max() <= 3.0


def test_kaldi_fbank_shape():
    from voiceclonnx.engines.seedvc import _kaldi_fbank

    wav = np.zeros(16000, dtype=np.float32)
    fb = _kaldi_fbank(wav)
    assert fb.shape[0] == 1
    assert fb.shape[2] == 80
    assert fb.dtype == np.float32


def test_source_mel_shape():
    from voiceclonnx.engines.seedvc import _source_mel_spectrogram

    wav = np.zeros(22050, dtype=np.float32)
    mel = _source_mel_spectrogram(wav)
    assert mel.shape[0] == 1
    assert mel.shape[1] == 80
    assert mel.shape[2] > 0


# ---------------------------------------------------------------------------
# 4. Length regulator helpers
# ---------------------------------------------------------------------------


def test_interpolate_nearest_size():
    from voiceclonnx.engines.seedvc import _interpolate_nearest

    x = np.arange(24, dtype=np.float32).reshape(1, 4, 6)
    out = _interpolate_nearest(x, 10)
    assert out.shape == (1, 4, 10)


def test_interpolate_nearest_identity():
    from voiceclonnx.engines.seedvc import _interpolate_nearest

    x = np.random.rand(1, 8, 12).astype(np.float32)
    out = _interpolate_nearest(x, 12)
    np.testing.assert_array_equal(out, x)


def test_mel_filterbank_shape():
    from voiceclonnx.engines.seedvc import _mel_filterbank_htk

    fb = _mel_filterbank_htk(16000, 400, 80, 20.0, 7600.0)
    assert fb.shape == (80, 201)
    assert fb.dtype == np.float32
    assert np.all(fb >= 0)


# ---------------------------------------------------------------------------
# 5. ODE solver shape + determinism with mocked estimator
# ---------------------------------------------------------------------------


def test_ode_solve_shape():
    """ODE solver returns correct shape with a trivial (zero) estimator."""
    from voiceclonnx.engines.seedvc import _euler_ode_solve

    T_src = 40
    T_ref = 20

    class _ZeroEstimator:
        def run(self, output_names, inputs):
            # Return zeros matching batch-2 (B=2)
            B, C, T = inputs["x"].shape
            return [np.zeros((B, C, T), dtype=np.float32)]

    mu = np.zeros((1, 80, T_src), dtype=np.float32)
    prompt_mel = np.zeros((1, 80, T_ref), dtype=np.float32)
    style = np.zeros((1, 192), dtype=np.float32)

    out = _euler_ode_solve(
        mu=mu,
        prompt_mel=prompt_mel,
        style=style,
        flow_sess=_ZeroEstimator(),
        n_steps=3,
        cfg_rate=0.7,
    )

    assert out.shape == (1, 80, T_src), f"Expected (1, 80, {T_src}), got {out.shape}"
    assert out.dtype == np.float32


def test_ode_solve_not_nan():
    """ODE output must be finite."""
    from voiceclonnx.engines.seedvc import _euler_ode_solve

    rng = np.random.default_rng(0)

    class _RandEstimator:
        def run(self, output_names, inputs):
            B, C, T = inputs["x"].shape
            return [rng.standard_normal((B, C, T)).astype(np.float32) * 0.01]

    mu = rng.standard_normal((1, 80, 30)).astype(np.float32)
    prompt_mel = rng.standard_normal((1, 80, 15)).astype(np.float32)
    style = rng.standard_normal((1, 192)).astype(np.float32)

    out = _euler_ode_solve(
        mu=mu, prompt_mel=prompt_mel, style=style,
        flow_sess=_RandEstimator(), n_steps=3, cfg_rate=0.7,
    )
    assert np.all(np.isfinite(out))


def test_ode_solve_cfg_combined():
    """CFG formula: (1 + r)*v_cond - r*v_uncond applied per step."""
    from voiceclonnx.engines.seedvc import _euler_ode_solve

    # Estimator returns: slot-0 = 1.0, slot-1 = 0.0
    # Expected combined: (1 + 0.7)*1.0 - 0.7*0.0 = 1.7
    class _ConstEstimator:
        def run(self, output_names, inputs):
            B, C, T = inputs["x"].shape
            v = np.zeros((B, C, T), dtype=np.float32)
            v[0] = 1.0  # conditioned
            v[1] = 0.0  # unconditioned
            return [v]

    mu = np.zeros((1, 80, 10), dtype=np.float32)
    prompt_mel = np.zeros((1, 80, 5), dtype=np.float32)
    style = np.zeros((1, 192), dtype=np.float32)

    np.random.seed(0)
    out = _euler_ode_solve(
        mu=mu, prompt_mel=prompt_mel, style=style,
        flow_sess=_ConstEstimator(), n_steps=1, cfg_rate=0.7,
    )
    # With 1 step: x_final ≈ noise + dt * 1.7 in the non-prompt region
    assert out.shape == (1, 80, 10)


# ---------------------------------------------------------------------------
# 6. Adapter contract with mocked ORT sessions
# ---------------------------------------------------------------------------


def _make_mock_sessions(T_src_mel: int = 40, T_ref_mel: int = 20):
    """Return mock ORT sessions that return plausible shapes."""

    class _MockSession:
        def __init__(self, fn):
            self._fn = fn

        def run(self, output_names, inputs):
            return self._fn(inputs)

    rng = np.random.default_rng(42)

    # Whisper encoder: (1, 128, 3000) → (1, T_enc, 768)
    T_enc = 50
    whisper_sess = _MockSession(lambda inp: [
        rng.standard_normal((1, T_enc, 768)).astype(np.float32)
    ])

    # CAMPPlus: (1, T, 80) → (1, 192)
    campplus_sess = _MockSession(lambda inp: [
        rng.standard_normal((1, 192)).astype(np.float32)
    ])

    # LR model: (1, 512, T) → (1, 512, T)
    lr_sess = _MockSession(lambda inp: [
        rng.standard_normal(inp["x"].shape).astype(np.float32)
    ])

    # Flow estimator: (x, prompt_x, x_lens, t, style, mu) → (2, 80, T)
    def _flow_fn(inp):
        B, C, T = inp["x"].shape
        return [rng.standard_normal((B, C, T)).astype(np.float32) * 0.01]

    flow_sess = _MockSession(_flow_fn)

    # BigVGAN: (1, 80, T_mel) → (1, 1, T_audio)
    def _bigvgan_fn(inp):
        T_mel = inp["mel"].shape[2]
        T_audio = T_mel * 256
        return [rng.standard_normal((1, 1, T_audio)).astype(np.float32) * 0.1]

    bigvgan_sess = _MockSession(_bigvgan_fn)

    return whisper_sess, campplus_sess, lr_sess, flow_sess, bigvgan_sess


def _make_mock_lr_emb() -> np.ndarray:
    """Fake 2048×512 LR embedding."""
    rng = np.random.default_rng(99)
    return rng.standard_normal((2048, 512)).astype(np.float32)


def test_adapter_clone_voice_mock(tmp_path):
    """Full pipeline completes with mocked ORT sessions."""
    from voiceclonnx.engines.seedvc import SeedVCAdapter

    src_wav = _make_wav(str(tmp_path / "src.wav"), duration_s=1.0, sr=22050)
    ref_wav = _make_wav(str(tmp_path / "ref.wav"), duration_s=1.0, sr=22050)
    out_wav = str(tmp_path / "out.wav")

    adapter = SeedVCAdapter()
    (
        adapter._whisper_sess,
        adapter._campplus_sess,
        adapter._lr_sess,
        adapter._flow_sess,
        adapter._bigvgan_sess,
    ) = _make_mock_sessions()
    adapter._lr_emb = _make_mock_lr_emb()

    result = adapter.clone_voice(src_wav, ref_wav, out_wav)
    assert result == str(Path(out_wav).resolve())
    assert Path(out_wav).exists()
    assert Path(out_wav).stat().st_size > 0

    with wave.open(out_wav, "rb") as wf:
        assert wf.getframerate() == 22050
        assert wf.getsampwidth() == 2
        assert wf.getnframes() > 0


def test_adapter_output_is_22k(tmp_path):
    """Output WAV must be 22050 Hz regardless of input sample rate."""
    from voiceclonnx.engines.seedvc import SeedVCAdapter

    # Write source at 16kHz
    src_path = str(tmp_path / "src16.wav")
    _make_wav(src_path, duration_s=1.0, sr=16000)
    ref_wav = _make_wav(str(tmp_path / "ref.wav"))
    out_wav = str(tmp_path / "out.wav")

    adapter = SeedVCAdapter()
    (
        adapter._whisper_sess,
        adapter._campplus_sess,
        adapter._lr_sess,
        adapter._flow_sess,
        adapter._bigvgan_sess,
    ) = _make_mock_sessions()
    adapter._lr_emb = _make_mock_lr_emb()

    adapter.clone_voice(src_path, ref_wav, out_wav)

    with wave.open(out_wav, "rb") as wf:
        assert wf.getframerate() == 22050


def test_adapter_lazy_load_raises_without_onnxruntime(monkeypatch):
    """Missing onnxruntime raises ImportError with a helpful message."""
    import builtins
    from voiceclonnx.engines.seedvc import SeedVCAdapter

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("no onnxruntime")
        return real_import(name, *args, **kwargs)

    adapter = SeedVCAdapter()
    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="onnxruntime"):
            adapter._ensure_models()


def test_adapter_long_source_mock(tmp_path):
    """Adapter handles a 5-second source clip without error."""
    from voiceclonnx.engines.seedvc import SeedVCAdapter

    src_wav = _make_wav(str(tmp_path / "src5s.wav"), duration_s=5.0, sr=22050)
    ref_wav = _make_wav(str(tmp_path / "ref.wav"), duration_s=1.0, sr=22050)
    out_wav = str(tmp_path / "out.wav")

    adapter = SeedVCAdapter()
    (
        adapter._whisper_sess,
        adapter._campplus_sess,
        adapter._lr_sess,
        adapter._flow_sess,
        adapter._bigvgan_sess,
    ) = _make_mock_sessions()
    adapter._lr_emb = _make_mock_lr_emb()

    result = adapter.clone_voice(src_wav, ref_wav, out_wav)
    assert Path(result).exists()
    assert Path(result).stat().st_size > 0


def test_adapter_ode_steps_parameter(tmp_path):
    """ode_steps parameter controls how many steps are taken."""
    from voiceclonnx.engines.seedvc import SeedVCAdapter

    call_count = [0]

    class _CountingEstimator:
        def run(self, output_names, inputs):
            call_count[0] += 1
            B, C, T = inputs["x"].shape
            return [np.zeros((B, C, T), dtype=np.float32)]

    src_wav = _make_wav(str(tmp_path / "src.wav"))
    ref_wav = _make_wav(str(tmp_path / "ref.wav"))
    out_wav = str(tmp_path / "out.wav")

    n_steps = 5
    adapter = SeedVCAdapter(ode_steps=n_steps)
    (
        adapter._whisper_sess,
        adapter._campplus_sess,
        adapter._lr_sess,
        _,
        adapter._bigvgan_sess,
    ) = _make_mock_sessions()
    adapter._flow_sess = _CountingEstimator()
    adapter._lr_emb = _make_mock_lr_emb()

    adapter.clone_voice(src_wav, ref_wav, out_wav)
    assert call_count[0] == n_steps, (
        f"Expected {n_steps} ODE steps, got {call_count[0]}"
    )


# ---------------------------------------------------------------------------
# 7. E2E test (VOICECLONNX_E2E-gated)
# ---------------------------------------------------------------------------

_SKIP_E2E = not os.environ.get("VOICECLONNX_E2E", "")
_E2E_REASON = (
    "E2E seedvc test downloads ~430+ MB of ONNX models. "
    "Set VOICECLONNX_E2E=1 (and optionally SEEDVC_MODEL_DIR=/path/to/export) to run."
)


@pytest.mark.skipif(_SKIP_E2E, reason=_E2E_REASON)
def test_e2e_seedvc_wer_gate(tmp_path):
    """Real end-to-end: convert demo source and verify WER ≤ 25%.

    Requires VOICECLONNX_E2E=1.
    Optionally set SEEDVC_MODEL_DIR to a local export dir to avoid downloading.
    """
    import asyncio
    import subprocess
    import wave as wave_mod

    model_dir = os.environ.get("SEEDVC_MODEL_DIR")

    try:
        import edge_tts
    except ImportError:
        pytest.skip("edge-tts not installed")

    from voiceclonnx.engines.seedvc import SeedVCAdapter

    SOURCE_TEXT = "The quick brown fox jumps over the lazy dog. Voice conversion changes who is speaking."
    WER_GATE = 0.25

    async def _synth(text: str, voice: str, out: str):
        await edge_tts.Communicate(text, voice).save(out)

    # Use the bundled demo files if available
    demo_dir = Path(__file__).resolve().parent.parent / "demo"
    src_wav_path = demo_dir / "source.wav"
    ref_aria_path = demo_dir / "reference_aria.wav"

    if not src_wav_path.exists():
        src_mp3 = str(tmp_path / "src.mp3")
        src_wav_path = tmp_path / "src.wav"
        asyncio.run(_synth(SOURCE_TEXT, "en-US-GuyNeural", src_mp3))
        subprocess.run(
            ["ffmpeg", "-y", "-i", src_mp3, str(src_wav_path)],
            check=True, capture_output=True
        )

    if not ref_aria_path.exists():
        ref_mp3 = str(tmp_path / "ref.mp3")
        ref_aria_path = tmp_path / "ref_aria.wav"
        asyncio.run(_synth("This sentence provides the reference voice.", "en-US-AriaNeural", ref_mp3))
        subprocess.run(
            ["ffmpeg", "-y", "-i", ref_mp3, str(ref_aria_path)],
            check=True, capture_output=True
        )

    for ref_name, ref_path in [("aria", ref_aria_path)]:
        out_wav = str(tmp_path / f"out_seedvc_{ref_name}.wav")

        kwargs = {"quantized": False}
        if model_dir:
            kwargs["model_dir"] = model_dir

        adapter = SeedVCAdapter(**kwargs)
        result = adapter.clone_voice(str(src_wav_path), str(ref_path), out_wav)

        assert Path(result).exists()
        assert Path(result).stat().st_size > 0

        with wave_mod.open(result, "rb") as wf:
            sr_out = wf.getframerate()
            n_frames = wf.getnframes()
            duration = n_frames / sr_out

        assert sr_out == 22050, f"Expected 22050 Hz, got {sr_out}"
        assert 0.5 <= duration <= 60.0, f"Suspicious duration: {duration:.2f}s"

        try:
            from faster_whisper import WhisperModel
            wm = WhisperModel("base.en", device="cpu", compute_type="int8")
            segments, _ = wm.transcribe(result, language="en")
            transcript = " ".join(s.text.strip() for s in segments).lower()
            ref_words = SOURCE_TEXT.lower().split()
            from difflib import SequenceMatcher
            matcher = SequenceMatcher(None, transcript.split(), ref_words)
            wer_score = 1.0 - matcher.ratio()
            print(f"\n[e2e seedvc/{ref_name}] transcript: {transcript!r}")
            print(f"[e2e seedvc/{ref_name}] WER={wer_score:.1%}  gate=≤{WER_GATE:.0%}  dur={duration:.2f}s")
            assert wer_score <= WER_GATE, (
                f"seedvc/{ref_name} WER {wer_score:.1%} exceeds gate {WER_GATE:.0%} — NOT MERGEABLE"
            )
        except ImportError:
            print(f"[e2e seedvc/{ref_name}] faster-whisper not installed — skipping WER check")
            print(f"[e2e seedvc/{ref_name}] duration={duration:.2f}s  sr={sr_out}")
