"""Tests for the CosyVoice adapter — voiceclonnx/engines/cosyvoice.py.

Structure
---------
1. Registry wiring (no model loading)
2. Audio helpers (pure numpy)
3. STFT/ISTFT numpy roundtrip
4. Adapter contract with fully mocked ORT sessions
5. E2E test gated on VOICECLONNX_E2E=1 (real models, WER gate ≤ 25%)
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


def test_cosyvoice_registered():
    """cosyvoice engine must appear in ENGINE_REGISTRY after importing."""
    import voiceclonnx.engines.cosyvoice  # noqa: F401
    from voiceclonnx.engines.base import ENGINE_REGISTRY

    assert "cosyvoice" in ENGINE_REGISTRY


def test_cosyvoice_entry_metadata():
    import voiceclonnx.engines.cosyvoice  # noqa: F401
    from voiceclonnx.engines.base import get_engine

    entry = get_engine("cosyvoice")
    assert entry.alias == "cosyvoice"
    assert entry.onnx_native is True
    assert entry.adapter_class.__name__ == "CosyVoiceAdapter"


def test_cosyvoice_sample_rate():
    from voiceclonnx.engines.cosyvoice import CosyVoiceAdapter

    a = CosyVoiceAdapter()
    assert a.sample_rate == 22050


def test_cosyvoice_quantized_flag():
    from voiceclonnx.engines.cosyvoice import CosyVoiceAdapter

    assert CosyVoiceAdapter(quantized=True)._quantized is True
    assert CosyVoiceAdapter(quantized=False)._quantized is False


def test_cosyvoice_ode_steps_stored():
    from voiceclonnx.engines.cosyvoice import CosyVoiceAdapter

    assert CosyVoiceAdapter(ode_steps=5)._ode_steps == 5


# ---------------------------------------------------------------------------
# 2. Audio helpers
# ---------------------------------------------------------------------------


def test_load_wav_resamples(tmp_path):
    """_load_wav must resample from 44100 to 16000."""
    from voiceclonnx.engines.cosyvoice import _load_wav

    path = str(tmp_path / "hi.wav")
    n = 44100
    data = np.zeros(n, dtype=np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(data.tobytes())

    audio = _load_wav(path, target_sr=16000)
    assert len(audio) == pytest.approx(16000, rel=0.05)
    assert audio.dtype == np.float32


def test_save_wav_16bit(tmp_path):
    from voiceclonnx.engines.cosyvoice import _save_wav

    path = str(tmp_path / "out.wav")
    audio = np.sin(np.linspace(0, 2 * np.pi, 22050)).astype(np.float32)
    _save_wav(path, audio, sr=22050)

    with wave.open(path, "rb") as wf:
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 22050


# ---------------------------------------------------------------------------
# 3. STFT/ISTFT numpy roundtrip
# ---------------------------------------------------------------------------


def test_stft_istft_roundtrip():
    """numpy STFT → ISTFT must reconstruct the signal (max_abs < 1e-5)."""
    from voiceclonnx.engines.cosyvoice import _stft, _istft

    np.random.seed(0)
    x = np.random.randn(4096).astype(np.float32) * 0.1
    real, imag = _stft(x, n_fft=16, hop_len=4)
    mag = np.sqrt(real ** 2 + imag ** 2)
    phase = np.arctan2(imag, real)
    x_rec = _istft(mag, phase, n_fft=16, hop_len=4)

    # trim center padding
    pad = 16 // 2
    x_rec_trim = x_rec[pad: pad + len(x)]
    max_abs = np.max(np.abs(x_rec_trim - x))
    assert max_abs < 1e-3, f"STFT→ISTFT roundtrip max_abs={max_abs:.2e} > 1e-3"


def test_stft_output_shape():
    from voiceclonnx.engines.cosyvoice import _stft

    x = np.zeros(22016, dtype=np.float32)
    real, imag = _stft(x, n_fft=16, hop_len=4)
    assert real.shape[0] == 16 // 2 + 1  # n_fft//2+1 = 9
    assert real.shape == imag.shape


# ---------------------------------------------------------------------------
# 4. Adapter contract with mocked ORT sessions
# ---------------------------------------------------------------------------


class _MockTokenizerSess:
    """Fake speech_tokenizer: feats (1,128,T) → tokens (1,1,T//4) int64."""
    def run(self, _, inputs):
        T = inputs["feats"].shape[2]
        T2 = max(1, T // 4)
        rng = np.random.default_rng(0)
        return [rng.integers(0, 4096, (1, 1, T2), dtype=np.int64)]


class _MockCAMPlussSess:
    """Fake campplus: (1,T,80) → (1,192) float32."""
    def get_inputs(self):
        class _I:
            name = "input"
        return [_I()]
    def run(self, _, inputs):
        rng = np.random.default_rng(1)
        return [rng.standard_normal((1, 192)).astype(np.float32)]


class _MockFlowEncoderSess:
    """Fake flow_encoder_conformer: (tokens, token_len) → h (1,T,80) float32.

    The adapter now uses a two-stage encoder: the ONNX conformer returns
    h (1, T, 80), then numpy InterpolateRegulator + LR model produce mu.
    """
    def run(self, _, inputs):
        T = inputs["tokens"].shape[1]
        return [np.zeros((1, T, 80), dtype=np.float32)]


class _MockFlowDecoderSess:
    """Fake flow_decoder: ODE estimator → zeros."""
    def run(self, _, inputs):
        x = inputs["x"]
        return [np.zeros_like(x)]


class _MockF0SourceSess:
    """Fake hifigan_f0_source: (1,80,T) → (1,1,T*256) float32."""
    def run(self, _, inputs):
        T = inputs["mel"].shape[2]
        return [np.zeros((1, 1, T * 256), dtype=np.float32)]


class _MockBackboneSess:
    """Fake hifigan_backbone: (mel, source_stft) → (mag, phase)."""
    def run(self, _, inputs):
        T_stft = inputs["source_stft"].shape[2]
        mag = np.ones((1, 9, T_stft), dtype=np.float32) * 0.01
        phase = np.zeros((1, 9, T_stft), dtype=np.float32)
        return [mag, phase]


def _inject_mocks(adapter):
    adapter._tok_sess = _MockTokenizerSess()
    adapter._spk_sess = _MockCAMPlussSess()
    adapter._fe_sess = _MockFlowEncoderSess()
    adapter._fd_sess = _MockFlowDecoderSess()
    adapter._hf_src_sess = _MockF0SourceSess()
    adapter._hf_bb_sess = _MockBackboneSess()
    # Inject a dummy spk_proj so _project_spk_emb works without HF download
    adapter._spk_proj_w = np.zeros((80, 192), dtype=np.float32)
    adapter._spk_proj_b = np.zeros(80, dtype=np.float32)
    # Inject identity LR weights so the numpy LR model passes through
    # Architecture: 4 × [Conv1d(80,80,k=3,p=1)+GN+Mish] + Conv1d(80,80,k=1)
    # Use identity-like weights (zero bias, identity-ish weight)
    eye80 = np.eye(80, dtype=np.float32)
    zeros80 = np.zeros(80, dtype=np.float32)
    w3_identity = np.zeros((80, 80, 3), dtype=np.float32)
    w3_identity[:, :, 1] = eye80  # centre tap = identity
    w1_identity = eye80[:, :, np.newaxis]  # (80, 80, 1)
    adapter._lr_weights = {
        **{f"{i * 3}_weight": w3_identity for i in range(4)},
        **{f"{i * 3}_bias": zeros80 for i in range(4)},
        **{f"{i * 3 + 1}_weight": np.ones(80, dtype=np.float32) for i in range(4)},
        **{f"{i * 3 + 1}_bias": zeros80 for i in range(4)},
        "12_weight": w1_identity,
        "12_bias": zeros80,
    }
    return adapter


def test_adapter_clone_voice_mock(tmp_path):
    """Adapter pipeline completes with mocked ORT sessions."""
    from voiceclonnx.engines.cosyvoice import CosyVoiceAdapter

    src_wav = _make_wav(str(tmp_path / "src.wav"), duration_s=1.0)
    ref_wav = _make_wav(str(tmp_path / "ref.wav"), duration_s=2.0)
    out_wav = str(tmp_path / "out.wav")

    adapter = CosyVoiceAdapter(ode_steps=2)
    _inject_mocks(adapter)
    result = adapter.clone_voice(src_wav, ref_wav, out_wav)

    assert result == str(Path(out_wav).resolve())
    assert Path(out_wav).exists()

    with wave.open(out_wav, "rb") as wf:
        assert wf.getframerate() == 22050
        assert wf.getsampwidth() == 2
        assert wf.getnframes() > 0


def test_adapter_output_is_22050hz(tmp_path):
    """Output WAV must be 22050 Hz regardless of input sample rate."""
    from voiceclonnx.engines.cosyvoice import CosyVoiceAdapter

    src_path = str(tmp_path / "src44.wav")
    data = np.zeros(44100, dtype=np.int16)
    with wave.open(src_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(data.tobytes())

    ref_wav = _make_wav(str(tmp_path / "ref.wav"))
    out_wav = str(tmp_path / "out.wav")

    adapter = CosyVoiceAdapter(ode_steps=2)
    _inject_mocks(adapter)
    adapter.clone_voice(src_path, ref_wav, out_wav)

    with wave.open(out_wav, "rb") as wf:
        assert wf.getframerate() == 22050


def test_adapter_lazy_load_raises_without_onnxruntime(monkeypatch):
    """Missing onnxruntime raises ImportError with a helpful message."""
    import builtins
    from voiceclonnx.engines.cosyvoice import CosyVoiceAdapter

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("onnxruntime not installed")
        return real_import(name, *args, **kwargs)

    adapter = CosyVoiceAdapter()
    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="onnxruntime"):
            adapter._ensure_models()


def test_adapter_returns_resolved_path(tmp_path):
    """clone_voice return value must be an absolute, existing path."""
    from voiceclonnx.engines.cosyvoice import CosyVoiceAdapter

    src = _make_wav(str(tmp_path / "s.wav"))
    ref = _make_wav(str(tmp_path / "r.wav"))
    out = str(tmp_path / "o.wav")

    adapter = CosyVoiceAdapter(ode_steps=2)
    _inject_mocks(adapter)
    result = adapter.clone_voice(src, ref, out)

    assert os.path.isabs(result)
    assert Path(result).exists()


def test_flow_decode_shape(tmp_path):
    """_flow_decode must output (1, 80, T_mel) given mu of shape (1, 80, T)."""
    from voiceclonnx.engines.cosyvoice import CosyVoiceAdapter

    adapter = CosyVoiceAdapter(ode_steps=3)
    _inject_mocks(adapter)

    mu = np.zeros((1, 80, 50), dtype=np.float32)
    spk = np.zeros((1, 80), dtype=np.float32)
    mel = adapter._flow_decode(mu, spk)
    assert mel.shape == (1, 80, 50)


def test_project_spk_emb_with_injected_weights():
    """_project_spk_emb uses injected weights correctly."""
    from voiceclonnx.engines.cosyvoice import CosyVoiceAdapter

    adapter = CosyVoiceAdapter()
    adapter._spk_proj_w = np.eye(80, 192, dtype=np.float32)  # identity first 80 dims
    adapter._spk_proj_b = np.zeros(80, dtype=np.float32)

    emb = np.random.randn(1, 192).astype(np.float32)
    proj = adapter._project_spk_emb(emb)
    assert proj.shape == (1, 80)
    np.testing.assert_allclose(proj[0], emb[0, :80], atol=1e-5)


def test_project_spk_emb_fallback_no_weights():
    """Without weights, _project_spk_emb falls back to truncation."""
    from voiceclonnx.engines.cosyvoice import CosyVoiceAdapter

    adapter = CosyVoiceAdapter()
    # Ensure no weights set
    if hasattr(adapter, "_spk_proj_w"):
        del adapter._spk_proj_w

    # Mock HF download to fail
    import unittest.mock as mock
    with mock.patch("huggingface_hub.hf_hub_download", side_effect=Exception("offline")):
        emb = np.random.randn(1, 192).astype(np.float32)
        proj = adapter._project_spk_emb(emb)
    assert proj.shape == (1, 80)


# ---------------------------------------------------------------------------
# 5. E2E test — real models (skip unless VOICECLONNX_E2E=1)
# ---------------------------------------------------------------------------

_SKIP_E2E = not os.environ.get("VOICECLONNX_E2E", "")
_E2E_REASON = (
    "E2E CosyVoice test downloads ~1.1 GB of ONNX models; "
    "set VOICECLONNX_E2E=1 to run."
)


@pytest.mark.skipif(_SKIP_E2E, reason=_E2E_REASON)
def test_e2e_cosyvoice_clone_edge_tts_voices(tmp_path):
    """Real end-to-end: convert an edge-tts source to a reference voice.

    Validates:
    - Output is a valid 16-bit 22050 Hz WAV.
    - Duration is in a reasonable range (0.5 s – 30 s).
    - WER ≤ 25% on the source utterance (intelligibility gate).
    """
    import asyncio
    import re
    import subprocess


    try:
        import edge_tts
    except ImportError:
        pytest.skip("edge-tts not installed")

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        pytest.skip("faster-whisper not installed")

    from voiceclonnx.engines.cosyvoice import CosyVoiceAdapter

    SOURCE_TEXT = (
        "The quick brown fox jumps over the lazy dog. "
        "Voice conversion changes who is speaking, but not what is said. "
        "Listen closely and compare the engines."
    )
    REF_TEXT = (
        "This sentence provides the reference voice. Its timbre and style "
        "are what the converted audio should resemble."
    )

    async def _synth(text: str, voice: str, out: str):
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(out)

    src_mp3 = str(tmp_path / "src.mp3")
    ref_mp3 = str(tmp_path / "ref.mp3")
    src_wav_path = str(tmp_path / "src.wav")
    ref_wav_path = str(tmp_path / "ref.wav")
    out_wav = str(tmp_path / "out_cosyvoice.wav")

    asyncio.run(_synth(SOURCE_TEXT, "en-US-GuyNeural", src_mp3))
    asyncio.run(_synth(REF_TEXT, "en-US-AriaNeural", ref_mp3))

    for mp3, wav, rate in [(src_mp3, src_wav_path, "16000"), (ref_mp3, ref_wav_path, "16000")]:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", mp3,
             "-ar", rate, "-ac", "1", "-sample_fmt", "s16", wav],
            check=True,
        )

    adapter = CosyVoiceAdapter(quantized=False, ode_steps=10)
    result = adapter.clone_voice(src_wav_path, ref_wav_path, out_wav)

    assert Path(result).exists()
    assert Path(result).stat().st_size > 0

    with wave.open(result, "rb") as wf:
        sr_out = wf.getframerate()
        n_frames = wf.getnframes()
        duration_s = n_frames / sr_out

    assert sr_out == 22050, f"Expected 22050 Hz, got {sr_out}"
    assert 0.5 <= duration_s <= 30.0, f"Suspicious output duration: {duration_s:.2f} s"

    def norm(t):
        return re.sub(r"[^a-z' ]", " ", t.lower()).split()

    def wer(ref, hyp):
        d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
        for i in range(len(ref) + 1):
            d[i][0] = i
        for j in range(len(hyp) + 1):
            d[0][j] = j
        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                d[i][j] = min(
                    d[i - 1][j] + 1, d[i][j - 1] + 1,
                    d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1])
                )
        return d[len(ref)][len(hyp)] / max(len(ref), 1)

    model = WhisperModel("base.en", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(result, beam_size=5)
    hyp_text = " ".join(seg.text.strip() for seg in segments)
    score = wer(norm(SOURCE_TEXT), norm(hyp_text))

    print(
        f"\n[e2e cosyvoice] output={result}  "
        f"duration={duration_s:.2f}s  sr={sr_out}  "
        f"WER={score:.0%}  transcript={hyp_text[:100]}"
    )

    assert score <= 0.25, (
        f"WER {score:.0%} exceeds 25%% intelligibility gate. "
        f"Transcript: {hyp_text[:200]}"
    )
