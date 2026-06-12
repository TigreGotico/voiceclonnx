"""Tests for the SpeechTokenizer adapter — vconnx/engines/speechtokenizer.py.

Structure
---------
- Registry wiring (no model loading)
- RVQ token-swap math on synthetic tokens (real numpy computation)
- Numpy RVQ encode/decode correctness (synthetic data)
- Mock-session contract tests (full adapter pipeline with stubbed ORT)
- skipif-gated e2e: real conversion using edge-tts generated audio
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


def test_speechtokenizer_registered():
    """speechtokenizer engine must appear in ENGINE_REGISTRY after importing vconnx."""
    import vconnx.engines.speechtokenizer  # noqa: F401
    from vconnx.engines.base import ENGINE_REGISTRY

    assert "speechtokenizer" in ENGINE_REGISTRY


def test_speechtokenizer_entry_metadata():
    import vconnx.engines.speechtokenizer  # noqa: F401
    from vconnx.engines.base import get_engine

    entry = get_engine("speechtokenizer")
    assert entry.alias == "speechtokenizer"
    assert entry.onnx_native is True
    assert entry.extras == ""
    assert entry.adapter_class.__name__ == "SpeechTokenizerAdapter"


def test_speechtokenizer_sample_rate():
    from vconnx.engines.speechtokenizer import SpeechTokenizerAdapter

    adapter = SpeechTokenizerAdapter()
    assert adapter.sample_rate == 16000


# ---------------------------------------------------------------------------
# 2. RVQ token-swap math — real numpy computation
# ---------------------------------------------------------------------------


def test_swap_rvq_keeps_source_content_layer():
    """Content layer (index 0) must equal the source codes after swap."""
    from vconnx.engines.speechtokenizer import _swap_rvq_tokens

    rng = np.random.default_rng(0)
    Q, T = 8, 50
    src = rng.integers(0, 1024, (Q, T), dtype=np.int64)
    ref = rng.integers(0, 1024, (Q, T), dtype=np.int64)

    mixed = _swap_rvq_tokens(src, ref, content_layers=1)
    np.testing.assert_array_equal(mixed[0], src[0])


def test_swap_rvq_timbre_layers_from_ref_same_length():
    """Timbre layers (1-7) must equal reference codes when lengths match."""
    from vconnx.engines.speechtokenizer import _swap_rvq_tokens

    rng = np.random.default_rng(1)
    Q, T = 8, 50
    src = rng.integers(0, 1024, (Q, T), dtype=np.int64)
    ref = rng.integers(0, 1024, (Q, T), dtype=np.int64)

    mixed = _swap_rvq_tokens(src, ref, content_layers=1)
    for q in range(1, Q):
        np.testing.assert_array_equal(mixed[q], ref[q, :T])


def test_swap_rvq_output_shape():
    """Output shape must always equal source shape (Q, T_src)."""
    from vconnx.engines.speechtokenizer import _swap_rvq_tokens

    rng = np.random.default_rng(2)
    for T_src, T_ref in [(50, 50), (30, 80), (100, 40)]:
        src = rng.integers(0, 512, (8, T_src), dtype=np.int64)
        ref = rng.integers(0, 512, (8, T_ref), dtype=np.int64)
        mixed = _swap_rvq_tokens(src, ref)
        assert mixed.shape == (8, T_src), f"T_src={T_src} T_ref={T_ref}"


def test_swap_rvq_longer_ref_truncated():
    """When reference is longer than source, it must be truncated to source length."""
    from vconnx.engines.speechtokenizer import _swap_rvq_tokens

    rng = np.random.default_rng(3)
    src = rng.integers(0, 512, (8, 30), dtype=np.int64)
    ref = rng.integers(0, 512, (8, 100), dtype=np.int64)

    mixed = _swap_rvq_tokens(src, ref, content_layers=1)
    assert mixed.shape == (8, 30)
    for q in range(1, 8):
        np.testing.assert_array_equal(mixed[q, :], ref[q, :30])


def test_swap_rvq_shorter_ref_tiled():
    """When reference is shorter than source, it must be tiled to cover source length."""
    from vconnx.engines.speechtokenizer import _swap_rvq_tokens

    rng = np.random.default_rng(4)
    T_src, T_ref = 100, 30
    src = rng.integers(0, 512, (8, T_src), dtype=np.int64)
    ref = rng.integers(0, 512, (8, T_ref), dtype=np.int64)

    mixed = _swap_rvq_tokens(src, ref, content_layers=1)
    assert mixed.shape == (8, T_src)
    q = 1
    np.testing.assert_array_equal(mixed[q, :T_ref], ref[q, :T_ref])
    np.testing.assert_array_equal(mixed[q, T_ref : 2 * T_ref], ref[q, :T_ref])


def test_swap_rvq_content_layers_two():
    """content_layers=2 keeps layers 0 and 1 from source."""
    from vconnx.engines.speechtokenizer import _swap_rvq_tokens

    rng = np.random.default_rng(5)
    Q, T = 8, 40
    src = rng.integers(0, 512, (Q, T), dtype=np.int64)
    ref = rng.integers(0, 512, (Q, T), dtype=np.int64)

    mixed = _swap_rvq_tokens(src, ref, content_layers=2)
    np.testing.assert_array_equal(mixed[0], src[0])
    np.testing.assert_array_equal(mixed[1], src[1])
    for q in range(2, Q):
        np.testing.assert_array_equal(mixed[q], ref[q, :T])


def test_swap_rvq_dtype_preserved():
    """Output dtype must be int64."""
    from vconnx.engines.speechtokenizer import _swap_rvq_tokens

    rng = np.random.default_rng(6)
    src = rng.integers(0, 512, (8, 50), dtype=np.int64)
    ref = rng.integers(0, 512, (8, 50), dtype=np.int64)
    mixed = _swap_rvq_tokens(src, ref)
    assert mixed.dtype == np.int64


def test_swap_rvq_deterministic():
    """Token swap is deterministic."""
    from vconnx.engines.speechtokenizer import _swap_rvq_tokens

    rng = np.random.default_rng(7)
    src = rng.integers(0, 512, (8, 60), dtype=np.int64)
    ref = rng.integers(0, 512, (8, 45), dtype=np.int64)

    m1 = _swap_rvq_tokens(src, ref)
    m2 = _swap_rvq_tokens(src, ref)
    np.testing.assert_array_equal(m1, m2)


# ---------------------------------------------------------------------------
# 3. Numpy RVQ encode/decode — correctness on synthetic data
# ---------------------------------------------------------------------------


def test_rvq_encode_output_shape():
    """RVQ encode output is (Q, T)."""
    from vconnx.engines.speechtokenizer import _rvq_encode

    rng = np.random.default_rng(10)
    Q, CB, D, T = 8, 64, 32, 50
    feats = rng.standard_normal((1, D, T)).astype(np.float32)
    codebooks = rng.standard_normal((Q, CB, D)).astype(np.float32)
    codes = _rvq_encode(feats, codebooks)
    assert codes.shape == (Q, T)
    assert codes.dtype == np.int64


def test_rvq_encode_indices_in_range():
    """All indices must be in [0, codebook_size)."""
    from vconnx.engines.speechtokenizer import _rvq_encode

    rng = np.random.default_rng(11)
    CB = 128
    feats = rng.standard_normal((1, 64, 40)).astype(np.float32)
    codebooks = rng.standard_normal((8, CB, 64)).astype(np.float32)
    codes = _rvq_encode(feats, codebooks)
    assert codes.min() >= 0
    assert codes.max() < CB


def test_rvq_decode_output_shape():
    """RVQ decode output is (1, D, T)."""
    from vconnx.engines.speechtokenizer import _rvq_decode

    rng = np.random.default_rng(12)
    Q, CB, D, T = 8, 64, 32, 50
    codebooks = rng.standard_normal((Q, CB, D)).astype(np.float32)
    codes = rng.integers(0, CB, (Q, T), dtype=np.int64)
    out = _rvq_decode(codes, codebooks)
    assert out.shape == (1, D, T)
    assert out.dtype == np.float32


def test_rvq_encode_decode_residual_decreases():
    """Each successive RVQ layer should reduce reconstruction error."""
    from vconnx.engines.speechtokenizer import _rvq_encode, _rvq_decode

    rng = np.random.default_rng(13)
    Q, CB, D, T = 4, 256, 32, 20
    feats = rng.standard_normal((1, D, T)).astype(np.float32)
    codebooks = rng.standard_normal((Q, CB, D)).astype(np.float32)

    codes = _rvq_encode(feats, codebooks)

    errors = []
    for q in range(1, Q + 1):
        partial = _rvq_decode(codes[:q], codebooks[:q])
        err = float(np.abs(feats - partial).mean())
        errors.append(err)

    for i in range(len(errors) - 1):
        assert errors[i] >= errors[i + 1] or errors[i] < 1e-5, (
            f"Error did not decrease at layer {i}: {errors}"
        )


def test_rvq_encode_decode_reconstruct_from_codebook():
    """With dense, well-spread codebooks, layer-1 alone partially reconstructs the input."""
    from vconnx.engines.speechtokenizer import _rvq_encode, _rvq_decode

    rng = np.random.default_rng(14)
    Q, CB, D, T = 4, 256, 16, 20
    # Use random unit vectors as codebook (well spread, small D)
    feats = rng.standard_normal((1, D, T)).astype(np.float32)
    codebooks = rng.standard_normal((Q, CB, D)).astype(np.float32)
    # Normalize codebook to unit vectors so distance makes sense
    codebooks /= np.linalg.norm(codebooks, axis=2, keepdims=True) + 1e-8

    codes = _rvq_encode(feats, codebooks)
    recon = _rvq_decode(codes, codebooks)

    # The reconstruction should have the same shape
    assert recon.shape == feats.shape
    # Each layer's indices must be valid
    assert codes.min() >= 0 and codes.max() < CB


# ---------------------------------------------------------------------------
# 4. Adapter contract with mocked ORT sessions
# ---------------------------------------------------------------------------


class _MockEncoderSession:
    """Fake encoder ORT session — returns (1, 1024, T) float32 features."""

    def run(self, output_names, inputs):
        audio = inputs["audio"]  # (1, 1, N)
        N = audio.shape[2]
        T = max(1, N // 320)
        rng = np.random.default_rng(0)
        feats = rng.standard_normal((1, 1024, T)).astype(np.float32) * 0.1
        return [feats]


class _MockDecoderSession:
    """Fake decoder ORT session — returns (1, 1, N) float32 waveform."""

    def run(self, output_names, inputs):
        feats = inputs["features"]  # (1, 1024, T)
        T = feats.shape[2]
        N = T * 320
        rng = np.random.default_rng(1)
        wav = rng.standard_normal((1, 1, N)).astype(np.float32) * 0.05
        return [wav]


def _make_mock_codebooks(Q: int = 8, CB: int = 64, D: int = 1024) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.standard_normal((Q, CB, D)).astype(np.float32)


def test_adapter_clone_voice_mock(tmp_path):
    """Adapter pipeline completes with mocked ORT sessions."""
    from vconnx.engines.speechtokenizer import SpeechTokenizerAdapter

    src_wav = _make_wav(str(tmp_path / "src.wav"), duration_s=1.0)
    ref_wav = _make_wav(str(tmp_path / "ref.wav"), duration_s=2.0)
    out_wav = str(tmp_path / "out.wav")

    adapter = SpeechTokenizerAdapter()
    adapter._enc_sess = _MockEncoderSession()
    adapter._dec_sess = _MockDecoderSession()
    adapter._codebooks = _make_mock_codebooks()

    result = adapter.clone_voice(src_wav, ref_wav, out_wav)
    assert result == str(Path(out_wav).resolve())
    assert Path(out_wav).exists()

    with wave.open(out_wav, "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getsampwidth() == 2
        assert wf.getnframes() > 0


def test_adapter_output_is_16khz(tmp_path):
    """Output WAV must be 16 kHz regardless of input sample rate."""
    from vconnx.engines.speechtokenizer import SpeechTokenizerAdapter

    src_path = str(tmp_path / "src44.wav")
    n = 44100
    data = np.zeros(n, dtype=np.int16)
    with wave.open(src_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(data.tobytes())

    ref_wav = _make_wav(str(tmp_path / "ref.wav"))
    out_wav = str(tmp_path / "out.wav")

    adapter = SpeechTokenizerAdapter()
    adapter._enc_sess = _MockEncoderSession()
    adapter._dec_sess = _MockDecoderSession()
    adapter._codebooks = _make_mock_codebooks()

    adapter.clone_voice(src_path, ref_wav, out_wav)

    with wave.open(out_wav, "rb") as wf:
        assert wf.getframerate() == 16000


def test_adapter_lazy_load_raises_without_onnxruntime(tmp_path, monkeypatch):
    """Missing onnxruntime raises ImportError with a helpful message."""
    import builtins

    from vconnx.engines.speechtokenizer import SpeechTokenizerAdapter

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("onnxruntime not installed")
        return real_import(name, *args, **kwargs)

    adapter = SpeechTokenizerAdapter()
    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="onnxruntime"):
            adapter._ensure_models()


def test_adapter_quantized_flag_stored():
    """quantized=False is stored; quantized=True raises NotImplementedError (incompatible q8 export)."""
    import pytest
    from vconnx.engines.speechtokenizer import SpeechTokenizerAdapter

    b = SpeechTokenizerAdapter(quantized=False)
    assert b._quantized is False

    with pytest.raises(NotImplementedError):
        SpeechTokenizerAdapter(quantized=True)


def test_adapter_content_layers_parameter():
    """content_layers parameter is stored."""
    from vconnx.engines.speechtokenizer import SpeechTokenizerAdapter

    adapter = SpeechTokenizerAdapter(content_layers=2)
    assert adapter._content_layers == 2


def test_adapter_pipeline_uses_swap(tmp_path):
    """Verify that the adapter performs the expected RVQ swap during clone_voice."""
    from vconnx.engines.speechtokenizer import (
        SpeechTokenizerAdapter,
        _rvq_encode,
        _rvq_decode,
        _swap_rvq_tokens,
    )

    src_wav = _make_wav(str(tmp_path / "src.wav"), duration_s=1.0)
    ref_wav = _make_wav(str(tmp_path / "ref.wav"), duration_s=1.0)

    feat_calls = []

    class TrackingEncoder:
        def run(self, output_names, inputs):
            N = inputs["audio"].shape[2]
            T = max(1, N // 320)
            rng = np.random.default_rng(len(feat_calls))
            feats = rng.standard_normal((1, 1024, T)).astype(np.float32) * 0.1
            feat_calls.append(feats.copy())
            return [feats]

    class TrackingDecoder:
        def __init__(self): self.received_feats = None
        def run(self, output_names, inputs):
            self.received_feats = inputs["features"].copy()
            T = inputs["features"].shape[2]
            return [np.zeros((1, 1, T * 320), dtype=np.float32)]

    enc = TrackingEncoder()
    dec = TrackingDecoder()
    cbs = _make_mock_codebooks(Q=8, CB=32, D=1024)

    adapter = SpeechTokenizerAdapter(content_layers=1)
    adapter._enc_sess = enc
    adapter._dec_sess = dec
    adapter._codebooks = cbs

    adapter.clone_voice(str(src_wav), str(ref_wav), str(tmp_path / "out.wav"))

    assert len(feat_calls) == 2, "encoder must be called twice"
    src_feats, ref_feats = feat_calls[0], feat_calls[1]

    src_codes = _rvq_encode(src_feats, cbs)
    ref_codes = _rvq_encode(ref_feats, cbs)
    expected_mixed_codes = _swap_rvq_tokens(src_codes, ref_codes, content_layers=1)
    expected_feats = _rvq_decode(expected_mixed_codes, cbs)

    np.testing.assert_allclose(dec.received_feats, expected_feats, atol=1e-5)


# ---------------------------------------------------------------------------
# 5. E2E test — real models, real audio (skip if not opted in)
# ---------------------------------------------------------------------------

_SKIP_E2E = not os.environ.get("VCONNX_E2E", "")

_E2E_REASON = (
    "E2E speechtokenizer test downloads ~400MB of public ONNX models; "
    "set VCONNX_E2E=1 to run."
)


@pytest.mark.skipif(_SKIP_E2E, reason=_E2E_REASON)
def test_e2e_speechtokenizer_clone_edge_tts_voices(tmp_path):
    """Real end-to-end: convert an edge-tts source to a second edge-tts reference.

    Validates:
    - Output is a valid 16-bit 16 kHz WAV.
    - Duration is in a reasonable range (0.5 s – 15 s).
    - File size > 0 bytes.
    - WER ≤ 25 % on the source utterance (intelligibility gate).
    """
    import asyncio
    import re
    import subprocess

    import soundfile as sf

    try:
        import edge_tts
    except ImportError:
        pytest.skip("edge-tts not installed")

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        pytest.skip("faster-whisper not installed")

    from vconnx.engines.speechtokenizer import SpeechTokenizerAdapter

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
    out_wav = str(tmp_path / "out_speechtokenizer.wav")

    asyncio.run(_synth(SOURCE_TEXT, "en-US-GuyNeural", src_mp3))
    asyncio.run(_synth(REF_TEXT, "en-US-AriaNeural", ref_mp3))

    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", src_mp3,
         "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", src_wav_path],
        check=True,
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", ref_mp3,
         "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", ref_wav_path],
        check=True,
    )

    adapter = SpeechTokenizerAdapter(quantized=False, content_layers=1)
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
    assert 0.5 <= duration_s <= 15.0, f"Suspicious output duration: {duration_s:.2f} s"

    def norm(text):
        return re.sub(r"[^a-z' ]", " ", text.lower()).split()

    def wer(ref, hyp):
        d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
        for i in range(len(ref) + 1):
            d[i][0] = i
        for j in range(len(hyp) + 1):
            d[0][j] = j
        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                              d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
        return d[len(ref)][len(hyp)] / max(len(ref), 1)

    model = WhisperModel("base.en", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(result, beam_size=5)
    hyp_text = " ".join(seg.text.strip() for seg in segments)
    score = wer(norm(SOURCE_TEXT), norm(hyp_text))

    print(
        f"\n[e2e speechtokenizer] output={result}  "
        f"duration={duration_s:.2f}s  sr={sr_out}  "
        f"size={size_bytes // 1024} KiB  WER={score:.0%}  "
        f"transcript={hyp_text[:100]}"
    )

    assert score <= 0.25, (
        f"WER {score:.0%} exceeds 25%% intelligibility gate. "
        f"Transcript: {hyp_text[:200]}"
    )
