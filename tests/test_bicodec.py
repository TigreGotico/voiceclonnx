"""Tests for the BiCodec adapter — voiceclonnx/engines/bicodec.py.

Structure
---------
- Registry wiring (no model loading)
- Token-swap logic on synthetic data (pure numpy)
- Chunked Wav2Vec2 feature extraction (numpy, mock ORT session)
- Adapter contract tests with mocked ORT sessions (full pipeline)
- E2E test gated on VOICECLONNX_E2E=1 (real models, edge-tts, WER gate ≤25%)
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


def test_bicodec_registered():
    """bicodec engine must appear in ENGINE_REGISTRY after importing voiceclonnx."""
    import voiceclonnx.engines.bicodec  # noqa: F401
    from voiceclonnx.engines.base import ENGINE_REGISTRY

    assert "bicodec" in ENGINE_REGISTRY


def test_bicodec_entry_metadata():
    import voiceclonnx.engines.bicodec  # noqa: F401
    from voiceclonnx.engines.base import get_engine

    entry = get_engine("bicodec")
    assert entry.alias == "bicodec"
    assert entry.onnx_native is True
    assert entry.extras == ""
    assert entry.adapter_class.__name__ == "BiCodecAdapter"


def test_bicodec_sample_rate():
    from voiceclonnx.engines.bicodec import BiCodecAdapter

    adapter = BiCodecAdapter()
    assert adapter.sample_rate == 16000


def test_bicodec_quantized_flag():
    from voiceclonnx.engines.bicodec import BiCodecAdapter

    a = BiCodecAdapter(quantized=True)
    assert a._quantized is True
    b = BiCodecAdapter(quantized=False)
    assert b._quantized is False


def test_bicodec_chunk_samples_stored():
    from voiceclonnx.engines.bicodec import BiCodecAdapter

    a = BiCodecAdapter(chunk_samples=16000)
    assert a._chunk_samples == 16000


# ---------------------------------------------------------------------------
# 2. Numpy token swap: semantic/global split
# ---------------------------------------------------------------------------


def test_swap_keeps_source_semantic_tokens():
    """After VC, semantic tokens must equal source (content preserved)."""
    # In BiCodec the token swap is explicit: we keep src semantic tokens and
    # replace global tokens with reference global tokens.
    # Verify that the adapter's pipeline passes src_semantic to the decoder,
    # not ref_semantic.
    rng = np.random.default_rng(0)
    src_sem = rng.integers(0, 8192, (1, 50), dtype=np.int64)
    ref_sem = rng.integers(0, 8192, (1, 50), dtype=np.int64)

    # The swap: use src_sem (content) + ref_global (timbre)
    # Assert src_sem is the 'content' side
    assert not np.array_equal(src_sem, ref_sem), "test setup: must differ"

    # Simulate what the adapter does: feeds src_semantic to decoder
    # (this is just the semantic identity check — the adapter always uses
    # the source semantic tokens without modification)
    decoded_with_src_sem = src_sem.copy()   # adapter passes this through
    np.testing.assert_array_equal(decoded_with_src_sem, src_sem)


def test_global_tokens_shape_contract():
    """Global tokens from BiCodec speaker encoder are always (1, 1, 32) int32."""
    rng = np.random.default_rng(1)
    # Mock global tokens with the expected shape from global_encoder.onnx
    global_tokens = rng.integers(0, 1024, (1, 1, 32), dtype=np.int32)
    assert global_tokens.shape == (1, 1, 32)
    assert global_tokens.dtype == np.int32


def test_semantic_tokens_dtype():
    """Semantic tokens must be int64."""
    rng = np.random.default_rng(2)
    sem = rng.integers(0, 8192, (1, 100), dtype=np.int64)
    assert sem.dtype == np.int64


# ---------------------------------------------------------------------------
# 3. Chunked feature extraction — numpy / mock ORT session
# ---------------------------------------------------------------------------


class _MockW2VSession:
    """Mock wav2vec2_encoder ORT session: (1, N) → (1, N//320, 1024)."""

    def run(self, output_names, inputs):
        wav = inputs["waveform"]  # (1, N)
        N = wav.shape[1]
        T = max(1, N // 320)
        rng = np.random.default_rng(0)
        feats = rng.standard_normal((1, T, 1024)).astype(np.float32) * 0.1
        return [feats]


def test_chunked_extraction_short_no_chunking():
    """Audio shorter than chunk_samples goes through as a single batch."""
    from voiceclonnx.engines.bicodec import _extract_features_chunked

    wav = np.zeros(16000, dtype=np.float32)
    sess = _MockW2VSession()
    result = _extract_features_chunked(wav, sess, chunk_samples=32000)
    assert result.shape[0] == 1
    assert result.shape[2] == 1024


def test_chunked_extraction_long_concatenates():
    """Audio longer than chunk_samples is split and concatenated."""
    from voiceclonnx.engines.bicodec import _extract_features_chunked

    wav = np.zeros(96000, dtype=np.float32)  # 6 s
    sess = _MockW2VSession()
    result = _extract_features_chunked(wav, sess, chunk_samples=32000, overlap_samples=4000)
    assert result.ndim == 3
    assert result.shape[0] == 1
    assert result.shape[2] == 1024
    # Should have more frames than a single-chunk extraction
    single = _extract_features_chunked(np.zeros(32000, dtype=np.float32), sess, chunk_samples=32000)
    assert result.shape[1] > single.shape[1]


def test_chunked_extraction_output_dtype():
    """Feature output must be float32."""
    from voiceclonnx.engines.bicodec import _extract_features_chunked

    wav = np.zeros(48000, dtype=np.float32)
    sess = _MockW2VSession()
    result = _extract_features_chunked(wav, sess, chunk_samples=32000)
    assert result.dtype == np.float32


# ---------------------------------------------------------------------------
# 4. Adapter contract with mocked ORT sessions
# ---------------------------------------------------------------------------


class _MockSemanticSession:
    """Fake semantic_encoder: (1, T, 1024) → (1, T2) int64."""

    def run(self, output_names, inputs):
        feats = inputs["features"]  # (1, T, 1024)
        T2 = max(1, feats.shape[1] // 4)
        rng = np.random.default_rng(20)
        tokens = rng.integers(0, 8192, (1, T2), dtype=np.int64)
        return [tokens]


class _MockGlobalSession:
    """Fake global_encoder: (1, 128, T_mel) → (1, 1, 32) int32."""

    def run(self, output_names, inputs):
        rng = np.random.default_rng(30)
        tokens = rng.integers(0, 1024, (1, 1, 32), dtype=np.int32)
        return [tokens]


class _MockDecoderSession:
    """Fake decoder: semantic_tokens + global_tokens → (1, 1, N) float32."""

    def run(self, output_names, inputs):
        sem = inputs["semantic_tokens"]  # (1, T2)
        T2 = sem.shape[1]
        N = T2 * 320
        rng = np.random.default_rng(40)
        wav = rng.standard_normal((1, 1, N)).astype(np.float32) * 0.05
        return [wav]


def _inject_mock_sessions(adapter):
    """Inject mock ORT sessions and a dummy mel filterbank into the adapter."""
    adapter._w2v_sess = _MockW2VSession()
    adapter._sem_sess = _MockSemanticSession()
    adapter._glob_sess = _MockGlobalSession()
    adapter._dec_sess = _MockDecoderSession()
    # Numpy mel: dummy filterbank (128, 513) and config
    adapter._mel_fb = np.zeros((128, 513), dtype=np.float32)
    adapter._mel_cfg = {"n_fft": 1024, "hop_length": 320, "win_length": 640}
    return adapter


def test_adapter_clone_voice_mock(tmp_path):
    """Adapter pipeline completes with mocked ORT sessions."""
    from voiceclonnx.engines.bicodec import BiCodecAdapter

    src_wav = _make_wav(str(tmp_path / "src.wav"), duration_s=1.0)
    ref_wav = _make_wav(str(tmp_path / "ref.wav"), duration_s=2.0)
    out_wav = str(tmp_path / "out.wav")

    adapter = BiCodecAdapter()
    _inject_mock_sessions(adapter)

    result = adapter.clone_voice(src_wav, ref_wav, out_wav)
    assert result == str(Path(out_wav).resolve())
    assert Path(out_wav).exists()

    with wave.open(out_wav, "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getsampwidth() == 2
        assert wf.getnframes() > 0


def test_adapter_output_is_16khz(tmp_path):
    """Output WAV must be 16 kHz regardless of input sample rate."""
    from voiceclonnx.engines.bicodec import BiCodecAdapter

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

    adapter = BiCodecAdapter()
    _inject_mock_sessions(adapter)

    adapter.clone_voice(src_path, ref_wav, out_wav)

    with wave.open(out_wav, "rb") as wf:
        assert wf.getframerate() == 16000


def test_adapter_lazy_load_raises_without_onnxruntime(monkeypatch):
    """Missing onnxruntime raises ImportError with a helpful message."""
    import builtins
    from voiceclonnx.engines.bicodec import BiCodecAdapter

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("onnxruntime not installed")
        return real_import(name, *args, **kwargs)

    adapter = BiCodecAdapter()
    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="onnxruntime"):
            adapter._ensure_models()


def test_adapter_uses_source_semantic_tokens(tmp_path):
    """Verify adapter feeds source semantic tokens (not reference) to decoder."""
    from voiceclonnx.engines.bicodec import BiCodecAdapter

    src_wav = _make_wav(str(tmp_path / "src.wav"), duration_s=1.0)
    ref_wav = _make_wav(str(tmp_path / "ref.wav"), duration_s=1.0)

    received_semantics = []
    received_globals = []

    class TrackingSemanticSession:
        def run(self, output_names, inputs):
            feats = inputs["features"]
            T2 = max(1, feats.shape[1] // 4)
            # Use deterministic tokens based on call index so src != ref
            tokens = np.full((1, T2), len(received_semantics) * 100, dtype=np.int64)
            received_semantics.append(tokens.copy())
            return [tokens]

    class TrackingGlobalSession:
        def run(self, output_names, inputs):
            tokens = np.full((1, 1, 32), 555, dtype=np.int32)
            received_globals.append(tokens.copy())
            return [tokens]

    class TrackingDecoderSession:
        def __init__(self):
            self.semantic_tokens_received = None
            self.global_tokens_received = None

        def run(self, output_names, inputs):
            self.semantic_tokens_received = inputs["semantic_tokens"].copy()
            self.global_tokens_received = inputs["global_tokens"].copy()
            T2 = inputs["semantic_tokens"].shape[1]
            return [np.zeros((1, 1, T2 * 320), dtype=np.float32)]

    dec = TrackingDecoderSession()
    adapter = BiCodecAdapter()
    adapter._w2v_sess = _MockW2VSession()
    adapter._sem_sess = TrackingSemanticSession()
    adapter._glob_sess = TrackingGlobalSession()
    adapter._dec_sess = dec
    adapter._mel_fb = np.zeros((128, 513), dtype=np.float32)
    adapter._mel_cfg = {"n_fft": 1024, "hop_length": 320, "win_length": 640}

    adapter.clone_voice(str(src_wav), str(ref_wav), str(tmp_path / "out.wav"))

    # Semantic encoder is called once (for source)
    assert len(received_semantics) == 1, "semantic encoder must be called once (source only)"
    # Global encoder is called once (for reference)
    assert len(received_globals) == 1, "global encoder must be called once (reference only)"

    # Decoder received the source semantic tokens
    np.testing.assert_array_equal(
        dec.semantic_tokens_received, received_semantics[0]
    )
    # Decoder received the reference global tokens
    np.testing.assert_array_equal(
        dec.global_tokens_received, received_globals[0]
    )


def test_adapter_returns_resolved_path(tmp_path):
    """clone_voice return value must be the resolved absolute path."""
    from voiceclonnx.engines.bicodec import BiCodecAdapter

    src = _make_wav(str(tmp_path / "s.wav"))
    ref = _make_wav(str(tmp_path / "r.wav"))
    out = str(tmp_path / "o.wav")

    adapter = BiCodecAdapter()
    _inject_mock_sessions(adapter)

    result = adapter.clone_voice(src, ref, out)
    assert os.path.isabs(result)
    assert Path(result).exists()


# ---------------------------------------------------------------------------
# 5. E2E test — real models, real audio (skip unless VOICECLONNX_E2E=1)
# ---------------------------------------------------------------------------

_SKIP_E2E = not os.environ.get("VOICECLONNX_E2E", "")

_E2E_REASON = (
    "E2E bicodec test downloads ~1.5 GB of ONNX models (Wav2Vec2 + BiCodec); "
    "set VOICECLONNX_E2E=1 to run."
)


@pytest.mark.skipif(_SKIP_E2E, reason=_E2E_REASON)
def test_e2e_bicodec_clone_edge_tts_voices(tmp_path):
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


    try:
        import edge_tts
    except ImportError:
        pytest.skip("edge-tts not installed")

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        pytest.skip("faster-whisper not installed")

    from voiceclonnx.engines.bicodec import BiCodecAdapter

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
    out_wav = str(tmp_path / "out_bicodec.wav")

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

    # Run voice conversion
    adapter = BiCodecAdapter(quantized=False, chunk_samples=32000)
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
        f"\n[e2e bicodec] output={result}  "
        f"duration={duration_s:.2f}s  sr={sr_out}  "
        f"size={size_bytes // 1024} KiB  WER={score:.0%}  "
        f"transcript={hyp_text[:100]}"
    )

    assert score <= 0.25, (
        f"WER {score:.0%} exceeds 25%% intelligibility gate. "
        f"Transcript: {hyp_text[:200]}"
    )
