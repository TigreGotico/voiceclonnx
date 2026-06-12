"""Tests for the FACodec adapter — vconnx/engines/facodec.py.

Structure
---------
- Registry wiring (no model loading)
- Prosody mel computation correctness (pure numpy)
- Mock-session contract tests (full adapter pipeline with stubbed ORT)
- E2E test: real conversion + WER gate (set VCONNX_E2E=1)
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


def test_facodec_registered():
    import vconnx.engines.facodec  # noqa: F401
    from vconnx.engines.base import ENGINE_REGISTRY

    assert "facodec" in ENGINE_REGISTRY


def test_facodec_entry_metadata():
    import vconnx.engines.facodec  # noqa: F401
    from vconnx.engines.base import get_engine

    entry = get_engine("facodec")
    assert entry.alias == "facodec"
    assert entry.onnx_native is True
    assert entry.extras == ""
    assert entry.adapter_class.__name__ == "FACodecAdapter"


def test_facodec_sample_rate():
    from vconnx.engines.facodec import FACodecAdapter

    assert FACodecAdapter().sample_rate == 16000


def test_facodec_quantized_flag():
    from vconnx.engines.facodec import FACodecAdapter

    assert FACodecAdapter(quantized=True)._quantized is True
    assert FACodecAdapter(quantized=False)._quantized is False


# ---------------------------------------------------------------------------
# 2. Prosody mel — shape + value sanity
# ---------------------------------------------------------------------------


def test_prosody_mel_shape():
    from vconnx.engines.facodec import _compute_prosody_mel

    audio = np.zeros(16000, dtype=np.float32)
    mel = _compute_prosody_mel(audio)
    assert mel.ndim == 3
    assert mel.shape[0] == 1       # batch
    assert mel.shape[1] == 20      # 20 mel bins
    assert mel.shape[2] > 0        # frames


def test_prosody_mel_dtype():
    from vconnx.engines.facodec import _compute_prosody_mel

    audio = np.random.randn(8000).astype(np.float32)
    mel = _compute_prosody_mel(audio)
    assert mel.dtype == np.float32


def test_prosody_mel_frame_count():
    """Number of frames should match T = floor(N/hop) for padded input."""
    from vconnx.engines.facodec import _compute_prosody_mel, _FA_HOP

    sr = 16000
    audio = np.zeros(sr, dtype=np.float32)
    mel = _compute_prosody_mel(audio)
    # FACodecEncoderV2 hop=200 means T≈80 for 1s
    T = mel.shape[2]
    assert 70 <= T <= 100, f"Unexpected frame count: {T}"


def test_prosody_mel_log_compressed():
    """All values should be finite and log-compressed (no positive-infinity)."""
    from vconnx.engines.facodec import _compute_prosody_mel

    audio = np.random.randn(16000).astype(np.float32) * 0.5
    mel = _compute_prosody_mel(audio)
    assert np.all(np.isfinite(mel))


def test_prosody_mel_silence_vs_signal():
    """Silence should produce lower mel values than a sine tone."""
    from vconnx.engines.facodec import _compute_prosody_mel

    silence = np.zeros(16000, dtype=np.float32)
    tone = np.sin(2 * np.pi * 440 * np.arange(16000) / 16000).astype(np.float32) * 0.9
    mel_silence = _compute_prosody_mel(silence)
    mel_tone = _compute_prosody_mel(tone)
    assert mel_silence.mean() < mel_tone.mean()


def test_prosody_mel_filterbank_cached():
    """Calling _get_mel_filterbank twice returns the same object."""
    from vconnx.engines.facodec import _get_mel_filterbank

    fb1 = _get_mel_filterbank()
    fb2 = _get_mel_filterbank()
    assert fb1 is fb2


def test_prosody_mel_filterbank_shape():
    from vconnx.engines.facodec import _get_mel_filterbank, _FA_N_MELS, _FA_N_FFT

    fb = _get_mel_filterbank()
    assert fb.shape == (_FA_N_MELS, _FA_N_FFT // 2 + 1)


# ---------------------------------------------------------------------------
# 3. Mock-session adapter contract
# ---------------------------------------------------------------------------

_T_SRC = 80   # frames for 1 s at hop=200
_T_REF = 120  # frames for 1.5 s
_N_QUANT = 6  # prosody(1) + content(2) + residual(3)
_D = 256


class _MockEncoder:
    """Returns (1, D, T) float32 features; T varies with input length."""

    def run(self, output_names, inputs):
        wav = inputs["wav"]  # (1,1,N)
        N = wav.shape[2]
        T = max(1, N // 200)
        rng = np.random.default_rng(N)
        return [rng.standard_normal((1, _D, T)).astype(np.float32)]


class _MockTimbre:
    """Returns (1, D) float32 speaker embedding."""

    def run(self, output_names, inputs):
        rng = np.random.default_rng(42)
        return [rng.standard_normal((1, _D)).astype(np.float32)]


class _MockQuantize:
    """Returns (N_QUANT, 1, T) int64 VQ ids."""

    def run(self, output_names, inputs):
        T = inputs["enc_feats"].shape[2]
        rng = np.random.default_rng(7)
        return [rng.integers(0, 512, (_N_QUANT, 1, T), dtype=np.int64)]


class _MockDecoder:
    """Returns (1, 1, N) float32 waveform."""

    def __init__(self):
        self.received_vq_ids = None
        self.received_spk_embs = None

    def run(self, output_names, inputs):
        self.received_vq_ids = inputs["vq_ids"]
        self.received_spk_embs = inputs["spk_embs"]
        T = inputs["vq_ids"].shape[2]
        N = T * 200
        return [np.zeros((1, 1, N), dtype=np.float32)]


def _make_adapter(dec=None) -> "tuple[FACodecAdapter, _MockDecoder]":  # noqa: F821
    from vconnx.engines.facodec import FACodecAdapter

    if dec is None:
        dec = _MockDecoder()
    adapter = FACodecAdapter()
    adapter._enc_sess = _MockEncoder()
    adapter._timbre_sess = _MockTimbre()
    adapter._quant_sess = _MockQuantize()
    adapter._dec_sess = dec
    return adapter, dec


def test_adapter_clone_voice_mock(tmp_path):
    """Pipeline completes and produces a valid 16-bit 16 kHz WAV."""
    adapter, _ = _make_adapter()
    src = _make_wav(str(tmp_path / "src.wav"), duration_s=1.0)
    ref = _make_wav(str(tmp_path / "ref.wav"), duration_s=1.5)
    out = str(tmp_path / "out.wav")

    result = adapter.clone_voice(src, ref, out)
    assert result == str(Path(out).resolve())
    assert Path(out).exists()

    with wave.open(out, "rb") as wf:
        assert wf.getframerate() == 16000
        assert wf.getsampwidth() == 2
        assert wf.getnframes() > 0


def test_adapter_output_sr(tmp_path):
    """Output WAV must be 16 kHz even when input is 44.1 kHz."""
    adapter, _ = _make_adapter()
    # Write 44.1 kHz source
    src_path = str(tmp_path / "src44.wav")
    n = 44100
    with wave.open(src_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(np.zeros(n, dtype=np.int16).tobytes())
    ref = _make_wav(str(tmp_path / "ref.wav"))
    out = str(tmp_path / "out.wav")
    adapter.clone_voice(src_path, ref, out)
    with wave.open(out, "rb") as wf:
        assert wf.getframerate() == 16000


def test_adapter_encoder_called_twice(tmp_path):
    """Encoder must be called for both source and reference."""

    call_count = [0]

    class CountingEncoder:
        def run(self, output_names, inputs):
            call_count[0] += 1
            wav = inputs["wav"]
            T = max(1, wav.shape[2] // 200)
            return [np.zeros((1, _D, T), dtype=np.float32)]

    from vconnx.engines.facodec import FACodecAdapter

    adapter = FACodecAdapter()
    adapter._enc_sess = CountingEncoder()
    adapter._timbre_sess = _MockTimbre()
    adapter._quant_sess = _MockQuantize()
    adapter._dec_sess = _MockDecoder()

    src = _make_wav(str(tmp_path / "s.wav"))
    ref = _make_wav(str(tmp_path / "r.wav"))
    adapter.clone_voice(src, ref, str(tmp_path / "o.wav"))
    assert call_count[0] == 2, f"Expected 2 encoder calls, got {call_count[0]}"


def test_adapter_decoder_receives_ref_timbre(tmp_path):
    """Decoder must receive the reference speaker embedding."""
    dec = _MockDecoder()
    adapter, dec = _make_adapter(dec=dec)
    src = _make_wav(str(tmp_path / "s.wav"))
    ref = _make_wav(str(tmp_path / "r.wav"))
    adapter.clone_voice(src, ref, str(tmp_path / "o.wav"))

    assert dec.received_spk_embs is not None
    assert dec.received_spk_embs.shape == (1, _D)


def test_adapter_decoder_receives_source_vq_ids(tmp_path):
    """Decoder must receive the source VQ ids (shape (6,1,T))."""
    dec = _MockDecoder()
    adapter, dec = _make_adapter(dec=dec)
    src = _make_wav(str(tmp_path / "s.wav"))
    ref = _make_wav(str(tmp_path / "r.wav"))
    adapter.clone_voice(src, ref, str(tmp_path / "o.wav"))

    assert dec.received_vq_ids is not None
    assert dec.received_vq_ids.shape[0] == _N_QUANT
    assert dec.received_vq_ids.shape[1] == 1


def test_adapter_lazy_load_raises_without_onnxruntime(monkeypatch):
    """Missing onnxruntime raises ImportError with a helpful message."""
    import builtins
    from vconnx.engines.facodec import FACodecAdapter

    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ImportError("onnxruntime not installed")
        return real_import(name, *args, **kwargs)

    adapter = FACodecAdapter()
    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", mock_import)
        with pytest.raises(ImportError, match="onnxruntime"):
            adapter._ensure_models()


def test_adapter_deterministic(tmp_path):
    """Two calls with the same inputs produce the same output."""
    adapter, _ = _make_adapter()
    src = _make_wav(str(tmp_path / "s.wav"))
    ref = _make_wav(str(tmp_path / "r.wav"))
    out1 = str(tmp_path / "o1.wav")
    out2 = str(tmp_path / "o2.wav")
    adapter.clone_voice(src, ref, out1)
    adapter.clone_voice(src, ref, out2)

    import soundfile as sf

    a1, _ = sf.read(out1)
    a2, _ = sf.read(out2)
    np.testing.assert_array_equal(a1, a2)


# ---------------------------------------------------------------------------
# 4. E2E test — real models, real audio (set VCONNX_E2E=1 to run)
# ---------------------------------------------------------------------------

_SKIP_E2E = not os.environ.get("VCONNX_E2E", "")
_E2E_REASON = (
    "E2E FACodec test downloads ~200MB of public ONNX models; "
    "set VCONNX_E2E=1 to run."
)


@pytest.mark.skipif(_SKIP_E2E, reason=_E2E_REASON)
def test_e2e_facodec_clone_edge_tts_voices(tmp_path):
    """Real end-to-end: convert an edge-tts source to a second edge-tts reference.

    Validates:
    - Output is a valid 16-bit 16 kHz WAV.
    - Duration in 0.5–15 s range.
    - WER ≤ 25% on source utterance (intelligibility gate).
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

    from vconnx.engines.facodec import FACodecAdapter

    SOURCE_TEXT = (
        "The quick brown fox jumps over the lazy dog. "
        "Voice conversion changes who is speaking, but not what is said. "
        "Listen closely and compare the engines."
    )
    REF_TEXT = (
        "This sentence provides the reference voice. Its timbre and style "
        "are what the converted audio should resemble."
    )

    async def _synth(text, voice, out):
        await edge_tts.Communicate(text, voice).save(out)

    src_mp3 = str(tmp_path / "src.mp3")
    ref_mp3 = str(tmp_path / "ref.mp3")
    src_wav = str(tmp_path / "src.wav")
    ref_wav = str(tmp_path / "ref.wav")
    out_wav = str(tmp_path / "out_facodec.wav")

    asyncio.run(_synth(SOURCE_TEXT, "en-US-GuyNeural", src_mp3))
    asyncio.run(_synth(REF_TEXT, "en-US-AriaNeural", ref_mp3))

    for mp3, wav in [(src_mp3, src_wav), (ref_mp3, ref_wav)]:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", mp3,
             "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", wav],
            check=True,
        )

    adapter = FACodecAdapter(quantized=False)
    result = adapter.clone_voice(src_wav, ref_wav, out_wav)

    assert Path(result).exists()
    size_bytes = Path(result).stat().st_size
    assert size_bytes > 0

    with wave.open(result, "rb") as wf:
        sr_out = wf.getframerate()
        n_frames = wf.getnframes()
        sampwidth = wf.getsampwidth()
        duration_s = n_frames / sr_out

    assert sr_out == 16000
    assert sampwidth == 2
    assert 0.5 <= duration_s <= 15.0, f"Duration out of range: {duration_s:.2f} s"

    def norm(text):
        return re.sub(r"[^a-z' ]", " ", text.lower()).split()

    def wer(ref_words, hyp_words):
        d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
        for i in range(len(ref_words) + 1):
            d[i][0] = i
        for j in range(len(hyp_words) + 1):
            d[0][j] = j
        for i in range(1, len(ref_words) + 1):
            for j in range(1, len(hyp_words) + 1):
                d[i][j] = min(
                    d[i - 1][j] + 1,
                    d[i][j - 1] + 1,
                    d[i - 1][j - 1] + (ref_words[i - 1] != hyp_words[j - 1]),
                )
        return d[len(ref_words)][len(hyp_words)] / max(len(ref_words), 1)

    model = WhisperModel("base.en", device="cpu", compute_type="int8")
    segments, _ = model.transcribe(result, beam_size=5)
    hyp_text = " ".join(seg.text.strip() for seg in segments)
    score = wer(norm(SOURCE_TEXT), norm(hyp_text))

    print(
        f"\n[e2e facodec] output={result} "
        f"duration={duration_s:.2f}s sr={sr_out} "
        f"size={size_bytes // 1024} KiB WER={score:.0%} "
        f"transcript={hyp_text[:120]}"
    )

    assert score <= 0.25, (
        f"WER {score:.0%} exceeds 25% intelligibility gate. "
        f"Transcript: {hyp_text[:200]}"
    )
