"""End-to-end test: real Chatterbox ONNX voice conversion.

Skipped unless chatterbox_onnx is importable.  Runs a real voice_convert
call using edge-tts output as source audio.  Timeout-bound for CI.

Run locally with:
    pytest tests/test_e2e_chatterbox.py -v -s
"""

import asyncio
import os
import tempfile
import time
import wave
from pathlib import Path

import pytest

pytest.importorskip("chatterbox_onnx", reason="chatterbox_onnx not installed")
pytest.importorskip("edge_tts", reason="edge-tts not installed")


@pytest.fixture(scope="module")
def tts_wav(tmp_path_factory):
    """Generate a short WAV via edge-tts for use as source + reference."""
    import edge_tts

    tmp = tmp_path_factory.mktemp("e2e_audio")
    src_path = str(tmp / "source.wav")
    ref_path = str(tmp / "reference.wav")

    async def _gen(text, path):
        comm = edge_tts.Communicate(text, "en-US-AriaNeural")
        # edge-tts produces mp3; use a temp mp3 then convert via soundfile/librosa
        mp3_path = path.replace(".wav", ".mp3")
        await comm.save(mp3_path)
        import librosa
        import soundfile as sf
        y, sr = librosa.load(mp3_path, sr=24000, mono=True)
        sf.write(path, y, sr, subtype="PCM_16")

    asyncio.run(_gen("Hello, this is a voice cloning test using vconnx.", src_path))
    asyncio.run(_gen("The quick brown fox jumps over the lazy dog.", ref_path))
    return src_path, ref_path


@pytest.mark.timeout(360)
def test_chatterbox_voice_convert(tts_wav, tmp_path):
    from vconnx import VoiceCloner

    src_path, ref_path = tts_wav
    out_path = str(tmp_path / "converted.wav")

    t0 = time.time()
    cloner = VoiceCloner(engine="chatterbox", quantized=True, max_new_tokens=256)
    result = cloner.clone_voice(src_path, ref_path, out_path)
    elapsed = time.time() - t0

    assert result == out_path, "clone_voice must return out_path"
    assert Path(out_path).exists(), "output file must exist"
    assert Path(out_path).stat().st_size > 1000, "output must be non-trivial"
    assert cloner.sample_rate == 24000

    # Verify WAV is valid
    with wave.open(out_path, "rb") as wf:
        nframes = wf.getnframes()
        sr = wf.getframerate()

    assert sr == 24000, f"expected 24000 Hz, got {sr}"
    assert nframes > 0

    print(f"\n[e2e] voice_convert completed in {elapsed:.1f}s")
    print(f"[e2e] output: {out_path}  frames={nframes}  sr={sr}")
