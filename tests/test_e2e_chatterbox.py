"""End-to-end test: real Chatterbox ONNX voice conversion.

Skipped unless VOICECLONNX_E2E=1 is set; also requires edge-tts + ffmpeg for
audio generation.  Runs a real voice_convert call using native voiceclonnx
chatterbox adapter (no chatterbox_onnx package needed).

Run locally with:
    VOICECLONNX_E2E=1 pytest tests/test_e2e_chatterbox.py -v -s
"""

import asyncio
import os
import subprocess
import time
import wave
from pathlib import Path

import pytest

_SKIP_E2E = not os.environ.get("VOICECLONNX_E2E", "")
_E2E_REASON = (
    "E2E chatterbox test downloads large ONNX models and requires ffmpeg+edge-tts; "
    "set VOICECLONNX_E2E=1 to run."
)


def _synth_wav(text: str, voice: str, out_wav: str) -> None:
    """Generate a WAV via edge-tts + ffmpeg (no librosa required)."""
    import edge_tts

    mp3_path = out_wav.replace(".wav", ".mp3")

    async def _run():
        await edge_tts.Communicate(text, voice).save(mp3_path)

    asyncio.run(_run())
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", mp3_path,
         "-ar", "24000", "-ac", "1", "-sample_fmt", "s16", out_wav],
        check=True,
    )
    Path(mp3_path).unlink(missing_ok=True)


@pytest.fixture(scope="module")
def tts_wav(tmp_path_factory):
    """Generate short WAVs via edge-tts for use as source and reference."""
    pytest.importorskip("edge_tts", reason="edge-tts not installed")

    tmp = tmp_path_factory.mktemp("e2e_audio")
    src_path = str(tmp / "source.wav")
    ref_path = str(tmp / "reference.wav")

    _synth_wav("Hello, this is a voice cloning test using voiceclonnx.", "en-US-AriaNeural", src_path)
    _synth_wav("The quick brown fox jumps over the lazy dog.", "en-GB-SoniaNeural", ref_path)
    return src_path, ref_path


@pytest.mark.skipif(_SKIP_E2E, reason=_E2E_REASON)
@pytest.mark.timeout(360)
def test_chatterbox_voice_convert(tts_wav, tmp_path):
    from voiceclonnx import VoiceCloner

    src_path, ref_path = tts_wav
    out_path = str(tmp_path / "converted.wav")

    t0 = time.time()
    cloner = VoiceCloner(engine="chatterbox")
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
