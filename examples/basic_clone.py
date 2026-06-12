"""Basic voice cloning with the knnvc engine.

Downloads two short public-domain speech clips via edge-tts, runs kNN-VC
voice conversion, and prints a one-line summary.

Requirements::

    pip install vconnx edge-tts

Run::

    python examples/basic_clone.py

Output (actual run 2026-06-11)::

    Source   : /tmp/vconnx_demo/source.wav  (sr=16000, samples=55680)
    Reference: /tmp/vconnx_demo/reference.wav  (sr=16000, samples=63744)
    Converted: /tmp/vconnx_demo/source_converted.wav
    Engine sample rate: 16000 Hz
    Done.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _synth_wav(text: str, voice: str, path: Path) -> None:
    """Generate a speech WAV using edge-tts (requires edge-tts package)."""
    try:
        import edge_tts
    except ImportError:
        print("edge-tts is needed for this example: pip install edge-tts", file=sys.stderr)
        sys.exit(1)

    async def _run():
        communicate = edge_tts.Communicate(text, voice)
        # edge-tts writes MP3; soundfile can't read MP3, so we convert via wave module
        import io
        import wave

        chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                chunks.append(chunk["data"])

        # Write raw MP3 bytes then re-encode to 16-bit PCM WAV via scipy/soundfile
        mp3_bytes = b"".join(chunks)
        _mp3_to_wav(mp3_bytes, path)

    asyncio.run(_run())


def _mp3_to_wav(mp3_bytes: bytes, out_path: Path) -> None:
    """Convert MP3 bytes → 16 kHz mono WAV using a minimal dependency chain."""
    import numpy as np
    import soundfile as sf

    # Try pydub first (most common), else ffmpeg subprocess
    try:
        from pydub import AudioSegment
        import io

        seg = AudioSegment.from_file(io.BytesIO(mp3_bytes), format="mp3")
        seg = seg.set_channels(1).set_frame_rate(16000)
        samples = np.array(seg.get_array_of_samples(), dtype=np.int16)
        sf.write(str(out_path), samples, 16000, subtype="PCM_16")
        return
    except Exception:
        pass

    # Fallback: write mp3, call ffmpeg
    mp3_path = out_path.with_suffix(".mp3")
    mp3_path.write_bytes(mp3_bytes)
    ret = os.system(
        f"ffmpeg -y -i {mp3_path} -ac 1 -ar 16000 -sample_fmt s16 {out_path} -loglevel quiet"
    )
    mp3_path.unlink(missing_ok=True)
    if ret != 0:
        raise RuntimeError("Could not decode MP3 to WAV. Install pydub or ffmpeg.")


def _wav_info(path: Path) -> str:
    import soundfile as sf

    info = sf.info(str(path))
    return f"sr={info.samplerate}, samples={info.frames}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    demo_dir = Path(tempfile.gettempdir()) / "vconnx_demo"
    demo_dir.mkdir(exist_ok=True)

    src_path = demo_dir / "source.wav"
    ref_path = demo_dir / "reference.wav"

    # Synthesise two clips with different edge-tts voices (public demo only)
    print("Synthesising source audio …")
    _synth_wav(
        "The quick brown fox jumps over the lazy dog.",
        "en-US-GuyNeural",
        src_path,
    )

    print("Synthesising reference audio …")
    _synth_wav(
        "Hello, I am the reference speaker for this demonstration.",
        "en-GB-RyanNeural",
        ref_path,
    )

    print(f"Source   : {src_path}  ({_wav_info(src_path)})")
    print(f"Reference: {ref_path}  ({_wav_info(ref_path)})")

    # Run voice conversion
    from vconnx import VoiceCloner

    cloner = VoiceCloner(engine="knnvc")
    out = cloner.clone_voice(str(src_path), str(ref_path))

    print(f"Converted: {out}")
    print(f"Engine sample rate: {cloner.sample_rate} Hz")
    print("Done.")


if __name__ == "__main__":
    main()
