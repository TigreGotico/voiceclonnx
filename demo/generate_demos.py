#!/usr/bin/env python3
"""Regenerate the bundled engine-comparison demos.

Synthesizes a fixed source utterance and reference voices with edge-tts,
then runs every registered voiceclonnx engine on the same (source, reference)
pairs, writing ``outputs/<engine>__<reference>.wav``.  The wav files are
committed so listeners can compare engines without running any code.

Run after adding an engine:
    python demo/generate_demos.py [--engines knnvc chatterbox]
"""
import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_DIR.parent))  # prefer the checkout over any installed copy
OUT_DIR = DEMO_DIR / "outputs"

SOURCE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Voice conversion changes who is speaking, but not what is said. "
    "Listen closely and compare the engines."
)
REF_TEXT = (
    "This sentence provides the reference voice. Its timbre and style "
    "are what the converted audio should resemble."
)

SOURCE = ("source", "en-US-GuyNeural", SOURCE_TEXT)
REFERENCES = [
    ("reference_aria", "en-US-AriaNeural", REF_TEXT),
    ("reference_sonia", "en-GB-SoniaNeural", REF_TEXT),
]

# any-to-ONE engines take a voice MODEL as reference, not audio: one demo
# output per (engine, model) instead of per reference wav.
MODEL_REFERENCES = {
    "rvc": [("woman1", "ozada/onnx_rvc::woman_1.onnx")],
}


def synth(name: str, voice: str, text: str) -> Path:
    wav = DEMO_DIR / f"{name}.wav"
    if wav.exists():
        return wav
    mp3 = DEMO_DIR / f"{name}.mp3"
    import edge_tts

    async def _run():
        await edge_tts.Communicate(text, voice).save(str(mp3))

    asyncio.run(_run())
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
         "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(wav)],
        check=True,
    )
    mp3.unlink()
    return wav


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engines", nargs="*", default=None,
                    help="engine aliases to run (default: all registered)")
    args = ap.parse_args()

    from voiceclonnx import VoiceCloner
    from voiceclonnx.engines.base import ENGINE_REGISTRY

    engines = args.engines or sorted(ENGINE_REGISTRY)
    OUT_DIR.mkdir(exist_ok=True)

    src = synth(*SOURCE)
    refs = [(name, synth(name, voice, text))
            for name, voice, text in REFERENCES]

    failures = []
    for engine in engines:
        try:
            cloner = VoiceCloner(engine=engine)
        except Exception as exc:
            print(f"[demo] {engine}: cannot instantiate ({exc}); skipped")
            failures.append(engine)
            continue
        if engine in MODEL_REFERENCES:
            for ref_name, model_ref in MODEL_REFERENCES[engine]:
                out = OUT_DIR / f"{engine}__{ref_name}.wav"
                print(f"[demo] {engine} ← model:{ref_name} → {out.name}")
                cloner.clone_voice(str(src), model_ref, str(out))
            continue
        for ref_name, ref_wav in refs:
            out = OUT_DIR / f"{engine}__{ref_name.replace('reference_', '')}.wav"
            print(f"[demo] {engine} ← {ref_name} → {out.name}")
            cloner.clone_voice(str(src), str(ref_wav), str(out))
    print("[demo] done.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
