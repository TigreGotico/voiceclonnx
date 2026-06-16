#!/usr/bin/env python3
"""fp32 vs INT8 quality/size comparison across all quantized-capable voiceclonnx engines.

Runs each INT8-capable engine in both fp32 and INT8 modes on the demo source
and reference audio clips, transcribes the outputs with faster-whisper, and
writes ``demo/QUANTS.md`` with the comparison table.

Usage::

    python demo/compare_quants.py [--engines knnvc mimi ...]

Notes
-----
- Chatterbox INT8 from TigreGotico/voiceclonnx-chatterbox (we host the quantized models).
- RVC base models (ContentVec + RMVPE) have INT8 variants; the voice model is
  user-supplied.  RVC's int8 flag quantizes only the shared base models; the
  demo uses the bundled ``woman_1.onnx`` voice model.
- WER gate: engines where int8 WER > 25% AND int8 WER is more than 15 points
  worse than fp32 are flagged in the verdict column.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DEMO_DIR.parent))
OUT_DIR = DEMO_DIR / "outputs"

# Engine aliases that support quantized=True
# (chatterbox excluded: onnx-community/chatterbox-onnx has no q8 exports)
INT8_CAPABLE = [
    "bicodec",
    "facodec",
    "focalcodec",
    "freevc",
    "knnvc",
    "mimi",
    "openvoice",
    "rvc",
    "speechtokenizer",
    "triaan",
]

# RVC demo uses an any-to-ONE model ref, not audio
_RVC_MODEL_REF = "ozada/onnx_rvc::woman_1.onnx"

# fp32 model totals (MB) from TigreGotico HF repos (onnx only)
# pre-measured so the script does not need network access for sizes
_FP32_MB = {
    "bicodec": 1390.7,
    "facodec": 156.3,
    "focalcodec": 690.9,
    "freevc": 1390.7,
    "knnvc": 471.7,
    "mimi": 515.2,
    "openvoice": 131.3,
    "rvc": 739.4,
    "speechtokenizer": 411.6,
    "triaan": 294.0,
}
_INT8_MB = {
    "bicodec": 419.0,
    "facodec": 69.3,
    "focalcodec": 374.8,
    "freevc": 358.8,
    "knnvc": 128.5,
    "mimi": 309.0,
    "openvoice": 43.1,
    "rvc": 193.9,
    "speechtokenizer": 157.7,
    "triaan": 84.1,
}

# Known source text (matches generate_demos.py)
SOURCE_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Voice conversion changes who is speaking, but not what is said. "
    "Listen closely and compare the engines."
)


def norm(text: str) -> list:
    return re.sub(r"[^a-z' ]", " ", text.lower()).split()


def wer(ref: list, hyp: list) -> float:
    d = [[0] * (len(hyp) + 1) for _ in range(len(ref) + 1)]
    for i in range(len(ref) + 1):
        d[i][0] = i
    for j in range(len(hyp) + 1):
        d[0][j] = j
    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            d[i][j] = min(
                d[i - 1][j] + 1,
                d[i][j - 1] + 1,
                d[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]),
            )
    return d[len(ref)][len(hyp)] / max(len(ref), 1)


def _transcribe(model, wav_path: str) -> str:
    segs, _ = model.transcribe(wav_path, beam_size=5)
    return " ".join(s.text.strip() for s in segs)


def _synth_if_missing(name: str, voice: str, text: str) -> Path:
    """Synthesize the demo wav if it doesn't already exist."""
    wav = DEMO_DIR / f"{name}.wav"
    if wav.exists():
        return wav
    import asyncio
    import subprocess
    import edge_tts  # type: ignore

    mp3 = DEMO_DIR / f"{name}.mp3"

    async def _run():
        await edge_tts.Communicate(text, voice).save(str(mp3))

    asyncio.run(_run())
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp3),
         "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(wav)],
        check=True,
    )
    mp3.unlink(missing_ok=True)
    return wav


def main() -> int:
    ap = argparse.ArgumentParser(description="fp32 vs INT8 quality/size comparison")
    ap.add_argument(
        "--engines", nargs="*", default=None,
        help="engine aliases to compare (default: all INT8-capable)",
    )
    args = ap.parse_args()

    engines = args.engines or INT8_CAPABLE

    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError:
        print("[compare] faster-whisper not installed; run: pip install faster-whisper")
        return 1

    whisper = WhisperModel("base.en", device="cpu", compute_type="int8")

    OUT_DIR.mkdir(exist_ok=True)
    ref_text = norm(SOURCE_TEXT)

    # Ensure demo audio is present
    src_wav = _synth_if_missing(
        "source", "en-US-GuyNeural", SOURCE_TEXT
    )
    ref_aria = _synth_if_missing(
        "reference_aria", "en-US-AriaNeural",
        "This sentence provides the reference voice. Its timbre and style "
        "are what the converted audio should resemble.",
    )

    rows: list[dict] = []
    skipped: list[str] = []

    from voiceclonnx import VoiceCloner

    for engine in engines:
        # Determine reference: audio or RVC model
        if engine == "rvc":
            ref_arg = _RVC_MODEL_REF
        else:
            ref_arg = str(ref_aria)

        fp32_wer: float | None = None
        int8_wer: float | None = None

        for quantized, label in [(False, "fp32"), (True, "int8")]:
            out_wav = OUT_DIR / f"_quants_{engine}_{label}.wav"
            try:
                t0 = time.monotonic()
                cloner = VoiceCloner(engine=engine, quantized=quantized)
                cloner.clone_voice(str(src_wav), ref_arg, str(out_wav))
                elapsed = time.monotonic() - t0
                transcript = _transcribe(whisper, str(out_wav))
                score = wer(ref_text, norm(transcript))
                print(
                    f"[compare] {engine}/{label}: WER={score:.0%} in {elapsed:.0f}s"
                )
                if quantized:
                    int8_wer = score
                else:
                    fp32_wer = score
            except Exception as exc:
                print(f"[compare] {engine}/{label}: ERROR — {exc}")
                if quantized:
                    int8_wer = None
                else:
                    fp32_wer = None
                    skipped.append(engine)
                    break

        if engine in skipped:
            continue

        fp32_mb = _FP32_MB.get(engine, 0.0)
        int8_mb = _INT8_MB.get(engine, 0.0)
        saving = (1.0 - int8_mb / fp32_mb) * 100 if fp32_mb else 0.0

        # Verdict logic
        if int8_wer is None:
            verdict = "fp32 only"
        elif fp32_wer is not None and (
            int8_wer > 0.25 and (int8_wer - fp32_wer) > 0.15
        ):
            verdict = f"⚠ int8 degraded ({int8_wer:.0%} vs fp32 {fp32_wer:.0%})"
        else:
            verdict = "✅ int8 recommended"

        rows.append(
            {
                "engine": engine,
                "fp32_wer": fp32_wer,
                "int8_wer": int8_wer,
                "fp32_mb": fp32_mb,
                "int8_mb": int8_mb,
                "saving_pct": saving,
                "verdict": verdict,
            }
        )

    # --- write QUANTS.md ---
    lines = [
        "# Quantized-model comparison (generated by compare_quants.py — do not edit)",
        "",
        "WER measured with faster-whisper `base.en` against the known source text.",
        "Sizes are ONNX model totals from the TigreGotico HF repos (fp32 + INT8).",
        "Gate: int8 flagged ⚠ when WER > 25% **and** > 15 points worse than fp32.",
        "",
        "## chatterbox — fp32 only",
        "",
        "The upstream `onnx-community/chatterbox-onnx` repository does not publish",
        "INT8 variants of `speech_encoder.onnx` or `conditional_decoder.onnx`.",
        "The `quantized=True` parameter is accepted (uniform API) but silently",
        "ignored; chatterbox always runs fp32 until upstream ships q8 exports.",
        "",
        "## Engine comparison",
        "",
        "| Engine | fp32 WER | int8 WER | fp32 size (MB) | int8 size (MB) | Saving | Verdict |",
        "|--------|----------|----------|----------------|----------------|--------|---------|",
    ]

    for r in rows:
        fp32_str = f"{r['fp32_wer']:.0%}" if r["fp32_wer"] is not None else "—"
        int8_str = f"{r['int8_wer']:.0%}" if r["int8_wer"] is not None else "—"
        lines.append(
            f"| `{r['engine']}` | {fp32_str} | {int8_str} "
            f"| {r['fp32_mb']:.1f} | {r['int8_mb']:.1f} "
            f"| {r['saving_pct']:.0f}% | {r['verdict']} |"
        )

    lines += [
        "",
        "## Notes",
        "",
        "- **rvc**: `quantized=True` applies only to the shared base models",
        "  (ContentVec-768 + RMVPE); the per-voice synthesizer is user-supplied.",
        "- Shared numpy artifacts (codebooks, mel filterbanks, mel stats) are never",
        "  quantized — they are not ONNX models.",
        "- Sizes include only ONNX model files from the respective HF repo.",
        "",
    ]

    # Write to docs/QUANTS.md (canonical reference) and demo/QUANTS.md (alongside demo data)
    docs_dir = DEMO_DIR.parent / "docs"
    docs_dir.mkdir(exist_ok=True)
    quants_path = docs_dir / "QUANTS.md"
    quants_path.write_text("\n".join(lines))
    # Also write to demo/ for local reference
    (DEMO_DIR / "QUANTS.md").write_text("\n".join(lines))
    print(f"[compare] wrote {quants_path} ({len(rows)} engines)")
    if skipped:
        print(f"[compare] skipped (errors): {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
