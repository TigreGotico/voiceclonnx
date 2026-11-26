"""Write docs/QUANTS.md from demo/quants.json, which demo/compare_quants.py writes.

    python scripts/render_quants.py           # write the page
    python scripts/render_quants.py --check   # exit 1 when the page differs from the JSON
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "demo" / "quants.json"
PAGE = ROOT / "docs" / "QUANTS.md"


def render(data: dict) -> str:
    words = len(data["source_text"].split())
    lines = [
        "# Quantized-model comparison",
        "<!-- agent-hint",
        "covers: fp32 and int8 WER of each engine",
        "source: demo/quants.json",
        "generated: yes, by scripts/render_quants.py. Edit the generator or rerun demo/compare_quants.py",
        "verify: python scripts/render_quants.py --check",
        "-->",
        "",
        f"This table gives the word error rate (WER) of each engine's fp32 and int8 ONNX models. "
        f"The ASR model is faster-whisper `{data['asr'].split()[-1]}`. The source text is one sentence "
        f"of {words} words, so one word error is {100 / words:.1f} WER points.",
        "",
        "> **Sample size.** Each WER comes from one sentence. A difference of one or two words is noise. "
        "The table gives no int8 recommendation until a larger run is available.",
        "",
        "| Engine | fp32 WER | int8 WER | fp32 errors | int8 errors |",
        "|--------|---------:|---------:|------------:|------------:|",
    ]
    for r in data["rows"]:
        lines.append(f"| `{r['engine']}` | {r['fp32_wer']:.0%} | {r['int8_wer']:.0%} "
                     f"| {round(r['fp32_wer'] * words)} | {round(r['int8_wer'] * words)} |")
    if data["skipped"]:
        lines += ["", "Not measured: " + ", ".join(f"`{e}`" for e in data["skipped"]) + "."]
    lines += [
        "",
        "Model sizes are not in this table. `demo/compare_quants.py` holds them as fixed values, "
        "not as a measurement, and has no value for some engines.",
        "",
        "## Notes",
        "",
        "- **rvc**: `quantized=True` applies only to the shared base models "
        "(ContentVec-768 and RMVPE). The per-voice synthesizer is user-supplied.",
        "- Shared numpy artifacts (codebooks, mel filterbanks, mel stats) are never quantized, "
        "because they are not ONNX models.",
        "",
        "---",
        "[← api](api.md) · [Home](index.md) · [converting →](converting.md)",
        "",
    ]
    return "\n".join(lines)


def main(argv) -> int:
    page = render(json.loads(DATA.read_text(encoding="utf-8")))
    if "--check" in argv:
        if PAGE.read_text(encoding="utf-8") != page:
            print(f"{PAGE.relative_to(ROOT)} differs from {DATA.relative_to(ROOT)}: run python scripts/render_quants.py")
            return 1
        return 0
    PAGE.write_text(page, encoding="utf-8")
    print(f"rows in {len(json.loads(DATA.read_text())['rows'])} -> {PAGE.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
