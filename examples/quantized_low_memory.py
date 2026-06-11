"""Low-memory voice cloning with INT8 quantized ONNX models.

Demonstrates quantized variants for both knnvc and openvoice engines.
INT8 models trade a small amount of quality for significantly reduced RAM
and faster CPU inference.

Model footprints:
  knnvc   fp32: ~450 MB  |  int8: ~123 MB  (73% reduction)
  openvoice fp32: varies  |  int8: smaller

Requirements::

    pip install "vconnx[knnvc]" "vconnx[openvoice]"

Run::

    python examples/quantized_low_memory.py source.wav reference.wav
"""

from __future__ import annotations

import sys
from pathlib import Path


def _clone(engine: str, quantized: bool, src: str, ref: str, out: str) -> None:
    """Run one conversion and print timing + file size."""
    import time

    from vconnx import VoiceCloner

    label = f"{engine} {'int8' if quantized else 'fp32':>4}"
    print(f"  [{label}] loading …", end="", flush=True)

    t0 = time.perf_counter()
    cloner = VoiceCloner(engine=engine, quantized=quantized)
    result = cloner.clone_voice(src, ref, out)
    elapsed = time.perf_counter() - t0

    size_kb = Path(result).stat().st_size / 1024
    print(f"\r  [{label}] {elapsed:.1f}s → {result} ({size_kb:.0f} KB, {cloner.sample_rate} Hz)")


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        print("Usage: python examples/quantized_low_memory.py <source.wav> <reference.wav>")
        sys.exit(1)

    src = sys.argv[1]
    ref = sys.argv[2]
    out_dir = Path(src).parent

    print("knnvc — fp32 vs int8:")
    _clone("knnvc", False, src, ref, str(out_dir / "out_knnvc_fp32.wav"))
    _clone("knnvc", True, src, ref, str(out_dir / "out_knnvc_int8.wav"))

    print("\nopenvoice — fp32 vs int8:")
    _clone("openvoice", False, src, ref, str(out_dir / "out_openvoice_fp32.wav"))
    _clone("openvoice", True, src, ref, str(out_dir / "out_openvoice_int8.wav"))

    print("\nDone. Compare the four output files to assess quality vs speed.")


if __name__ == "__main__":
    main()
