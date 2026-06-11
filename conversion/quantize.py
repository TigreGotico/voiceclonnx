"""INT8 dynamic quantization pass for vconnx ONNX models.

Applies ``onnxruntime.quantization.quantize_dynamic`` to produce a ``_q8.onnx``
variant alongside the original full-precision file, then prints a size and
(optionally) RTF comparison report.

Usage
-----
Command-line::

    python -m conversion.quantize path/to/model.onnx [--output path/to/model_q8.onnx]

Programmatic::

    from conversion.quantize import quantize_model, QuantReport

    report = quantize_model("encoder.onnx")
    print(report.summary())
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------


@dataclass
class QuantReport:
    """Size and (optional) latency comparison after quantization."""

    original_path: str
    quantized_path: str
    original_size_mb: float
    quantized_size_mb: float
    size_reduction_pct: float
    original_latency_ms: Optional[float] = None
    quantized_latency_ms: Optional[float] = None

    def summary(self) -> str:
        lines = [
            "Quantization report",
            f"  original : {self.original_path}  ({self.original_size_mb:.1f} MB)",
            f"  quantized: {self.quantized_path}  ({self.quantized_size_mb:.1f} MB)",
            f"  size reduction: {self.size_reduction_pct:.1f}%",
        ]
        if self.original_latency_ms is not None:
            lines.append(f"  latency original : {self.original_latency_ms:.1f} ms")
        if self.quantized_latency_ms is not None:
            lines.append(f"  latency quantized: {self.quantized_latency_ms:.1f} ms")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core quantization helper
# ---------------------------------------------------------------------------


def quantize_model(
    input_path: Union[str, Path],
    output_path: Optional[Union[str, Path]] = None,
    weight_type: str = "QInt8",
    nodes_to_exclude: Optional[List[str]] = None,
    benchmark_inputs: Optional[Dict[str, np.ndarray]] = None,
    benchmark_runs: int = 10,
) -> QuantReport:
    """Apply INT8 dynamic quantization and return a :class:`QuantReport`.

    Parameters
    ----------
    input_path:
        Path to the source ``.onnx`` model.
    output_path:
        Destination for the quantized model.  Defaults to the input path
        with a ``_q8`` suffix inserted before the extension.
    weight_type:
        ORT quantization weight type.  ``"QInt8"`` (default) or ``"QUInt8"``.
    nodes_to_exclude:
        Optional list of node names to skip during quantization.
    benchmark_inputs:
        Optional dict of ``{input_name: np.ndarray}`` for latency measurement.
    benchmark_runs:
        Number of warm-up + timed inference runs for the latency benchmark.

    Returns
    -------
    QuantReport
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_stem(input_path.stem + "_q8")
    output_path = Path(output_path)

    qt = QuantType.QInt8 if weight_type == "QInt8" else QuantType.QUInt8

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=qt,
        nodes_to_exclude=nodes_to_exclude or [],
    )

    orig_mb = input_path.stat().st_size / 1024 ** 2
    quant_mb = output_path.stat().st_size / 1024 ** 2
    reduction = (1.0 - quant_mb / orig_mb) * 100.0 if orig_mb > 0 else 0.0

    orig_lat: Optional[float] = None
    quant_lat: Optional[float] = None

    if benchmark_inputs is not None:
        orig_lat = _bench(str(input_path), benchmark_inputs, benchmark_runs)
        quant_lat = _bench(str(output_path), benchmark_inputs, benchmark_runs)

    return QuantReport(
        original_path=str(input_path),
        quantized_path=str(output_path),
        original_size_mb=orig_mb,
        quantized_size_mb=quant_mb,
        size_reduction_pct=reduction,
        original_latency_ms=orig_lat,
        quantized_latency_ms=quant_lat,
    )


def _bench(onnx_path: str, inputs: Dict[str, np.ndarray], runs: int) -> float:
    """Return average inference latency in milliseconds over *runs* runs."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    # warm-up
    for _ in range(max(2, runs // 5)):
        sess.run(None, inputs)
    start = time.perf_counter()
    for _ in range(runs):
        sess.run(None, inputs)
    elapsed_ms = (time.perf_counter() - start) / runs * 1000.0
    return elapsed_ms


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Dynamic INT8 quantization for vconnx ONNX models.")
    p.add_argument("input", help="Path to the source .onnx file.")
    p.add_argument("--output", default=None, help="Destination path (default: <stem>_q8.onnx).")
    p.add_argument(
        "--weight-type",
        default="QInt8",
        choices=["QInt8", "QUInt8"],
        help="Quantization weight type (default: QInt8).",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    report = quantize_model(
        input_path=args.input,
        output_path=args.output,
        weight_type=args.weight_type,
    )
    print(report.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
