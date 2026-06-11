"""Parity harness: compare a PyTorch reference path vs an ONNX runtime path.

Usage
-----
Command-line (per-component)::

    python -m conversion.parity \\
        --onnx path/to/model.onnx \\
        --input-npy path/to/input.npy \\
        --ref-npy path/to/reference_output.npy \\
        --report path/to/parity_report.json

Programmatic::

    from conversion.parity import compare_outputs, ParityReport, check_tolerance

    report = compare_outputs(torch_outputs, ort_outputs)
    check_tolerance(report, max_abs_tol=1e-3, mean_abs_tol=1e-4)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ComponentReport:
    """Parity metrics for a single output tensor."""

    name: str
    max_abs_delta: float
    mean_abs_delta: float
    shape: Tuple[int, ...]
    passed: bool = True


@dataclass
class ParityReport:
    """Aggregated parity report across all outputs of one model component."""

    components: List[ComponentReport] = field(default_factory=list)
    overall_passed: bool = True
    max_abs_tol: float = 1e-3
    mean_abs_tol: float = 1e-4

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict:
        d = {
            "overall_passed": self.overall_passed,
            "tolerances": {
                "max_abs": self.max_abs_tol,
                "mean_abs": self.mean_abs_tol,
            },
            "components": [asdict(c) for c in self.components],
        }
        return d

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ParityReport":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        components = [ComponentReport(**c) for c in d.get("components", [])]
        return cls(
            components=components,
            overall_passed=d.get("overall_passed", True),
            max_abs_tol=d.get("tolerances", {}).get("max_abs", 1e-3),
            mean_abs_tol=d.get("tolerances", {}).get("mean_abs", 1e-4),
        )

    def summary(self) -> str:
        lines = [f"Parity report — {'PASS' if self.overall_passed else 'FAIL'}"]
        for c in self.components:
            status = "OK" if c.passed else "FAIL"
            lines.append(
                f"  [{status}] {c.name}: max_abs={c.max_abs_delta:.3e}  mean_abs={c.mean_abs_delta:.3e}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core comparison helpers
# ---------------------------------------------------------------------------


def _to_numpy(x) -> np.ndarray:
    """Convert a tensor-like (torch, numpy, list) to a numpy float32 array."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    try:
        import torch  # noqa: F401

        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy().astype(np.float32)
    except ImportError:
        pass
    return np.array(x, dtype=np.float32)


def compare_outputs(
    ref_outputs: Union[Sequence, np.ndarray],
    ort_outputs: Union[Sequence, np.ndarray],
    names: Optional[Sequence[str]] = None,
    max_abs_tol: float = 1e-3,
    mean_abs_tol: float = 1e-4,
) -> ParityReport:
    """Compare reference outputs vs ORT outputs and return a :class:`ParityReport`.

    Parameters
    ----------
    ref_outputs:
        Sequence of reference (torch) output tensors / arrays.
    ort_outputs:
        Sequence of onnxruntime output arrays (same length/order as ref).
    names:
        Optional output names for reporting.  Defaults to ``["out_0", "out_1", …]``.
    max_abs_tol:
        Tolerance for maximum absolute difference per component.
    mean_abs_tol:
        Tolerance for mean absolute difference per component.

    Returns
    -------
    ParityReport
    """
    if not isinstance(ref_outputs, (list, tuple)):
        ref_outputs = [ref_outputs]
    if not isinstance(ort_outputs, (list, tuple)):
        ort_outputs = [ort_outputs]

    if len(ref_outputs) != len(ort_outputs):
        raise ValueError(
            f"Output count mismatch: ref={len(ref_outputs)} vs ort={len(ort_outputs)}"
        )

    if names is None:
        names = [f"out_{i}" for i in range(len(ref_outputs))]

    report = ParityReport(max_abs_tol=max_abs_tol, mean_abs_tol=mean_abs_tol)

    for name, ref, ort in zip(names, ref_outputs, ort_outputs):
        ref_np = _to_numpy(ref)
        ort_np = _to_numpy(ort)

        if ref_np.shape != ort_np.shape:
            report.components.append(
                ComponentReport(
                    name=name,
                    max_abs_delta=float("inf"),
                    mean_abs_delta=float("inf"),
                    shape=tuple(ref_np.shape),
                    passed=False,
                )
            )
            report.overall_passed = False
            continue

        diff = np.abs(ref_np - ort_np)
        max_d = float(diff.max())
        mean_d = float(diff.mean())
        passed = max_d <= max_abs_tol and mean_d <= mean_abs_tol

        report.components.append(
            ComponentReport(
                name=name,
                max_abs_delta=max_d,
                mean_abs_delta=mean_d,
                shape=tuple(ref_np.shape),
                passed=passed,
            )
        )
        if not passed:
            report.overall_passed = False

    return report


def check_tolerance(
    report: ParityReport,
    max_abs_tol: Optional[float] = None,
    mean_abs_tol: Optional[float] = None,
) -> None:
    """Raise :class:`AssertionError` if the report fails the given tolerances.

    If *max_abs_tol* / *mean_abs_tol* are omitted the thresholds embedded in
    the report are used.
    """
    max_tol = max_abs_tol if max_abs_tol is not None else report.max_abs_tol
    mean_tol = mean_abs_tol if mean_abs_tol is not None else report.mean_abs_tol

    failures = []
    for c in report.components:
        if c.max_abs_delta > max_tol or c.mean_abs_delta > mean_tol:
            failures.append(
                f"{c.name}: max_abs={c.max_abs_delta:.3e} (tol {max_tol:.3e}), "
                f"mean_abs={c.mean_abs_delta:.3e} (tol {mean_tol:.3e})"
            )

    if failures:
        raise AssertionError("Parity check FAILED:\n" + "\n".join(failures))


# ---------------------------------------------------------------------------
# ORT inference helper
# ---------------------------------------------------------------------------


def run_ort(
    onnx_path: Union[str, Path],
    inputs: Dict[str, np.ndarray],
    providers: Optional[List[str]] = None,
) -> List[np.ndarray]:
    """Run an ONNX model with onnxruntime and return the outputs.

    Parameters
    ----------
    onnx_path:
        Path to the ``.onnx`` file.
    inputs:
        Dict mapping input name → numpy array.
    providers:
        ORT execution providers (defaults to ``["CPUExecutionProvider"]``).
    """
    import onnxruntime as ort

    if providers is None:
        providers = ["CPUExecutionProvider"]

    sess = ort.InferenceSession(str(onnx_path), providers=providers)
    return sess.run(None, inputs)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare a torch reference output with an ONNX runtime output."
    )
    p.add_argument("--onnx", required=True, help="Path to the .onnx file.")
    p.add_argument(
        "--input-npy",
        nargs="+",
        required=True,
        help="One or more .npy input arrays (same order as ONNX input names).",
    )
    p.add_argument(
        "--ref-npy",
        nargs="+",
        required=True,
        help="One or more .npy reference output arrays.",
    )
    p.add_argument("--report", default="parity_report.json", help="Output JSON report path.")
    p.add_argument("--max-abs-tol", type=float, default=1e-3)
    p.add_argument("--mean-abs-tol", type=float, default=1e-4)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)

    sess_inputs = []
    import onnxruntime as ort

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_names = [inp.name for inp in sess.get_inputs()]

    inputs: Dict[str, np.ndarray] = {}
    for name, npy_path in zip(input_names, args.input_npy):
        inputs[name] = np.load(npy_path)

    ort_outputs = sess.run(None, inputs)
    ref_outputs = [np.load(p) for p in args.ref_npy]

    report = compare_outputs(
        ref_outputs,
        ort_outputs,
        max_abs_tol=args.max_abs_tol,
        mean_abs_tol=args.mean_abs_tol,
    )
    report.save(args.report)
    print(report.summary())

    if not report.overall_passed:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
