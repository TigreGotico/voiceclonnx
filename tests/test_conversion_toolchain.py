"""Unit tests for the conversion toolchain (pure logic — no model downloads).

These tests use tiny synthetic ONNX graphs so they run in CI without any
large model artifacts.

Covered:
- OutputLayout path helpers + makedirs
- write_manifest / read_manifest round-trip
- write_provenance content
- compare_outputs / check_tolerance math
- ComponentReport pass/fail logic
- QuantReport summary formatting
- push_models dry-run (no HF auth required)
- Tiny synthetic end-to-end: export_model → run_ort → compare_outputs → quantize_model
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Helpers to build a tiny synthetic ONNX graph without torch
# ---------------------------------------------------------------------------


def _make_tiny_onnx(output_path: str) -> None:
    """Write a minimal valid ONNX model: identity float32[1,4] → float32[1,4]."""
    import onnx
    from onnx import TensorProto, helper

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [None, 4])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [None, 4])
    node = helper.make_node("Identity", inputs=["X"], outputs=["Y"])
    graph = helper.make_graph([node], "tiny", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8
    onnx.save(model, output_path)


# ---------------------------------------------------------------------------
# OutputLayout
# ---------------------------------------------------------------------------


class TestOutputLayout:
    def test_engine_dir(self, tmp_path):
        from conversion.export_base import OutputLayout

        layout = OutputLayout.for_engine("rvc", tmp_path)
        assert layout.engine_dir == tmp_path / "rvc"

    def test_component_path(self, tmp_path):
        from conversion.export_base import OutputLayout

        layout = OutputLayout.for_engine("rvc", tmp_path)
        p = layout.component_path("encoder.onnx")
        assert p == tmp_path / "rvc" / "encoder.onnx"

    def test_makedirs_creates_dir(self, tmp_path):
        from conversion.export_base import OutputLayout

        layout = OutputLayout.for_engine("freevc", tmp_path)
        assert not layout.engine_dir.exists()
        layout.makedirs()
        assert layout.engine_dir.is_dir()

    def test_makedirs_idempotent(self, tmp_path):
        from conversion.export_base import OutputLayout

        layout = OutputLayout.for_engine("freevc", tmp_path)
        layout.makedirs()
        layout.makedirs()  # should not raise


# ---------------------------------------------------------------------------
# write_manifest / read_manifest
# ---------------------------------------------------------------------------


class TestManifest:
    def test_roundtrip(self, tmp_path):
        from conversion.export_base import OutputLayout, read_manifest, write_manifest

        layout = OutputLayout.for_engine("knn-vc", tmp_path)
        layout.makedirs()
        components = {"encoder": "encoder.onnx", "vocoder": "vocoder.onnx"}
        sample_rates = {"output": 16000}
        meta = {"opset": 14}

        path = write_manifest(layout, components, sample_rates, meta)
        assert path.exists()

        loaded = read_manifest(layout.engine_dir)
        assert loaded["engine"] == "knn-vc"
        assert loaded["components"] == components
        assert loaded["sample_rates"] == sample_rates
        assert loaded["metadata"] == meta

    def test_manifest_filename(self, tmp_path):
        from conversion.export_base import MANIFEST_FILENAME, OutputLayout, write_manifest

        layout = OutputLayout.for_engine("test", tmp_path)
        layout.makedirs()
        write_manifest(layout, {}, {})
        assert (layout.engine_dir / MANIFEST_FILENAME).exists()

    def test_manifest_no_metadata(self, tmp_path):
        from conversion.export_base import OutputLayout, read_manifest, write_manifest

        layout = OutputLayout.for_engine("test", tmp_path)
        layout.makedirs()
        write_manifest(layout, {"m": "model.onnx"}, {"out": 22050})
        loaded = read_manifest(layout.engine_dir)
        assert "metadata" not in loaded


# ---------------------------------------------------------------------------
# write_provenance
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_provenance_contains_fields(self, tmp_path):
        from conversion.export_base import write_provenance

        p = write_provenance(
            engine_dir=tmp_path,
            upstream_repo_url="https://huggingface.co/org/model",
            upstream_ref="v1.2.3",
            license_text="MIT License\nCopyright 2024",
        )
        text = p.read_text()
        assert "https://huggingface.co/org/model" in text
        assert "v1.2.3" in text
        assert "MIT License" in text

    def test_provenance_extra_fields(self, tmp_path):
        from conversion.export_base import write_provenance

        p = write_provenance(
            engine_dir=tmp_path,
            upstream_repo_url="https://example.com/model",
            upstream_ref="abc123",
            extra={"custom_key": "custom_value"},
        )
        assert "custom_key" in p.read_text()
        assert "custom_value" in p.read_text()


# ---------------------------------------------------------------------------
# compare_outputs / check_tolerance
# ---------------------------------------------------------------------------


class TestParityCompare:
    def test_identical_outputs_pass(self):
        from conversion.parity import check_tolerance, compare_outputs

        a = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
        report = compare_outputs([a], [a.copy()])
        assert report.overall_passed
        assert report.components[0].max_abs_delta == pytest.approx(0.0)
        check_tolerance(report)  # should not raise

    def test_within_tolerance_pass(self):
        from conversion.parity import check_tolerance, compare_outputs

        ref = np.ones((2, 4), dtype=np.float32)
        ort = ref + 5e-4  # below 1e-3
        report = compare_outputs([ref], [ort], max_abs_tol=1e-3, mean_abs_tol=1e-3)
        assert report.overall_passed

    def test_above_tolerance_fail(self):
        from conversion.parity import check_tolerance, compare_outputs

        ref = np.zeros((1, 8), dtype=np.float32)
        ort = ref + 0.1  # well above 1e-3
        report = compare_outputs([ref], [ort])
        assert not report.overall_passed
        with pytest.raises(AssertionError, match="Parity check FAILED"):
            check_tolerance(report)

    def test_shape_mismatch_fails(self):
        from conversion.parity import compare_outputs

        ref = np.zeros((1, 4), dtype=np.float32)
        ort = np.zeros((1, 8), dtype=np.float32)
        report = compare_outputs([ref], [ort])
        assert not report.overall_passed
        assert report.components[0].max_abs_delta == float("inf")

    def test_output_count_mismatch_raises(self):
        from conversion.parity import compare_outputs

        with pytest.raises(ValueError, match="Output count mismatch"):
            compare_outputs([np.zeros((1,))], [np.zeros((1,)), np.zeros((1,))])

    def test_named_outputs(self):
        from conversion.parity import compare_outputs

        a = np.array([1.0, 2.0])
        b = np.array([1.0, 2.0])
        report = compare_outputs([a], [b], names=["stress_logits"])
        assert report.components[0].name == "stress_logits"

    def test_report_serialisation(self, tmp_path):
        from conversion.parity import ParityReport, compare_outputs

        ref = np.array([[0.1, 0.2]])
        ort = ref + 1e-5
        report = compare_outputs([ref], [ort])
        path = tmp_path / "report.json"
        report.save(path)
        loaded = ParityReport.load(path)
        assert loaded.overall_passed == report.overall_passed
        assert len(loaded.components) == 1

    def test_summary_contains_pass_fail(self):
        from conversion.parity import compare_outputs

        ref = np.zeros((1, 4))
        ort = ref + 0.5  # fail
        report = compare_outputs([ref], [ort])
        summary = report.summary()
        assert "FAIL" in summary


# ---------------------------------------------------------------------------
# QuantReport summary
# ---------------------------------------------------------------------------


class TestQuantReport:
    def test_summary_formatting(self):
        from conversion.quantize import QuantReport

        r = QuantReport(
            original_path="/tmp/m.onnx",
            quantized_path="/tmp/m_q8.onnx",
            original_size_mb=10.0,
            quantized_size_mb=2.5,
            size_reduction_pct=75.0,
        )
        s = r.summary()
        assert "75.0%" in s
        assert "10.0 MB" in s
        assert "2.5 MB" in s


# ---------------------------------------------------------------------------
# push_models dry-run
# ---------------------------------------------------------------------------


class TestPushModelsDryRun:
    def test_dry_run_no_upload(self, tmp_path, capsys):
        from conversion.push_models import push_engine

        eng_dir = tmp_path / "rvc"
        eng_dir.mkdir()
        (eng_dir / "model.onnx").write_bytes(b"\x00" * 16)
        (eng_dir / "PROVENANCE.md").write_text("# PROVENANCE\n")

        push_engine(engine_dir=eng_dir, engine_name="rvc", dry_run=True)
        out = capsys.readouterr().out
        assert "dry-run" in out.lower() or "[dry-run]" in out
        assert "model.onnx" in out

    def test_dry_run_missing_token_ok(self, tmp_path):
        from conversion.push_models import push_engine

        eng_dir = tmp_path / "freevc"
        eng_dir.mkdir()
        (eng_dir / "x.onnx").write_bytes(b"\x00" * 8)

        # Should not raise even without HF_TOKEN in dry-run mode
        push_engine(engine_dir=eng_dir, engine_name="freevc", dry_run=True)


# ---------------------------------------------------------------------------
# Synthetic end-to-end: tiny torch module → export → ORT → parity → quantize
# ---------------------------------------------------------------------------


class TestSyntheticEndToEnd:
    """Validates the full toolchain with a 2-layer MLP (no model downloads)."""

    @pytest.fixture()
    def tiny_model_and_input(self):
        """Return (torch.nn.Module in eval, dummy_input tensor, numpy array)."""
        torch = pytest.importorskip("torch")

        class TinyMLP(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(8, 4)

            def forward(self, x):
                return self.fc(x)

        model = TinyMLP()
        model.eval()
        dummy = torch.randn(2, 8)
        return model, dummy

    def test_export_creates_onnx(self, tmp_path, tiny_model_and_input):
        from conversion.export_base import OutputLayout, export_model

        model, dummy = tiny_model_and_input
        layout = OutputLayout.for_engine("test-mlp", tmp_path)
        layout.makedirs()

        onnx_path = export_model(
            model=model,
            dummy_inputs=(dummy,),
            output_path=layout.component_path("mlp.onnx"),
            input_names=["x"],
            output_names=["y"],
            dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
        )
        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 0

    def test_parity_passes(self, tmp_path, tiny_model_and_input):
        import torch
        from conversion.export_base import OutputLayout, export_model
        from conversion.parity import check_tolerance, compare_outputs, run_ort

        model, dummy = tiny_model_and_input
        layout = OutputLayout.for_engine("test-mlp", tmp_path)
        layout.makedirs()

        onnx_path = export_model(
            model=model,
            dummy_inputs=(dummy,),
            output_path=layout.component_path("mlp.onnx"),
            input_names=["x"],
            output_names=["y"],
            dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
        )

        with torch.no_grad():
            ref_out = model(dummy).numpy()

        ort_out = run_ort(onnx_path, {"x": dummy.numpy()})
        report = compare_outputs([ref_out], ort_out, max_abs_tol=1e-4, mean_abs_tol=1e-5)
        check_tolerance(report)
        assert report.overall_passed

    def test_quantize_produces_q8(self, tmp_path, tiny_model_and_input):
        from conversion.export_base import OutputLayout, export_model
        from conversion.quantize import quantize_model

        model, dummy = tiny_model_and_input
        layout = OutputLayout.for_engine("test-mlp", tmp_path)
        layout.makedirs()

        onnx_path = export_model(
            model=model,
            dummy_inputs=(dummy,),
            output_path=layout.component_path("mlp.onnx"),
            input_names=["x"],
            output_names=["y"],
            dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
        )

        report = quantize_model(onnx_path)
        q8_path = Path(report.quantized_path)
        assert q8_path.exists()
        assert q8_path.stat().st_size > 0
        assert "_q8" in q8_path.name

    def test_q8_loads_in_ort(self, tmp_path, tiny_model_and_input):
        import onnxruntime as ort
        from conversion.export_base import OutputLayout, export_model
        from conversion.quantize import quantize_model

        model, dummy = tiny_model_and_input
        layout = OutputLayout.for_engine("test-mlp", tmp_path)
        layout.makedirs()

        onnx_path = export_model(
            model=model,
            dummy_inputs=(dummy,),
            output_path=layout.component_path("mlp.onnx"),
            input_names=["x"],
            output_names=["y"],
            dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
        )
        report = quantize_model(onnx_path)
        sess = ort.InferenceSession(report.quantized_path, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"x": dummy.numpy()})
        assert out[0].shape == (2, 4)

    def test_full_pipeline_with_manifest_and_provenance(self, tmp_path, tiny_model_and_input):
        """Full pipeline: export → manifest → provenance → parity → quantize."""
        import torch
        from conversion.export_base import (
            OutputLayout,
            export_model,
            read_manifest,
            write_manifest,
            write_provenance,
        )
        from conversion.parity import check_tolerance, compare_outputs, run_ort
        from conversion.quantize import quantize_model

        model, dummy = tiny_model_and_input
        layout = OutputLayout.for_engine("synth-e2e", tmp_path)
        layout.makedirs()

        onnx_path = export_model(
            model=model,
            dummy_inputs=(dummy,),
            output_path=layout.component_path("net.onnx"),
            input_names=["x"],
            output_names=["y"],
            dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}},
        )

        write_manifest(
            layout,
            components={"net": "net.onnx"},
            sample_rates={"output": 22050},
            metadata={"opset": 14},
        )

        write_provenance(
            engine_dir=layout.engine_dir,
            upstream_repo_url="https://huggingface.co/test/synth",
            upstream_ref="v0.1",
            license_text="Apache-2.0",
        )

        with torch.no_grad():
            ref = model(dummy).numpy()
        ort_out = run_ort(onnx_path, {"x": dummy.numpy()})
        report = compare_outputs([ref], ort_out)
        check_tolerance(report)

        q_report = quantize_model(onnx_path)
        assert Path(q_report.quantized_path).exists()

        manifest = read_manifest(layout.engine_dir)
        assert manifest["engine"] == "synth-e2e"
        assert "PROVENANCE.md" in [f.name for f in layout.engine_dir.iterdir()]


class TestDistributablePolicy:
    def test_manifest_distributable_default_true(self, tmp_path):
        from conversion.export_base import OutputLayout, write_manifest
        import json
        layout = OutputLayout(base_dir=tmp_path, engine_name="eng")
        layout.engine_dir.mkdir(parents=True, exist_ok=True)
        p = write_manifest(layout, {"m": "m.onnx"}, {"output": 16000})
        assert json.loads(p.read_text())["distributable"] is True

    def test_push_engine_refuses_local_only(self, tmp_path):
        from conversion.export_base import OutputLayout, write_manifest
        from conversion.push_models import push_engine
        import pytest
        layout = OutputLayout(base_dir=tmp_path, engine_name="eng")
        layout.engine_dir.mkdir(parents=True, exist_ok=True)
        write_manifest(layout, {"m": "m.onnx"}, {"output": 16000},
                       distributable=False)
        with pytest.raises(RuntimeError, match="local-only"):
            push_engine(layout.engine_dir, "eng", dry_run=True)
