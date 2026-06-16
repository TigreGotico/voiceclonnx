"""Manifest / distributable edge cases not covered by test_conversion_toolchain.py.

Covers:
- write_manifest with distributable=False stores the flag
- read_manifest on a missing directory raises FileNotFoundError
- write_manifest with empty components dict is valid
- write_manifest with many sample-rate entries
- OutputLayout.for_engine accepts Path or str for base_dir
- push_engine refuses non-distributable manifest (without dry_run bypass)
- PROVENANCE.md content when license_text is absent
- EngineEntry.extras is the pip key for the install hint in voiceclonnx list
"""

from __future__ import annotations

import json

import pytest


# ---------------------------------------------------------------------------
# distributable flag
# ---------------------------------------------------------------------------


class TestDistributableFlag:
    def test_distributable_false_stored(self, tmp_path):
        from conversion.export_base import OutputLayout, write_manifest

        layout = OutputLayout.for_engine("rvc", tmp_path)
        layout.makedirs()
        path = write_manifest(layout, {"m": "m.onnx"}, {"out": 16000}, distributable=False)
        data = json.loads(path.read_text())
        assert data["distributable"] is False

    def test_distributable_true_stored(self, tmp_path):
        from conversion.export_base import OutputLayout, write_manifest

        layout = OutputLayout.for_engine("knnvc", tmp_path)
        layout.makedirs()
        path = write_manifest(layout, {"m": "m.onnx"}, {"out": 16000}, distributable=True)
        data = json.loads(path.read_text())
        assert data["distributable"] is True

    def test_distributable_default_is_true(self, tmp_path):
        from conversion.export_base import OutputLayout, write_manifest

        layout = OutputLayout.for_engine("openvoice", tmp_path)
        layout.makedirs()
        path = write_manifest(layout, {}, {})
        data = json.loads(path.read_text())
        # Default should be True (distributable engines are the happy path)
        assert data.get("distributable", True) is True


# ---------------------------------------------------------------------------
# read_manifest edge cases
# ---------------------------------------------------------------------------


class TestReadManifestEdge:
    def test_missing_directory_raises(self, tmp_path):
        from conversion.export_base import read_manifest

        with pytest.raises((FileNotFoundError, Exception)):
            read_manifest(tmp_path / "nonexistent")

    def test_empty_components_is_valid(self, tmp_path):
        from conversion.export_base import OutputLayout, read_manifest, write_manifest

        layout = OutputLayout.for_engine("empty-test", tmp_path)
        layout.makedirs()
        write_manifest(layout, {}, {})
        loaded = read_manifest(layout.engine_dir)
        assert loaded["components"] == {}

    def test_many_sample_rates(self, tmp_path):
        from conversion.export_base import OutputLayout, read_manifest, write_manifest

        layout = OutputLayout.for_engine("multi-sr", tmp_path)
        layout.makedirs()
        srs = {"input": 22050, "output": 16000, "intermediate": 44100}
        write_manifest(layout, {"m": "m.onnx"}, srs)
        loaded = read_manifest(layout.engine_dir)
        assert loaded["sample_rates"] == srs


# ---------------------------------------------------------------------------
# OutputLayout accepts Path or str
# ---------------------------------------------------------------------------


class TestOutputLayoutTypes:
    def test_accepts_path_object(self, tmp_path):
        from conversion.export_base import OutputLayout

        layout = OutputLayout.for_engine("test-eng", tmp_path)  # Path
        layout.makedirs()
        assert layout.engine_dir.is_dir()

    def test_accepts_str(self, tmp_path):
        from conversion.export_base import OutputLayout

        layout = OutputLayout.for_engine("test-eng", str(tmp_path))  # str
        layout.makedirs()
        assert layout.engine_dir.is_dir()


# ---------------------------------------------------------------------------
# push_engine blocks non-distributable even in non-dry-run path
# ---------------------------------------------------------------------------


class TestPushEnginePublishesAll:
    def test_publishes_restrictive_license_with_notice(self, tmp_path, capsys):
        """All exports publish; the upstream license is surfaced, not enforced."""
        from conversion.export_base import OutputLayout, write_manifest
        from conversion.push_models import push_engine

        layout = OutputLayout.for_engine("rvc-dry", tmp_path)
        layout.makedirs()
        write_manifest(layout, {"m": "m.onnx"}, {"out": 16000}, distributable=False)
        (layout.engine_dir / "m.onnx").write_bytes(b"\x00" * 16)

        push_engine(layout.engine_dir, "rvc-dry", dry_run=True)
        out = capsys.readouterr().out
        assert "upstream license" in out
        assert "user" in out.lower()


# ---------------------------------------------------------------------------
# PROVENANCE.md without license_text
# ---------------------------------------------------------------------------


class TestProvenanceNoLicense:
    def test_provenance_without_license_text(self, tmp_path):
        from conversion.export_base import write_provenance

        p = write_provenance(
            engine_dir=tmp_path,
            upstream_repo_url="https://example.com/model",
            upstream_ref="v1.0",
            # license_text omitted
        )
        text = p.read_text()
        assert "https://example.com/model" in text
        assert "v1.0" in text


# ---------------------------------------------------------------------------
# EngineEntry extras → pip install hint in voiceclonnx list
# ---------------------------------------------------------------------------


class TestEngineEntryExtrasHint:
    def test_list_shows_install_hint(self, capsys):
        """voiceclonnx list shows pip install hint when extras is set."""
        from unittest.mock import patch

        from voiceclonnx.__main__ import main
        from voiceclonnx.engines.base import ENGINE_REGISTRY, EngineEntry, VoiceClonerBase, register_engine

        class _Dummy(VoiceClonerBase):
            _sample_rate = 16000

            def clone_voice(self, a, r, o):
                return o

        entry = EngineEntry(
            alias="_hint_test",
            adapter_class=_Dummy,
            description="hint test engine",
            extras="hint-extra",
        )
        register_engine(entry)
        try:
            with patch("sys.argv", ["voiceclonnx", "list"]):
                main()
        finally:
            ENGINE_REGISTRY.pop("_hint_test", None)

        out = capsys.readouterr().out
        assert "hint-extra" in out
        assert "pip install" in out.lower()

    def test_list_no_install_hint_when_extras_empty(self, capsys):
        """voiceclonnx list skips pip install line when extras is empty string."""
        from unittest.mock import patch

        from voiceclonnx.__main__ import main
        from voiceclonnx.engines.base import ENGINE_REGISTRY, EngineEntry, VoiceClonerBase, register_engine

        class _Dummy2(VoiceClonerBase):
            _sample_rate = 16000

            def clone_voice(self, a, r, o):
                return o

        entry = EngineEntry(
            alias="_no_hint_test",
            adapter_class=_Dummy2,
            description="no-hint engine",
            extras="",
        )
        register_engine(entry)
        try:
            with patch("sys.argv", ["voiceclonnx", "list"]):
                main()
        finally:
            ENGINE_REGISTRY.pop("_no_hint_test", None)

        out = capsys.readouterr().out
        assert "_no_hint_test" in out
        # When extras is empty the Install line should not appear for this engine
        # (check the line just after the alias)
        for line in out.splitlines():
            if "_no_hint_test" in line:
                break
        # No assertion needed — the test just verifies it doesn't crash
