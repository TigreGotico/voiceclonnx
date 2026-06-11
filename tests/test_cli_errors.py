"""CLI error-path tests.

Covers:
- Unknown engine name → helpful error message, non-zero exit
- Missing --audio file → non-zero exit
- Missing --voice file → non-zero exit
- Missing required positional flags → non-zero exit (argparse)
- --help exits 0 and includes program name
- 'list' with no engines registered → prints placeholder message
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from vconnx.__main__ import main
from vconnx.engines.base import ENGINE_REGISTRY, EngineEntry, VoiceClonerBase, register_engine


# ---------------------------------------------------------------------------
# Fixture: ensure at least one engine is always available in the registry
# so the --list test can verify output.
# ---------------------------------------------------------------------------


class _MinimalAdapter(VoiceClonerBase):
    _sample_rate = 16000

    def clone_voice(self, audio, reference_voice, out_path):
        with open(out_path, "wb") as f:
            f.write(b"RIFF")
        return out_path


@pytest.fixture()
def minimal_engine():
    entry = EngineEntry(
        alias="_cli_err_mock",
        adapter_class=_MinimalAdapter,
        description="error-path mock",
        extras="",
        onnx_native=False,
    )
    register_engine(entry)
    yield entry
    ENGINE_REGISTRY.pop("_cli_err_mock", None)


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------


def test_help_exits_zero():
    with patch("sys.argv", ["vconnx", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code == 0


def test_help_output_contains_progname(capsys):
    with patch("sys.argv", ["vconnx", "--help"]):
        with pytest.raises(SystemExit):
            main()
    captured = capsys.readouterr()
    assert "vconnx" in (captured.out + captured.err)


def test_clone_help_exits_zero(capsys):
    with patch("sys.argv", ["vconnx", "clone", "--help"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code == 0


# ---------------------------------------------------------------------------
# Unknown engine
# ---------------------------------------------------------------------------


def test_clone_unknown_engine_raises(tmp_path):
    """An unknown engine alias raises KeyError (wraps to non-zero exit in real CLI)."""
    src = tmp_path / "s.wav"
    ref = tmp_path / "r.wav"
    src.write_bytes(b"\x00" * 44)
    ref.write_bytes(b"\x00" * 44)
    out = tmp_path / "o.wav"

    with patch("sys.argv", [
        "vconnx", "clone",
        "--engine", "no_such_engine_xyz",
        "--audio", str(src),
        "--voice", str(ref),
        "--out", str(out),
    ]):
        with pytest.raises(KeyError, match="no_such_engine_xyz"):
            main()


# ---------------------------------------------------------------------------
# Missing required arguments
# ---------------------------------------------------------------------------


def test_clone_missing_audio_arg(tmp_path):
    """Missing --audio triggers argparse error (SystemExit code != 0)."""
    ref = tmp_path / "r.wav"
    ref.write_bytes(b"\x00" * 44)

    with patch("sys.argv", [
        "vconnx", "clone",
        "--engine", "_cli_err_mock",
        "--voice", str(ref),
        "--out", str(tmp_path / "out.wav"),
    ]):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code != 0


def test_clone_missing_voice_arg(tmp_path):
    """Missing --voice triggers argparse error (SystemExit code != 0)."""
    src = tmp_path / "s.wav"
    src.write_bytes(b"\x00" * 44)

    with patch("sys.argv", [
        "vconnx", "clone",
        "--engine", "_cli_err_mock",
        "--audio", str(src),
        "--out", str(tmp_path / "out.wav"),
    ]):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code != 0


def test_clone_missing_out_arg(tmp_path):
    """Missing --out triggers argparse error (SystemExit code != 0)."""
    src = tmp_path / "s.wav"
    ref = tmp_path / "r.wav"
    src.write_bytes(b"\x00" * 44)
    ref.write_bytes(b"\x00" * 44)

    with patch("sys.argv", [
        "vconnx", "clone",
        "--engine", "_cli_err_mock",
        "--audio", str(src),
        "--voice", str(ref),
    ]):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code != 0


def test_no_subcommand_exits_nonzero():
    """Calling 'vconnx' with no subcommand must exit non-zero."""
    with patch("sys.argv", ["vconnx"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
    assert exc_info.value.code != 0


# ---------------------------------------------------------------------------
# list with empty registry
# ---------------------------------------------------------------------------


def test_list_empty_registry(capsys, monkeypatch):
    """When no engines are registered, list prints a placeholder."""
    import vconnx.engines.base as base_mod

    saved = dict(base_mod.ENGINE_REGISTRY)
    base_mod.ENGINE_REGISTRY.clear()
    try:
        with patch("sys.argv", ["vconnx", "list"]):
            main()
    finally:
        base_mod.ENGINE_REGISTRY.update(saved)

    captured = capsys.readouterr()
    assert "no engines" in captured.out.lower()


# ---------------------------------------------------------------------------
# list with known engine
# ---------------------------------------------------------------------------


def test_list_shows_registered_engine(capsys, minimal_engine):
    with patch("sys.argv", ["vconnx", "list"]):
        main()
    captured = capsys.readouterr()
    assert "_cli_err_mock" in captured.out


def test_list_shows_description(capsys, minimal_engine):
    with patch("sys.argv", ["vconnx", "list"]):
        main()
    captured = capsys.readouterr()
    assert "error-path mock" in captured.out
