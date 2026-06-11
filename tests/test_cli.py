"""Tests for the vconnx CLI."""

import os
import pytest
from unittest.mock import patch, MagicMock
from vconnx.__main__ import main
from vconnx.engines.base import ENGINE_REGISTRY, EngineEntry, VoiceClonerBase, register_engine


class MockAdapter(VoiceClonerBase):
    _sample_rate = 16000

    def clone_voice(self, audio, reference_voice, out_path):
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        with open(out_path, "wb") as f:
            f.write(b"RIFF")
        return out_path


@pytest.fixture(autouse=True)
def mock_engine():
    entry = EngineEntry(
        alias="_mock",
        adapter_class=MockAdapter,
        description="CLI mock engine",
        extras="",
        onnx_native=False,
    )
    register_engine(entry)
    yield
    ENGINE_REGISTRY.pop("_mock", None)


def test_cli_clone(tmp_path, capsys):
    src = tmp_path / "src.wav"
    ref = tmp_path / "ref.wav"
    out = tmp_path / "out.wav"
    src.write_bytes(b"\x00" * 44)
    ref.write_bytes(b"\x00" * 44)

    with patch("sys.argv", [
        "vconnx", "clone",
        "--engine", "_mock",
        "--audio", str(src),
        "--voice", str(ref),
        "--out", str(out),
    ]):
        main()

    captured = capsys.readouterr()
    assert "Saved:" in captured.out
    assert out.exists()


def test_cli_list(capsys):
    with patch("sys.argv", ["vconnx", "list"]):
        main()
    captured = capsys.readouterr()
    assert "_mock" in captured.out


def test_cli_no_tts_subcommand():
    """The 'tts' subcommand must not exist."""
    import argparse
    with patch("sys.argv", ["vconnx", "tts", "--text", "hello", "--voice", "x.wav", "--out", "o.wav"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code != 0
