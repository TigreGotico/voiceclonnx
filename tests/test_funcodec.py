"""Tests for the FunCodec engine investigation (issue #38 — BLOCKED).

The FunCodec engine is blocked because no publicly available checkpoint
satisfies the semantic RVQ paradigm required for zero-shot voice conversion.
See docs/engines/funcodec.md for the full investigation findings.

These tests document the verified negative result:
- FunCodec is NOT registered (no adapter exists).
- The export script exits with an error indicating the block.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def test_funcodec_not_registered():
    """funcodec must NOT appear in ENGINE_REGISTRY — no adapter exists yet."""
    import voiceclonnx
    from voiceclonnx.engines.base import ENGINE_REGISTRY

    assert "funcodec" not in ENGINE_REGISTRY, (
        "funcodec appeared in ENGINE_REGISTRY unexpectedly; "
        "the engine is blocked pending a qualifying semantic checkpoint."
    )


def test_export_funcodec_stub_exits_nonzero():
    """The export stub must exit with a non-zero code indicating the block."""
    result = subprocess.run(
        [sys.executable, "-m", "conversion.export_funcodec"],
        capture_output=True,
        text=True,
        cwd=__file__.replace("/tests/test_funcodec.py", ""),
    )
    assert result.returncode != 0, (
        "export_funcodec.py returned 0 unexpectedly — the stub should exit 1"
    )
    assert "BLOCKED" in result.stdout or "BLOCKED" in result.stderr


def test_funcodec_investigation_docs_present():
    """docs/engines/funcodec.md must exist and document the blocking finding."""
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    doc = repo_root / "docs" / "engines" / "funcodec.md"
    assert doc.exists(), f"Missing {doc}"
    text = doc.read_text()
    # Must contain the key blocking verdict
    assert "BLOCKED" in text
    assert "PPG" in text or "ppg" in text
    assert "codec_semantic_aug" in text
