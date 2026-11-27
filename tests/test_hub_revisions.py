"""Every Hub download in the package names a revision."""

import ast
from pathlib import Path

import voiceclonnx

PKG = Path(voiceclonnx.__file__).parent


def test_every_hf_hub_download_passes_revision():
    calls = []
    unpinned = []
    for path in PKG.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
            if name not in ("hf_hub_download", "list_repo_files", "snapshot_download"):
                continue
            calls.append(f"{path.name}:{node.lineno}")
            if not any(kw.arg == "revision" for kw in node.keywords):
                unpinned.append(f"{path.name}:{node.lineno}")
    assert calls
    assert unpinned == []
