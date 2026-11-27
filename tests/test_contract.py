"""Family contract: every Hub download of a built-in model is pinned to a commit sha."""
import ast
import importlib
import re
from pathlib import Path

import voiceclonnx

SHA = re.compile(r"[0-9a-f]{40}")
PKG = Path(voiceclonnx.__file__).parent


def _revisions():
    """Yield (location, revision value) for each Hub download call in the package."""
    for path in PKG.rglob("*.py"):
        module = None
        for node in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "hf_hub_download"):
                continue
            rev = next(kw.value for kw in node.keywords if kw.arg == "revision")
            if isinstance(rev, ast.Constant):
                yield f"{path.name}:{node.lineno}", rev.value
                continue
            if not isinstance(rev, ast.Name):
                yield f"{path.name}:{node.lineno}", ast.unparse(rev)
                continue
            if module is None:
                name = ".".join(path.relative_to(PKG.parent).with_suffix("").parts)
                module = importlib.import_module(name)
            yield f"{path.name}:{node.lineno}", getattr(module, rev.id)


# An RVC voice is a user model named owner/repo[@revision]; the user picks its revision.
USER_REFERENCE = ("rvc.py", "revision or None")


def test_every_hub_download_pins_a_commit_sha():
    found = [(w, r) for w, r in _revisions() if (w.split(":")[0], r) != USER_REFERENCE]
    assert len(found) >= 10
    for where, rev in found:
        assert isinstance(rev, str) and SHA.fullmatch(rev), f"{where}: {rev!r}"
