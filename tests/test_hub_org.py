"""Every Hub repo id constant in the engines points at the voiceclonnx org."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ID_NAMES = {"_HF_REPO_ID", "_HF_MODEL_ID", "_HF_BASE_REPO"}


def _repo_ids():
    ids = {}
    for path in sorted((ROOT / "voiceclonnx" / "engines").glob("*.py")):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ID_NAMES:
                        ids[path.stem] = node.value.value
    return ids


def test_repo_ids_use_new_org():
    ids = _repo_ids()
    assert len(ids) == 10
    for engine, repo_id in ids.items():
        assert repo_id.startswith("voiceclonnx/"), (engine, repo_id)
        assert "TigreGotico/voiceclonnx-" not in repo_id, (engine, repo_id)


def test_push_models_targets_new_org():
    tree = ast.parse((ROOT / "conversion" / "push_models.py").read_text())
    consts = {
        t.id: n.value.value
        for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
        for t in n.targets if isinstance(t, ast.Name)
    }
    assert consts["HF_NAMESPACE"] == "voiceclonnx"
