"""Every engine module must self-register on package import.

Guards against the recurring bug where a new adapter module exists but is
missing from the auto-import list in ``vconnx/__init__.py`` (mock tests import
adapter modules directly, which hides the gap).
"""
from pathlib import Path

ENGINES_DIR = Path(__file__).resolve().parent.parent / "vconnx" / "engines"
NON_ENGINE_MODULES = {"__init__", "base"}


def test_every_engine_module_is_registered():
    import vconnx  # noqa: F401 — triggers auto-imports
    from vconnx.engines.base import ENGINE_REGISTRY

    modules = {p.stem for p in ENGINES_DIR.glob("*.py")} - NON_ENGINE_MODULES
    registered_modules = {
        entry.adapter_class.__module__.rsplit(".", 1)[-1]
        for entry in ENGINE_REGISTRY.values()
    }
    missing = modules - registered_modules
    assert not missing, (
        f"engine modules without a registry entry (add the auto-import to "
        f"vconnx/__init__.py): {sorted(missing)}"
    )
