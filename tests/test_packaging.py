"""The wheel carries the ``voiceclonnx`` package and no directory beside it."""

from __future__ import annotations

from pathlib import Path

from setuptools.config.pyprojecttoml import read_configuration

_ROOT = Path(__file__).resolve().parent.parent


def test_wheel_packages_are_only_the_voiceclonnx_tree():
    config = read_configuration(_ROOT / "pyproject.toml", expand=True)
    packages = set(config["tool"]["setuptools"]["packages"])

    expected = {
        ".".join(init.parent.relative_to(_ROOT).parts)
        for init in (_ROOT / "voiceclonnx").rglob("__init__.py")
    }
    assert packages == expected
