"""Make the ``conversion`` development toolchain importable to the tests.

``conversion/`` is the dev-time export and validation toolchain. Its own
``__init__`` records that the runtime package never imports it, and the wheel
does not ship it: ``packages.find`` includes only ``voiceclonnx*``. Seven test
modules still test it, and they import it as ``from conversion.export_base
import ...``.

That name resolves under pytest's default ``prepend`` import mode only because
the mode puts the repository root at ``sys.path[0]`` -- which also lets the
checkout shadow the installed ``voiceclonnx``, so the suite can stop testing the
artefact it ships. Under ``--import-mode=importlib`` neither the root nor the
test directory is on ``sys.path`` and the name does not resolve at all.

So load the package from its path and register it, rather than put a directory
on ``sys.path``. Its internal absolute imports (``conversion.parity``,
``conversion.quantize``) resolve through the normal machinery because the parent
package is in ``sys.modules`` and carries its own search location. Nothing joins
``sys.path``, so the installed ``voiceclonnx`` stays the one under test.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_CONVERSION = Path(__file__).resolve().parent.parent / "conversion"

if "conversion" not in sys.modules and _CONVERSION.is_dir():
    _spec = importlib.util.spec_from_file_location(
        "conversion",
        _CONVERSION / "__init__.py",
        submodule_search_locations=[str(_CONVERSION)],
    )
    _module = importlib.util.module_from_spec(_spec)
    # Registered before exec_module, so a submodule importing its own parent
    # mid-execution finds it rather than starting a second copy.
    sys.modules["conversion"] = _module
    _spec.loader.exec_module(_module)
