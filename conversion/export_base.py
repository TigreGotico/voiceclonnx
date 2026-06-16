"""Shared helpers for exporting PyTorch voice-conversion models to ONNX.

This module is **dev-time only** — it requires ``torch`` and ``onnx``.
Neither is a runtime dependency of ``voiceclonnx``.

Typical usage
-------------
::

    from conversion.export_base import export_model, write_manifest, OutputLayout

    layout = OutputLayout.for_engine("knn-vc", base_dir="/tmp/out")
    layout.makedirs()

    export_model(
        model=my_torch_module,
        dummy_inputs=(torch.zeros(1, 512),),
        output_path=layout.component_path("encoder.onnx"),
        input_names=["features"],
        output_names=["embedding"],
        dynamic_axes={"features": {0: "batch"}, "embedding": {0: "batch"}},
    )

    write_manifest(
        layout=layout,
        components={"encoder": "encoder.onnx"},
        sample_rates={"output": 16000},
        metadata={"opset": 14},
    )
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default ONNX opset used across all exports — can be overridden per call.
DEFAULT_OPSET: int = 14

#: Filename for the per-engine manifest written alongside the ONNX files.
MANIFEST_FILENAME: str = "config.json"


# ---------------------------------------------------------------------------
# Output layout
# ---------------------------------------------------------------------------


@dataclass
class OutputLayout:
    """Directory layout for one exported engine.

    Parameters
    ----------
    engine_name:
        Short engine identifier, e.g. ``"knn-vc"`` or ``"rvc"``.
    base_dir:
        Root staging directory.  Engine files land in ``base_dir/engine_name/``.
    """

    engine_name: str
    base_dir: Path

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir)

    # ------------------------------------------------------------------
    # Class-method constructors
    # ------------------------------------------------------------------

    @classmethod
    def for_engine(cls, engine_name: str, base_dir: Union[str, Path]) -> "OutputLayout":
        """Return a layout rooted at *base_dir/engine_name*."""
        return cls(engine_name=engine_name, base_dir=Path(base_dir))

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    @property
    def engine_dir(self) -> Path:
        """Directory that holds all files for this engine."""
        return self.base_dir / self.engine_name

    def component_path(self, filename: str) -> Path:
        """Absolute path to *filename* inside the engine directory."""
        return self.engine_dir / filename

    def manifest_path(self) -> Path:
        return self.component_path(MANIFEST_FILENAME)

    def makedirs(self) -> None:
        """Create the engine output directory (and parents) if needed."""
        self.engine_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def write_manifest(
    layout: OutputLayout,
    components: Dict[str, str],
    sample_rates: Dict[str, int],
    metadata: Optional[Dict[str, Any]] = None,
    distributable: bool = True,
) -> Path:
    """Write a ``config.json`` manifest next to the ONNX files.

    Parameters
    ----------
    layout:
        The output layout; ``layout.engine_dir`` must already exist.
    components:
        Mapping of logical name → filename, e.g.
        ``{"encoder": "encoder.onnx", "vocoder": "vocoder.onnx"}``.
    sample_rates:
        Mapping of role → sample rate in Hz, e.g. ``{"output": 16000}``.
    metadata:
        Optional free-form key/value pairs embedded in the manifest.
    distributable:
        Informational flag recording the upstream weight-license situation.
        All exports are published either way — the model card states the
        upstream license and downstream users judge fit for their use case.

    Returns
    -------
    Path
        Path to the written manifest file.
    """
    manifest: Dict[str, Any] = {
        "engine": layout.engine_name,
        "components": components,
        "sample_rates": sample_rates,
        "distributable": distributable,
    }
    if metadata:
        manifest["metadata"] = metadata

    path = layout.manifest_path()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return path


def read_manifest(engine_dir: Union[str, Path]) -> Dict[str, Any]:
    """Load and return the ``config.json`` manifest from *engine_dir*."""
    path = Path(engine_dir) / MANIFEST_FILENAME
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# torch → ONNX export wrapper
# ---------------------------------------------------------------------------


def export_model(
    model: Any,
    dummy_inputs: Union[Tuple[Any, ...], Any],
    output_path: Union[str, Path],
    input_names: List[str],
    output_names: List[str],
    dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
    opset_version: int = DEFAULT_OPSET,
    export_params: bool = True,
    external_data_threshold_bytes: int = 2 * 1024 ** 3,
) -> Path:
    """Export a PyTorch module to ONNX, handling the >2 GB external-data case.

    Parameters
    ----------
    model:
        A ``torch.nn.Module`` already in ``eval()`` mode.
    dummy_inputs:
        A tuple of dummy tensors (or a single tensor) that match the model's
        ``forward`` signature.
    output_path:
        Destination ``.onnx`` file path.  Parent directory must exist.
    input_names:
        Ordered list of input tensor names.
    output_names:
        Ordered list of output tensor names.
    dynamic_axes:
        ONNX dynamic-axes dict, e.g.
        ``{"input": {0: "batch"}, "output": {0: "batch"}}``.
    opset_version:
        ONNX opset.  Defaults to :data:`DEFAULT_OPSET`.
    export_params:
        Whether to embed weights in the ONNX file (default ``True``).
    external_data_threshold_bytes:
        When the serialised model is larger than this, ``save_as_external_data``
        is enabled automatically.  Default 2 GiB.

    Returns
    -------
    Path
        Absolute path to the written ``.onnx`` file.
    """
    import torch  # local import — export-time only

    output_path = Path(output_path)

    if not isinstance(dummy_inputs, tuple):
        dummy_inputs = (dummy_inputs,)

    # Check approximate size via state dict to decide on external data.
    total_params = sum(p.numel() * p.element_size() for p in model.parameters())
    use_external = total_params > external_data_threshold_bytes

    # Use legacy TorchScript-based export (dynamo=False) for compatibility with
    # models that have data-dependent shapes in torchaudio / WavLM internals.
    torch.onnx.export(
        model,
        dummy_inputs,
        str(output_path),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes or {},
        opset_version=opset_version,
        export_params=export_params,
        dynamo=False,
    )

    if use_external:
        # Re-save with external-data flag so the .onnx header stays <2 GB.
        try:
            import onnx
            from onnx.external_data_helper import convert_model_to_external_data

            loaded = onnx.load(str(output_path))
            convert_model_to_external_data(
                loaded,
                all_tensors_to_one_file=True,
                location=output_path.name + ".data",
            )
            onnx.save(loaded, str(output_path))
        except ImportError:
            pass  # onnx package not installed — skip external-data step

    return output_path.resolve()


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def _git_sha(repo_path: Optional[str] = None) -> str:
    """Return the current HEAD SHA of *repo_path* (or the voiceclonnx repo)."""
    try:
        cmd = ["git", "rev-parse", "HEAD"]
        if repo_path:
            cmd = ["git", "-C", repo_path, "rev-parse", "HEAD"]
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def write_provenance(
    engine_dir: Union[str, Path],
    upstream_repo_url: str,
    upstream_ref: str,
    license_text: Optional[str] = None,
    extra: Optional[Dict[str, str]] = None,
) -> Path:
    """Write a PROVENANCE.md file recording export lineage.

    Parameters
    ----------
    engine_dir:
        Directory where PROVENANCE.md will be written.
    upstream_repo_url:
        URL of the upstream model repository.
    upstream_ref:
        Upstream commit hash or tag that was used.
    license_text:
        Contents of the upstream model's licence file (embedded verbatim).
    extra:
        Additional free-form fields (key: value) appended to the document.
    """
    import datetime
    import platform

    try:
        import torch
        torch_ver = torch.__version__
    except ImportError:
        torch_ver = "n/a"

    try:
        import onnx
        onnx_ver = onnx.__version__
    except ImportError:
        onnx_ver = "n/a"

    engine_dir = Path(engine_dir)
    export_sha = _git_sha(str(Path(__file__).parent))
    now = datetime.datetime.utcnow().isoformat(timespec="seconds") + "Z"

    lines = [
        "# PROVENANCE",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| upstream_repo | {upstream_repo_url} |",
        f"| upstream_ref | {upstream_ref} |",
        f"| export_script_sha | {export_sha} |",
        f"| export_date | {now} |",
        f"| torch_version | {torch_ver} |",
        f"| onnx_version | {onnx_ver} |",
        f"| platform | {platform.platform()} |",
    ]

    if extra:
        for k, v in extra.items():
            lines.append(f"| {k} | {v} |")

    if license_text:
        lines += ["", "## Upstream licence", "", "```", license_text.strip(), "```"]

    path = engine_dir / "PROVENANCE.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
