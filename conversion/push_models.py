"""Upload exported ONNX artifacts to ``TigreGotico/vconnx-models`` on Hugging Face.

Each engine lives in its own subdirectory on the Hub:
``TigreGotico/vconnx-models/<engine>/``.

Requirements
------------
- ``HF_TOKEN`` env var (write access to the TigreGotico org).
- ``huggingface_hub`` package (already a runtime dep of vconnx).

Usage
-----
Command-line (dry-run — prints what would be uploaded)::

    python -m conversion.push_models path/to/out/knn-vc --engine knn-vc --dry-run

Actual upload::

    HF_TOKEN=hf_... python -m conversion.push_models path/to/out/knn-vc --engine knn-vc

Programmatic::

    from conversion.push_models import push_engine

    push_engine(engine_dir="out/knn-vc", engine_name="knn-vc", dry_run=True)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, Union

HF_REPO_ID = "TigreGotico/vconnx-models"


# ---------------------------------------------------------------------------
# HF helpers
# ---------------------------------------------------------------------------


def _ensure_repo(token: str, dry_run: bool) -> None:
    """Create the HF repo if it doesn't exist (private by default)."""
    from huggingface_hub import HfApi, RepositoryNotFoundError

    api = HfApi(token=token)
    try:
        api.repo_info(repo_id=HF_REPO_ID, repo_type="model")
        print(f"[push] HF repo {HF_REPO_ID!r} exists.")
    except RepositoryNotFoundError:
        if dry_run:
            print(f"[dry-run] Would create PRIVATE HF repo {HF_REPO_ID!r}.")
            return
        api.create_repo(
            repo_id=HF_REPO_ID,
            repo_type="model",
            private=True,
            exist_ok=True,
        )
        print(f"[push] Created PRIVATE HF repo {HF_REPO_ID!r}.")


def push_engine(
    engine_dir: Union[str, Path],
    engine_name: str,
    token: Optional[str] = None,
    dry_run: bool = False,
    commit_message: Optional[str] = None,
) -> None:
    """Upload *engine_dir* to ``TigreGotico/vconnx-models/<engine_name>/``.

    Parameters
    ----------
    engine_dir:
        Local directory containing ONNX files + PROVENANCE.md + config.json.
    engine_name:
        Target subdirectory name on the Hub (e.g. ``"knn-vc"``).
    token:
        HF write token.  Falls back to the ``HF_TOKEN`` environment variable.
    dry_run:
        If ``True``, print what would be uploaded without touching HF.
    commit_message:
        Optional commit message for the Hub upload.
    """

    engine_dir = Path(engine_dir)
    manifest_file = engine_dir / "config.json"
    if manifest_file.is_file():
        import json
        if json.loads(manifest_file.read_text()).get("distributable", True) is False:
            raise RuntimeError(
                f"{engine_name}: manifest is marked distributable=false "
                "(local-only weights — upstream license does not allow "
                "redistribution). Refusing to upload; this engine's models "
                "stay on the machine that converted them."
            )
    engine_dir = Path(engine_dir)

    if not engine_dir.is_dir():
        raise FileNotFoundError(f"Engine directory not found: {engine_dir}")

    tok = token or os.environ.get("HF_TOKEN", "")

    if not tok:
        if dry_run:
            tok = "dry-run-no-token"
        else:
            raise EnvironmentError(
                "HF_TOKEN is required for upload.  "
                "Set the environment variable or pass --token."
            )

    # ------------------------------------------------------------------
    # Auth check
    # ------------------------------------------------------------------
    if not dry_run:
        from huggingface_hub import HfApi

        api = HfApi(token=tok)
        try:
            user = api.whoami()
            print(f"[push] Authenticated as {user.get('name', '?')!r}.")
        except Exception as exc:
            raise EnvironmentError(f"HF auth failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Ensure repo exists
    # ------------------------------------------------------------------
    if dry_run:
        print(f"[dry-run] Would ensure HF repo {HF_REPO_ID!r} exists (private).")
    else:
        _ensure_repo(tok, dry_run=False)

    # ------------------------------------------------------------------
    # Enumerate files and (dry-run) report or upload
    # ------------------------------------------------------------------
    files = sorted(engine_dir.rglob("*"))
    files = [f for f in files if f.is_file()]

    path_in_repo_prefix = engine_name

    if dry_run:
        print(f"[dry-run] Would upload {len(files)} file(s) to {HF_REPO_ID}/{path_in_repo_prefix}:")
        for f in files:
            rel = f.relative_to(engine_dir)
            print(f"  {f}  →  {path_in_repo_prefix}/{rel}")
        return

    from huggingface_hub import HfApi

    api = HfApi(token=tok)
    msg = commit_message or f"export: upload {engine_name} ONNX artifacts"
    api.upload_folder(
        folder_path=str(engine_dir),
        repo_id=HF_REPO_ID,
        repo_type="model",
        path_in_repo=path_in_repo_prefix,
        commit_message=msg,
    )
    print(f"[push] Uploaded {len(files)} file(s) to {HF_REPO_ID}/{path_in_repo_prefix}.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Upload vconnx ONNX artifacts to HF.")
    p.add_argument("engine_dir", help="Local engine directory (output of export_base).")
    p.add_argument("--engine", required=True, help="Engine name (HF subdirectory).")
    p.add_argument("--token", default=None, help="HF write token (default: $HF_TOKEN).")
    p.add_argument("--dry-run", action="store_true", help="Print files without uploading.")
    p.add_argument("--message", default=None, help="HF commit message.")
    return p


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    push_engine(
        engine_dir=args.engine_dir,
        engine_name=args.engine,
        token=args.token,
        dry_run=args.dry_run,
        commit_message=args.message,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
