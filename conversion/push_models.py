"""Upload exported ONNX artifacts to Hugging Face — one PUBLIC repo per engine.

Each engine lives in its own subdirectory on the Hub:
``TigreGotico/vconnx-<engine>`` (public), grouped in the vconnx collection.

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

HF_NAMESPACE = "TigreGotico"
COLLECTION_SLUG = "TigreGotico/vconnx-pure-onnx-voice-conversion-6a2ac089852d9b90a66c4509"


def engine_repo_id(engine_name: str) -> str:
    return f"{HF_NAMESPACE}/vconnx-{engine_name}"


# ---------------------------------------------------------------------------
# HF helpers
# ---------------------------------------------------------------------------


def _ensure_repo(repo_id: str, token: str, dry_run: bool) -> None:
    """Create the public per-engine HF repo if it doesn't exist."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    repo_exists = False
    try:
        api.repo_info(repo_id=repo_id, repo_type="model")
        repo_exists = True
        print(f"[push] HF repo {repo_id!r} exists.")
    except Exception:
        pass  # repo not found or other transient error

    if repo_exists:
        return

    if dry_run:
        print(f"[dry-run] Would create PUBLIC HF repo {repo_id!r}.")
        return

    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=False,
        exist_ok=True,
    )
    print(f"[push] Created PUBLIC HF repo {repo_id!r}.")
    try:
        from huggingface_hub import add_collection_item
        add_collection_item(COLLECTION_SLUG, item_id=repo_id,
                            item_type="model", exists_ok=True)
        print(f"[push] Added {repo_id!r} to the vconnx collection.")
    except Exception as exc:  # collection add is best-effort
        print(f"[push] Could not add to collection: {exc}")


def push_engine(
    engine_dir: Union[str, Path],
    engine_name: str,
    token: Optional[str] = None,
    dry_run: bool = False,
    commit_message: Optional[str] = None,
) -> None:
    """Upload *engine_dir* to the public ``TigreGotico/vconnx-<engine_name>`` repo.

    Parameters
    ----------
    engine_dir:
        Local directory containing ONNX files + PROVENANCE.md + config.json.
    engine_name:
        Engine name; the target repo becomes ``TigreGotico/vconnx-<name>``.
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
    repo_id = engine_repo_id(engine_name)
    if dry_run:
        print(f"[dry-run] Would ensure PUBLIC HF repo {repo_id!r} exists.")
    else:
        _ensure_repo(repo_id, tok, dry_run=False)

    # ------------------------------------------------------------------
    # Enumerate files and (dry-run) report or upload
    # ------------------------------------------------------------------
    files = sorted(engine_dir.rglob("*"))
    files = [f for f in files if f.is_file()]

    if dry_run:
        print(f"[dry-run] Would upload {len(files)} file(s) to {repo_id}:")
        for f in files:
            rel = f.relative_to(engine_dir)
            print(f"  {f}  →  {rel}")
        return

    from huggingface_hub import HfApi

    api = HfApi(token=tok)
    msg = commit_message or f"export: upload {engine_name} ONNX artifacts"
    api.upload_folder(
        folder_path=str(engine_dir),
        repo_id=repo_id,
        repo_type="model",
        commit_message=msg,
    )
    print(f"[push] Uploaded {len(files)} file(s) to {repo_id}.")


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
