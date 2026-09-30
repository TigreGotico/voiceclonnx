"""Regression test: fp32 chatterbox models must load from real files, not
cache symlinks whose external-data sidecar resolves into a different shard
of a hash-sharded blob store.

Hugging Face Hub caches can shard their blob store by the first two hex
characters of the blob hash (``blobs/<xx>/<hash>``). A model's main ``.onnx``
and its ``.onnx_data`` sidecar hash independently, so they usually land in
different shard directories even though both are symlinked from the same
snapshot directory. onnxruntime's HF-cache symlink fallback only accepts a
sidecar that resolves under the *same* real directory as the main file, so it
rejects this layout as "External data path escapes model directory" even
though both files belong to the same model.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np

from voiceclonnx.engines.chatterbox import ChatterboxAdapter


def _make_external_data_onnx(path: Path, data_filename: str) -> None:
    """Write a tiny ONNX model whose sole initializer lives in an external file."""
    import onnx
    from onnx import TensorProto, helper

    weight = np.arange(2048, dtype=np.float32)
    initializer = helper.make_tensor("W", TensorProto.FLOAT, weight.shape, weight.tobytes(), raw=True)
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [2048])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [2048])
    node = helper.make_node("Add", inputs=["X", "W"], outputs=["Y"])
    graph = helper.make_graph([node], "tiny", [X], [Y], initializer=[initializer])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 14)])
    model.ir_version = 8

    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model, str(path),
        save_as_external_data=True, all_tensors_to_one_file=True, location=data_filename,
    )


def _build_sharded_cache(tmp_path: Path):
    """Lay out a shared, hash-sharded blob store (``blobs/<xx>/<hash>``) with
    snapshot symlinks pointing into it, a model and its ``.onnx_data`` sidecar
    landing in different shard directories -- the collision that trips
    onnxruntime's symlink fallback.
    """
    real_dir = tmp_path / "generated" / "onnx"
    _make_external_data_onnx(real_dir / "speech_encoder.onnx", "speech_encoder.onnx_data")
    _make_external_data_onnx(real_dir / "conditional_decoder.onnx", "conditional_decoder.onnx_data")

    blobs = tmp_path / "cache" / "blobs"
    snapshot = tmp_path / "cache" / "models--TigreGotico--voiceclonnx-chatterbox" / "snapshots" / "abc123" / "onnx"
    snapshot.mkdir(parents=True)

    # Fabricated hashes landing in different two-hex shards for a model and its
    # sidecar, matching a real observed collision ("36..." vs "e8...").
    hashes = {
        "speech_encoder.onnx": "36" + "a" * 62,
        "speech_encoder.onnx_data": "e8" + "b" * 62,
        "conditional_decoder.onnx": "17" + "c" * 62,
        "conditional_decoder.onnx_data": "9f" + "d" * 62,
    }
    cache_paths = {}
    for fname, blob_hash in hashes.items():
        shard = blobs / blob_hash[:2]
        shard.mkdir(parents=True, exist_ok=True)
        blob_path = shard / blob_hash
        blob_path.write_bytes((real_dir / fname).read_bytes())
        link_path = snapshot / fname
        link_path.symlink_to(blob_path)
        cache_paths[f"onnx/{fname}"] = str(link_path)
    return cache_paths, real_dir


def _fake_hf_hub_download(cache_paths, real_dir):
    """Mirror huggingface_hub: without ``local_dir`` return the (symlinked)
    cache path; with ``local_dir`` write a real file, as ``local_dir`` does.
    """

    def _download(repo_id, filename, local_dir=None, **kw):
        if local_dir is not None:
            dst = Path(local_dir) / filename
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes((real_dir / Path(filename).name).read_bytes())
            return str(dst)
        return cache_paths[filename]

    return _download


def test_ensure_models_loads_past_a_sharded_blob_store(tmp_path):
    """clone_voice must load fp32 chatterbox models even when the HF cache
    shards blobs by hash prefix, landing a model and its external-data
    sidecar in different shard directories."""
    cache_paths, real_dir = _build_sharded_cache(tmp_path)

    with patch("huggingface_hub.hf_hub_download", side_effect=_fake_hf_hub_download(cache_paths, real_dir)):
        adapter = ChatterboxAdapter()
        adapter._ensure_models()

    assert adapter._speech_enc_sess is not None
    assert adapter._cond_dec_sess is not None
