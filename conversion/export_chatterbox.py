"""Chatterbox VC ONNX quantization script for vconnx.

Downloads fp32 ONNX artifacts from ``onnx-community/chatterbox-onnx`` (Apache-2.0),
applies INT8 dynamic quantization, and uploads all four files to
``TigreGotico/vconnx-chatterbox``.

No PyTorch dependency — this is a pure quantization pass (onnx + onnxruntime only).
The fp32 models are re-hosted verbatim from the upstream onnx-community repo.

Usage
-----
::

    python -m conversion.export_chatterbox --output-dir /tmp/cb-out [--no-push]

Quantization notes
------------------
``speech_encoder.onnx`` contains two ``Gemm(transB=1)`` nodes in the S3 VQ
codebook (``project_down``).  ORT's ``quantize_dynamic`` preprocessor decomposes
``Gemm`` into ``MatMul + Add`` *without* transposing the weight — a known ORT
preprocessing bug that corrupts the resulting MatMul graph and causes an
``[ShapeInferenceError] Incompatible dimensions`` failure at session-load time.

The fix applied here: pre-transpose the ``project_down.weight`` initializer and
rewrite the two ``Gemm`` nodes as ``MatMul + Add`` *before* invoking
``quantize_dynamic``.  This makes the graph compatible with ORT's MatMul
quantizer while preserving numerical identity with the original Gemm.

``conditional_decoder.onnx`` contains 20 ``If`` nodes with subgraph nodes.
Quantizing MatMul ops *inside* those ``If`` subgraphs causes ORT's session
initializer to hang.  The subgraph node names are enumerated and passed to
``nodes_to_exclude`` so only main-graph MatMul ops are quantized.
"""

from __future__ import annotations

import argparse
import copy
import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _download_fp32(output_dir: str) -> tuple[str, str]:
    """Download fp32 models from onnx-community/chatterbox-onnx.

    Returns ``(enc_onnx_path, dec_onnx_path)``.
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("pip install huggingface_hub") from exc

    src_repo = "onnx-community/chatterbox-onnx"
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    files = [
        "onnx/speech_encoder.onnx",
        "onnx/speech_encoder.onnx_data",
        "onnx/conditional_decoder.onnx",
        "onnx/conditional_decoder.onnx_data",
    ]

    cached = {}
    for f in files:
        print(f"  Downloading {f}...")
        cached[f] = hf_hub_download(repo_id=src_repo, filename=f)
        print(f"    -> {cached[f]}")

    return cached["onnx/speech_encoder.onnx"], cached["onnx/conditional_decoder.onnx"]


def _patch_encoder_gemm(onnx_path: str, patched_path: str) -> None:
    """Rewrite Gemm(transB=1) → MatMul+Add with transposed weight in-place.

    ORT's quantize_dynamic preprocessing decomposes ``Gemm(transB=1)`` into
    ``MatMul(A, B) + bias`` without transposing B.  This produces a graph where
    ``MatMul((N,K), (8,K))`` has mismatched K dimensions, causing a load-time
    ``ShapeInferenceError``.

    The fix: pre-transpose B in the initializer and emit a plain ``MatMul(A, B_T)``
    so that the shape is correct before quantize_dynamic sees the graph.
    """
    import onnx
    from onnx import numpy_helper, helper

    print("  Patching Gemm(transB=1) nodes in speech_encoder...")
    model = onnx.load(onnx_path)

    gemm_nodes = [n for n in model.graph.node if n.op_type == "Gemm"]
    if not gemm_nodes:
        # Nothing to patch — save as-is
        import shutil
        shutil.copy2(onnx_path, patched_path)
        return

    init_map = {init.name: init for init in model.graph.initializer}
    extra_inits: list = []
    new_nodes: list = []

    for n in model.graph.node:
        if n.op_type != "Gemm":
            new_nodes.append(n)
            continue

        trans_b = next((a.i for a in n.attribute if a.name == "transB"), 0)
        A, B, bias = n.input[0], n.input[1], n.input[2]
        out = n.output[0]

        if trans_b and B in init_map:
            B_arr = numpy_helper.to_array(init_map[B])
            B_T_name = B + "_T"
            B_T_init = numpy_helper.from_array(B_arr.T, name=B_T_name)
            extra_inits.append(B_T_init)
            mm_out = out + "_mm"
            new_nodes.append(helper.make_node("MatMul", [A, B_T_name], [mm_out]))
            new_nodes.append(helper.make_node("Add", [mm_out, bias], [out]))
            print(f"    Replaced {n.name}: weight {B_arr.shape} -> {B_arr.T.shape}")
        else:
            new_nodes.append(n)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(extra_inits)

    import onnx
    onnx.save(model, patched_path)
    print(f"    Saved patched encoder: {patched_path}")


def _collect_if_subgraph_nodes(onnx_path: str) -> List[str]:
    """Return names of all nodes inside ``If`` subgraphs."""
    import onnx

    model = onnx.load(onnx_path, load_external_data=False)
    names: list[str] = []
    for n in model.graph.node:
        if n.op_type == "If":
            for attr in n.attribute:
                if attr.type == 5:  # GRAPH
                    for subnode in attr.g.node:
                        names.append(subnode.name)
    return names


def _quantize(
    input_path: str,
    output_path: str,
    nodes_to_exclude: Optional[List[str]] = None,
) -> None:
    """Apply INT8 dynamic quantization (MatMul ops only)."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print(f"  Quantizing {Path(input_path).name} -> {Path(output_path).name}...")
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
        op_types_to_quantize=["MatMul"],
        nodes_to_exclude=nodes_to_exclude or [],
    )
    size_mb = os.path.getsize(output_path) / 1024 ** 2
    print(f"    Done: {size_mb:.1f} MB")


def _validate(
    enc_fp32: str,
    enc_q8: str,
    dec_fp32: str,
    dec_q8: str,
    sample_audio: Optional[str] = None,
) -> None:
    """Smoke-test that q8 sessions load and produce outputs with correct shapes."""
    import onnxruntime as ort

    print("  Validating q8 models...")
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 4
    opts.intra_op_num_threads = 4

    enc_sess = ort.InferenceSession(enc_q8, sess_options=opts)
    dec_sess = ort.InferenceSession(dec_q8, sess_options=opts)

    # Synthetic 1-second test input
    audio = np.random.randn(1, 24000).astype(np.float32)
    _, tokens, x_vec, feat = enc_sess.run(None, {"audio_values": audio})
    speech_tokens = np.concatenate([tokens, tokens], axis=1)
    wav = dec_sess.run(None, {
        "speech_tokens": speech_tokens,
        "speaker_embeddings": x_vec,
        "speaker_features": feat,
    })[0]
    assert wav.ndim == 2 and wav.shape[0] == 1, f"unexpected waveform shape: {wav.shape}"
    print(f"    Encoder q8: tokens={tokens.shape}, x_vec={x_vec.shape}")
    print(f"    Decoder q8: waveform={wav.shape}")
    print("    Validation PASSED")


def _push(output_dir: str, dst_repo: str, commit_msg: str) -> None:
    """Upload all ONNX artifacts to HF Hub."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id=dst_repo, repo_type="model", private=True, exist_ok=True)

    out = Path(output_dir)
    for fpath in sorted(out.rglob("*.onnx")) + sorted(out.rglob("*.onnx_data")) + sorted(out.glob("*.md")):
        relative = fpath.relative_to(out)
        print(f"  Uploading {relative} ({fpath.stat().st_size/1024/1024:.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(fpath),
            path_in_repo=str(relative),
            repo_id=dst_repo,
            repo_type="model",
            commit_message=commit_msg,
        )
    print(f"  Pushed to https://huggingface.co/{dst_repo}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Chatterbox VC quantization for vconnx.")
    p.add_argument("--output-dir", default="/tmp/vconnx-chatterbox", help="Local staging dir.")
    p.add_argument("--no-push", action="store_true", help="Skip HF Hub upload.")
    p.add_argument("--dst-repo", default="TigreGotico/vconnx-chatterbox")
    args = p.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    onnx_dir = out / "onnx"
    onnx_dir.mkdir(exist_ok=True)

    print("Step 1: Downloading fp32 models from onnx-community/chatterbox-onnx...")
    enc_fp32_cached, dec_fp32_cached = _download_fp32(args.output_dir)

    print("\nStep 2: Patching speech_encoder Gemm nodes...")
    enc_patched = str(onnx_dir / "speech_encoder_patched.onnx")
    _patch_encoder_gemm(enc_fp32_cached, enc_patched)

    print("\nStep 3: Quantizing speech_encoder...")
    enc_q8 = str(onnx_dir / "speech_encoder_q8.onnx")
    _quantize(enc_patched, enc_q8)
    os.unlink(enc_patched)

    print("\nStep 4: Collecting decoder If-subgraph nodes to exclude...")
    dec_exclude = _collect_if_subgraph_nodes(dec_fp32_cached)
    print(f"  Excluding {len(dec_exclude)} If-subgraph nodes from quantization")

    print("\nStep 5: Quantizing conditional_decoder...")
    dec_q8 = str(onnx_dir / "conditional_decoder_q8.onnx")
    _quantize(dec_fp32_cached, dec_q8, nodes_to_exclude=dec_exclude)

    print("\nStep 6: Validating q8 models...")
    _validate(enc_fp32_cached, enc_q8, dec_fp32_cached, dec_q8)

    if not args.no_push:
        print(f"\nStep 7: Uploading to {args.dst_repo}...")
        _push(args.output_dir, args.dst_repo, "export: quantized chatterbox VC ONNX artifacts")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
