"""Chatterbox execution-provider selection (no weights required).

The adapter constructs lazily — sessions load on first use — so the provider plumbing can be
checked without downloading the ONNX models.
"""
import pytest
from unittest.mock import MagicMock, patch

from voiceclonnx.engines.chatterbox import ChatterboxAdapter


def test_defaults_to_cpu():
    assert ChatterboxAdapter()._providers is None


def test_stores_explicit_providers():
    gpu = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert ChatterboxAdapter(providers=gpu)._providers == gpu


def _load_with_mocks(adapter):
    # onnxruntime + hf_hub_download are imported inside _ensure_models, so patch the source
    # modules, not local names.
    with patch("huggingface_hub.hf_hub_download", return_value="/tmp/x.onnx"), \
         patch("onnxruntime.InferenceSession") as sess:
        adapter._ensure_models()
    return sess


def test_providers_are_passed_to_the_sessions():
    """The configured providers must reach both InferenceSession calls."""
    gpu = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    sess = _load_with_mocks(ChatterboxAdapter(quantized=True, providers=gpu))
    assert sess.call_count == 2
    for call in sess.call_args_list:
        assert call.kwargs["providers"] == gpu


def test_default_passes_cpu_provider():
    sess = _load_with_mocks(ChatterboxAdapter(quantized=True))
    for call in sess.call_args_list:
        assert call.kwargs["providers"] == ["CPUExecutionProvider"]
