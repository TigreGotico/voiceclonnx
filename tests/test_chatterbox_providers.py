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


def test_failed_decoder_load_leaves_adapter_uninitialised():
    """A decoder that fails to load must not leave the encoder set.

    Otherwise `_ensure_models`' early-return guard short-circuits every retry while the decoder
    is still None, and each clone dies on `NoneType.run` rather than re-raising the real error.
    """
    a = ChatterboxAdapter(quantized=True)
    calls = []

    def _sess(path, **kw):
        calls.append(path)
        if len(calls) == 2:            # the decoder
            raise RuntimeError("simulated session init failure (e.g. OOM)")
        return MagicMock()

    with patch("huggingface_hub.hf_hub_download", return_value="/tmp/x.onnx"), \
         patch("onnxruntime.InferenceSession", side_effect=_sess):
        with pytest.raises(RuntimeError, match="simulated session init failure"):
            a._ensure_models()

    assert a._speech_enc_sess is None, "encoder must not be published when the decoder fails"
    assert a._cond_dec_sess is None

    # a retry must genuinely retry both sessions, not short-circuit
    with patch("huggingface_hub.hf_hub_download", return_value="/tmp/x.onnx"), \
         patch("onnxruntime.InferenceSession", return_value=MagicMock()) as ok:
        a._ensure_models()
    assert ok.call_count == 2
    assert a._speech_enc_sess is not None and a._cond_dec_sess is not None
