"""Unit tests: quantized=True selects _q8 ONNX filenames for every engine.

Patches are applied at the library level (huggingface_hub.hf_hub_download,
onnxruntime.InferenceSession) because all engine adapters import these inside
_ensure_models() as deferred local imports.  No network access or real models
are required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _dl_filenames(dl_mock: MagicMock) -> list[str]:
    """Return the ``filename`` argument from each hf_hub_download call."""
    names = []
    for c in dl_mock.call_args_list:
        if "filename" in c.kwargs:
            names.append(c.kwargs["filename"])
        elif len(c.args) >= 2:
            names.append(c.args[1])
    return names


def _ort_session_factory():
    """Return a fresh MagicMock that looks like an ORT InferenceSession."""
    sess = MagicMock()
    # Default return value for sess.run — engines may check shapes
    sess.run.return_value = [np.zeros((1, 1, 1), dtype="float32")]
    return sess


# ---------------------------------------------------------------------------
# Shared patch context: hub + ort + numpy (for engines that load .npy files)
# ---------------------------------------------------------------------------

_COMMON_PATCHES = [
    patch("huggingface_hub.hf_hub_download", return_value="/tmp/fake.onnx"),
    patch("onnxruntime.InferenceSession", side_effect=lambda *a, **kw: _ort_session_factory()),
    patch("numpy.load", return_value=np.zeros((2, 80), dtype="float32")),
]


def _with_patches(fn):
    """Call *fn()* inside all common patches; return result."""
    ctx = __import__("contextlib").ExitStack()
    mocks = [ctx.enter_context(p) for p in _COMMON_PATCHES]
    with ctx:
        return fn(), mocks[0]  # (result, dl_mock)


# ---------------------------------------------------------------------------
# Per-engine helpers
# ---------------------------------------------------------------------------


def _bicodec_filenames(quantized: bool) -> list[str]:
    from vconnx.engines.bicodec import BiCodecAdapter

    def run():
        a = BiCodecAdapter(quantized=quantized)
        with patch("builtins.open", MagicMock(
            return_value=MagicMock(
                __enter__=lambda s, *a: MagicMock(
                    read=lambda: '{"n_fft":1024,"hop_length":320,"win_length":640}'
                ),
                __exit__=lambda *a: None,
            ),
        )):
            a._ensure_models()

    _, dl = _with_patches(run)
    return _dl_filenames(dl)


def _simple_filenames(adapter_cls, ensure_method: str = "_ensure_models") -> callable:
    """Return a factory that calls *adapter_cls(quantized=q)._ensure_models()."""
    def _fn(quantized: bool) -> list[str]:
        def run():
            a = adapter_cls(quantized=quantized)
            getattr(a, ensure_method)()
        _, dl = _with_patches(run)
        return _dl_filenames(dl)
    return _fn


# ---------------------------------------------------------------------------
# Tests: bicodec
# ---------------------------------------------------------------------------


class TestBicodecFilenames:
    def test_fp32_uses_no_q8(self):
        names = _bicodec_filenames(quantized=False)
        assert not any("q8" in n for n in names), f"unexpected q8 in {names}"

    def test_int8_uses_q8_for_all_onnx(self):
        names = _bicodec_filenames(quantized=True)
        onnx_names = [n for n in names if n.endswith(".onnx")]
        assert onnx_names, "no onnx files requested"
        assert all("q8" in n for n in onnx_names), f"not all ONNX q8: {onnx_names}"


# ---------------------------------------------------------------------------
# Tests: facodec
# ---------------------------------------------------------------------------


class TestFacodecFilenames:
    @staticmethod
    def _filenames(quantized: bool) -> list[str]:
        from vconnx.engines.facodec import FACodecAdapter
        return _simple_filenames(FACodecAdapter)(quantized)

    def test_fp32_uses_no_q8(self):
        assert not any("q8" in n for n in self._filenames(False))

    def test_int8_uses_q8_for_all_onnx(self):
        names = self._filenames(True)
        assert all("q8" in n for n in names), f"not all q8: {names}"


# ---------------------------------------------------------------------------
# Tests: focalcodec
# ---------------------------------------------------------------------------


class TestFocalcodecFilenames:
    @staticmethod
    def _filenames(quantized: bool) -> list[str]:
        from vconnx.engines.focalcodec import FocalCodecAdapter
        return _simple_filenames(FocalCodecAdapter)(quantized)

    def test_fp32_uses_no_q8(self):
        assert not any("q8" in n for n in self._filenames(False))

    def test_int8_uses_q8_for_all_onnx(self):
        names = self._filenames(True)
        assert all("q8" in n for n in names), f"not all q8: {names}"


# ---------------------------------------------------------------------------
# Tests: freevc
# ---------------------------------------------------------------------------


class TestFreevcFilenames:
    @staticmethod
    def _filenames(quantized: bool) -> list[str]:
        from vconnx.engines.freevc import FreeVCAdapter
        return _simple_filenames(FreeVCAdapter)(quantized)

    def test_fp32_uses_no_q8(self):
        names = self._filenames(False)
        assert not any("q8" in n for n in names)

    def test_int8_uses_q8_for_all_onnx(self):
        names = self._filenames(True)
        onnx_names = [n for n in names if n.endswith(".onnx")]
        assert all("q8" in n for n in onnx_names), f"not all q8: {onnx_names}"


# ---------------------------------------------------------------------------
# Tests: knnvc
# ---------------------------------------------------------------------------


class TestKnnvcFilenames:
    @staticmethod
    def _filenames(quantized: bool) -> list[str]:
        from vconnx.engines.knnvc import KNNVCAdapter
        return _simple_filenames(KNNVCAdapter)(quantized)

    def test_fp32_uses_no_q8(self):
        assert not any("q8" in n for n in self._filenames(False))

    def test_int8_uses_q8_for_all_onnx(self):
        names = self._filenames(True)
        assert all("q8" in n for n in names), f"not all q8: {names}"


# ---------------------------------------------------------------------------
# Tests: mimi
# ---------------------------------------------------------------------------


class TestMimiFilenames:
    @staticmethod
    def _filenames(quantized: bool) -> list[str]:
        from vconnx.engines.mimi import MimiAdapter
        return _simple_filenames(MimiAdapter)(quantized)

    def test_fp32_uses_no_q8(self):
        assert not any("q8" in n for n in self._filenames(False))

    def test_int8_uses_q8_for_all_onnx(self):
        names = self._filenames(True)
        assert all("q8" in n for n in names), f"not all q8: {names}"


# ---------------------------------------------------------------------------
# Tests: openvoice
# ---------------------------------------------------------------------------


class TestOpenvoiceFilenames:
    @staticmethod
    def _filenames(quantized: bool) -> list[str]:
        from vconnx.engines.openvoice import OpenVoiceV2Adapter
        return _simple_filenames(OpenVoiceV2Adapter)(quantized)

    def test_fp32_uses_no_q8(self):
        assert not any("q8" in n for n in self._filenames(False))

    def test_int8_uses_q8_for_all_onnx(self):
        names = self._filenames(True)
        assert all("q8" in n for n in names), f"not all q8: {names}"


# ---------------------------------------------------------------------------
# Tests: rvc
# ---------------------------------------------------------------------------


class TestRvcFilenames:
    @staticmethod
    def _filenames(quantized: bool) -> list[str]:
        from vconnx.engines.rvc import RVCAdapter
        return _simple_filenames(RVCAdapter, "_ensure_base_models")(quantized)

    def test_fp32_uses_no_q8(self):
        names = self._filenames(False)
        onnx_names = [n for n in names if n.endswith(".onnx")]
        assert onnx_names, f"expected onnx files, got: {names}"
        assert not any("q8" in n for n in onnx_names)

    def test_int8_uses_q8_base_models(self):
        names = self._filenames(True)
        onnx_names = [n for n in names if n.endswith(".onnx")]
        assert all("q8" in n for n in onnx_names), (
            f"expected q8 base models, got: {onnx_names}"
        )


# ---------------------------------------------------------------------------
# Tests: speechtokenizer
# ---------------------------------------------------------------------------


class TestSpeechtokenizerFilenames:
    @staticmethod
    def _filenames(quantized: bool) -> list[str]:
        from vconnx.engines.speechtokenizer import SpeechTokenizerAdapter
        return _simple_filenames(SpeechTokenizerAdapter)(quantized)

    def test_fp32_uses_no_q8(self):
        names = self._filenames(False)
        onnx_names = [n for n in names if n.endswith(".onnx")]
        assert not any("q8" in n for n in onnx_names)

    def test_int8_raises_not_implemented(self):
        """speechtokenizer INT8 export has incompatible interface — must raise NotImplementedError."""
        from vconnx.engines.speechtokenizer import SpeechTokenizerAdapter

        with pytest.raises(NotImplementedError, match="quantized=True is not supported"):
            SpeechTokenizerAdapter(quantized=True)


# ---------------------------------------------------------------------------
# Tests: triaan
# ---------------------------------------------------------------------------


class TestTriaanFilenames:
    @staticmethod
    def _filenames(quantized: bool) -> list[str]:
        from vconnx.engines.triaan import TriAANVCAdapter
        return _simple_filenames(TriAANVCAdapter)(quantized)

    def test_fp32_uses_no_q8(self):
        names = self._filenames(False)
        onnx_names = [n for n in names if n.endswith(".onnx")]
        assert not any("q8" in n for n in onnx_names)

    def test_int8_uses_q8_for_all_onnx(self):
        names = self._filenames(True)
        onnx_names = [n for n in names if n.endswith(".onnx")]
        assert all("q8" in n for n in onnx_names), f"not all q8: {onnx_names}"


# ---------------------------------------------------------------------------
# Tests: chatterbox — quantized=True accepted but silently ignored (fp32-only)
# ---------------------------------------------------------------------------


class TestChatterboxQuantizedIgnored:
    @staticmethod
    def _filenames(quantized: bool) -> list[str]:
        from vconnx.engines.chatterbox import ChatterboxAdapter
        return _simple_filenames(ChatterboxAdapter)(quantized)

    def test_fp32_no_q8_files(self):
        names = self._filenames(False)
        assert not any("q8" in n for n in names)

    def test_quantized_true_also_loads_fp32_only(self):
        """Even with quantized=True chatterbox loads fp32 (no q8 upstream)."""
        names = self._filenames(True)
        assert not any("q8" in n for n in names), (
            "chatterbox loaded q8 files — update this test if upstream ships INT8 exports"
        )

    def test_quantized_attr_stored(self):
        from vconnx.engines.chatterbox import ChatterboxAdapter

        a = ChatterboxAdapter(quantized=True)
        assert a._quantized is True, "quantized flag should be stored for API completeness"
