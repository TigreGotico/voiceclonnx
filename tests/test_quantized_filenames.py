"""Unit tests: quantized=True selects _q8 ONNX filenames for every engine.

Patches are applied at the library level (huggingface_hub.hf_hub_download,
onnxruntime.InferenceSession) because all engine adapters import these inside
_ensure_models() as deferred local imports.  No network access or real models
are required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np


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
    from voiceclonnx.engines.bicodec import BiCodecAdapter

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
        from voiceclonnx.engines.facodec import FACodecAdapter
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
        from voiceclonnx.engines.focalcodec import FocalCodecAdapter
        return _simple_filenames(FocalCodecAdapter)(quantized)

    def test_fp32_uses_no_q8(self):
        assert not any("q8" in n for n in self._filenames(False))

    def test_int8_uses_q8_for_all_onnx(self):
        names = self._filenames(True)
        assert all("q8" in n for n in names), f"not all q8: {names}"


# ---------------------------------------------------------------------------
# Tests: knnvc
# ---------------------------------------------------------------------------


class TestKnnvcFilenames:
    @staticmethod
    def _filenames(quantized: bool) -> list[str]:
        from voiceclonnx.engines.knnvc import KNNVCAdapter
        return _simple_filenames(KNNVCAdapter)(quantized)

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
        from voiceclonnx.engines.openvoice import OpenVoiceV2Adapter
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
        from voiceclonnx.engines.rvc import RVCAdapter
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
# Tests: triaan
# ---------------------------------------------------------------------------


class TestTriaanFilenames:
    @staticmethod
    def _filenames(quantized: bool) -> list[str]:
        from voiceclonnx.engines.triaan import TriAANVCAdapter
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
# Tests: chatterbox — quantized=True loads _q8 files from TigreGotico/voiceclonnx-chatterbox
# ---------------------------------------------------------------------------


class TestChatterboxFilenames:
    @staticmethod
    def _filenames(quantized: bool) -> list[str]:
        from voiceclonnx.engines.chatterbox import ChatterboxAdapter
        return _simple_filenames(ChatterboxAdapter)(quantized)

    def test_fp32_no_q8_files(self):
        names = self._filenames(False)
        onnx_names = [n for n in names if ".onnx" in n]
        assert onnx_names, f"expected onnx files, got: {names}"
        assert not any("q8" in n for n in onnx_names), (
            f"fp32 mode should not request q8 files: {onnx_names}"
        )

    def test_fp32_loads_external_data_sidecars(self):
        """fp32 mode downloads the .onnx_data sidecar files."""
        names = self._filenames(False)
        data_names = [n for n in names if ".onnx_data" in n]
        assert data_names, f"expected .onnx_data sidecars, got: {names}"

    def test_int8_loads_q8_onnx_files(self):
        """quantized=True requests the _q8 ONNX files (no external-data sidecars)."""
        names = self._filenames(True)
        onnx_names = [n for n in names if ".onnx" in n and ".onnx_data" not in n]
        assert onnx_names, f"expected onnx files, got: {names}"
        assert all("q8" in n for n in onnx_names), (
            f"int8 mode should request only q8 files: {onnx_names}"
        )

    def test_int8_no_external_data_sidecars(self):
        """quantized=True must NOT download .onnx_data sidecars (q8 files are self-contained)."""
        names = self._filenames(True)
        data_names = [n for n in names if ".onnx_data" in n]
        assert not data_names, (
            f"int8 mode should not request .onnx_data sidecars: {data_names}"
        )

    def test_quantized_attr_stored(self):
        from voiceclonnx.engines.chatterbox import ChatterboxAdapter

        a = ChatterboxAdapter(quantized=True)
        assert a._quantized is True
