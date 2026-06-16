"""Tests for the LSCodec adapter."""
import os
from unittest.mock import MagicMock

import numpy as np
import pytest

import voiceclonnx
from voiceclonnx.engines.lscodec import LSCodecAdapter, _vq, _PROMPT_WIN, _OUT_SR


def test_registered():
    assert "lscodec" in voiceclonnx.ENGINE_REGISTRY
    entry = voiceclonnx.ENGINE_REGISTRY["lscodec"]
    assert entry.adapter_class is LSCodecAdapter
    assert entry.onnx_native is True


def test_vq_nearest_neighbour():
    # codebook rows are well separated; each query should map to the nearest row
    codebook = np.array([[0.0, 0.0], [10.0, 10.0], [-5.0, 5.0]], dtype=np.float32)
    means = np.array([[0.1, -0.1], [9.0, 11.0], [-5.1, 4.9]], dtype=np.float32)
    out = _vq(means, codebook)
    assert np.allclose(out[0], codebook[0])
    assert np.allclose(out[1], codebook[1])
    assert np.allclose(out[2], codebook[2])


def test_vq_shape():
    rng = np.random.default_rng(0)
    codebook = rng.standard_normal((300, 64)).astype(np.float32)
    means = rng.standard_normal((57, 64)).astype(np.float32)
    out = _vq(means, codebook)
    assert out.shape == (57, 64)


def test_sample_rate():
    assert LSCodecAdapter._sample_rate == _OUT_SR == 24000


def _mock_adapter():
    ad = LSCodecAdapter()
    L = 40
    ad._enc_sess = MagicMock()
    ad._enc_sess.run.return_value = [np.zeros((1, L, 64), dtype=np.float32)]
    ad._wl_sess = MagicMock()
    ad._wl_sess.run.return_value = [np.zeros((1, 199, 1024), dtype=np.float32)]
    ad._voc_sess = MagicMock()
    ad._voc_sess.run.return_value = [np.zeros((1, L * 480), dtype=np.float32)]
    ad._codebook = np.random.default_rng(1).standard_normal((300, 64)).astype(np.float32)
    return ad


def test_clone_voice_contract(tmp_path):
    """clone_voice wires encoder->VQ->wavlm->vocoder and writes a 24 kHz wav."""
    import soundfile as sf

    ad = _mock_adapter()
    src = tmp_path / "src.wav"
    ref = tmp_path / "ref.wav"
    sf.write(src, np.zeros(16000, dtype=np.float32), 16000)
    sf.write(ref, np.zeros(16000, dtype=np.float32), 16000)
    out = tmp_path / "out.wav"
    ad.clone_voice(str(src), str(ref), str(out))
    assert out.exists()
    _, sr = sf.read(str(out))
    assert sr == 24000

    # reference is padded/cropped to the fixed WavLM window
    wl_input = ad._wl_sess.run.call_args[0][1]["wav"]
    assert wl_input.shape == (1, _PROMPT_WIN)


@pytest.mark.skipif(
    os.environ.get("VOICECLONNX_E2E") != "1",
    reason="E2E download/inference gated on VOICECLONNX_E2E=1",
)
def test_e2e_lscodec(tmp_path):
    from voiceclonnx import VoiceCloner

    cloner = VoiceCloner(engine="lscodec")
    out = cloner.clone_voice("demo/source.wav", "demo/reference_aria.wav", str(tmp_path / "o.wav"))
    assert os.path.exists(out)
