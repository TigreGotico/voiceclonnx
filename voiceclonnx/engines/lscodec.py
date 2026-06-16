"""LSCodec adapter for voiceclonnx.

LSCodec (Guo et al., Interspeech 2025) is a low-bitrate, **speaker-decoupled**
discrete speech codec.  Because the discrete content space is trained to be
speaker-agnostic, voice conversion is direct: encode the source to content
tokens, then resynthesise with a vocoder conditioned on the *target* speaker's
WavLM features.

Pipeline (pure onnxruntime + numpy, zero torch at runtime):

1. **Encoder** (``lscodec_encoder.onnx``) — raw 16 kHz source audio → 64-dim
   continuous ``means`` at 50 Hz.
2. **Numpy VQ** — nearest-neighbour of ``means`` to the 300-entry codebook
   (``codebook.npy``); the matched codebook vectors form the VQ sequence.
3. **WavLM-Large layer-6** (``wavlm_l6.onnx``) — the *reference* (target) clip →
   1024-dim prompt features.  Exported at a fixed 4 s window, so the reference
   is padded / cropped to 64000 samples.
4. **CTXVEC2WAV vocoder** (``lscodec_vocoder.onnx``) — VQ sequence + prompt
   features → 24 kHz waveform.

Verified: the ONNX pipeline reproduces the upstream torch reference (speaker
embedding cosine ≈ 0.97) and transfers the target voice (target-similarity
≈ 0.54, well above the no-conversion floor).  See demo/SPEAKER_SIMILARITY.md.

References
----------
- https://github.com/X-LANCE/LSCodec-Inference  (MIT)
- https://arxiv.org/abs/2410.15764
- https://huggingface.co/TigreGotico/voiceclonnx-lscodec
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from voiceclonnx.engines.base import EngineEntry, VoiceClonerBase, register_engine

_CONTENT_SR = 16000
_PROMPT_SR = 16000
_OUT_SR = 24000
_PROMPT_WIN = 16000 * 4   # WavLM ONNX is exported at a fixed 4 s window

_HF_REPO_ID = "TigreGotico/voiceclonnx-lscodec"

_ENC_FP32, _ENC_INT8 = "lscodec_encoder.onnx", "lscodec_encoder_q8.onnx"
_WL_FP32, _WL_INT8 = "wavlm_l6.onnx", "wavlm_l6_q8.onnx"
_VOC_FP32, _VOC_INT8 = "lscodec_vocoder.onnx", "lscodec_vocoder_q8.onnx"
_CODEBOOK = "codebook.npy"


def _load_wav(path: str, target_sr: int) -> np.ndarray:
    """Load *path* as float32 mono resampled to *target_sr*."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        n_out = int(round(len(audio) * target_sr / sr))
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n_out),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
    return audio.astype(np.float32)


def _save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    import soundfile as sf

    audio16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    sf.write(str(path), audio16, sr, subtype="PCM_16")


def _vq(means: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """Euclidean nearest-neighbour quantisation: (L,64) means -> (L,64) vectors."""
    dist = (means ** 2).sum(1, keepdims=True) - 2.0 * means @ codebook.T + (codebook ** 2).sum(1)
    idx = dist.argmin(1)
    return codebook[idx]


class LSCodecAdapter(VoiceClonerBase):
    """Voice-cloning adapter backed by LSCodec ONNX models.

    Parameters
    ----------
    quantized:
        Use INT8 quantized ONNX models (default ``False`` — fp32 is the
        supported quality path).
    **cfg:
        Additional keyword arguments stored but unused at runtime.
    """

    _sample_rate = _OUT_SR

    def __init__(self, quantized: bool = False, **cfg) -> None:
        super().__init__(**cfg)
        self._quantized = quantized
        self._enc_sess = None
        self._wl_sess = None
        self._voc_sess = None
        self._codebook = None

    def _ensure_models(self) -> None:
        if self._enc_sess is not None:
            return
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for engine='lscodec'. "
                "Install it with: pip install voiceclonnx"
            ) from exc
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("huggingface_hub is required.") from exc

        enc = _ENC_INT8 if self._quantized else _ENC_FP32
        wl = _WL_INT8 if self._quantized else _WL_FP32
        voc = _VOC_INT8 if self._quantized else _VOC_FP32

        enc_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=enc)
        wl_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=wl)
        voc_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=voc)
        cb_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=_CODEBOOK)

        opts = ort.SessionOptions()
        n = os.cpu_count() or 4
        opts.inter_op_num_threads = n
        opts.intra_op_num_threads = n
        providers = ["CPUExecutionProvider"]
        self._enc_sess = ort.InferenceSession(enc_path, sess_options=opts, providers=providers)
        self._wl_sess = ort.InferenceSession(wl_path, sess_options=opts, providers=providers)
        self._voc_sess = ort.InferenceSession(voc_path, sess_options=opts, providers=providers)
        cb = np.load(cb_path, allow_pickle=True)
        self._codebook = cb.reshape(-1, cb.shape[-1]).astype(np.float32)  # (300, 64)

    def clone_voice(self, audio: str, reference_voice: str, out_path: str) -> str:
        """Convert *audio* to sound like *reference_voice* using LSCodec."""
        self._ensure_models()

        # Content path: source -> means -> numpy VQ
        src = _load_wav(str(audio), _CONTENT_SR)
        means = self._enc_sess.run(None, {"audio": src[np.newaxis, np.newaxis, :]})[0][0]  # (L,64)
        vqvec = _vq(means, self._codebook)[np.newaxis].astype(np.float32)  # (1,L,64)

        # Prompt path: reference -> fixed 4 s window -> WavLM-Large layer 6
        ref = _load_wav(str(reference_voice), _PROMPT_SR)
        if len(ref) < _PROMPT_WIN:
            ref = np.pad(ref, (0, _PROMPT_WIN - len(ref)))
        else:
            ref = ref[:_PROMPT_WIN]
        prompt = self._wl_sess.run(None, {"wav": ref[np.newaxis, :]})[0]  # (1,199,1024)

        # Vocoder
        wav = self._voc_sess.run(None, {"vqvec": vqvec, "prompt": prompt})[0][0]  # (T,)

        out_path = str(Path(out_path).resolve())
        _save_wav(out_path, wav.astype(np.float32), sr=_OUT_SR)
        return out_path


register_engine(
    EngineEntry(
        alias="lscodec",
        adapter_class=LSCodecAdapter,
        description=(
            "LSCodec: speaker-decoupled discrete speech codec (Interspeech 2025). "
            "Encoder + numpy VQ + WavLM-prompt CTXVEC2WAV vocoder. Zero-shot "
            "any-to-any VC at 24 kHz. ONNX from TigreGotico/voiceclonnx-lscodec. (MIT)"
        ),
        extras="",
        onnx_native=True,
    )
)
