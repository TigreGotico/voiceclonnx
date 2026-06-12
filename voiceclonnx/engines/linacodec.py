"""LinaCodec adapter for voiceclonnx.

LinaCodec (Yatharth Sharma, 2024) is a highly compressed audio codec that
produces 48 kHz audio at only 12.5 tokens/second.  This adapter implements
codec-based any-to-any voice conversion:

    encode(source) → content_embedding (semantic tokens)
    encode(reference) → global_embedding (speaker identity)
    decode(content=source, global=reference) → 48 kHz converted audio

Voice-conversion pipeline
--------------------------
1. Resample both source and reference to 24 kHz.
2. For each waveform, resample to 16 kHz for the WavLM SSL extractors.
3. **Acoustic SSL**: WavLM Base Plus layers 1–2 average → ``acoustic_ssl_encoder.onnx``
   → acoustic features (1, T_ssl, 768).
4. **Content SSL**: Distilled WavLM layers 6+9 average → ``distill_wavlm_encoder.onnx``
   → semantic features (1, T_ssl, 768).
5. **Normalise** semantic features: zero-mean, unit-variance across time axis.
6. **Content encoder**: ``content_encoder.onnx`` (local Transformer + conv-downsample
   + FSQ quantizer) → (content_embedding, content_tokens).
7. **Global encoder**: ``global_encoder.onnx`` (ConvNext + AttentiveStatsPool)
   → global_embedding (1, 128).
8. **Mel decoder**: ``mel_decoder.onnx`` (mel_prenet + mel_decoder conditioned on
   global_embedding + mel_postnet) → mel spectrogram (1, 100, T_mel).
9. **Vocos backbone**: ``vocos_backbone.onnx`` → (mag_24k, phase_24k, mag_48k, phase_48k).
10. **numpy ISTFT** (n_fft=1024, hop=256, center padding) → 24 kHz and 48 kHz waveforms.
11. **Linkwitz-Riley crossover merge** (cutoff 4 kHz) → 48 kHz output.

LICENSE NOTE
-----------
The ONNX artifacts in ``TigreGotico/voiceclonnx-linacodec`` were exported from
upstream code that includes:
- A Transformer module adapted from Meta's Llama-3 (Llama 3 Community License).
- A distill_wavlm module derived from torchaudio (BSD-2-Clause).

The upstream code was used ONLY at conversion time via an external git clone
(``/tmp/LinaCodec``).  NO LinaCodec source code is present in this MIT-licensed
repository.  This runtime adapter contains ZERO upstream code — pure onnxruntime
and numpy only.

Runtime requirements: ``onnxruntime``, ``numpy``, ``soundfile``, ``huggingface_hub``.
No torch dependency.

References
----------
- https://github.com/ysharma3501/LinaCodec
- https://huggingface.co/YatharthS/LinaCodec
- https://huggingface.co/TigreGotico/voiceclonnx-linacodec
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np

from voiceclonnx.engines.base import EngineEntry, VoiceClonerBase, register_engine

# Output sample rate
_LINA_SR = 48000
_MODEL_SR = 24000   # LinaCodec model input sample rate
_SSL_SR = 16000     # WavLM SSL input sample rate
_N_FFT = 1024
_HOP_LENGTH = 256
_N_MELS = 100

_HF_REPO_ID = "TigreGotico/voiceclonnx-linacodec"

# ONNX component filenames (fp32 and INT8)
_ACOUSTIC_FP32 = "acoustic_ssl_encoder.onnx"
_ACOUSTIC_INT8 = "acoustic_ssl_encoder_q8.onnx"
_DISTILL_FP32 = "distill_wavlm_encoder.onnx"
_DISTILL_INT8 = "distill_wavlm_encoder_q8.onnx"
_CONTENT_FP32 = "content_encoder.onnx"
_CONTENT_INT8 = "content_encoder_q8.onnx"
_GLOBAL_FP32 = "global_encoder.onnx"
_GLOBAL_INT8 = "global_encoder_q8.onnx"
_MEL_FP32 = "mel_decoder.onnx"
_MEL_INT8 = "mel_decoder_q8.onnx"
_VOCOS_FP32 = "vocos_backbone.onnx"
_VOCOS_INT8 = "vocos_backbone_q8.onnx"


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------


def _load_wav(path: str, target_sr: int) -> np.ndarray:
    """Load WAV as float32 mono, resampled to *target_sr*."""
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        n_out = int(len(audio) * target_sr / sr)
        audio = np.interp(
            np.linspace(0, len(audio) - 1, n_out),
            np.arange(len(audio)),
            audio,
        ).astype(np.float32)
    return audio


def _save_wav(path: str, audio: np.ndarray, sr: int) -> None:
    """Write float32 audio as 16-bit PCM WAV."""
    import soundfile as sf

    audio_i16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    sf.write(str(path), audio_i16, sr, subtype="PCM_16")


# ---------------------------------------------------------------------------
# numpy audio helpers
# ---------------------------------------------------------------------------


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Simple linear interpolation resampler (no torch required)."""
    if src_sr == dst_sr:
        return audio
    n_out = int(len(audio) * dst_sr / src_sr)
    return np.interp(
        np.linspace(0, len(audio) - 1, n_out),
        np.arange(len(audio)),
        audio,
    ).astype(np.float32)


def _normalize_ssl(features: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Zero-mean unit-variance normalisation across time axis (B, T, C)."""
    mean = features.mean(axis=1, keepdims=True)
    std = features.std(axis=1, keepdims=True)
    return (features - mean) / (std + eps)


# ---------------------------------------------------------------------------
# numpy ISTFT
# ---------------------------------------------------------------------------


def _numpy_istft(
    magnitude: np.ndarray,
    phase: np.ndarray,
    n_fft: int = 1024,
    hop_length: int = 256,
    padding: str = "center",
) -> np.ndarray:
    """
    Pure numpy ISTFT matching vocos ISTFTHead output.

    Args:
        magnitude: (B, n_fft//2+1, T) float32 — magnitude spectrum
        phase: (B, n_fft//2+1, T) float32 — phase angle (radians, after sin activation)
        n_fft, hop_length: STFT parameters
        padding: "center" trims n_fft//2 from each end of raw OLA output

    Returns:
        waveform: (B, T_audio) float32
    """
    B, n_bins, T_frames = magnitude.shape
    win = np.hanning(n_fft + 1)[:-1].astype(np.float32)

    results = []
    for b in range(B):
        spec = magnitude[b] * np.exp(1j * phase[b])  # (n_bins, T)

        output_length = (T_frames - 1) * hop_length + n_fft
        out = np.zeros(output_length, dtype=np.float32)
        norm = np.zeros(output_length, dtype=np.float32)

        for t in range(T_frames):
            frame = np.fft.irfft(spec[:, t], n=n_fft).real.astype(np.float32)
            frame = frame * win
            start = t * hop_length
            out[start:start + n_fft] += frame
            norm[start:start + n_fft] += win ** 2

        norm = np.where(norm < 1e-8, 1.0, norm)
        out /= norm

        if padding == "center":
            trim = n_fft // 2
            out = out[trim:-trim]

        results.append(out)

    return np.stack(results, axis=0)


# ---------------------------------------------------------------------------
# numpy Linkwitz-Riley crossover merge
# ---------------------------------------------------------------------------


def _numpy_linkwitz_riley(
    path1_48k: np.ndarray,
    path2_48k: np.ndarray,
    sample_rate: int = 48000,
    cutoff: int = 4000,
    transition_bins: int = 8,
) -> np.ndarray:
    """
    Pure numpy Linkwitz-Riley frequency crossover merge.

    Merges low-frequency content from path2 with high-frequency from path1.
    Matches the upstream ``crossover_merge_linkwitz_riley`` torch implementation.

    Args:
        path1_48k: (..., T) — high-frequency source (head_48k ISTFT output)
        path2_48k: (..., T) — low-frequency source (head_24k resampled to 48k)
        sample_rate: 48000
        cutoff: crossover frequency in Hz (default 4000)
        transition_bins: width of the crossover transition

    Returns:
        merged: (..., T) float32
    """
    min_len = min(path1_48k.shape[-1], path2_48k.shape[-1])
    p1 = path1_48k[..., :min_len]
    p2 = path2_48k[..., :min_len]

    spec1 = np.fft.rfft(p1, axis=-1)
    spec2 = np.fft.rfft(p2, axis=-1)

    n_bins = spec1.shape[-1]
    cutoff_bin = int((cutoff / (sample_rate / 2)) * n_bins)

    mask = np.ones(n_bins, dtype=np.float32)
    half = transition_bins // 2
    start = max(0, cutoff_bin - half)
    end = min(n_bins, cutoff_bin + half)

    x = np.linspace(-1, 1, end - start, dtype=np.float32)
    fade = 3 * ((x + 1) / 2) ** 2 - 2 * ((x + 1) / 2) ** 3

    mask[:start] = 0.0
    mask[start:end] = fade
    mask[end:] = 1.0

    merged = spec1 * mask + spec2 * (1.0 - mask)
    return np.fft.irfft(merged, n=min_len, axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Mel length calculation (matches LinaCodecModel._calculate_target_mel_length)
# ---------------------------------------------------------------------------


def _calculate_target_mel_length(audio_length: int, hop_length: int = 256,
                                  n_fft: int = 1024, padding: str = "center") -> int:
    if padding == "center":
        return audio_length // hop_length + 1
    elif padding == "same":
        return audio_length // hop_length
    else:
        return (audio_length - n_fft) // hop_length + 1


def _calculate_original_audio_length(token_length: int) -> int:
    """Estimate audio length from content token count.

    Matches LinaCodecModel._calculate_original_audio_length with downsample_factor=4
    and WavLM hop size 320 at 16kHz, upsampled to 24kHz.

    WavLM Base Plus: hop_size = 320 (at 16kHz).
    SSL produces feature_length = token_length * downsample_factor = token_length * 4.
    min_input_16k = (feature_length - 1) * 320 + (final_kernel) ≈ feature_length * 320.
    Convert back to 24kHz: n_samples_24k ≈ min_input_16k * 24000 / 16000.
    """
    # Empirical: ~24000 samples/s at 12.5 tokens/s → ~1920 samples/token at 24kHz
    return token_length * 4 * 320 * 24000 // 16000  # ≈ token_length * 1920


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class LinaCodecAdapter(VoiceClonerBase):
    """Voice-cloning adapter backed by LinaCodec ONNX models (48 kHz output).

    VC pipeline: encode source → content_embedding; encode reference → global_embedding;
    decode(content=source, global=reference) → mel → Vocos vocoder → 48 kHz audio.

    Parameters
    ----------
    quantized:
        Use INT8 quantized ONNX models (default ``False`` — fp32 for best quality).
    model_dir:
        Local directory with exported ONNX files.  When ``None``, downloads from
        ``TigreGotico/voiceclonnx-linacodec`` on HF Hub.
    **cfg:
        Additional keyword arguments stored but unused at runtime.
    """

    _sample_rate = _LINA_SR

    def __init__(self, quantized: bool = False, model_dir: Optional[str] = None, **cfg) -> None:
        super().__init__(**cfg)
        self._quantized = quantized
        self._model_dir = model_dir

        # ORT sessions — lazily initialised
        self._acoustic_sess = None
        self._distill_sess = None
        self._content_sess = None
        self._global_sess = None
        self._mel_sess = None
        self._vocos_sess = None

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _ensure_models(self) -> None:
        if self._acoustic_sess is not None:
            return

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for engine='linacodec'. "
                "Install with: pip install voiceclonnx"
            ) from exc

        def _get(fname_fp32: str, fname_q8: str) -> str:
            name = fname_q8 if self._quantized else fname_fp32
            if self._model_dir:
                return str(Path(self._model_dir) / name)
            try:
                from huggingface_hub import hf_hub_download
                return hf_hub_download(repo_id=_HF_REPO_ID, filename=name)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not download {name} from {_HF_REPO_ID}. "
                    "If you have a local export, pass model_dir='/path/to/linacodec/'. "
                    f"Original error: {exc}"
                ) from exc

        n_threads = os.cpu_count() or 4
        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = n_threads
        sess_opts.intra_op_num_threads = n_threads
        providers = ["CPUExecutionProvider"]

        def _load(fp32: str, q8: str) -> ort.InferenceSession:
            return ort.InferenceSession(_get(fp32, q8), sess_options=sess_opts, providers=providers)

        self._acoustic_sess = _load(_ACOUSTIC_FP32, _ACOUSTIC_INT8)
        self._distill_sess = _load(_DISTILL_FP32, _DISTILL_INT8)
        self._content_sess = _load(_CONTENT_FP32, _CONTENT_INT8)
        self._global_sess = _load(_GLOBAL_FP32, _GLOBAL_INT8)
        self._mel_sess = _load(_MEL_FP32, _MEL_INT8)
        self._vocos_sess = _load(_VOCOS_FP32, _VOCOS_INT8)

    # ------------------------------------------------------------------
    # Encode: waveform → (content_embedding, global_embedding)
    # ------------------------------------------------------------------

    def _encode(self, wav_24k: np.ndarray):
        """Encode a 24kHz waveform into (content_embedding, global_embedding).

        Args:
            wav_24k: (N,) float32 waveform at 24kHz

        Returns:
            content_embedding: (1, T, 768) float32
            global_embedding: (1, 128) float32
        """
        # Resample to SSL rate
        wav_16k = _resample_linear(wav_24k, _MODEL_SR, _SSL_SR)[np.newaxis]  # (1, N_16k)

        # Acoustic SSL features for global branch
        acoustic_feats = self._acoustic_sess.run(
            None, {"waveform_16k": wav_16k}
        )[0]  # (1, T_ssl, 768)

        # Distilled WavLM features for content branch
        semantic_feats = self._distill_sess.run(
            None, {"waveform_16k": wav_16k}
        )[0]  # (1, T_ssl, 768)

        # Normalize semantic features
        semantic_feats = _normalize_ssl(semantic_feats)

        # Content encoder: local Transformer + FSQ
        content_emb, _tokens = self._content_sess.run(
            None, {"local_ssl_features": semantic_feats}
        )  # (1, T_tokens, 768), (1, T_tokens)

        # Global encoder: speaker identity
        global_emb = self._global_sess.run(
            None, {"acoustic_features": acoustic_feats}
        )[0]  # (1, 128)

        return content_emb, global_emb

    # ------------------------------------------------------------------
    # Decode: (content_embedding, global_embedding) → 48kHz waveform
    # ------------------------------------------------------------------

    def _decode(self, content_emb: np.ndarray, global_emb: np.ndarray) -> np.ndarray:
        """Decode content + global embeddings to 48kHz waveform.

        Args:
            content_emb: (1, T_tokens, 768) float32
            global_emb: (1, 128) float32

        Returns:
            waveform: (T_audio,) float32 at 48kHz
        """
        # Mel decoder (mel_length is baked as a constant inside the ONNX model
        # based on the content_embedding sequence length — matches the export)
        mel = self._mel_sess.run(
            None,
            {
                "content_embedding": content_emb,
                "global_embedding": global_emb,
            }
        )[0]  # (1, 100, T_mel)

        # Vocos backbone: (mag_24k, phase_24k, mag_48k, phase_48k)
        mag_24k, phase_24k, mag_48k, phase_48k = self._vocos_sess.run(
            None, {"mel_spectrogram": mel}
        )

        # Apply ISTFTHead activations (as in vocos.heads.ISTFTHead):
        # S = exp(mag) * (cos(p) + i*sin(p)) — phase angle is the raw network output
        mag_24k = np.exp(np.clip(mag_24k, -10, 10))
        # phase_24k: raw angle (radians); no sin activation
        mag_48k = np.exp(np.clip(mag_48k, -10, 10))
        # phase_48k: raw angle (radians); no sin activation

        # numpy ISTFT
        audio_24k = _numpy_istft(mag_24k, phase_24k, n_fft=_N_FFT, hop_length=_HOP_LENGTH)  # (1, T_24k)
        audio_48k = _numpy_istft(mag_48k, phase_48k, n_fft=_N_FFT, hop_length=_HOP_LENGTH)  # (1, T_48k)

        # Resample 24kHz → 48kHz for Linkwitz-Riley merge
        audio_24k_up = _resample_linear(audio_24k[0], 24000, 48000)  # (T_48k,)
        audio_48k_mono = audio_48k[0]  # (T_48k,)

        # Linkwitz-Riley crossover merge
        merged = _numpy_linkwitz_riley(
            audio_48k_mono[np.newaxis],
            audio_24k_up[np.newaxis],
            cutoff=4000,
        )[0]  # (T,)

        return merged

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clone_voice(
        self,
        audio: str,
        reference_voice: str,
        out_path: str,
    ) -> str:
        """Convert *audio* to sound like *reference_voice* using LinaCodec.

        VC recipe: source content_embedding + reference global_embedding → mel → 48kHz audio.

        Parameters
        ----------
        audio:
            Path to source WAV file (any sample rate; resampled to 24kHz internally).
        reference_voice:
            Path to reference speaker WAV file.
        out_path:
            Destination path for the 16-bit 48kHz output WAV.

        Returns
        -------
        str
            Absolute path to the written output file.
        """
        self._ensure_models()

        src_wav = _load_wav(str(audio), target_sr=_MODEL_SR)
        ref_wav = _load_wav(str(reference_voice), target_sr=_MODEL_SR)

        # Encode source: get content embedding
        src_content_emb, _ = self._encode(src_wav)

        # Encode reference: get global (speaker) embedding
        _, ref_global_emb = self._encode(ref_wav)

        # Decode: source content + reference speaker → 48kHz audio
        waveform = self._decode(src_content_emb, ref_global_emb)

        out_path = str(Path(out_path).resolve())
        _save_wav(out_path, waveform, sr=_LINA_SR)
        return out_path


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_engine(
    EngineEntry(
        alias="linacodec",
        adapter_class=LinaCodecAdapter,
        description=(
            "LinaCodec: codec-based any-to-any VC at 48 kHz, 12.5 tokens/sec. "
            "VC recipe: source content_embedding + reference global_embedding → mel → Vocos → 48kHz. "
            "ONNX artifacts from TigreGotico/voiceclonnx-linacodec. "
            "Transformer adapted from Meta Llama-3 (Llama 3 Community License); "
            "distill_wavlm from torchaudio (BSD-2-Clause). "
            "Highest sample rate engine on the board."
        ),
        extras="",
        onnx_native=True,
    )
)
