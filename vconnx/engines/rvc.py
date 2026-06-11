"""RVC (Retrieval-based Voice Conversion) adapter for vconnx.

RVC (RVC-Project, MIT license) is the most widely-deployed community voice
conversion system.  Unlike any-to-any systems such as kNN-VC, **RVC is
any-to-ONE**: the target speaker identity is baked into the voice model at
training time.  Each `.pth` / `.onnx` model represents exactly one target
voice.

Architecture (ONNX inference path):
  1. **ContentVec encoder** (``contentvec_768l12.onnx``) — 16 kHz audio →
     768-dim content features at 50 Hz (12-layer HuBERT variant).
  2. **RMVPE F0 predictor** (``rmvpe.onnx``) — 16 kHz audio → per-frame
     fundamental frequency (Hz), which encodes pitch.
  3. **Synthesizer / net_g** (user-supplied, per-voice ``<voice>.onnx``) —
     (features, F0, speaker_id) → 40 kHz or 48 kHz waveform.

All neural components run via onnxruntime.  Pitch post-processing is pure
numpy — no torch at inference.

**Any-to-ONE semantics — important note**:
  ``reference_voice`` does NOT accept a reference audio file.  It accepts
  the **path to an RVC voice model** (either a local ``.onnx`` / ``.pth``
  file or a Hugging Face repo ID ``owner/repo-name``).  The target speaker
  identity is already encoded in that model.  Providing actual reference
  audio has no effect in RVC; the default model shipped with this adapter
  demonstrates the API shape.

  Config key ``default_model`` sets the model used when ``reference_voice``
  is ``None`` or empty.

Base models (shared ContentVec encoder + RMVPE F0 predictor):
  ``TigreGotico/vconnx-rvc`` (public, MIT)

Voice models are user-supplied (community .pth files converted to ONNX via
``convert_rvc_model.py``, or pulled from HF repos that redistribute ONNX
files).

Requires: ``pip install vconnx[rvc]``
  → onnxruntime, numpy, soundfile, librosa

References
----------
- https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
- https://deepwiki.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/3.4-onnx-export-and-inference
- https://gudgud96.github.io/2024/09/26/annotated-rvc/
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np

from vconnx.engines.base import EngineEntry, VoiceClonerBase, register_engine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ContentVec encoder frame rate and expected sample rate
_RVC_INPUT_SR = 16000  # ContentVec/RMVPE input: 16 kHz
_RVC_HOP = 320         # 320-sample hop → 50 Hz feature rate at 16 kHz

# HF repo holding shared base models (ContentVec + RMVPE)
_HF_BASE_REPO = "TigreGotico/vconnx-rvc"

# Filenames within the base repo
_CONTENTVEC_FP32 = "contentvec_768l12.onnx"
_CONTENTVEC_INT8 = "contentvec_768l12_q8.onnx"
_RMVPE_FP32 = "rmvpe.onnx"
_RMVPE_INT8 = "rmvpe_q8.onnx"

# RMVPE mel constants (matches RMVPE's internal preprocessing)
_RMVPE_N_MEL = 128
_RMVPE_N_FFT = 1024
_RMVPE_HOP = 160          # 10 ms hop at 16 kHz
_RMVPE_WIN = 1024
_RMVPE_FMIN = 30.0
_RMVPE_FMAX = 8000.0


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------


def _load_wav(path: str, target_sr: int = _RVC_INPUT_SR) -> np.ndarray:
    """Load *path* as float32 mono at *target_sr* Hz."""
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
    """Write float32 *audio* as 16-bit PCM WAV at *sr* Hz."""
    import soundfile as sf

    audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    sf.write(str(path), audio_int16, sr, subtype="PCM_16")


# ---------------------------------------------------------------------------
# Pure-numpy RMVPE mel preprocessing
# ---------------------------------------------------------------------------


def _stft_mag(audio: np.ndarray, n_fft: int, hop: int, win: int) -> np.ndarray:
    """Compute STFT magnitude spectrogram (no torch, no librosa required).

    Returns
    -------
    np.ndarray
        shape (n_fft//2 + 1, T) float32
    """
    window = np.hanning(win).astype(np.float32)
    pad = (win - hop) // 2
    audio_padded = np.pad(audio, (pad, pad + win), mode="reflect")

    n_frames = 1 + (len(audio_padded) - win) // hop
    frames = np.lib.stride_tricks.as_strided(
        audio_padded,
        shape=(n_frames, win),
        strides=(audio_padded.strides[0] * hop, audio_padded.strides[0]),
    ).copy()

    windowed = frames * window[np.newaxis, :]
    spec = np.fft.rfft(windowed, n=n_fft, axis=1)  # (T, n_fft//2+1)
    mag = np.abs(spec).T.astype(np.float32)         # (n_fft//2+1, T)
    return mag


def _mel_filterbank(
    n_fft: int, n_mel: int, sr: int, fmin: float, fmax: float
) -> np.ndarray:
    """Build a mel filterbank matrix (librosa-compatible HTK scale)."""
    # Convert Hz to mel (HTK formula)
    def hz2mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel2hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    n_fft_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sr / 2.0, n_fft_bins)

    mel_min = hz2mel(fmin)
    mel_max = hz2mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mel + 2)
    freq_points = mel2hz(mel_points)

    fb = np.zeros((n_mel, n_fft_bins), dtype=np.float32)
    for m in range(n_mel):
        lo, center, hi = freq_points[m], freq_points[m + 1], freq_points[m + 2]
        for k, f in enumerate(fft_freqs):
            if lo <= f <= center:
                fb[m, k] = (f - lo) / (center - lo)
            elif center < f <= hi:
                fb[m, k] = (hi - f) / (hi - center)
    return fb


def _audio_to_rmvpe_mel(audio: np.ndarray, sr: int = _RVC_INPUT_SR) -> np.ndarray:
    """Convert raw 16 kHz audio to the log-mel spectrogram expected by RMVPE.

    Returns
    -------
    np.ndarray
        (1, n_mel, T) float32  (batch=1, 128 mel bins, time frames)
    """
    mag = _stft_mag(audio, n_fft=_RMVPE_N_FFT, hop=_RMVPE_HOP, win=_RMVPE_WIN)

    fb = _mel_filterbank(_RMVPE_N_FFT, _RMVPE_N_MEL, sr, _RMVPE_FMIN, _RMVPE_FMAX)
    mel = fb @ mag                            # (128, T)
    mel = np.log(np.clip(mel, 1e-5, None))   # log mel
    return mel[np.newaxis, :, :].astype(np.float32)  # (1, 128, T)


# ---------------------------------------------------------------------------
# Pitch utilities (post-RMVPE)
# ---------------------------------------------------------------------------


def _rmvpe_decode(raw: np.ndarray, cents_per_bin: float = 20.0) -> np.ndarray:
    """Decode raw RMVPE logits to F0 in Hz.

    RMVPE outputs a softmax over 360 pitch classes (20 cents/bin, 0–7200 cents
    above 32.7 Hz / C1).  The centre frequency of each bin is:
        f = 32.7 * 2^(bin * 20 / 1200)

    Parameters
    ----------
    raw:
        (T, 360) float32 — softmax probabilities per frame.

    Returns
    -------
    np.ndarray
        (T,) float32 — F0 in Hz; 0.0 for unvoiced frames.
    """
    n_bins = raw.shape[-1]
    bins = np.arange(n_bins, dtype=np.float32)
    # Weighted mean bin index
    voiced = raw.max(axis=-1) > 0.003   # simple voiced/unvoiced threshold
    cents = (raw * bins).sum(axis=-1) * cents_per_bin
    # Clamp cents to a range that won't overflow float32
    cents = np.clip(cents, 0.0, 9600.0)
    f0 = 32.7 * (2.0 ** (cents / 1200.0))
    f0[~voiced] = 0.0
    return f0.astype(np.float32)


def _interpolate_f0(f0: np.ndarray) -> np.ndarray:
    """Linearly interpolate across unvoiced (0-valued) frames."""
    voiced_mask = f0 > 0
    if not voiced_mask.any():
        return f0  # all unvoiced — leave as-is
    indices = np.arange(len(f0))
    f0_interp = np.interp(indices, indices[voiced_mask], f0[voiced_mask])
    return f0_interp.astype(np.float32)


def _resample_f0(f0: np.ndarray, src_hop: int, tgt_hop: int, tgt_len: int) -> np.ndarray:
    """Resample F0 from RMVPE frame rate to ContentVec frame rate."""
    src_t = np.arange(len(f0)) * src_hop
    tgt_t = np.arange(tgt_len) * tgt_hop
    return np.interp(tgt_t, src_t, f0).astype(np.float32)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class RVCAdapter(VoiceClonerBase):
    """Voice-cloning adapter backed by RVC ONNX models.

    Any-to-ONE semantics
    --------------------
    RVC is **any-to-ONE** — the target speaker is baked into the voice model
    (``net_g``).  The ``reference_voice`` parameter accepts:

    * A **local path** to a converted ``.onnx`` RVC voice model.
    * A **Hugging Face repo ID** (``owner/repo-name``), from which the
      ``model.onnx`` file is downloaded automatically.
    * ``None`` / empty string → falls back to ``default_model`` config key or
      the built-in demo voice.

    Reference *audio* files are never passed here; they have no meaning in
    the RVC inference pipeline.

    Parameters
    ----------
    default_model:
        Default RVC voice model (local path or HF repo ID) used when
        ``reference_voice`` is ``None``.  Set in config or at construction.
    quantized:
        If ``True``, use INT8 quantized ContentVec / RMVPE base models.
    sample_rate:
        Override the output sample rate (Hz).  RVC v2 models typically output
        40000 Hz (40k) or 48000 Hz (48k).  The adapter reads this from the
        synthesizer ONNX metadata when possible; otherwise use this parameter.
    speaker_id:
        Speaker index for multi-speaker synthesizer models (default 0).
    f0_up_key:
        Pitch shift in semitones applied to the extracted F0 before synthesis
        (default 0 — no shift).
    **cfg:
        Additional keyword arguments stored but not used.
    """

    _sample_rate = 40000  # RVC v2 40k default; overridden from model metadata

    def __init__(
        self,
        default_model: Optional[str] = None,
        quantized: bool = False,
        sample_rate: int = 40000,
        speaker_id: int = 0,
        f0_up_key: int = 0,
        **cfg,
    ):
        super().__init__(**cfg)
        self._default_model = default_model
        self._quantized = quantized
        self._sample_rate = sample_rate
        self._speaker_id = speaker_id
        self._f0_up_key = f0_up_key

        # Lazy-loaded sessions
        self._cv_sess = None    # ContentVec encoder
        self._rmvpe_sess = None  # RMVPE F0 predictor
        self._net_g_sess = None  # Synthesizer (per-voice)
        self._net_g_model_ref: Optional[str] = None  # track which model is loaded

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def _ensure_base_models(self) -> None:
        """Load ContentVec and RMVPE base sessions (shared, model-independent)."""
        if self._cv_sess is not None:
            return

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for engine='rvc'. "
                "Install it with: pip install vconnx[rvc]"
            ) from exc

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("huggingface_hub is required.") from exc

        cv_file = _CONTENTVEC_INT8 if self._quantized else _CONTENTVEC_FP32
        rmvpe_file = _RMVPE_INT8 if self._quantized else _RMVPE_FP32

        cv_path = hf_hub_download(repo_id=_HF_BASE_REPO, filename=cv_file)
        rmvpe_path = hf_hub_download(repo_id=_HF_BASE_REPO, filename=rmvpe_file)

        opts = ort.SessionOptions()
        n = os.cpu_count() or 4
        opts.inter_op_num_threads = n
        opts.intra_op_num_threads = n
        providers = ["CPUExecutionProvider"]

        self._cv_sess = ort.InferenceSession(cv_path, sess_options=opts, providers=providers)
        self._rmvpe_sess = ort.InferenceSession(rmvpe_path, sess_options=opts, providers=providers)

    def _ensure_net_g(self, model_ref: str) -> None:
        """Load the per-voice synthesizer from *model_ref* (path or HF repo ID)."""
        if self._net_g_sess is not None and self._net_g_model_ref == model_ref:
            return

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for engine='rvc'. "
                "Install it with: pip install vconnx[rvc]"
            ) from exc

        onnx_path = self._resolve_model_path(model_ref)

        opts = ort.SessionOptions()
        n = os.cpu_count() or 4
        opts.inter_op_num_threads = n
        opts.intra_op_num_threads = n
        providers = ["CPUExecutionProvider"]

        sess = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)

        # Attempt to read output sample rate from model metadata
        meta = sess.get_modelmeta()
        if meta and meta.custom_metadata_map:
            sr_str = meta.custom_metadata_map.get("sample_rate", "")
            if sr_str.isdigit():
                self._sample_rate = int(sr_str)

        self._net_g_sess = sess
        self._net_g_model_ref = model_ref

    def _resolve_model_path(self, model_ref: str) -> Path:
        """Resolve *model_ref* to a local .onnx path.

        Accepts:
        - Local file path (absolute or relative)
        - HF repo ID ``owner/repo`` → downloads the first suitable .onnx
        - ``owner/repo::filename.onnx`` → downloads a specific file from the repo
        """
        # Handle explicit file syntax: "owner/repo::filename.onnx"
        if "::" in str(model_ref):
            repo_id, filename = str(model_ref).split("::", 1)
            try:
                from huggingface_hub import hf_hub_download
                return Path(hf_hub_download(repo_id=repo_id, filename=filename))
            except Exception as exc:
                raise FileNotFoundError(
                    f"Could not download {filename!r} from {repo_id!r}: {exc}"
                ) from exc

        p = Path(model_ref)
        if p.exists() and p.suffix == ".onnx":
            return p

        # Check if it looks like a HF repo ID (contains exactly one slash,
        # no .onnx extension, does not start with / or .)
        if (
            "/" in str(model_ref)
            and not p.exists()
            and not str(model_ref).startswith(".")
        ):
            try:
                from huggingface_hub import hf_hub_download, list_repo_files
                # Try common filenames first
                for candidate in ("model.onnx", "net_g.onnx", "voice.onnx"):
                    try:
                        return Path(hf_hub_download(repo_id=model_ref, filename=candidate))
                    except Exception:
                        pass
                # Fall back to the first .onnx file in the repo
                repo_files = list(list_repo_files(model_ref))
                onnx_files = [f for f in repo_files if f.endswith(".onnx")
                              and not any(skip in f for skip in ("rmvpe", "vec", "hubert", "contentvec"))]
                if onnx_files:
                    return Path(hf_hub_download(repo_id=model_ref, filename=onnx_files[0]))
                raise FileNotFoundError(f"No .onnx voice model found in repo {model_ref!r}")
            except FileNotFoundError:
                raise
            except Exception as exc:
                raise FileNotFoundError(
                    f"Could not find or download RVC model from {model_ref!r}: {exc}"
                ) from exc

        raise FileNotFoundError(
            f"RVC voice model not found: {model_ref!r}. "
            "Pass a local .onnx path or a Hugging Face repo ID (owner/repo)."
        )

    # ------------------------------------------------------------------
    # Feature extraction (ContentVec)
    # ------------------------------------------------------------------

    def _extract_features(self, audio: np.ndarray) -> np.ndarray:
        """Run ContentVec encoder; return (T, 768) features.

        Supports two community ONNX formats:
        - TigreGotico/vconnx-rvc export: input ``input_values`` (1, T),
          optional ``attention_mask`` (1, T); output ``hidden_states`` (1, T, 768)
        - ozada/onnx_rvc vec-768-layer-12 format: input ``source`` (1, 1, T);
          output ``embed`` (1, T, 768)
        """
        inp_f32 = audio[np.newaxis, :].astype(np.float32)  # (1, T)
        cv_inputs = self._cv_sess.get_inputs()
        input_names = [i.name for i in cv_inputs]

        if "source" in input_names:
            # ozada-format: (1, 1, T)
            feed = {"source": inp_f32[:, np.newaxis, :]}
        else:
            # vconnx-rvc format: (1, T) + optional attention_mask
            feed = {"input_values": inp_f32}
            if "attention_mask" in input_names:
                feed["attention_mask"] = np.ones((1, inp_f32.shape[1]), dtype=np.int64)

        out = self._cv_sess.run(None, feed)
        # Both formats produce (batch, T_frames, 768) — take batch 0
        return out[0][0]  # (T_frames, 768)

    # ------------------------------------------------------------------
    # F0 extraction (RMVPE)
    # ------------------------------------------------------------------

    def _extract_f0(self, audio: np.ndarray, n_feature_frames: int) -> np.ndarray:
        """Run RMVPE to get F0 in Hz, resampled to *n_feature_frames* frames.

        Returns
        -------
        np.ndarray
            (n_feature_frames,) float32 — F0 in Hz; 0.0 for unvoiced.
        """
        mel = _audio_to_rmvpe_mel(audio, sr=_RVC_INPUT_SR)  # (1, 128, T_mel)

        # Community RMVPE ONNX (Politrees/RVC_resources predictors/rmvpe.onnx)
        # expects (batch, n_mel, T) — the 3D form.
        # T must be a multiple of 32 (5-level DeepUnet pooling); pad if needed.
        T_mel = mel.shape[2]
        pad_to = ((T_mel + 31) // 32) * 32
        if pad_to > T_mel:
            mel = np.pad(mel, ((0, 0), (0, 0), (0, pad_to - T_mel)), mode="reflect")

        out = self._rmvpe_sess.run(None, {"input": mel})
        # Output: (1, T_padded, 360) probabilities — trim to original T
        raw = out[0][0, :T_mel, :]  # (T_mel, 360)

        f0 = _rmvpe_decode(raw)            # (T_rmvpe,) Hz
        f0 = _interpolate_f0(f0)

        # Apply pitch shift (semitones)
        if self._f0_up_key != 0:
            voiced_mask = f0 > 0
            f0[voiced_mask] *= 2.0 ** (self._f0_up_key / 12.0)

        # Resample to match ContentVec frame count
        f0_resampled = _resample_f0(
            f0,
            src_hop=_RMVPE_HOP,
            tgt_hop=_RVC_HOP,
            tgt_len=n_feature_frames,
        )
        return f0_resampled

    # ------------------------------------------------------------------
    # Synthesis (net_g)
    # ------------------------------------------------------------------

    def _synthesize(
        self,
        features: np.ndarray,  # (T, 768)
        f0: np.ndarray,        # (T,) Hz
        f0_coarse: np.ndarray, # (T,) quantised pitch index
    ) -> np.ndarray:
        """Run the RVC VITS synthesizer; return (samples,) float32 waveform."""
        # net_g input layout (standard RVC ONNX export):
        #   phone:       (1, T, 768) float32   — ContentVec features
        #   phone_lengths: (1,) int64
        #   pitch:       (1, T) int64          — coarse F0 index
        #   pitchf:      (1, T) float32        — fine F0 in Hz
        #   ds:          (1,) int64            — speaker index
        #   rnd:         (1, 192, T) float32   — noise for stochastic decoder
        T = features.shape[0]
        phone = features[np.newaxis, :, :].astype(np.float32)          # (1, T, 768)
        phone_lengths = np.array([T], dtype=np.int64)                  # (1,)
        pitch = f0_coarse[np.newaxis, :].astype(np.int64)              # (1, T)
        pitchf = f0[np.newaxis, :].astype(np.float32)                  # (1, T)
        ds = np.array([self._speaker_id], dtype=np.int64)              # (1,)
        rng = np.random.default_rng(42)
        rnd = rng.standard_normal((1, 192, T)).astype(np.float32)      # (1, 192, T)

        out = self._net_g_sess.run(
            None,
            {
                "phone": phone,
                "phone_lengths": phone_lengths,
                "pitch": pitch,
                "pitchf": pitchf,
                "ds": ds,
                "rnd": rnd,
            },
        )
        # Output: (1, 1, samples) → (samples,)
        # Some models name the output 'audio' (ozada format) or 'waveform' (vconnx)
        return out[0][0, 0]  # first output, first batch, first channel

    # ------------------------------------------------------------------
    # F0 quantization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _f0_to_coarse(f0: np.ndarray, f0_min: float = 50.0, f0_max: float = 1100.0, bins: int = 256) -> np.ndarray:
        """Quantise continuous F0 to coarse pitch index in [1, bins-1].

        Unvoiced frames (f0 == 0) map to index 0.
        """
        log_min = np.log(f0_min)
        log_max = np.log(f0_max)
        coarse = np.zeros_like(f0, dtype=np.int64)
        voiced = f0 > 0
        safe_f0 = np.where(voiced, np.clip(f0, f0_min, f0_max), f0_min)
        log_f0 = np.log(safe_f0)
        idx = np.round((log_f0 - log_min) / (log_max - log_min) * (bins - 2)).astype(np.int64)
        coarse[voiced] = np.clip(idx[voiced] + 1, 1, bins - 1)
        return coarse

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clone_voice(
        self,
        audio: str,
        reference_voice: Optional[str],
        out_path: str,
    ) -> str:
        """Convert *audio* to the target voice encoded in *reference_voice* model.

        Parameters
        ----------
        audio:
            Path to the source WAV file (any sample rate).
        reference_voice:
            **Path to an RVC voice model** (local ``.onnx`` file or HF repo ID
            ``owner/repo``), NOT a reference audio file.  RVC bakes the target
            speaker into the model weights; reference audio is never used.
            Falls back to ``default_model`` (constructor parameter) if ``None``.
        out_path:
            Destination path for the converted WAV.

        Returns
        -------
        str
            Absolute path to the written output WAV.

        Notes
        -----
        Output sample rate matches the synthesizer model (typically 40 kHz for
        RVC v2 40k models, 48 kHz for v2 48k models).  Check
        ``adapter.sample_rate`` after the first call.
        """
        model_ref = reference_voice or self._default_model
        if not model_ref:
            raise ValueError(
                "reference_voice must be a path to an RVC .onnx model or a HF repo ID. "
                "Set default_model in the constructor if you want a fallback."
            )

        self._ensure_base_models()
        self._ensure_net_g(model_ref)

        # Load source audio at 16 kHz (ContentVec / RMVPE input rate)
        src_wav = _load_wav(str(audio), target_sr=_RVC_INPUT_SR)

        # 1. Extract ContentVec features
        features = self._extract_features(src_wav)   # (T, 768)
        T = features.shape[0]

        # 2. Extract F0 (RMVPE)
        f0 = self._extract_f0(src_wav, n_feature_frames=T)   # (T,) Hz

        # 3. Quantise F0 to coarse pitch index
        f0_coarse = self._f0_to_coarse(f0)   # (T,) int64

        # 4. Synthesize
        waveform = self._synthesize(features, f0, f0_coarse)  # (samples,)

        # 5. Write output
        out_path = str(Path(out_path).resolve())
        _save_wav(out_path, waveform, sr=self._sample_rate)
        return out_path


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_engine(
    EngineEntry(
        alias="rvc",
        adapter_class=RVCAdapter,
        description=(
            "RVC (Retrieval-based Voice Conversion): ContentVec encoder + RMVPE F0 predictor "
            "+ VITS-based synthesizer.  Any-to-ONE: the target speaker is baked into the model. "
            "reference_voice = path to an RVC .onnx voice model or HF repo ID.  "
            "Base models from TigreGotico/vconnx-rvc (MIT).  "
            "(RVC-Project, MIT license)"
        ),
        extras="rvc",
        onnx_native=True,
    )
)
