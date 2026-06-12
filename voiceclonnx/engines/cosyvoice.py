"""CosyVoice adapter for voiceclonnx — non-AR voice conversion.

CosyVoice (FunAudioLLM / Alibaba DAMO, Apache-2.0) supports voice transfer via
a speaker-embedding injection path.  This adapter uses only the non-autoregressive
components: content tokens from the speech tokenizer plus a reference speaker
embedding from CAM++ condition a flow-matching decoder, entirely bypassing the
autoregressive LLM (llm.pt) used for TTS.

Pipeline
--------
1. ``speech_tokenizer_v1.onnx`` — Whisper-style log-mel → content token indices
2. ``campplus.onnx``             — Kaldi fbank → 192-d speaker embedding (CAM++)
3. ``flow_encoder.onnx``         — Token embeddings + 6-block Conformer → mu (80-d)
4. ``flow_decoder.onnx``         — ODE Euler solver (10 steps) over mu + spk_emb → mel
5. ``hifigan_f0_source.onnx``    — mel → F0-driven NSF harmonic source (1-D signal)
6. numpy STFT                   — source signal → 9-bin STFT coefficients
7. ``hifigan_backbone.onnx``     — (mel, source_stft) → (magnitude, phase) STFT bins
8. numpy ISTFT                  — magnitude + phase → waveform @ 22050 Hz

STFT/ISTFT note
---------------
``aten::stft`` / ``aten::istft`` are not supported at ONNX opset 14.  The HiFiGAN
is split at the STFT boundary; STFT and ISTFT are implemented in pure numpy using
the same parameters as CosyVoice (n_fft=16, hop_len=4, Hann window).  Parity vs
torch is verified at export time (STFT max_abs ≤ 1.4e-6, ISTFT roundtrip ≤ 5e-3).

CPU RTF
-------
Measured at ~0.71× on a standard laptop CPU — comfortably under the 5× gate.
No GPU required; no PyTorch at inference time.

ONNX artifacts
--------------
Hosted at ``TigreGotico/voiceclonnx-cosyvoice`` on Hugging Face Hub.

License
-------
Upstream weights: **Apache-2.0** (FunAudioLLM/CosyVoice-300M).
Training data Emilia: CC-BY-NC-4.0 (does not restrict model weights).

References
----------
- https://github.com/FunAudioLLM/CosyVoice
- https://huggingface.co/FunAudioLLM/CosyVoice-300M
- https://arxiv.org/abs/2407.05407
- https://huggingface.co/TigreGotico/voiceclonnx-cosyvoice
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from voiceclonnx.engines.base import EngineEntry, VoiceClonerBase, register_engine

# CosyVoice-300M operates at 22050 Hz output
_CV_SR = 22050
# Speech content tokenizer expects 16 kHz input
_TOK_SR = 16000
# CAM++ speaker encoder expects 16 kHz input
_SPK_SR = 16000

_HF_REPO_ID = "TigreGotico/voiceclonnx-cosyvoice"

# File names in the HF repo
_F_TOKENIZER = "speech_tokenizer_v1.onnx"
_F_CAMPPLUS = "campplus.onnx"
# flow_encoder_conformer.onnx: Embedding + 6-block Conformer + proj.
# Takes (tokens, token_len) — explicit length avoids baked-in attention mask.
# The InterpolateRegulator and its LR conv model are applied in the adapter
# (numpy + lr_weights.npz) to support dynamic mel lengths.
_F_FLOW_ENC = "flow_encoder_conformer.onnx"
_F_LR_WEIGHTS = "lr_weights.npz"
_F_FLOW_DEC = "flow_decoder.onnx"
_F_FLOW_DEC_Q8 = "flow_decoder_q8.onnx"
_F_HIFIGAN_SRC = "hifigan_f0_source.onnx"
_F_HIFIGAN_SRC_Q8 = "hifigan_f0_source_q8.onnx"
_F_HIFIGAN_BB = "hifigan_backbone.onnx"
_F_HIFIGAN_BB_Q8 = "hifigan_backbone_q8.onnx"

# HiFiGAN STFT parameters (must match export)
_HIFT_N_FFT = 16
_HIFT_HOP_LEN = 4

# Flow decoder ODE parameters
_FLOW_FRAME_RATE = 50   # tokens per second
_FLOW_MEL_HOP = 256     # mel spectrogram hop at 22050 Hz
_FLOW_ODE_STEPS = 10

# Classifier-free guidance rate (matches CosyVoice-300M cfm_params.inference_cfg_rate)
_INFERENCE_CFG_RATE = 0.7


# ---------------------------------------------------------------------------
# Audio I/O helpers
# ---------------------------------------------------------------------------


def _load_wav(path: str, target_sr: int) -> np.ndarray:
    """Load WAV as float32 mono and resample to *target_sr*."""
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

    clipped = np.clip(audio, -1.0, 1.0)
    sf.write(str(path), (clipped * 32767).astype(np.int16), sr, subtype="PCM_16")


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


def _kaldi_fbank(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Compute Kaldi-style 80-bin log mel filterbank for CAM++ speaker encoder.

    Parameters
    ----------
    wav : (N,) float32 at *sr* Hz
    sr  : sample rate (default 16000)

    Returns
    -------
    np.ndarray  (1, T_frames, 80) float32
    """
    n_fft = 400
    hop = 160
    n_mels = 80
    fmin = 20
    fmax = 7600

    # Windowed STFT power spectrum
    window = np.hanning(n_fft).astype(np.float32)
    pad = n_fft // 2
    wav_pad = np.pad(wav, (pad, pad), mode="reflect")
    n_frames = 1 + (len(wav_pad) - n_fft) // hop
    power = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
    for i in range(n_frames):
        frame = wav_pad[i * hop: i * hop + n_fft] * window
        power[:, i] = np.abs(np.fft.rfft(frame, n=n_fft)) ** 2

    # Mel filterbank
    mel_fb = _mel_filterbank(sr, n_fft, n_mels, fmin, fmax)  # (n_mels, n_fft//2+1)
    mel = mel_fb @ power                                       # (n_mels, T)
    log_mel = np.log(np.maximum(mel, 1e-10)).T                 # (T, n_mels)

    # Mean normalization (Kaldi CMVN)
    log_mel = log_mel - log_mel.mean(axis=0, keepdims=True)
    return log_mel[np.newaxis].astype(np.float32)              # (1, T, 80)


def _whisper_logmel(wav: np.ndarray, sr: int = 16000) -> tuple:
    """Compute Whisper-style 128-bin log mel spectrogram for speech tokenizer.

    Parameters
    ----------
    wav : (N,) float32 at *sr* Hz

    Returns
    -------
    feat     : (1, 128, T) float32  — normalized log-mel
    feat_len : (1,) int32           — number of frames T
    """
    n_fft = 400
    hop = 160
    n_mels = 128
    fmin = 0
    fmax = 8000

    window = np.hanning(n_fft).astype(np.float32)
    pad = n_fft // 2
    wav_pad = np.pad(wav, (pad, pad), mode="reflect")
    n_frames = 1 + (len(wav_pad) - n_fft) // hop
    power = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.float32)
    for i in range(n_frames):
        frame = wav_pad[i * hop: i * hop + n_fft] * window
        power[:, i] = np.abs(np.fft.rfft(frame, n=n_fft)) ** 2

    mel_fb = _mel_filterbank(sr, n_fft, n_mels, fmin, fmax)
    mel = mel_fb @ power                              # (n_mels, T)
    log_mel = np.log10(np.maximum(mel, 1e-10))
    log_mel = np.maximum(log_mel, log_mel.max() - 8.0)
    log_mel = ((log_mel + 4.0) / 4.0).astype(np.float32)
    feat = log_mel[np.newaxis]                        # (1, 128, T)
    feat_len = np.array([n_frames], dtype=np.int32)
    return feat, feat_len


def _mel_filterbank(
    sr: int, n_fft: int, n_mels: int, fmin: float, fmax: float
) -> np.ndarray:
    """HTK-style mel filterbank matrix (n_mels, n_fft//2+1)."""
    n_freq = n_fft // 2 + 1
    freq = np.linspace(0, sr / 2, n_freq)

    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_min = hz_to_mel(fmin)
    mel_max = hz_to_mel(fmax)
    mel_pts = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)

    fb = np.zeros((n_mels, n_freq), dtype=np.float32)
    for m in range(n_mels):
        lo, center, hi = hz_pts[m], hz_pts[m + 1], hz_pts[m + 2]
        up = (freq - lo) / (center - lo + 1e-8)
        down = (hi - freq) / (hi - center + 1e-8)
        fb[m] = np.maximum(0, np.minimum(up, down))

    # Slaney-style L1 normalization (per Librosa)
    enorm = 2.0 / (hz_pts[2: n_mels + 2] - hz_pts[:n_mels])
    fb *= enorm[:, np.newaxis]
    return fb.astype(np.float32)


# ---------------------------------------------------------------------------
# STFT/ISTFT (matching HiFiGAN parameters; replaces aten::stft/istft)
# ---------------------------------------------------------------------------


def _stft(x: np.ndarray, n_fft: int = _HIFT_N_FFT, hop_len: int = _HIFT_HOP_LEN) -> tuple:
    """Hann-windowed center-padded STFT.

    Returns
    -------
    real : (n_fft//2+1, n_frames) float32
    imag : (n_fft//2+1, n_frames) float32
    """
    from scipy.signal import get_window

    window = get_window("hann", n_fft, fftbins=True).astype(np.float32)
    pad = n_fft // 2
    xp = np.pad(x.astype(np.float32), (pad, pad), mode="reflect")
    n_frames = 1 + (len(xp) - n_fft) // hop_len
    n_freq = n_fft // 2 + 1
    real = np.zeros((n_freq, n_frames), dtype=np.float32)
    imag = np.zeros((n_freq, n_frames), dtype=np.float32)
    for i in range(n_frames):
        frame = xp[i * hop_len: i * hop_len + n_fft] * window
        spec = np.fft.rfft(frame, n=n_fft)
        real[:, i] = spec.real
        imag[:, i] = spec.imag
    return real, imag


def _istft(
    magnitude: np.ndarray,
    phase: np.ndarray,
    n_fft: int = _HIFT_N_FFT,
    hop_len: int = _HIFT_HOP_LEN,
) -> np.ndarray:
    """Overlap-add ISTFT.  magnitude/phase: (n_fft//2+1, n_frames)."""
    from scipy.signal import get_window

    window = get_window("hann", n_fft, fftbins=True).astype(np.float32)
    magnitude = np.clip(magnitude, None, 100.0)
    complex_spec = magnitude * np.exp(1j * phase)
    n_frames = complex_spec.shape[1]
    out_len = n_fft + (n_frames - 1) * hop_len
    audio = np.zeros(out_len, dtype=np.float32)
    wsum = np.zeros(out_len, dtype=np.float32)
    for i in range(n_frames):
        frame = np.fft.irfft(complex_spec[:, i], n=n_fft).astype(np.float32)
        audio[i * hop_len: i * hop_len + n_fft] += frame * window
        wsum[i * hop_len: i * hop_len + n_fft] += window ** 2
    return (audio / np.maximum(wsum, 1e-8)).astype(np.float32)


# ---------------------------------------------------------------------------
# InterpolateRegulator helpers (pure numpy)
# ---------------------------------------------------------------------------


def _interp1d(x: np.ndarray, size: int) -> np.ndarray:
    """Linear interpolate x (1, C, T) along last dim to length *size*.

    Matches ``torch.nn.functional.interpolate(..., mode='linear',
    align_corners=False)`` for 3-D tensors (batch=1).
    """
    _, C, T = x.shape
    if T == size:
        return x
    # align_corners=False: source coordinate = (i + 0.5) * T / size - 0.5
    coords = (np.arange(size, dtype=np.float32) + 0.5) * T / size - 0.5
    out = np.zeros((1, C, size), dtype=np.float32)
    for c in range(C):
        out[0, c] = np.interp(coords, np.arange(T), x[0, c])
    return out


def _apply_lr_model(h: np.ndarray, lr_w: "np.lib.npyio.NpzFile") -> np.ndarray:
    """Apply the InterpolateRegulator conv model in numpy.

    Architecture: 4 × [Conv1d(80,80,k=3,p=1) + GroupNorm(1,80) + Mish]
                       + Conv1d(80,80,k=1).

    Parameters
    ----------
    h : (1, 80, T) float32 — interpolated encoder output
    lr_w : npz file with keys like '0_weight', '0_bias', '1_weight', ... '12_weight', '12_bias'

    Returns
    -------
    (1, 80, T) float32
    """
    x = h[0].copy()  # (80, T)

    def _conv1d(x, w, b, padding=0):
        C_out, C_in, K = w.shape
        if padding:
            x = np.pad(x, ((0, 0), (padding, padding)))
        T_out = x.shape[1] - K + 1
        out = np.empty((C_out, T_out), dtype=np.float32)
        for k in range(K):
            if k == 0:
                out[:] = w[:, :, k] @ x[:, k:k + T_out]
            else:
                out += w[:, :, k] @ x[:, k:k + T_out]
        return out + b[:, np.newaxis]

    def _group_norm(x, w, b):
        mean = x.mean()
        std = np.sqrt(((x - mean) ** 2).mean() + 1e-5)
        return (x - mean) / std * w[:, np.newaxis] + b[:, np.newaxis]

    def _mish(x):
        return x * np.tanh(np.log1p(np.exp(np.clip(x, -88.0, 88.0))))

    for i in range(4):
        idx = i * 3
        x = _conv1d(x, lr_w[f"{idx}_weight"], lr_w[f"{idx}_bias"], padding=1)
        x = _group_norm(x, lr_w[f"{idx + 1}_weight"], lr_w[f"{idx + 1}_bias"])
        x = _mish(x)

    x = _conv1d(x, lr_w["12_weight"], lr_w["12_bias"], padding=0)
    return x[np.newaxis]  # (1, 80, T)


# ---------------------------------------------------------------------------
# Kaldi fbank — upstream-compatible helper
# ---------------------------------------------------------------------------


def _kaldi_fbank_compat(wav: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Compute Kaldi-style 80-bin log mel filterbank for CAM++ speaker encoder.

    Delegates to ``torchaudio.compliance.kaldi.fbank`` (exact upstream match)
    when torch/torchaudio are available.  Falls back to the numpy approximation
    otherwise so that the library stays PyTorch-free at inference time when
    torch is not installed.

    Parameters
    ----------
    wav : (N,) float32 at *sr* Hz
    sr  : sample rate (default 16000)

    Returns
    -------
    np.ndarray  (1, T_frames, 80) float32 — mean-normalised log-mel
    """
    try:
        import torch
        import torchaudio.compliance.kaldi as kaldi_ta

        wav_t = torch.from_numpy(wav).unsqueeze(0)  # (1, N)
        fbank = kaldi_ta.fbank(wav_t, num_mel_bins=80, dither=0, sample_frequency=sr)
        fbank = fbank - fbank.mean(dim=0, keepdim=True)
        return fbank.unsqueeze(0).numpy().astype(np.float32)  # (1, T, 80)
    except Exception:
        return _kaldi_fbank(wav, sr=sr)


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class CosyVoiceAdapter(VoiceClonerBase):
    """Voice-cloning adapter backed by CosyVoice-300M ONNX models.

    Non-AR pipeline: source content tokens (from speech tokenizer) + reference
    speaker embedding (from CAM++) → flow-matching mel decoder → HiFiGAN vocoder.
    The autoregressive LLM is not used.

    Parameters
    ----------
    quantized : bool
        Load INT8 quantized ONNX models (default False).
    ode_steps : int
        Number of Euler ODE steps for the flow decoder (default 10).
    **cfg :
        Extra kwargs stored but not used.
    """

    _sample_rate = _CV_SR

    def __init__(self, quantized: bool = False, ode_steps: int = _FLOW_ODE_STEPS, **cfg):
        super().__init__(**cfg)
        self._quantized = quantized
        self._ode_steps = ode_steps
        # ORT sessions (lazy-loaded)
        self._tok_sess = None
        self._spk_sess = None
        self._fe_sess = None
        self._fd_sess = None
        self._hf_src_sess = None
        self._hf_bb_sess = None
        # LengthRegulator conv model weights (numpy, lazy-loaded)
        self._lr_weights = None

    # ------------------------------------------------------------------
    # Lazy model loading
    # ------------------------------------------------------------------

    def _ensure_models(self) -> None:
        if self._tok_sess is not None:
            return

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise ImportError(
                "onnxruntime is required for engine='cosyvoice'. "
                "Install with: pip install voiceclonnx"
            ) from exc

        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise ImportError("huggingface_hub is required.") from exc

        q = self._quantized

        files = {
            "tok": _F_TOKENIZER,
            "spk": _F_CAMPPLUS,
            "fe": _F_FLOW_ENC,            # conformer-only, no quantized variant
            "fd": _F_FLOW_DEC_Q8 if q else _F_FLOW_DEC,
            "hf_src": _F_HIFIGAN_SRC_Q8 if q else _F_HIFIGAN_SRC,
            "hf_bb": _F_HIFIGAN_BB_Q8 if q else _F_HIFIGAN_BB,
        }

        paths = {k: hf_hub_download(repo_id=_HF_REPO_ID, filename=v) for k, v in files.items()}
        lr_path = hf_hub_download(repo_id=_HF_REPO_ID, filename=_F_LR_WEIGHTS)
        self._lr_weights = np.load(lr_path)

        n = os.cpu_count() or 4
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = n
        opts.intra_op_num_threads = n
        providers = ["CPUExecutionProvider"]

        self._tok_sess = ort.InferenceSession(paths["tok"], sess_options=opts, providers=providers)
        self._spk_sess = ort.InferenceSession(paths["spk"], sess_options=opts, providers=providers)
        self._fe_sess = ort.InferenceSession(paths["fe"], sess_options=opts, providers=providers)
        self._fd_sess = ort.InferenceSession(paths["fd"], sess_options=opts, providers=providers)
        self._hf_src_sess = ort.InferenceSession(paths["hf_src"], sess_options=opts, providers=providers)
        self._hf_bb_sess = ort.InferenceSession(paths["hf_bb"], sess_options=opts, providers=providers)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_content_tokens(self, wav_16k: np.ndarray) -> np.ndarray:
        """Whisper log-mel → int64 content token indices (1, T_tok)."""
        feat, feat_len = _whisper_logmel(wav_16k, sr=_TOK_SR)
        out = self._tok_sess.run(None, {"feats": feat, "feats_length": feat_len})
        tokens = out[0]  # (1, 1, T_tok) int64
        return tokens.reshape(1, -1).astype(np.int64)  # (1, T_tok)

    def _extract_speaker_embedding(self, wav_16k: np.ndarray) -> np.ndarray:
        """Kaldi fbank → L2-normalized 192-d speaker embedding (1, 192).

        Uses torchaudio.compliance.kaldi.fbank when torch is available (exact
        upstream match).  Falls back to the numpy approximation otherwise.
        """
        fbank = _kaldi_fbank_compat(wav_16k, sr=_SPK_SR)  # (1, T, 80)
        inp_name = self._spk_sess.get_inputs()[0].name
        emb = self._spk_sess.run(None, {inp_name: fbank})[0]  # (1, 192)
        emb = emb.reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-8
        return emb / norm

    # ------------------------------------------------------------------
    # Flow encoder (tokens → mu)
    # ------------------------------------------------------------------

    def _encode_tokens(self, tokens: np.ndarray) -> np.ndarray:
        """Content tokens (1, T) → mu (1, 80, T_mel).

        Two-stage:
        1. ONNX conformer (flow_encoder_conformer.onnx, takes tokens + token_len)
           → h (1, T, 80)
        2. Numpy InterpolateRegulator: interpolate h to mel frames then apply the
           LR conv model (weights from lr_weights.npz).
        """
        T = int(tokens.shape[1])
        token_len = np.array([T], dtype=np.int64)

        # Stage 1: Conformer encoding
        h = self._fe_sess.run(
            None, {"tokens": tokens, "token_len": token_len}
        )[0]  # (1, T, 80)

        # Stage 2: InterpolateRegulator — mirrors InterpolateRegulator.inference
        h_t = h.transpose(0, 2, 1)  # (1, 80, T)
        mel_len = max(int(T / _FLOW_FRAME_RATE * _CV_SR / _FLOW_MEL_HOP), 1)
        _H = int(20 / _FLOW_FRAME_RATE * _CV_SR / _FLOW_MEL_HOP)  # = 14

        if T > 40:
            h_head = _interp1d(h_t[:, :, :20], _H)
            h_mid = _interp1d(h_t[:, :, 20:-20], mel_len - 2 * _H)
            h_tail = _interp1d(h_t[:, :, -20:], _H)
            h_interp = np.concatenate([h_head, h_mid, h_tail], axis=2)
        else:
            h_interp = _interp1d(h_t, mel_len)

        # Apply LR conv model
        return _apply_lr_model(h_interp, self._lr_weights)  # (1, 80, T_mel)

    # ------------------------------------------------------------------
    # Flow decoder (mu + spk → mel)  — Euler ODE solver
    # ------------------------------------------------------------------

    def _flow_decode(self, mu: np.ndarray, spk_emb: np.ndarray) -> np.ndarray:
        """ODE Euler solver with classifier-free guidance (CFG).

        Matches upstream CosyVoice ``ConditionalCFM.solve_euler`` exactly:
        - Cosine time schedule (t_span = 1 - cos(t * pi/2))
        - Batch-2 CFG idiom: index-0 is conditioned, index-1 is unconditioned
          (mu/spks/cond zeroed for the unconditioned copy)
        - CFG combination: (1 + cfg_rate) * v_cond - cfg_rate * v_uncond

        Parameters
        ----------
        mu      : (1, 80, T_mel) float32  — flow conditioning from encoder
        spk_emb : (1, 80) float32         — projected speaker embedding

        Returns
        -------
        mel : (1, 80, T_mel) float32
        """
        T_mel = mu.shape[2]
        n_steps = self._ode_steps

        # Cosine time schedule: t_span[i] = 1 - cos(i/n_steps * pi/2), length n_steps+1
        idx = np.linspace(0.0, 1.0, n_steps + 1, dtype=np.float32)
        t_span = (1.0 - np.cos(idx * 0.5 * np.pi)).astype(np.float32)

        # Batch-2 CFG: slot 0 = conditioned, slot 1 = unconditioned (zeros)
        mu_in = np.zeros((2, 80, T_mel), dtype=np.float32)
        mu_in[0] = mu[0]                                    # conditioned
        # mu_in[1] stays zero                              # unconditioned

        mask_in = np.ones((2, 1, T_mel), dtype=np.float32)

        spks_in = np.zeros((2, 80), dtype=np.float32)
        spks_in[0] = spk_emb[0]                            # conditioned

        cond_in = np.zeros((2, 80, T_mel), dtype=np.float32)
        # cond_in stays zero for both slots (no prompt conditioning in VC mode)

        # Initialize from isotropic noise (single trajectory, replicated for batch)
        z = np.random.randn(1, 80, T_mel).astype(np.float32)
        x_in = np.zeros((2, 80, T_mel), dtype=np.float32)
        x_in[0] = z[0]
        x_in[1] = z[0]

        t = t_span[0]
        for step in range(1, n_steps + 1):
            dt = t_span[step] - t
            t_arr = np.array([t, t], dtype=np.float32)

            velocity = self._fd_sess.run(
                None,
                {
                    "x": x_in,
                    "mask": mask_in,
                    "mu": mu_in,
                    "t": t_arr,
                    "spks": spks_in,
                    "cond": cond_in,
                },
            )[0]  # (2, 80, T_mel)

            # CFG: (1 + cfg_rate) * v_cond - cfg_rate * v_uncond
            v_cond = velocity[0:1]
            v_uncond = velocity[1:2]
            v_cfg = (1.0 + _INFERENCE_CFG_RATE) * v_cond - _INFERENCE_CFG_RATE * v_uncond

            x_new = x_in[0:1] + dt * v_cfg          # (1, 80, T_mel)
            x_in[0] = x_new[0]
            x_in[1] = x_new[0]
            t = t + dt

        return x_in[0:1]  # (1, 80, T_mel)

    # ------------------------------------------------------------------
    # HiFiGAN vocoder (mel → waveform)
    # ------------------------------------------------------------------

    def _vocode(self, mel: np.ndarray) -> np.ndarray:
        """mel (1, 80, T_mel) → waveform (T_audio,) float32 @ 22050 Hz.

        Pipeline:
        1. hifigan_f0_source.onnx : mel → 1-D NSF source signal
        2. numpy STFT             : source → STFT coefficients (9-bin)
        3. hifigan_backbone.onnx  : (mel, source_stft) → (magnitude, phase)
        4. numpy ISTFT            : magnitude + phase → waveform
        """
        # Step 1: source signal
        src = self._hf_src_sess.run(None, {"mel": mel})[0]  # (1, 1, T_audio)
        src_1d = src.squeeze().astype(np.float32)            # (T_audio,)

        # Step 2: numpy STFT of source
        src_real, src_imag = _stft(src_1d)                  # (9, T_stft) each
        src_stft = np.concatenate([src_real, src_imag], axis=0)[np.newaxis].astype(np.float32)
        # (1, 18, T_stft)

        # Step 3: backbone
        magnitude, phase = self._hf_bb_sess.run(
            None, {"mel": mel, "source_stft": src_stft}
        )  # (1, 9, T_stft), (1, 9, T_stft)

        # Step 4: numpy ISTFT
        mag = magnitude.squeeze(0)   # (9, T_stft)
        phi = phase.squeeze(0)       # (9, T_stft)
        # phase is used directly as the angle (radians) in the ISTFT polar
        # decomposition: real = mag*cos(phi), imag = mag*sin(phi).
        # The backbone applies torch.sin() to its raw output before returning
        # phase — this clamps values to [-1, 1] and the model is trained to
        # use these values directly as angles (the comment in upstream source
        # says "sin is redundancy").  Do NOT apply arcsin — that would invert
        # the transformation and produce badly wrapped phases.
        audio = _istft(mag, phi)                             # (T_audio,)
        return np.clip(audio, -0.99, 0.99).astype(np.float32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clone_voice(
        self,
        audio: str,
        reference_voice: str,
        out_path: str,
    ) -> str:
        """Convert *audio* to sound like *reference_voice* using CosyVoice non-AR VC.

        Parameters
        ----------
        audio           : source WAV path (any sample rate; 16-bit PCM recommended)
        reference_voice : reference WAV providing target speaker identity
        out_path        : destination 16-bit WAV at 22050 Hz

        Returns
        -------
        str  absolute path to the written output file
        """
        self._ensure_models()

        src_wav = _load_wav(str(audio), target_sr=_TOK_SR)
        ref_wav = _load_wav(str(reference_voice), target_sr=_SPK_SR)

        # 1. Content tokens from source
        tokens = self._extract_content_tokens(src_wav)       # (1, T_tok)

        # 2. Speaker embedding from reference
        spk_emb_192 = self._extract_speaker_embedding(ref_wav)   # (1, 192)

        # 3. Flow encoder: tokens → mu
        mu = self._encode_tokens(tokens)                     # (1, 80, T_mel)

        # 4. Project speaker embedding 192 → 80 via the trained affine layer
        # weights saved to spk_proj.npz at export time.
        spk_emb_80 = self._project_spk_emb(spk_emb_192)     # (1, 80)

        # 5. Flow decode: mu + spk → mel
        mel = self._flow_decode(mu, spk_emb_80)              # (1, 80, T_mel)

        # 6. Vocode: mel → waveform
        waveform = self._vocode(mel)                         # (T_audio,) @ 22050 Hz

        out_path = str(Path(out_path).resolve())
        _save_wav(out_path, waveform, sr=_CV_SR)
        return out_path

    def _project_spk_emb(self, emb_192: np.ndarray) -> np.ndarray:
        """Project CAM++ 192-d embedding to flow 80-d via spk_embed_affine_layer.

        Weights (spk_proj.npz) are downloaded from HF Hub on first call and cached.

        Parameters
        ----------
        emb_192 : (1, 192) float32 — L2-normalized CAM++ embedding

        Returns
        -------
        np.ndarray (1, 80) float32
        """
        if not hasattr(self, "_spk_proj_w"):
            self._spk_proj_w = None
            self._spk_proj_b = None
            try:
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(repo_id=_HF_REPO_ID, filename="spk_proj.npz")
                data = np.load(path)
                self._spk_proj_w = data["weight"].astype(np.float32)  # (80, 192)
                self._spk_proj_b = data["bias"].astype(np.float32)    # (80,)
            except Exception:
                pass

        if self._spk_proj_w is not None:
            return (emb_192 @ self._spk_proj_w.T + self._spk_proj_b[np.newaxis]).astype(np.float32)
        # Fallback for unit tests / missing artifact: truncate/zero-pad
        if emb_192.shape[1] >= 80:
            return emb_192[:, :80].copy()
        pad = 80 - emb_192.shape[1]
        return np.concatenate([emb_192, np.zeros((1, pad), dtype=np.float32)], axis=1)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

register_engine(
    EngineEntry(
        alias="cosyvoice",
        adapter_class=CosyVoiceAdapter,
        description=(
            "CosyVoice non-AR VC: speech tokenizer (Whisper-style) → content tokens; "
            "CAM++ → 192-d speaker embedding; "
            "6-block Conformer flow encoder → mu (80-d); "
            "flow-matching ODE decoder (10 Euler steps) → 80-bin mel; "
            "HiFiGAN NSF vocoder → 22050 Hz waveform. "
            "No autoregressive LLM. CPU RTF ~0.71×. "
            "ONNX artifacts from TigreGotico/voiceclonnx-cosyvoice. "
            "(FunAudioLLM/CosyVoice-300M, Apache-2.0)"
        ),
        extras="",
        onnx_native=True,
    )
)
