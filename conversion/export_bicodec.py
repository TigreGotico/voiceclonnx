"""Export BiCodec (SparkTTS) components to ONNX for vconnx voice conversion.

BiCodec factorizes speech into two complementary token streams:
- Semantic tokens: Wav2Vec2-XLSR-53 (layers 11/14/16 avg) → convolutional
  encoder → single-codebook VQ; carry linguistic content.
- Global tokens: mel-spectrogram → ECAPA-TDNN + Perceiver → FSQ; carry
  timbre/speaker identity.

Voice-conversion recipe: encode source → keep semantic tokens → swap global
tokens from reference → decode.  Single forward pass per chunk, no AR LM.

ONNX components exported
------------------------
- ``wav2vec2_encoder.onnx`` : waveform (1, N) float32 → features (1, T, 1024) float32
  (averages hidden states 11, 14, 16 from Wav2Vec2-XLSR-53)
- ``semantic_encoder.onnx`` : features (1, T, 1024) float32 → semantic_tokens (1, T2) int64
  (BiCodec convolutional encoder + FactorizedVQ.tokenize)
- ``global_encoder.onnx`` : mel (1, 128, T_mel) float32 → global_tokens (1, 32, 1) int64
  (ECAPA-TDNN + Perceiver + FSQ tokenize)
- ``decoder.onnx`` : semantic_tokens (1, T2) int64, global_tokens (1, 32, 1) int64
  → waveform (1, 1, N) float32
  (quantizer.detokenize + speaker_encoder.detokenize + prenet + WaveGenerator)

Upstream
--------
- https://github.com/SparkAudio/Spark-TTS (Apache-2.0 code, CC BY-NC-SA 4.0 weights)
- https://huggingface.co/SparkAudio/Spark-TTS-0.5B

Usage::

    python -m conversion.export_bicodec --output-dir /tmp/bicodec-out
    python -m conversion.export_bicodec --output-dir /tmp/bicodec-out --no-push
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPARK_HF_REPO = "SparkAudio/Spark-TTS-0.5B"
BICODEC_UPSTREAM_URL = "https://github.com/SparkAudio/Spark-TTS"
BICODEC_UPSTREAM_REF = "main"
BICODEC_SR = 16000

# Wav2Vec2 hidden layer indices averaged for semantic features
WAV2VEC2_LAYERS = (11, 14, 16)

# Speaker token count (fixed, from config.yaml token_num=32)
GLOBAL_TOKEN_NUM = 32

CC_BY_NC_SA_4_LICENSE = """\
Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International

Copyright (c) 2025 SparkAudio

This work is licensed under the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International License.

You may not use the material for commercial purposes.
You must give appropriate credit, provide a link to the license, and indicate
if changes were made.  If you remix, transform, or build upon the material,
you must distribute your contributions under the same license as the original.

Full text: https://creativecommons.org/licenses/by-nc-sa/4.0/

NOTICE: These ONNX artifacts are derived from SparkAudio/Spark-TTS-0.5B
weights which are licensed CC BY-NC-SA 4.0.  Non-commercial use only.
Commercial use requires separate permission from the copyright holders.
"""


# ---------------------------------------------------------------------------
# Wrapper modules for ONNX tracing
# ---------------------------------------------------------------------------


def _patch_wav2vec2_for_tracing(wav2vec2_model):
    """Patch transformers Wav2Vec2 to work with TorchScript ONNX tracing.

    Applies the same class of patches documented for MimiModel export:

    1. ``sdpa_mask`` IndexError — ``create_bidirectional_mask`` passes a 0-d
       Tensor as ``q_length``; ``sdpa_mask`` then tries ``q_length.shape[0]``
       (IndexError).  Fix: extract ``int(q_length.item())`` when a 0-d Tensor.

    2. ``find_packed_sequence_indices`` + ``torch.diff`` — patch to return None.
    """
    import torch
    try:
        import transformers.masking_utils as mu

        # Patch 1: sdpa_mask q_length.shape[0] IndexError
        _orig_sdpa_mask = mu.sdpa_mask

        def _patched_sdpa_mask(q_length, kv_length, *args, **kwargs):
            if isinstance(q_length, torch.Tensor) and q_length.dim() == 0:
                q_length = int(q_length.item())
            return _orig_sdpa_mask(q_length, kv_length, *args, **kwargs)

        mu.sdpa_mask = _patched_sdpa_mask

        # Patch 2: torch.diff via find_packed_sequence_indices
        if hasattr(mu, "find_packed_sequence_indices"):
            mu.find_packed_sequence_indices = lambda *a, **kw: None
    except Exception:
        pass  # If transformers structure differs, skip — tracing may still work

    try:
        import transformers.models.wav2vec2.modeling_wav2vec2 as w2v_mod
        # Patch attention mask creation to use bidirectional (no causal mask)
        _orig_encoder_fwd = w2v_mod.Wav2Vec2Encoder.forward

        def _patched_encoder_fwd(self, *args, **kwargs):
            kwargs.pop("attention_mask", None)
            return _orig_encoder_fwd(self, *args, attention_mask=None, **kwargs)

        w2v_mod.Wav2Vec2Encoder.forward = _patched_encoder_fwd
    except Exception:
        pass


def _build_wav2vec2_wrapper(feature_extractor_model):
    """Wrap Wav2Vec2Model to output averaged hidden states 11/14/16.

    Input : waveform (1, N) float32
    Output: features (1, T, 1024) float32

    Applies tracing patches (sdpa_mask IndexError fix) before wrapping,
    consistent with the Mimi export approach documented in docs/converting.md.
    """
    import torch
    import torch.nn as nn

    # Apply tracing patches BEFORE building the wrapper so the patched
    # functions are in place when torch.onnx.export traces the module.
    _patch_wav2vec2_for_tracing(feature_extractor_model)

    class Wav2Vec2Wrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, waveform):
            # waveform: (1, N) float32
            outputs = self.model(waveform, output_hidden_states=True)
            # hidden_states: tuple of (1, T, 1024) tensors, length = num_layers+1
            h11 = outputs.hidden_states[11]
            h14 = outputs.hidden_states[14]
            h16 = outputs.hidden_states[16]
            feats = (h11 + h14 + h16) / 3.0
            return feats  # (1, T, 1024)

    return Wav2Vec2Wrapper(feature_extractor_model).eval()


def _build_semantic_encoder_wrapper(bicodec_model):
    """Wrap BiCodec encoder + quantizer.tokenize.

    Input : features (1, T, 1024) float32  — Wav2Vec2 output (time-first)
    Output: semantic_tokens (1, T2) int64

    The BiCodec encoder accepts (B, D, T) channel-first input and returns
    (B, D, T) channel-first output.  The quantizer.tokenize also expects (B, D, T).
    We transpose the Wav2Vec2 features (B, T, 1024) → (B, 1024, T) before the encoder.
    """
    import torch
    import torch.nn as nn

    class SemanticEncoderWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.encoder = model.encoder
            self.quantizer = model.quantizer

        def forward(self, features):
            # features: (1, T, 1024) time-first from Wav2Vec2
            # encoder expects (B, D, T) channel-first
            z = self.encoder(features.transpose(1, 2))  # (1, 1024, T2)
            # quantizer.tokenize expects (B, D, T) — z is already channel-first
            semantic_tokens = self.quantizer.tokenize(z)  # (1, T2) int64
            return semantic_tokens

    return SemanticEncoderWrapper(bicodec_model).eval()


def _build_global_encoder_wrapper(bicodec_model):
    """Wrap BiCodec speaker_encoder.tokenize with mel input.

    Input : mel (1, 128, T_mel) float32
    Output: global_tokens (1, 1, 32) int32

    Note: speaker_encoder.tokenize receives mel in time-first format (B, T, D),
    matching how BiCodec.tokenize calls it: self.speaker_encoder.tokenize(mel.transpose(1, 2)).
    We accept the channel-first mel (1, 128, T_mel) and transpose internally.
    """
    import torch
    import torch.nn as nn

    class GlobalEncoderWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.speaker_encoder = model.speaker_encoder

        def forward(self, mel):
            # mel: (1, 128, T_mel) channel-first
            # speaker_encoder.tokenize expects (B, T, D_mel) — transpose
            global_tokens = self.speaker_encoder.tokenize(mel.transpose(1, 2))  # (1, 1, 32)
            return global_tokens

    return GlobalEncoderWrapper(bicodec_model).eval()


def _build_decoder_wrapper(bicodec_model):
    """Wrap BiCodec.detokenize: semantic + global tokens → waveform.

    Inputs : semantic_tokens (1, T2) int64
             global_tokens   (1, 1, 32) int32
    Output : waveform (1, 1, N) float32

    This wraps BiCodec.detokenize directly to keep exact parity.
    """
    import torch
    import torch.nn as nn

    class DecoderWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self._model = model

        def forward(self, semantic_tokens, global_tokens):
            # semantic_tokens: (1, T2) int64
            # global_tokens:   (1, 1, 32) int32
            return self._model.detokenize(semantic_tokens, global_tokens)

    return DecoderWrapper(bicodec_model).eval()


# ---------------------------------------------------------------------------
# Mel spectrogram helper (numpy, for inference-time use without torchaudio)
# ---------------------------------------------------------------------------


def _build_mel_wrapper(bicodec_model):
    """Wrap BiCodec's torchaudio MelSpectrogram transformer.

    Input : waveform (1, 1, N) float32
    Output: mel (1, 128, T_mel) float32
    """
    import torch
    import torch.nn as nn

    class MelWrapper(nn.Module):
        def __init__(self, mel_transformer):
            super().__init__()
            self.mel_transformer = mel_transformer

        def forward(self, wav):
            # wav: (1, 1, N)  — squeeze channel for MelSpectrogram
            return self.mel_transformer(wav.squeeze(1))  # (1, 128, T_mel)

    return MelWrapper(bicodec_model.mel_transformer).eval()


# ---------------------------------------------------------------------------
# Main export
# ---------------------------------------------------------------------------


def export_bicodec(output_dir: str, no_push: bool = False) -> None:
    import torch
    import numpy as np
    from huggingface_hub import snapshot_download

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, "/tmp/sparktts-repo")

    from conversion.export_base import (
        OutputLayout,
        export_model,
        write_manifest,
        write_provenance,
    )
    from conversion.quantize import quantize_model

    # ------------------------------------------------------------------
    # 1. Download checkpoints
    # ------------------------------------------------------------------
    print(f"[export] downloading SparkTTS from {SPARK_HF_REPO} ...")
    print("[export]   (this is a 3.95 GB repo; Wav2Vec2 + BiCodec)")
    ckpt_dir = snapshot_download(
        SPARK_HF_REPO,
        ignore_patterns=["LLM/*"],  # skip the 2 GB LLM weights
    )
    ckpt_dir = Path(ckpt_dir)
    bicodec_dir = ckpt_dir / "BiCodec"
    wav2vec2_dir = ckpt_dir / "wav2vec2-large-xlsr-53"

    print(f"[export] checkpoint at {ckpt_dir}")

    # ------------------------------------------------------------------
    # 2. Load BiCodec model
    # ------------------------------------------------------------------
    print("[export] loading BiCodec ...")
    from sparktts.models.bicodec import BiCodec

    bicodec = BiCodec.load_from_checkpoint(bicodec_dir)
    bicodec.eval()

    # ------------------------------------------------------------------
    # 3. Load Wav2Vec2
    # ------------------------------------------------------------------
    print("[export] loading Wav2Vec2-XLSR-53 ...")
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

    wav2vec2_model = Wav2Vec2Model.from_pretrained(str(wav2vec2_dir))
    wav2vec2_model.config.output_hidden_states = True
    wav2vec2_model.eval()

    # ------------------------------------------------------------------
    # 4. Prepare output layout
    # ------------------------------------------------------------------
    layout = OutputLayout.for_engine("bicodec", base_dir=output_dir)
    layout.makedirs()

    # ------------------------------------------------------------------
    # 5. Dummy inputs
    # ------------------------------------------------------------------
    # 1 second of audio at 16 kHz
    N_DUMMY = 16000
    dummy_wav_1d = torch.zeros(1, N_DUMMY)          # for Wav2Vec2: (1, N)
    dummy_wav_3d = dummy_wav_1d.unsqueeze(0)        # (1, 1, N) — unused directly

    # Run Wav2Vec2 forward to get feature shape
    with torch.no_grad():
        w2v_out = wav2vec2_model(dummy_wav_1d, output_hidden_states=True)
        dummy_features = (
            w2v_out.hidden_states[11]
            + w2v_out.hidden_states[14]
            + w2v_out.hidden_states[16]
        ) / 3.0
    # dummy_features: (1, T_feat, 1024)
    print(f"[export] dummy feature shape: {dummy_features.shape}")

    # Semantic tokens from BiCodec
    with torch.no_grad():
        feat_T = dummy_features  # (1, T, 1024)
        z = bicodec.encoder(feat_T.transpose(1, 2))  # (1, 1024, T2) channel-first
        dummy_sem_tokens = bicodec.quantizer.tokenize(z)  # (1, T2) int64
    print(f"[export] dummy semantic_tokens shape: {dummy_sem_tokens.shape}")

    # Mel for global encoder — use BiCodec's mel_transformer on 3s reference
    N_REF = 16000 * 3
    dummy_ref_wav = torch.zeros(1, 1, N_REF)
    with torch.no_grad():
        dummy_mel = bicodec.mel_transformer(dummy_ref_wav.squeeze(1))  # (1, 128, T_mel)
    print(f"[export] dummy mel shape: {dummy_mel.shape}")

    # Global tokens - speaker_encoder.tokenize receives mel.transpose(1, 2) = (B, T_mel, 128)
    with torch.no_grad():
        dummy_global_tokens = bicodec.speaker_encoder.tokenize(dummy_mel.transpose(1, 2))  # (1, 1, 32)
    print(f"[export] dummy global_tokens shape: {dummy_global_tokens.shape}")

    # ------------------------------------------------------------------
    # 6. Export wav2vec2_encoder.onnx
    # ------------------------------------------------------------------
    print("[export] exporting wav2vec2_encoder ...")
    w2v_wrapper = _build_wav2vec2_wrapper(wav2vec2_model)
    # Test wrapper forward
    with torch.no_grad():
        w2v_test = w2v_wrapper(dummy_wav_1d)
    print(f"[export]   wrapper output shape: {w2v_test.shape}")

    w2v_path = export_model(
        model=w2v_wrapper,
        dummy_inputs=(dummy_wav_1d,),
        output_path=layout.component_path("wav2vec2_encoder.onnx"),
        input_names=["waveform"],
        output_names=["features"],
        dynamic_axes={
            "waveform": {1: "num_samples"},
            "features": {1: "num_frames"},
        },
        opset_version=14,
    )
    print(f"[export] wav2vec2_encoder saved: {w2v_path} ({w2v_path.stat().st_size // 1024 // 1024} MB)")

    # Parity check — wav2vec2_encoder
    import onnxruntime as ort

    sess_w2v = ort.InferenceSession(str(w2v_path), providers=["CPUExecutionProvider"])
    ort_features = sess_w2v.run(None, {"waveform": dummy_wav_1d.numpy()})[0]
    max_w2v = float(np.abs(w2v_test.numpy() - ort_features).max())
    mean_w2v = float(np.abs(w2v_test.numpy() - ort_features).mean())
    ok_w2v = max_w2v <= 1e-3 and mean_w2v <= 1e-4
    print(f"[parity] wav2vec2_encoder  max_abs={max_w2v:.2e}  mean_abs={mean_w2v:.2e}  "
          f"{'PASS' if ok_w2v else 'FAIL'}")

    # ------------------------------------------------------------------
    # 7. Export semantic_encoder.onnx
    # ------------------------------------------------------------------
    print("[export] exporting semantic_encoder ...")
    sem_wrapper = _build_semantic_encoder_wrapper(bicodec)
    with torch.no_grad():
        sem_test = sem_wrapper(dummy_features)
    print(f"[export]   wrapper output shape: {sem_test.shape}")

    sem_path = export_model(
        model=sem_wrapper,
        dummy_inputs=(dummy_features,),
        output_path=layout.component_path("semantic_encoder.onnx"),
        input_names=["features"],
        output_names=["semantic_tokens"],
        dynamic_axes={
            "features": {1: "num_frames"},
            "semantic_tokens": {1: "num_tokens"},
        },
        opset_version=14,
    )
    print(f"[export] semantic_encoder saved: {sem_path} ({sem_path.stat().st_size // 1024 // 1024} MB)")

    # Parity check — semantic_encoder (exact int64 match)
    sess_sem = ort.InferenceSession(str(sem_path), providers=["CPUExecutionProvider"])
    ort_sem = sess_sem.run(None, {"features": dummy_features.numpy()})[0]
    exact_sem = bool((sem_test.numpy() == ort_sem).all())
    max_sem = float(np.abs(sem_test.numpy().astype(float) - ort_sem.astype(float)).max())
    print(f"[parity] semantic_encoder  exact_int_match={exact_sem}  max_diff={max_sem:.2e}  "
          f"{'PASS' if exact_sem else 'FAIL'}")

    # ------------------------------------------------------------------
    # 8. Export global_encoder.onnx
    # ------------------------------------------------------------------
    print("[export] exporting global_encoder ...")
    glob_wrapper = _build_global_encoder_wrapper(bicodec)
    with torch.no_grad():
        glob_test = glob_wrapper(dummy_mel)
    print(f"[export]   wrapper output shape: {glob_test.shape}")

    glob_path = export_model(
        model=glob_wrapper,
        dummy_inputs=(dummy_mel,),
        output_path=layout.component_path("global_encoder.onnx"),
        input_names=["mel"],
        output_names=["global_tokens"],
        dynamic_axes={
            "mel": {2: "mel_frames"},
            # global_tokens shape is fixed (1, 1, 32) — no dynamic axes needed
        },
        opset_version=14,
    )
    print(f"[export] global_encoder saved: {glob_path} ({glob_path.stat().st_size // 1024 // 1024} MB)")

    # Parity check — global_encoder
    sess_glob = ort.InferenceSession(str(glob_path), providers=["CPUExecutionProvider"])
    ort_glob = sess_glob.run(None, {"mel": dummy_mel.numpy()})[0]
    exact_glob = bool((glob_test.numpy() == ort_glob).all())
    max_glob = float(np.abs(glob_test.numpy().astype(float) - ort_glob.astype(float)).max())
    print(f"[parity] global_encoder  exact_int_match={exact_glob}  max_diff={max_glob:.2e}  "
          f"{'PASS' if exact_glob else 'FAIL'}")

    # ------------------------------------------------------------------
    # 9. Save mel filterbank as numpy (instead of ONNX)
    #
    # torchaudio MelSpectrogram uses aten::stft which is only supported at
    # opset 17+; ONNX Runtime ships opset ≤ 18 support but the torchaudio
    # stft exporter was not stable at opset 14/17 on this transformers version.
    # Solution: export the mel filterbank matrix (fb) as a .npy file and
    # reimplement the STFT + mel projection in pure numpy in the adapter —
    # the same approach used for FocalCodec's Vocos ISTFT in this repo.
    # ------------------------------------------------------------------
    print("[export] saving mel filterbank (numpy, not ONNX — stft not in opset 14) ...")

    # Extract the learned Hann window and mel filterbank from the model
    # mel_transformer.spectrogram.window: (win_length,) float32
    # mel_transformer.mel_scale.fb: (n_fft//2+1, num_mels) float32
    # Both are Missing from the safetensors (torchaudio registers them as buffers
    # not parameters). We recompute them from the known config.

    mel_config = {
        "sample_rate": BICODEC_SR,
        "n_fft": 1024,
        "win_length": 640,
        "hop_length": 320,
        "mel_fmin": 10.0,
        "mel_fmax": None,
        "num_mels": 128,
        "norm": "slaney",
        "mel_scale": "slaney",
    }

    # Compute mel filterbank using librosa (matches torchaudio slaney norm)
    import librosa
    fb = librosa.filters.mel(
        sr=mel_config["sample_rate"],
        n_fft=mel_config["n_fft"],
        n_mels=mel_config["num_mels"],
        fmin=mel_config["mel_fmin"],
        fmax=mel_config["mel_fmax"],
        norm="slaney",
        htk=False,  # slaney scale, not HTK
    )  # (num_mels, n_fft//2+1) float32

    mel_fb_path = layout.component_path("mel_filterbank.npy")
    np.save(str(mel_fb_path), fb.astype(np.float32))
    print(f"[export] mel_filterbank.npy saved: {mel_fb_path} shape={fb.shape}")

    # Parity check: compare librosa-based numpy mel vs torchaudio mel on 3s dummy
    wav_np = dummy_ref_wav.squeeze().numpy()  # (N,)
    # numpy STFT (matches torchaudio with reflect padding, hann window)
    window_np = np.hanning(mel_config["win_length"]).astype(np.float32)
    n_fft = mel_config["n_fft"]
    hop = mel_config["hop_length"]
    win_len = mel_config["win_length"]
    # Reflect-pad both sides
    pad = n_fft // 2
    wav_padded = np.pad(wav_np, pad, mode="reflect")
    # STFT via numpy
    n_frames = 1 + (len(wav_padded) - n_fft) // hop
    stft_matrix = np.zeros((n_fft // 2 + 1, n_frames), dtype=np.complex64)
    for i in range(n_frames):
        frame = wav_padded[i * hop: i * hop + n_fft]
        # zero-pad frame to n_fft if win_len < n_fft
        windowed = np.zeros(n_fft, dtype=np.float32)
        windowed[:win_len] = frame[:win_len] * window_np
        fft = np.fft.rfft(windowed, n=n_fft)
        stft_matrix[:, i] = fft
    mag = np.abs(stft_matrix)  # (n_fft//2+1, T)
    mel_np = fb @ mag  # (num_mels, T)
    mel_np_3d = mel_np[np.newaxis, :, :]  # (1, 128, T)

    torch_mel = bicodec.mel_transformer(dummy_ref_wav.squeeze(1)).numpy()  # (1, 128, T)
    # Align T dimension (may differ by 1 frame due to padding)
    T_min = min(mel_np_3d.shape[2], torch_mel.shape[2])
    max_mel_p = float(np.abs(mel_np_3d[:, :, :T_min] - torch_mel[:, :, :T_min]).max())
    mean_mel_p = float(np.abs(mel_np_3d[:, :, :T_min] - torch_mel[:, :, :T_min]).mean())
    ok_mel = max_mel_p <= 5e-3 and mean_mel_p <= 1e-3  # looser tol for STFT vs torchaudio
    print(f"[parity] mel_numpy vs torchaudio  max_abs={max_mel_p:.2e}  mean_abs={mean_mel_p:.2e}  "
          f"{'PASS' if ok_mel else 'WARN: loose tolerance'}")
    # Save mel config
    import json
    mel_cfg_path = layout.component_path("mel_config.json")
    with open(mel_cfg_path, "w") as f:
        json.dump(mel_config, f, indent=2)

    # ------------------------------------------------------------------
    # 10. Export decoder.onnx
    # ------------------------------------------------------------------
    print("[export] exporting decoder ...")
    dec_wrapper = _build_decoder_wrapper(bicodec)
    with torch.no_grad():
        dec_test = dec_wrapper(dummy_sem_tokens, dummy_global_tokens)
    print(f"[export]   decoder wrapper output: {dec_test.shape}")

    dec_path = export_model(
        model=dec_wrapper,
        dummy_inputs=(dummy_sem_tokens, dummy_global_tokens),
        output_path=layout.component_path("decoder.onnx"),
        input_names=["semantic_tokens", "global_tokens"],
        output_names=["waveform"],
        dynamic_axes={
            "semantic_tokens": {1: "num_tokens"},
            "waveform": {2: "num_samples"},
            # global_tokens is fixed (1, 1, 32) — no dynamic axis
        },
        opset_version=14,
    )
    print(f"[export] decoder saved: {dec_path} ({dec_path.stat().st_size // 1024 // 1024} MB)")

    # Parity check — decoder
    sess_dec = ort.InferenceSession(str(dec_path), providers=["CPUExecutionProvider"])
    ort_wav = sess_dec.run(None, {
        "semantic_tokens": dummy_sem_tokens.numpy(),
        "global_tokens": dummy_global_tokens.numpy(),
    })[0]
    max_dec = float(np.abs(dec_test.numpy() - ort_wav).max())
    mean_dec = float(np.abs(dec_test.numpy() - ort_wav).mean())
    ok_dec = max_dec <= 1e-3 and mean_dec <= 1e-4
    print(f"[parity] decoder  max_abs={max_dec:.2e}  mean_abs={mean_dec:.2e}  "
          f"{'PASS' if ok_dec else 'FAIL'}")

    # Verify torch round-trip parity via detokenize
    with torch.no_grad():
        torch_roundtrip = bicodec.detokenize(dummy_sem_tokens, dummy_global_tokens)
    max_rt = float(np.abs(torch_roundtrip.numpy() - ort_wav).max())
    print(f"[parity] decoder vs torch.detokenize  max_abs={max_rt:.2e}")

    # ------------------------------------------------------------------
    # 11. Quantize (q8) all components
    # ------------------------------------------------------------------
    parity_results = {
        "wav2vec2_encoder": {"max_abs": max_w2v, "mean_abs": mean_w2v, "pass": ok_w2v},
        "semantic_encoder": {"exact_int_match": exact_sem, "max_diff": max_sem, "pass": exact_sem},
        "global_encoder": {"exact_int_match": exact_glob, "max_diff": max_glob, "pass": exact_glob},
        "mel_numpy_vs_torchaudio": {"max_abs": max_mel_p, "mean_abs": mean_mel_p, "pass": ok_mel},
        "decoder": {"max_abs": max_dec, "mean_abs": mean_dec, "pass": ok_dec},
    }

    print("[export] quantizing components to INT8 ...")
    for component_file in [
        "wav2vec2_encoder.onnx",
        "semantic_encoder.onnx",
        "global_encoder.onnx",
        "decoder.onnx",
    ]:
        comp_path = layout.component_path(component_file)
        try:
            report = quantize_model(str(comp_path))
            print(report.summary())
        except Exception as e:
            print(f"[warn] quantize {component_file}: {e}")

    # ------------------------------------------------------------------
    # 12. Write manifest and provenance
    # ------------------------------------------------------------------
    write_manifest(
        layout=layout,
        components={
            "wav2vec2_encoder": "wav2vec2_encoder.onnx",
            "wav2vec2_encoder_q8": "wav2vec2_encoder_q8.onnx",
            "semantic_encoder": "semantic_encoder.onnx",
            "semantic_encoder_q8": "semantic_encoder_q8.onnx",
            "global_encoder": "global_encoder.onnx",
            "global_encoder_q8": "global_encoder_q8.onnx",
            "mel_filterbank": "mel_filterbank.npy",
            "mel_config": "mel_config.json",
            "decoder": "decoder.onnx",
            "decoder_q8": "decoder_q8.onnx",
        },
        sample_rates={"input": BICODEC_SR, "output": BICODEC_SR},
        metadata={
            "opset": 14,
            "sample_rate": BICODEC_SR,
            "global_token_num": GLOBAL_TOKEN_NUM,
            "wav2vec2_layers": list(WAV2VEC2_LAYERS),
            "vc_recipe": (
                "source semantic tokens (content) + reference global tokens (timbre) → decode"
            ),
            "content_token_set": "semantic (Wav2Vec2-derived VQ)",
            "speaker_token_set": "global (ECAPA-TDNN + FSQ, 32 tokens fixed)",
            "parity": parity_results,
        },
        distributable=True,
    )

    write_provenance(
        engine_dir=layout.engine_dir,
        upstream_repo_url=BICODEC_UPSTREAM_URL,
        upstream_ref=BICODEC_UPSTREAM_REF,
        license_text=CC_BY_NC_SA_4_LICENSE,
        extra={
            "hf_model_id": SPARK_HF_REPO,
            "weight_license": "CC BY-NC-SA 4.0 — non-commercial use only",
            "code_license": "Apache-2.0",
            "vc_recipe": "source_semantic_tokens + reference_global_tokens",
            "wav2vec2_layers_averaged": "11, 14, 16",
            "global_token_count": str(GLOBAL_TOKEN_NUM),
        },
    )

    print(f"[export] artifacts written to {layout.engine_dir}")

    # ------------------------------------------------------------------
    # 13. Push to HF (public, with license stated on card)
    # ------------------------------------------------------------------
    if not no_push:
        from conversion.push_models import push_engine

        push_engine(
            engine_dir=str(layout.engine_dir),
            engine_name="bicodec",
            hf_repo_id="TigreGotico/vconnx-bicodec",
            commit_message="export: add bicodec ONNX artifacts (CC BY-NC-SA 4.0 weights)",
        )
        print("[export] pushed to TigreGotico/vconnx-bicodec")
    else:
        print("[export] --no-push: skipping HF upload")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description="Export BiCodec (SparkTTS) to ONNX")
    ap.add_argument("--output-dir", required=True, help="Staging output directory")
    ap.add_argument("--no-push", action="store_true", help="Skip HF upload")
    args = ap.parse_args()

    export_bicodec(output_dir=args.output_dir, no_push=args.no_push)


if __name__ == "__main__":
    main()
