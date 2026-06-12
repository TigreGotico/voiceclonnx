# Engine: bicodec

BiCodec (SparkAudio/Spark-TTS, 2025) — zero-shot any-to-any voice conversion
via explicit factorization of speech into semantic tokens (content) and global
tokens (speaker identity).  Voice conversion is a direct integer token swap,
requiring no auto-regressive LM.

Architecture:
1. **Wav2Vec2-XLSR-53** (`wav2vec2_encoder.onnx`) — encodes source waveform to
   (1, T, 1024) hidden states (average of layers 11, 14, 16).
2. **Semantic encoder** (`semantic_encoder.onnx`) — convolutional encoder +
   FactorizedVQ → (1, T2) int64 semantic token indices.  Carry the phoneme
   sequence (content).
3. **Mel filterbank** (`mel_filterbank.npy` + `mel_config.json`) — 128-bin
   Slaney mel filterbank computed in pure numpy from 16 kHz reference audio.
4. **Global encoder** (`global_encoder.onnx`) — ECAPA-TDNN + Perceiver
   resampler + FSQ → (1, 1, 32) int32 global token indices.  Fixed-length (32
   tokens) per utterance; carry timbre / speaker identity.
5. **Decoder** (`decoder.onnx`) — quantizer detokenize + WaveGenerator →
   (1, 1, N) float32 waveform.

Voice-conversion step: `source_semantic_tokens + reference_global_tokens → decoder → waveform`.

ONNX artifacts: [TigreGotico/voiceclonnx-bicodec](https://huggingface.co/TigreGotico/voiceclonnx-bicodec)
(public, **CC BY-NC-SA 4.0 weights — non-commercial use only**; code Apache-2.0).

Output sample rate: **16 kHz**.

---

## Install

```bash
pip install voiceclonnx
```

Dependencies: `onnxruntime`, `numpy`, `soundfile`, `huggingface_hub`.
Models are downloaded from HF Hub on first use (~1.5 GB fp32 combined, ~1 GB INT8).

---

## Config keys

| Key | Type | Default | Description |
|---|---|---|---|
| `quantized` | `bool` | `False` | Use INT8 quantized ONNX models. Reduces footprint from ~1.4 GB to ~419 MB; slight quality cost. See [QUANTS.md](../QUANTS.md) for the measured WER comparison. |
| `chunk_samples` | `int` | `32000` | Wav2Vec2 chunking window in samples (2 s at 16 kHz). Set to `0` to disable chunking (full audio in one pass; may degrade on long sequences). |

---

## Model sizes

| File | Size | Variant |
|---|---|---|
| `wav2vec2_encoder.onnx` | ~819 MB | fp32 |
| `wav2vec2_encoder_q8.onnx` | ~205 MB | INT8 |
| `semantic_encoder.onnx` | ~116 MB | fp32 |
| `semantic_encoder_q8.onnx` | ~34 MB | INT8 |
| `global_encoder.onnx` | ~22 MB | fp32 |
| `global_encoder_q8.onnx` | ~6 MB | INT8 |
| `decoder.onnx` | ~varies MB | fp32 |
| `decoder_q8.onnx` | ~varies MB | INT8 |
| `mel_filterbank.npy` | ~256 KB | numpy (128×513 float32) |
| `mel_config.json` | ~1 KB | JSON |

---

## Parity vs torch

| Component | Metric | Value | Result |
|---|---|---|---|
| wav2vec2_encoder | max_abs | 6.71e-4 | PASS |
| semantic_encoder | exact int match | True | PASS |
| global_encoder | exact int match | True | PASS |
| mel numpy vs torchaudio | max_abs | ≤5e-3 | PASS |
| decoder | max_abs | ≤1e-3 | PASS |

Mel spectrogram parity is checked with a looser tolerance (max_abs ≤5e-3) because
the numpy STFT implementation uses reflect-padding and Hann windowing to match
torchaudio's `MelSpectrogram(norm="slaney")` — opset 14 does not support
`aten::stft`, so the mel is computed in pure numpy using a pre-saved librosa
filterbank.

---

## Usage

### Python

```python
from voiceclonnx import VoiceCloner

# Default (fp32)
cloner = VoiceCloner(engine="bicodec")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 16000

# INT8 quantized (faster, smaller memory footprint)
cloner = VoiceCloner(engine="bicodec", quantized=True)
out = cloner.clone_voice("source.wav", "reference.wav", "out_q8.wav")

# Shorter chunking for memory-constrained machines
cloner = VoiceCloner(engine="bicodec", chunk_samples=16000)
```

### CLI

```bash
pip install voiceclonnx

voiceclonnx clone --engine bicodec \
             --audio source.wav \
             --voice reference.wav \
             --out out.wav
```

---

## License note

ONNX artifacts in `TigreGotico/voiceclonnx-bicodec` are derived from
`SparkAudio/Spark-TTS-0.5B` weights, which are released under
**CC BY-NC-SA 4.0**.  Non-commercial use only.  Attribution required.
See the model card on Hugging Face for the full license text.

---

## References

- Paper: [Spark-TTS: An Efficient LLM-Based Text-to-Speech Model with Single-Stream Decoupled Speech Tokens](https://arxiv.org/abs/2503.01710)
- Upstream repo: [SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS)
- Upstream checkpoint: [SparkAudio/Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B)

---

## Troubleshooting

**`ImportError: onnxruntime is required`**
Install with: `pip install voiceclonnx`.

**Output sounds noisy or robotic**
Ensure source and reference clips are clean mono audio at or near 16 kHz.  The
semantic / global token split is architecturally fixed (not a tunable dial);
ensure reference audio is at least 3 s long for a stable global token estimate.

**Out-of-memory on large files**
Use `quantized=True` (footprint drops ~40%) or reduce `chunk_samples` to process
smaller windows of the source waveform.

**Wav2Vec2 is slow**
Wav2Vec2-XLSR-53 is a large model (~300 MB of parameters).  On CPU, inference
takes a few seconds per second of audio.  Use `quantized=True` for faster throughput.

**First run is slow**
Models are downloaded from HF Hub on first use and cached in
`~/.cache/huggingface/hub`.  Total download is ~1.5 GB (fp32).
