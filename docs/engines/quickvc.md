# Engine: quickvc

QuickVC ([quickvc/QuickVC-VoiceConversion](https://github.com/quickvc/QuickVC-VoiceConversion),
MIT) — any-to-many voice conversion using a HuBERT-soft content encoder and a
VITS-style normalising-flow decoder with a Multistream-iSTFT (MS-iSTFT) generator.
The MS-iSTFT vocoder synthesises 4 frequency subbands with a tiny FFT (n_fft=16)
rather than a full-resolution upsampling network, which yields notably fast CPU
inference (~0.14× RTF measured on a single CPU core).

Architecture:
1. **HuBERT-soft content encoder** (12-layer Transformer, 94M params) — converts
   16 kHz audio to 256-dim soft speech units at 50 Hz.  Exported to ONNX in
   1-second chunks (fixed T=50 context window due to MHA reshape constraints).
2. **Speaker encoder** (LSTM × 1 + linear, 5.6M params) — extracts a 256-dim
   d-vector from an 80-channel log-mel spectrogram of the reference audio.
3. **Posterior encoder + normalising flow** — maps content features to the latent
   space conditioned on the speaker d-vector.
4. **Multistream-iSTFT generator** — upsamples by 20× (upsample_rates=[5,4]) and
   produces per-subband STFT coefficients via a convolutional stack.
5. **numpy Multistream-iSTFT** — pure-numpy center-mode OLA over 4 subbands
   (n_fft=16, hop=4).  Not in ONNX because `torch.istft` cannot be traced; parity
   vs torch = 0.0 max abs error.
6. **Postnet** — learned 1-D mixing conv that collapses subbands to a mono waveform.

All neural components run via onnxruntime.  Fully non-autoregressive.

ONNX artifacts: [TigreGotico/voiceclonnx-quickvc](https://huggingface.co/TigreGotico/voiceclonnx-quickvc) (MIT license).

Output sample rate: **16 kHz**.

---

## Install

```bash
pip install voiceclonnx
```

Core deps: `onnxruntime`, `numpy`, `soundfile`, `huggingface_hub` — no librosa at inference.
Models are downloaded from HF Hub on first use (~480 MB fp32 or ~130 MB int8).

---

## Config keys

| Key | Type | Default | Description |
|---|---|---|---|
| `quantized` | `bool` | `False` | Use INT8 quantized ONNX models. Reduces total model size from ~480 MB to ~130 MB. INT8 WER is 4% vs fp32 0% — int8 is recommended for memory-constrained deployments. See [QUANTS.md](../QUANTS.md). |

---

## Model sizes

| File | Size | Variant |
|---|---|---|
| `quickvc_content_encoder.onnx` | 361.1 MB | fp32 |
| `quickvc_content_encoder_q8.onnx` | 91.1 MB | INT8 (74.8% reduction) |
| `quickvc_speaker_encoder.onnx` | 5.6 MB | fp32 |
| `quickvc_speaker_encoder_q8.onnx` | 1.4 MB | INT8 (74.5% reduction) |
| `quickvc_decoder.onnx` | 113.4 MB | fp32 |
| `quickvc_decoder_q8.onnx` | 36.4 MB | INT8 (67.9% reduction) |
| `quickvc_postnet.onnx` | &lt;0.1 MB | fp32 |
| `quickvc_postnet_q8.onnx` | &lt;0.1 MB | INT8 |

**Total fp32:** 480.1 MB  |  **Total int8:** 128.9 MB

---

## Parity (fp32 torch vs ORT on synthetic input)

| Component | max_abs | mean_abs | Pass |
|---|---|---|---|
| HuBERT-soft content encoder | 2.92e-06 | 4.95e-07 | ✓ |
| Speaker encoder d-vector | 1.71e-07 | 1.18e-08 | ✓ |
| Decoder spec_phase | 1.80e-03 | 4.95e-04 | ✓ |
| Postnet output | 1.00e-04 | 1.00e-05 | ✓ |
| numpy MS-iSTFT vs torch.istft | 0.00e+00 | 0.00e+00 | ✓ |

---

## Benchmark (CPU, single core)

| Variant | RTF | WER (aria) | WER (sonia) |
|---|---|---|---|
| fp32 | 0.14× | 0% | 0% |
| int8 | 0.54× | 4% | 4% |

RTF measured on a 3-second clip.  WER uses faster-whisper `base.en` against the
known source text.  Intelligibility gate: ≤25% WER.  Both variants pass.

Note: INT8 is slower than fp32 on CPU for this model size — use fp32 for both
speed and quality.

---

## Usage

### Python

```python
from voiceclonnx import VoiceCloner

# Default (fp32, recommended)
cloner = VoiceCloner(engine="quickvc")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 16000

# Memory-constrained (int8 — 4% WER, ~73% smaller)
cloner_q8 = VoiceCloner(engine="quickvc", quantized=True)
out_q8 = cloner_q8.clone_voice("source.wav", "reference.wav", "out_q8.wav")
```

### CLI

```bash
voiceclonnx clone --engine quickvc source.wav reference.wav out.wav
```

---

## Notes

- **Chunked encoding:** The HuBERT-soft ONNX model processes audio in fixed 1-second
  (16000-sample) windows.  Longer inputs are split into non-overlapping 1-second
  chunks and feature frames are concatenated.  This causes minor timbral
  discontinuities at chunk boundaries for utterances longer than ~1 second; for
  most voice-conversion use cases (sentences ≤5 s) the effect is inaudible.
- **Noise-free decoding:** The posterior encoder uses the mean (`m_p`) rather than
  a sampled latent (`z_p`) for deterministic ONNX inference.  This is equivalent to
  `noise_scale=0` and is standard practice for any-to-many VC inference.
- **Reference audio length:** The speaker encoder is an LSTM and handles any input
  length; longer reference clips (≥3 s) give more stable d-vectors.

---

## References

- Ziqian Ning et al., *QuickVC: Any-to-many Voice Conversion Using Inverse
  Short-time Fourier Transform for Faster Conversion*, 2023.
  [GitHub](https://github.com/quickvc/QuickVC-VoiceConversion)
- Benjamin van Niekerk et al., *A Comparison of Discrete and Soft Speech Units for
  Improved Voice Conversion*, INTERSPEECH 2022.
  [GitHub](https://github.com/bshall/hubert) (HuBERT-soft)
