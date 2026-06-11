# Engine: freevc

FreeVC (Qian et al., ICASSP 2023) — zero-shot any-to-any voice conversion without
text annotations.  Uses WavLM-Large content features and a VITS-based decoder
conditioned on a GE2E speaker embedding.  Non-autoregressive and CPU-friendly.

Architecture:
1. **WavLM-Large encoder** (full final hidden states) — converts 16 kHz audio to
   1024-dim feature frames at 50 Hz.  FreeVC uses `extract_features()[0]`, the
   transformer's complete final output.  This differs from kNN-VC which extracts
   only layer 6; `wavlm_freevc.onnx` is a separate artifact.
2. **GE2E speaker encoder** (LSTM × 3 + linear) — produces a 256-dim d-vector from
   a log-mel spectrogram of the reference audio.
3. **VITS SynthesizerTrn** (prior encoder + normalising flow + HiFi-GAN generator)
   — decodes content features conditioned on the d-vector to a 16 kHz waveform.

All neural components run via onnxruntime.  Fully non-autoregressive; no diffusion
steps; streaming-friendly.

ONNX artifacts: [TigreGotico/vconnx-freevc](https://huggingface.co/TigreGotico/vconnx-freevc) (MIT license).

Output sample rate: **16 kHz**.

---

## Install

```bash
pip install "vconnx[freevc]"
```

Dependencies pulled in: `onnxruntime`, `numpy`, `soundfile`, `librosa`.
Models are downloaded from HF Hub on first use (~1.2 GB fp32 or ~342 MB int8).

---

## Config keys

| Key | Type | Default | Description |
|---|---|---|---|
| `quantized` | `bool` | `False` | Use INT8 quantized ONNX models. Reduces memory from ~1.3 GB to ~342 MB total; slightly lower quality. |

---

## Model sizes

| File | Size | Variant |
|---|---|---|
| `wavlm_freevc.onnx` | 1204.4 MB | fp32 |
| `wavlm_freevc_q8.onnx` | 303.5 MB | INT8 (74.8% reduction) |
| `speaker_encoder.onnx` | 5.4 MB | fp32 |
| `speaker_encoder_q8.onnx` | 1.4 MB | INT8 (74.5% reduction) |
| `freevc_decoder.onnx` | 116.4 MB | fp32 |
| `freevc_decoder_q8.onnx` | 37.3 MB | INT8 (68.0% reduction) |

---

## Parity (fp32 torch vs ORT on synthetic input)

| Component | max_abs | mean_abs | Pass |
|---|---|---|---|
| WavLM-Large last_hidden_state | 4.05e-05 | 3.96e-06 | ✓ |
| Speaker encoder embedding | 2.53e-07 | 3.80e-08 | ✓ |
| VITS decoder waveform | 6.80e-06 | 4.48e-07 | ✓ |

---

## Usage

### Python

```python
from vconnx import VoiceCloner

# Default (fp32)
cloner = VoiceCloner(engine="freevc")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 16000

# Quantized (low-memory)
cloner = VoiceCloner(engine="freevc", quantized=True)
out = cloner.clone_voice("source.wav", "reference.wav", "out_q8.wav")
```

### CLI

```bash
pip install "vconnx[freevc]"

vconnx clone --engine freevc \
             --audio source.wav \
             --voice reference.wav \
             --out out.wav
```

---

## WavLM artifact note

FreeVC and kNN-VC both use WavLM-Large but extract different outputs:

| Engine | Extraction | File |
|---|---|---|
| kNN-VC | Layer-6 hidden states (`hidden_states[7]`) | `TigreGotico/vconnx-knn-vc / wavlm_layer6.onnx` |
| FreeVC | Final transformer output (`last_hidden_state`) | `TigreGotico/vconnx-freevc / wavlm_freevc.onnx` |

These are **not interchangeable**.  The adapter downloads `wavlm_freevc.onnx` from
its own repo; the kNN-VC file is not referenced.

---

## References

- Paper: [FreeVC (Qian et al., ICASSP 2023)](https://arxiv.org/abs/2210.15418)
- Original code: [OlaWod/FreeVC](https://github.com/OlaWod/FreeVC)
- Export reference: [OpenVINO notebooks — FreeVC](https://docs.openvino.ai/2024/notebooks/freevc-voice-conversion-with-output.html)

---

## Troubleshooting

**`ImportError: onnxruntime is required`**
Install the extras group: `pip install "vconnx[freevc]"`.

**`ImportError: librosa`**
Same fix: `pip install "vconnx[freevc]"` pulls librosa.

**Output sounds muffled or robotic**
The conversion quality depends on having a clean reference clip that is at least
3–5 seconds long.  Ensure the reference has minimal background noise.

**Out-of-memory on large files**
Use `quantized=True` — total model footprint drops from ~1.3 GB to ~342 MB.

**First run is slow**
Models are downloaded from HF Hub on first use (~1.2 GB for fp32).  After the
initial download they are cached in `~/.cache/huggingface/hub`.
