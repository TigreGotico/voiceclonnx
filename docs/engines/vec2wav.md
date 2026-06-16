# Engine: vec2wav

vec2wav 2.0 (Guo et al., Interspeech 2024) — VC-native discrete-token neural vocoder.

Architecture:
1. **vq-wav2vec content encoder** — CNN feature extractor (16 kHz audio → pre-VQ
   features) + pure-numpy VQ discretisation (2 codebook groups × 320 entries × 256 dims)
   → concatenated (L, 512) VQ-vectors.
2. **WavLM-Large speaker encoder** (layer 6) — target 16 kHz audio → (T, 1024)
   continuous features; temporal mean → (1024,) speaker embedding for vocoder
   conditioning.
3. **CTXVEC2WAV frontend** (Conformer cross-attention decoder) — VQ-vectors (content)
   cross-attend to WavLM speaker features → (L, 184) hidden states.
4. **BigVGAN vocoder** (conditioned Snake-Beta activation with alias-free upsampling) —
   hidden states + mean speaker embedding → waveform at 24 kHz.

All neural components run via onnxruntime. The VQ discretisation step is pure numpy
(codebook nearest-neighbour lookup, no ONNX).

ONNX artifacts: [TigreGotico/voiceclonnx-vec2wav](https://huggingface.co/TigreGotico/voiceclonnx-vec2wav).

Output sample rate: **24 kHz**.

**License note:** Code Apache-2.0 (github.com/cantabile-kwok/vec2wav2.0).
Pretrained weights (huggingface.co/cantabile-kwok/vec2wav2.0) are **GPL-3.0**.
The ONNX artifacts in TigreGotico/voiceclonnx-vec2wav are derived from those
weights and therefore also carry the GPL-3.0 term.

---

## Install

```bash
pip install voiceclonnx
```

Dependencies pulled in: `onnxruntime`, `numpy`, `soundfile`.
Models are downloaded from HF Hub on first use.

---

## Config keys

| Key | Type | Default | Description |
|---|---|---|---|
| `quantized` | `bool` | `False` | Use INT8 quantized ONNX models. See [QUANTS.md](../QUANTS.md) for measured WER. |

---

## Model files

| File | Variant | Description |
|---|---|---|
| `vqwav2vec_encoder.onnx` | fp32 | vq-wav2vec CNN feature extractor |
| `vqwav2vec_encoder_q8.onnx` | INT8 | quantized CNN extractor |
| `vqwav2vec_codebook.npy` | — | VQ codebook [2, 320, 256] (numpy, never quantized) |
| `wavlm_speaker.onnx` | fp32 | WavLM-Large layer-6 speaker encoder |
| `wavlm_speaker_q8.onnx` | INT8 | quantized speaker encoder |
| `vec2wav_frontend.onnx` | fp32 | CTXVEC2WAV Conformer frontend |
| `vec2wav_frontend_q8.onnx` | INT8 | quantized frontend |
| `vec2wav_vocoder.onnx` | fp32 | BigVGAN vocoder |
| `vec2wav_vocoder_q8.onnx` | INT8 | quantized vocoder |

---

## Usage

### Python

```python
from voiceclonnx import VoiceCloner

cloner = VoiceCloner(engine="vec2wav")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 24000
```

INT8 (smaller, faster):

```python
cloner = VoiceCloner(engine="vec2wav", quantized=True)
```

### CLI

```bash
voiceclonnx convert --engine vec2wav source.wav reference.wav out.wav
```

---

## References

- Paper: [vec2wav 2.0: Advancing Voice Conversion via Discrete Token Vocoders](https://arxiv.org/abs/2409.01995) (Guo et al., Interspeech 2024)
- Code: [cantabile-kwok/vec2wav2.0](https://github.com/cantabile-kwok/vec2wav2.0) (Apache-2.0)
- Weights: [cantabile-kwok/vec2wav2.0](https://huggingface.co/cantabile-kwok/vec2wav2.0) (GPL-3.0)
