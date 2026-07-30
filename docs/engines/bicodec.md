# Engine: bicodec

**Family:** Factorized codec
**Sample rate:** 16 kHz
**WER:** 12%
**INT8:** available (slight quality cost)
**License:** CC BY-NC-SA 4.0 weights (non-commercial only); code Apache-2.0
**Model:** [TigreGotico/voiceclonnx-bicodec](https://huggingface.co/TigreGotico/voiceclonnx-bicodec)

---

## Overview

BiCodec (SparkAudio/Spark-TTS, 2025) performs zero-shot any-to-any voice
conversion via explicit factorization of speech into semantic tokens (content)
and global tokens (speaker identity). Voice conversion is a direct integer token
swap, with no autoregressive LM. Each segment needs a single forward pass.

## How it works

1. **Wav2Vec2-XLSR-53** (`wav2vec2_encoder.onnx`): encodes source waveform to
   (1, T, 1024) hidden states (average of layers 11, 14, 16).
2. **Semantic encoder** (`semantic_encoder.onnx`): convolutional encoder +
   FactorizedVQ → (1, T2) int64 semantic tokens (content / phoneme sequence).
3. **Mel filterbank** (`mel_filterbank.npy` + `mel_config.json`): 128-bin
   Slaney mel filterbank computed in pure numpy from 16 kHz reference audio.
4. **Global encoder** (`global_encoder.onnx`): ECAPA-TDNN + Perceiver
   resampler + FSQ → (1, 1, 32) int32 global tokens (fixed-length speaker identity).
5. **Decoder** (`decoder.onnx`): quantizer detokenize + WaveGenerator to
   (1, 1, N) float32 waveform.

Conversion step: `source_semantic_tokens + reference_global_tokens → decoder → waveform`.

## Config / params

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `quantized` | `bool` | `False` | Load INT8 `*_q8.onnx` models. Footprint drops from ~1.4 GB to ~419 MB, at a slight quality cost. |
| `chunk_samples` | `int` | `32000` | Wav2Vec2 chunking window (2 s at 16 kHz). Set `0` to disable (may degrade on long sequences). |

## Model and license

ONNX artifacts derived from `SparkAudio/Spark-TTS-0.5B` weights.
**License: CC BY-NC-SA 4.0, non-commercial use only. Attribution is required.**
See the model card on Hugging Face for the full license text.

| File | fp32 | INT8 |
|------|------|------|
| `wav2vec2_encoder.onnx` | ~819 MB | ~205 MB |
| `semantic_encoder.onnx` | ~116 MB | ~34 MB |
| `global_encoder.onnx` | ~22 MB | ~6 MB |
| `decoder.onnx` | varies | varies |
| `mel_filterbank.npy` | ~256 KB | ~256 KB |

## Sample rate

**16 kHz.** Source and reference are resampled to 16 kHz before processing.

## INT8 note

`quantized=True` loads the `*_q8.onnx` variants (~419 MB vs ~1.4 GB fp32).
Slight quality degradation is expected. See [QUANTS.md](../QUANTS.md).

## WER

**12%**: measured with faster-whisper `base.en` on demo clips.
See [demo/VERIFICATION.md](../../demo/VERIFICATION.md).

## CLI example

```bash
voiceclonnx clone --engine bicodec \
             --audio source.wav \
             --voice reference.wav \
             --out out.wav
```

## Python example

```python
from voiceclonnx import VoiceCloner

cloner = VoiceCloner(engine="bicodec")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 16000

# INT8, smaller footprint
cloner = VoiceCloner(engine="bicodec", quantized=True)

# Shorter chunk for memory-constrained hardware
cloner = VoiceCloner(engine="bicodec", chunk_samples=16000)
```

## Troubleshooting

**Output sounds noisy or robotic.** Use clean mono audio at or near 16 kHz
for source and reference. The reference should be at least 3 s long for a
stable global token estimate.

**Out-of-memory on large files.** Use `quantized=True` or reduce
`chunk_samples`.

**Wav2Vec2 is slow.** The encoder has about 300 MB of parameters. Use
`quantized=True` for faster throughput. Models are cached after the first
download (~1.5 GB).

## References

- Paper: [Spark-TTS: An Efficient LLM-Based Text-to-Speech Model](https://arxiv.org/abs/2503.01710)
- Upstream: [SparkAudio/Spark-TTS](https://github.com/SparkAudio/Spark-TTS)
- Upstream checkpoint: [SparkAudio/Spark-TTS-0.5B](https://huggingface.co/SparkAudio/Spark-TTS-0.5B)

---
[← cosyvoice](cosyvoice.md) · [Home](../index.md) · [knnvc →](knnvc.md)
