# Engine: lscodec

**Family:** Speaker-decoupled discrete codec
**Sample rate:** 24 kHz
**WER:** ~35% (moderate; content is intelligible but the 50 Hz / 300-token
bitrate degrades some words)
**Speaker similarity:** **~0.54** (best in class, see
[demo/SPEAKER_SIMILARITY.md](../../demo/SPEAKER_SIMILARITY.md))
**INT8:** available (`quantized=True`)
**License:** MIT (LSCodec code; WavLM-Large MIT)
**Model:** [TigreGotico/voiceclonnx-lscodec](https://huggingface.co/TigreGotico/voiceclonnx-lscodec)

---

> **Strong timbre, moderate intelligibility.** LSCodec prioritises voice
> identity over transcription: it transfers the **target voice** better than any
> other any-to-any engine here (cosine ≈ 0.54 against a no-conversion baseline of
> 0.09), at the cost of moderate WER. Reach for it when *who is speaking* matters
> more than perfect transcription.

## Overview

LSCodec (Guo et al., Interspeech 2025) is a low-bitrate discrete speech codec
trained with **speaker decoupling** as a primary objective. Its discrete
content space is speaker-agnostic by design, which makes voice conversion
direct and high-fidelity for timbre: encode the source to speaker-agnostic
tokens, then resynthesize conditioned on the target speaker.

## How it works

1. **Encoder** (`lscodec_encoder.onnx`): raw 16 kHz source audio to 64-dim
   continuous `means` at 50 Hz (conv feature extractor with a
   relative-self-attention conformer).
2. **Numpy VQ**: Euclidean nearest-neighbour of `means` to the 300-entry
   codebook (`codebook.npy`) → speaker-agnostic content vectors.
3. **WavLM-Large layer-6** (`wavlm_l6.onnx`): converts the *reference*
   (target) clip to 1024-dim prompt features. It is exported at a fixed
   **4 s window**, so references are padded or cropped to 64000 samples.
4. **CTXVEC2WAV vocoder** (`lscodec_vocoder.onnx`): converts content
   vectors and prompt features to a 24 kHz waveform.

All of these run via onnxruntime, with no torch at runtime.

## Verification

The ONNX pipeline reproduces the upstream PyTorch reference (speaker-embedding
cosine ≈ **0.97** ONNX↔torch) and transfers the target voice (target-similarity
≈ 0.54, source-similarity ≈ 0.19). Component parity vs torch: encoder max_abs
6e-3, WavLM 5e-4, vocoder 6e-6.

## Config / params

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `quantized` | `bool` | `False` | Load INT8 `*_q8.onnx` models (~154 MB vs ~555 MB fp32). fp32 is the supported quality path |

## INT8 note

INT8 weights are provided for all three models. fp32 is recommended for
best quality. See [QUANTS.md](../QUANTS.md).

## CLI example

```bash
voiceclonnx clone --engine lscodec \
             --audio source.wav \
             --voice reference.wav \
             --out out.wav
```

## Python example

```python
from voiceclonnx import VoiceCloner

cloner = VoiceCloner(engine="lscodec")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 24000
```

## References

- Upstream: [X-LANCE/LSCodec-Inference](https://github.com/X-LANCE/LSCodec-Inference) (MIT)
- Paper: [arXiv:2410.15764](https://arxiv.org/abs/2410.15764)
- Demo: [cantabile-kwok.github.io/LSCodec](https://cantabile-kwok.github.io/LSCodec/)

---
[← focalcodec](focalcodec.md) · [Home](../index.md) · [rvc →](rvc.md)
