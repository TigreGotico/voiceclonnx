# Engine: facodec

**Family:** Factorized codec
**Sample rate:** 16 kHz
**WER:** 0%
**INT8:** available (slight quality cost)
**License:** Apache-2.0
**Model:** [TigreGotico/voiceclonnx-facodec](https://huggingface.co/TigreGotico/voiceclonnx-facodec)

---

## Overview

FACodec (Factorized Audio Codec) is the core speech representation model from
NaturalSpeech 3 (Microsoft Research / Amphion, ICML 2024). It disentangles
speech into four independently controllable subspaces: content, prosody, timbre,
and acoustic detail. Voice conversion is zero-shot: encode source and reference,
swap only the timbre component from the reference, decode the combination.
No per-speaker fine-tuning required.

WER 0% — **recommended as the default engine for highest quality**.

## How it works

| Component | Description |
|-----------|-------------|
| Encoder (V2) | Convolutional downsampler (200× hop, 16 kHz → 80 Hz) |
| Timbre extractor | 4-layer Transformer encoder → mean-pool → 256-d speaker embedding |
| Quantizer | Hierarchical factorized VQ: prosody (1 codebook) + content (2) + residual (3) = 6 total |
| Decoder (V2) | Upsampling convolutional synthesizer with AdaIN timbre conditioning |

VC recipe:

```
enc_feats_src = encoder(wav_src)         # (1, 256, T_src)
enc_feats_ref = encoder(wav_ref)         # (1, 256, T_ref)
mel_src       = prosody_mel(wav_src)     # (1, 20, T_src) — pure numpy
vq_ids_src    = quantize(enc_feats_src, mel_src)   # (6, 1, T_src) int64
spk_embs_ref  = timbre(enc_feats_ref)    # (1, 256)
wav_out       = decode(vq_ids_src, spk_embs_ref)   # prosody+content from src, timbre from ref
```

The prosody mel (step 3) is computed in pure numpy — standard STFT mel (n_fft=1024,
hop=200, win=800, n_mels=80, sr=16000); first 20 bins used.

## Config / params

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `quantized` | `bool` | `False` | Load INT8 `*_q8.onnx` models. Footprint drops from ~156 MB to ~69 MB; slight quality cost. |

## Model and license

**Apache-2.0.** Upstream: NaturalSpeech 3 / Amphion.

| File | fp32 | INT8 |
|------|------|------|
| `facodec_encoder.onnx` | — | — |
| `facodec_timbre.onnx` | — | — |
| `facodec_quantize.onnx` | — | — |
| `facodec_decoder.onnx` | — | — |

Total: ~156 MB fp32 / ~69 MB INT8.

## Sample rate

**16 kHz.**

## INT8 note

`quantized=True` reduces footprint from ~156 MB to ~69 MB. Slight quality
degradation expected. See [QUANTS.md](../QUANTS.md) for the WER comparison.

## WER

**0%** — perfectly intelligible on demo clips.
See [demo/VERIFICATION.md](../../demo/VERIFICATION.md).

## CLI example

```bash
voiceclonnx clone --engine facodec \
             --audio source.wav \
             --voice reference.wav \
             --out out.wav
```

## Python example

```python
from voiceclonnx import VoiceCloner

cloner = VoiceCloner(engine="facodec")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 16000

# INT8 — smaller footprint
cloner = VoiceCloner(engine="facodec", quantized=True)
```

## Troubleshooting

**Output timbre not changing** — ensure the reference clip is clean and at least
2–3 s long. The timbre extractor mean-pools over the full utterance.

**First run is slow** — models download from HF Hub on first use; cached in
`~/.cache/huggingface/hub`.

## References

- Paper: [NaturalSpeech 3: Zero-Shot Polyglot Speech Synthesis](https://arxiv.org/abs/2403.03100)
- Upstream: [amphion/Amphion](https://github.com/open-mmlab/Amphion)
