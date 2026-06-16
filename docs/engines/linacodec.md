# Engine: linacodec

**Family:** Factorized codec + Transformer
**Sample rate:** 48 kHz
**WER:** 8–15%
**INT8:** ⚠ degrades (AdaLN + attention sensitive to INT8 — use fp32)
**License:** Llama 3 Community License / BSD-2-Clause (see model card)
**Model:** [TigreGotico/voiceclonnx-linacodec](https://huggingface.co/TigreGotico/voiceclonnx-linacodec)

---

## Overview

LinaCodec (Yatharth Sharma, 2024) is a codec-based any-to-any voice conversion
system that outputs at **48 kHz** — the highest sample rate of any voiceclonnx
engine. It encodes content via a Llama-3-style windowed Transformer with FSQ
quantization, encodes speaker identity via a ConvNeXT global encoder, and
decodes via a Vocos backbone with a Linkwitz-Riley 4 kHz crossover.

All six neural components run via onnxruntime; zero torch at runtime.

## How it works

1. **Acoustic SSL encoder** (`acoustic_ssl_encoder.onnx`) — WavLM Base Plus,
   layers 1–2 average at 16 kHz → 768-dim acoustic features (speaker identity).
2. **Distilled WavLM encoder** (`distill_wavlm_encoder.onnx`) — WavLM distillate,
   layers 6+9 average at 16 kHz → 768-dim semantic features (content).
3. **Content encoder** (`content_encoder.onnx`) — local Llama-3-style Transformer
   (window attention, RoPE) + Conv1d downsampler (factor 4) + FSQ quantizer
   → content tokens at 12.5 tokens/s.
4. **Global encoder** (`global_encoder.onnx`) — ConvNeXT encoder +
   attentive-statistics pooling → 128-dim speaker embedding.
5. **Mel decoder** (`mel_decoder.onnx`) — mel_prenet + Transformer decoder
   conditioned on global embedding (AdaLN-zero) + mel_postnet → 100-band mel.
6. **Vocos backbone** (`vocos_backbone.onnx`) — ConvNeXT backbone + UpSamplerBlock
   + dual ISTFT head (24 kHz + 48 kHz paths). ISTFT and Linkwitz-Riley crossover
   (cutoff 4 kHz) are computed in pure numpy.

VC recipe: `content = encode_content(source)` → `global = encode_speaker(reference)`
→ `mel = mel_decoder(content, global)` → `audio = vocos(mel)`.

## Config / params

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `quantized` | `bool` | `False` | Load INT8 `*_q8.onnx` models. **Not recommended**: INT8 degrades significantly (AdaLN and attention layers are sensitive). Use fp32 for production. |

## Model and license

The Transformer backbone in the exported ONNX weights derives from Meta's Llama-3
(Llama 3 Community License). The distilled WavLM module derives from torchaudio
(BSD-2-Clause). Both licenses are stated on the HF model card. The upstream code
was used only at export time (external-checkout pattern); no upstream code is
present in this MIT-licensed voiceclonnx repository. Review the HF model card
before commercial use.

| File | fp32 | INT8 |
|------|------|------|
| `acoustic_ssl_encoder.onnx` | 89.7 MB | 22.7 MB |
| `distill_wavlm_encoder.onnx` | 84.2 MB | 21.5 MB |
| `content_encoder.onnx` | 171.5 MB | 43.3 MB |
| `global_encoder.onnx` | 22.2 MB | 5.7 MB |
| `mel_decoder.onnx` | 265.6 MB | 73.5 MB |
| `vocos_backbone.onnx` | 60.8 MB | 19.2 MB |

Total: **694 MB** fp32 / **186 MB** INT8.

## Sample rate

**48 kHz** — the highest among all voiceclonnx engines.

## INT8 note

**INT8 is not recommended for this engine.** The AdaLN-zero conditioning and
attention layers in the mel decoder do not quantize well to INT8 (measured WER
degrades to ~100%). Use fp32. See [QUANTS.md](../QUANTS.md).

## WER

**8–15%** — measured with faster-whisper `base.en` on demo clips.
See [demo/VERIFICATION.md](../../demo/VERIFICATION.md).

## CLI example

```bash
voiceclonnx clone --engine linacodec \
             --audio source.wav \
             --voice reference.wav \
             --out out.wav
```

## Python example

```python
from voiceclonnx import VoiceCloner

cloner = VoiceCloner(engine="linacodec")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 48000
```

## Troubleshooting

**Output is very quiet or silent with `quantized=True`** — INT8 quantization
degrades this engine significantly. Use `quantized=False`.

**Robotic output on very short clips** — the FSQ content encoder runs at
12.5 tokens/s. Clips under 1 s produce only ~12 tokens; the mel decoder may
over-smooth. Prefer references of 3–10 s.

**First run is slow** — six models (~694 MB) download from HF Hub on first use.

## References

- Upstream: [ysharma3501/LinaCodec](https://github.com/ysharma3501/LinaCodec)
- Upstream weights: [YatharthS/LinaCodec](https://huggingface.co/YatharthS/LinaCodec)
