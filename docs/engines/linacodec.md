# Engine: linacodec

LinaCodec (Yatharth Sharma, 2024) — codec-based any-to-any voice conversion at 48 kHz.

Architecture (all neural components run via onnxruntime; zero torch at runtime):

1. **Acoustic SSL encoder** (`acoustic_ssl_encoder.onnx`) — WavLM Base Plus, layers 1–2
   average at 16 kHz → 768-dim acoustic features for speaker identity.
2. **Distilled WavLM encoder** (`distill_wavlm_encoder.onnx`) — WavLM distillate,
   layers 6+9 average at 16 kHz → 768-dim semantic features for content.
3. **Content encoder** (`content_encoder.onnx`) — local Llama-3-style Transformer
   (window attention, RoPE) + Conv1d downsampler (factor 4) + FSQ quantizer
   → content tokens at 12.5 tokens/s.
4. **Global encoder** (`global_encoder.onnx`) — ConvNeXT encoder +
   attentive-statistics pooling → 128-dim speaker embedding.
5. **Mel decoder** (`mel_decoder.onnx`) — mel_prenet + Transformer decoder
   conditioned on global embedding (AdaLN-zero) + mel_postnet → 100-band mel
   spectrogram.
6. **Vocos backbone** (`vocos_backbone.onnx`) — ConvNeXT backbone + UpSamplerBlock
   + dual ISTFT head: 24 kHz path (Vocos ISTFTHead) and 48 kHz path.
   Outputs raw magnitude/phase tensors; ISTFT and Linkwitz-Riley crossover
   (cutoff 4 kHz) are done in pure numpy.

VC recipe: `content_embedding = encode_content(source)`; `global_embedding =
encode_speaker(reference)`; `mel = mel_decoder(content, global)`; `audio =
vocos(mel)`.

ONNX artifacts: [TigreGotico/voiceclonnx-linacodec](https://huggingface.co/TigreGotico/voiceclonnx-linacodec).

**License note**: The Transformer backbone in the exported ONNX weights derives
from Meta's Llama-3 (Llama 3 Community License). The distilled WavLM module
derives from torchaudio (BSD-2-Clause). These licenses are stated on the HF model
card. The upstream code was used only at export time (external checkout pattern);
no upstream code is present in this MIT-licensed voiceclonnx repository. Users who
access the ONNX weights should review the licenses on the model card before use in
commercial products.

Output sample rate: **48 kHz** (highest among all voiceclonnx engines).

---

## Install

```bash
pip install voiceclonnx
```

Dependencies pulled in: `onnxruntime`, `numpy`, `soundfile`, `huggingface_hub`.
Models are downloaded from HF Hub on first use (~694 MB fp32 total).

---

## Config keys

| Key | Type | Default | Description |
|---|---|---|---|
| `quantized` | `bool` | `False` | Use INT8 quantized ONNX models. Reduces total size to ~186 MB. INT8 shows degraded intelligibility on this engine; fp32 is recommended. See [QUANTS.md](../QUANTS.md). |

---

## Model sizes

| File | Size | Variant |
|---|---|---|
| `acoustic_ssl_encoder.onnx` | 89.7 MB | fp32 |
| `acoustic_ssl_encoder_q8.onnx` | 22.7 MB | INT8 (−74.8%) |
| `distill_wavlm_encoder.onnx` | 84.2 MB | fp32 |
| `distill_wavlm_encoder_q8.onnx` | 21.5 MB | INT8 (−74.4%) |
| `content_encoder.onnx` | 171.5 MB | fp32 |
| `content_encoder_q8.onnx` | 43.3 MB | INT8 (−74.7%) |
| `global_encoder.onnx` | 22.2 MB | fp32 |
| `global_encoder_q8.onnx` | 5.7 MB | INT8 (−74.6%) |
| `mel_decoder.onnx` | 265.6 MB | fp32 |
| `mel_decoder_q8.onnx` | 73.5 MB | INT8 (−72.3%) |
| `vocos_backbone.onnx` | 60.8 MB | fp32 |
| `vocos_backbone_q8.onnx` | 19.2 MB | INT8 (−68.5%) |

Total: **694.0 MB** fp32 / **186.0 MB** INT8.

---

## Usage

### Python

```python
from voiceclonnx import VoiceCloner

# Default (fp32, 48 kHz output)
cloner = VoiceCloner(engine="linacodec")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 48000

# INT8 (smaller memory footprint; note degraded quality — see QUANTS.md)
cloner = VoiceCloner(engine="linacodec", quantized=True)
```

### CLI

```bash
voiceclonnx clone --engine linacodec \
             --audio source.wav \
             --voice reference.wav \
             --out out.wav
```

---

## References

- Upstream repo: [ysharma3501/LinaCodec](https://github.com/ysharma3501/LinaCodec)
- Upstream weights: [YatharthS/LinaCodec](https://huggingface.co/YatharthS/LinaCodec)
- voiceclonnx weights: [TigreGotico/voiceclonnx-linacodec](https://huggingface.co/TigreGotico/voiceclonnx-linacodec)

---

## Troubleshooting

**`ImportError: onnxruntime is required`**
Install the extras group: `pip install voiceclonnx`.

**Output is very quiet or silent with `quantized=True`**
INT8 quantization degrades this engine significantly (attention + AdaLN layers do
not quantize well to INT8). Use the default fp32 models (`quantized=False`).

**First run is slow**
Six models (~694 MB) are downloaded from HF Hub. After the initial download they
are cached in `~/.cache/huggingface/hub` and subsequent runs start quickly.

**Output sounds robotic on very short clips**
The FSQ content encoder operates at 12.5 tokens/s. Clips under 1 second produce
only 12 tokens and the mel decoder may over-smooth. Prefer references of 3–10 s.
