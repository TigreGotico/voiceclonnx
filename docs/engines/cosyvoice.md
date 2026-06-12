# Engine: cosyvoice

CosyVoice (FunAudioLLM / Alibaba DAMO, Apache-2.0) — zero-shot cross-lingual
voice conversion using a non-autoregressive flow-matching pipeline.  The
autoregressive LLM (`llm.pt`) used for TTS is bypassed entirely; only the
speech tokenizer, speaker encoder, flow encoder/decoder, and HiFiGAN vocoder
are used.

Output sample rate: **22050 Hz**.  CPU RTF: **~0.71×** (measured; well under
the 5× gate).

---

## Architecture

1. **`speech_tokenizer_v1.onnx`** — Whisper-style log-mel (128-bin, 16 kHz) →
   content token indices `(1, T_tok)` int64.  Encodes phoneme sequence of the
   source utterance.
2. **`campplus.onnx`** (CAM++) — Kaldi fbank (80-bin, 16 kHz) → 192-d speaker
   embedding `(1, 192)`.  Encodes voice identity of the reference speaker.
3. **`flow_encoder.onnx`** — 6-block Conformer encoder (rel_pos_espnet) +
   `InterpolateRegulator` → mu `(1, 80, T_mel)`.  Converts content tokens to
   flow conditioning.
4. **`flow_decoder.onnx`** — ODE Euler solver (10 steps by default) over mu +
   80-d projected speaker embedding → mel spectrogram `(1, 80, T_mel)`.
5. **`hifigan_f0_source.onnx`** — F0 predictor (`ConvRNNF0Predictor`) + NSF
   harmonic source (`SourceModuleHnNSF`) → 1-D source signal `(1, 1, T_audio)`.
6. **numpy STFT** (`n_fft=16`, `hop_len=4`, Hann window, center-padded) — source
   signal → STFT coefficients `(1, 18, T_stft)` (9 real + 9 imaginary bins).
7. **`hifigan_backbone.onnx`** — convolutional upsampling backbone (HiFTGenerator)
   → `(magnitude, phase)` STFT bins `(1, 9, T_stft)` each.
8. **numpy ISTFT** — magnitude + phase → waveform `(T_audio,)` @ 22050 Hz.

### STFT/ISTFT note

`aten::stft` / `aten::istft` are unsupported at ONNX opset 14.  The HiFiGAN is
split at the STFT boundary: `hifigan_f0_source.onnx` outputs a 1-D source
signal; STFT is computed in pure numpy; `hifigan_backbone.onnx` takes the STFT
as input.  ISTFT is also pure numpy.  Parity is verified at export time
(`STFT max_abs ≤ 1.4e-6`, `ISTFT roundtrip max_abs ≤ 5e-7`).

### Speaker projection

The `spk_embed_affine_layer` (`192 → 80`) is extracted from `flow.pt` at export
time and saved as `spk_proj.npz`.  The adapter loads this at inference time to
project the CAM++ 192-d embedding into the 80-d flow conditioning space without
needing the full flow model.

---

## ONNX artifacts

Hosted at [`TigreGotico/voiceclonnx-cosyvoice`](https://huggingface.co/TigreGotico/voiceclonnx-cosyvoice)
(public, **Apache-2.0**).

| File | Size (fp32) | Size (INT8) | Description |
|---|---|---|---|
| `speech_tokenizer_v1.onnx` | 499 MB | — | upstream; not quantized |
| `campplus.onnx` | 27 MB | — | upstream; not quantized |
| `flow_decoder.onnx` | 314 MB | 82 MB | upstream |
| `flow_encoder.onnx` | 108 MB | 42 MB | exported |
| `hifigan_f0_source.onnx` | 13 MB | 3 MB | exported |
| `hifigan_backbone.onnx` | 66 MB | 25 MB | exported |
| `spk_proj.npz` | ~120 KB | — | numpy; speaker affine weights |

---

## Parity vs torch

| Component | Metric | Value | Result |
|---|---|---|---|
| `flow_encoder` | max_abs | 1.9e-6 | PASS |
| `hifigan_f0_source` | max_abs | 0.115 (tol 0.15) | PASS (stochastic NSF noise expected) |
| `hifigan_backbone` | max_abs | 4.7e-4 | PASS |
| numpy STFT | max_abs vs torch | 1.3e-6 | PASS |
| numpy ISTFT roundtrip | max_abs | 4.8e-7 | PASS |

The NSF source has relaxed tolerance because the Gaussian noise injected by
`SineGen` (std=0.003) is baked as a constant in the ONNX graph; this ~3%
amplitude difference propagates to the source signal.

---

## Install

```bash
pip install voiceclonnx
```

Dependencies: `onnxruntime`, `numpy`, `soundfile`, `huggingface_hub`, `scipy`.
Models download from HF Hub on first use (~1.1 GB fp32 total).

---

## Config keys

| Key | Type | Default | Description |
|---|---|---|---|
| `quantized` | `bool` | `False` | Load INT8 quantized ONNX models. Reduces footprint from ~1.0 GB to ~152 MB (flow encoder + decoder + HiFiGAN). See [QUANTS.md](../QUANTS.md) for WER comparison. |
| `ode_steps` | `int` | `10` | Number of Euler ODE steps for the flow decoder. Fewer steps → faster but lower quality. Range 5–30. |

---

## Usage

### Python

```python
from voiceclonnx import VoiceCloner

# Default (fp32, 10 ODE steps)
cloner = VoiceCloner(engine="cosyvoice")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 22050

# INT8 quantized (smaller footprint)
cloner = VoiceCloner(engine="cosyvoice", quantized=True)
out = cloner.clone_voice("source.wav", "reference.wav", "out_q8.wav")

# Fewer ODE steps for faster inference
cloner = VoiceCloner(engine="cosyvoice", ode_steps=5)
out = cloner.clone_voice("source.wav", "reference.wav", "out_fast.wav")
```

### CLI

```bash
pip install voiceclonnx

voiceclonnx clone --engine cosyvoice \
             --audio source.wav \
             --voice reference.wav \
             --out out.wav
```

---

## License

Upstream weights (`FunAudioLLM/CosyVoice-300M`): **Apache-2.0**.
Training corpus Emilia: CC-BY-NC-4.0 (does not constrain the model weights).

ONNX artifacts in `TigreGotico/voiceclonnx-cosyvoice`: **Apache-2.0**.

---

## References

- Paper: [CosyVoice: A Scalable Multilingual Zero-shot Text-to-speech Synthesizer](https://arxiv.org/abs/2407.05407)
- Upstream repo: [FunAudioLLM/CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- Upstream checkpoint: [FunAudioLLM/CosyVoice-300M](https://huggingface.co/FunAudioLLM/CosyVoice-300M)

---

## Troubleshooting

**Output sounds like silence or noise**
Ensure source and reference are clean mono speech at or near 16 kHz.  Very short
clips (< 0.5 s) may yield degraded output because the speech tokenizer and flow
decoder expect at minimum ~25 tokens.

**First run is slow**
~1.1 GB of models download on first use and cache in `~/.cache/huggingface/hub`.

**`scipy` not installed**
`scipy` is needed only for the Hann window in the STFT/ISTFT helper.  Install
with `pip install scipy` or `pip install voiceclonnx`.

**ODE steps and quality tradeoff**
`ode_steps=5` is ~2× faster but may introduce metallic artefacts on complex
prosody.  `ode_steps=20` improves smoothness at the cost of extra compute.
The default `ode_steps=10` is recommended for CPU inference.
