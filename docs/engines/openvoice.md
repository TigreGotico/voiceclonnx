# Engine: openvoice

OpenVoice v2 tone-color converter (myshell-ai/OpenVoice, MIT license).

Architecture:
1. **Reference encoder** (`tone_ref_encoder.onnx`) — mel-spectrogram → 256-dim tone-color embedding. Run on both source and reference audio.
2. **Converter** (`tone_converter.onnx`) — (source_mel, src_tone, tgt_tone) → converted_mel. A flow-based AdaIN-conditioned network.
3. **Griffin-Lim vocoder** (pure numpy) — mel → waveform. Lightweight alternative to a neural vocoder; no extra ONNX model needed.

All neural components run via onnxruntime. Mel extraction and Griffin-Lim vocoder are
pure numpy — no torch at inference.

ONNX artifacts: [TigreGotico/vconnx-openvoice-v2](https://huggingface.co/TigreGotico/vconnx-openvoice-v2) (MIT license).

Output sample rate: **22050 Hz**.

---

## Install

```bash
pip install "vconnx[openvoice]"
```

Dependencies pulled in: `onnxruntime`, `numpy`, `soundfile`.
Models are downloaded from HF Hub on first use.

---

## Config keys

| Key | Type | Default | Description |
|---|---|---|---|
| `quantized` | `bool` | `False` | Use INT8 quantized ONNX models. Lower memory and faster on CPU; slight quality trade-off. |
| `gl_iters` | `int` | `32` | Griffin-Lim iterations for vocoding. More iterations improve quality at the cost of latency. 16 is faster; 64 is higher quality. |

---

## Mel-spectrogram parameters

The following parameters match the OpenVoice v2 training configuration and are fixed:

| Parameter | Value |
|---|---|
| Sample rate | 22050 Hz |
| n_mels | 80 |
| n_fft / win_length | 1024 |
| hop_length | 256 |
| f_min | 0 Hz |
| f_max | 8000 Hz |

---

## Usage

### Python

```python
from vconnx import VoiceCloner

# Default (fp32, 32 Griffin-Lim iterations)
cloner = VoiceCloner(engine="openvoice")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 22050

# Quantized with more vocoder iterations
cloner = VoiceCloner(engine="openvoice", quantized=True, gl_iters=64)
out = cloner.clone_voice("source.wav", "reference.wav", "out_hq.wav")
```

### CLI

```bash
pip install "vconnx[openvoice]"

vconnx clone --engine openvoice \
             --audio source.wav \
             --voice reference.wav \
             --out out.wav
```

---

## References

- Paper: [OpenVoice: Versatile Instant Voice Cloning](https://arxiv.org/abs/2312.01479)
- Original code: [myshell-ai/OpenVoice](https://github.com/myshell-ai/OpenVoice)

---

## Troubleshooting

**`ImportError: onnxruntime is required`**
Install the extras group: `pip install "vconnx[openvoice]"`.

**Output sounds muffled or phasey**
This is a known limitation of the Griffin-Lim vocoder. Increase `gl_iters` (e.g. 64)
for better quality. A neural vocoder replacement is a potential future enhancement.

**Output is out of tune or has pitch artefacts**
Ensure the reference audio is clean mono 22050 Hz speech. Avoid music or background
noise in the reference.

**First run is slow**
Models are downloaded on first use and cached in `~/.cache/huggingface/hub`.
