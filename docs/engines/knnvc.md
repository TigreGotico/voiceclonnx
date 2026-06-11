# Engine: knnvc

kNN-VC (Baas et al., Interspeech 2023) — zero-shot any-to-any voice conversion.

Architecture:
1. **WavLM-Large encoder** (layer 6) — converts 16 kHz audio to 1024-dim feature frames at 50 Hz.
2. **k-nearest-neighbour matching** (pure numpy) — replaces each source feature frame with the average of its k nearest reference frames (L2 distance).
3. **HiFi-GAN vocoder** — converts matched features back to waveform.

All neural components run via onnxruntime. The kNN step is pure numpy — no ONNX at matching time. Fully non-autoregressive and CPU-friendly.

ONNX artifacts: [TigreGotico/vconnx-knn-vc](https://huggingface.co/TigreGotico/vconnx-knn-vc) (MIT license).

Output sample rate: **16 kHz**.

---

## Install

```bash
pip install "vconnx[knnvc]"
```

Dependencies pulled in: `onnxruntime`, `numpy`, `soundfile`.
Models are downloaded from HF Hub on first use (~500 MB fp32 or ~123 MB int8).

---

## Config keys

| Key | Type | Default | Description |
|---|---|---|---|
| `quantized` | `bool` | `False` | Use INT8 quantized ONNX models. Reduces memory to ~123 MB total; slightly lower quality. |
| `k` | `int` | `4` | Number of nearest neighbours to average in the matching step. Higher values smooth the conversion; lower values preserve more source characteristics. |

---

## Model sizes

| File | Size | Variant |
|---|---|---|
| `wavlm_layer6.onnx` | 386.8 MB | fp32 |
| `wavlm_layer6_q8.onnx` | 97.5 MB | INT8 (75% reduction) |
| `hifigan_knnvc.onnx` | 63.1 MB | fp32 |
| `hifigan_knnvc_q8.onnx` | 25.1 MB | INT8 (60% reduction) |

---

## Usage

### Python

```python
from vconnx import VoiceCloner

# Default (fp32, k=4)
cloner = VoiceCloner(engine="knnvc")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 16000

# Quantized (low-memory)
cloner = VoiceCloner(engine="knnvc", quantized=True)
out = cloner.clone_voice("source.wav", "reference.wav", "out_q8.wav")

# Looser matching (k=8 averages more reference frames)
cloner = VoiceCloner(engine="knnvc", k=8)
```

### CLI

```bash
pip install "vconnx[knnvc]"

vconnx clone --engine knnvc \
             --audio source.wav \
             --voice reference.wav \
             --out out.wav
```

---

## References

- Paper: [kNN-VC (Baas et al., Interspeech 2023)](https://arxiv.org/abs/2305.18975)
- Original code: [bshall/knn-vc](https://github.com/bshall/knn-vc)

---

## Troubleshooting

**`ImportError: onnxruntime is required`**
Install the extras group: `pip install "vconnx[knnvc]"`.

**`ImportError: soundfile`**
Same fix: `pip install "vconnx[knnvc]"` pulls soundfile.

**Output sounds noisy or garbled**
The conversion quality depends on having a reference clip that is clean and close to
the target speaker's natural voice. Ensure the reference is at least 5 s long.

**Out-of-memory on large files**
Use `quantized=True` — total model footprint drops from ~450 MB to ~123 MB.
For very long utterances, consider chunking the source audio.

**First run is slow**
Models are downloaded from HF Hub on first use. After the initial download they are
cached in `~/.cache/huggingface/hub` and subsequent runs start quickly.
