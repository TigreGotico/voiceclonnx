# triaan — TriAAN-VC engine

Any-to-any voice conversion using Triple Adaptive Attention Normalization
([ICASSP 2023](https://arxiv.org/abs/2303.09057)).

## Install

```bash
pip install vconnx
```

## Usage

```python
from vconnx import VoiceCloner

vc = VoiceCloner(engine="triaan")
output_wav = vc.clone_voice(source_wav, target_wav)
```

Both `source_wav` and `target_wav` must be 16 000 Hz mono numpy float32 arrays.
The returned `output_wav` is also 16 000 Hz mono float32.

## Configuration keys

| Key | Default | Description |
|---|---|---|
| `model_dir` | `None` (auto-download) | Local path to the engine directory with all six `.onnx` files |
| `hf_repo_id` | `TigreGotico/vconnx-triaan-vc` | Hugging Face repository to download from |
| `use_quantized` | `False` | Use INT8 quantized models (smaller, slightly lower quality) |

## Architecture

TriAAN-VC converts voice in three stages, each a separate ONNX file:

1. **CPC encoder** (`cpc_encoder.onnx`) — 5-layer strided Conv1d (160× downsample,
   16 kHz → 100 Hz frame rate) + single LSTM layer; produces 256-dim content features
   from raw waveform.  Architecture from [facebookresearch/CPC\_audio](https://github.com/facebookresearch/CPC_audio);
   loaded from the `cpc.pt` checkpoint in the TriAAN-VC release.

2. **TriAAN-VC decoder** (`triaan_vc.onnx`) — ContentEncoder + SpeakerEncoder (with
   skip connections) + bidirectional GRU fusion + TriAANBlock decoder + PostNet.
   Triple Adaptive Attention Normalization combines:
   - **TAN** (Time-wise Adaptive Normalization): time-domain cross-attention on speaker features
   - **CAN** (Channel-wise Adaptive Normalization): channel-domain cross-attention
   - **GLAN** (Global Adaptive Normalization): global self-attention pooling

3. **ParallelWaveGAN vocoder** (`pwg_vocoder.onnx`) — WaveNet-style neural vocoder
   trained on VCTK; mel spectrogram → 16 kHz waveform.

## Model files

| File | Size | Description |
|---|---|---|
| `cpc_encoder.onnx` | 7.0 MB | CPC content encoder (FP32) |
| `cpc_encoder_q8.onnx` | 1.8 MB | CPC encoder INT8 |
| `triaan_vc.onnx` | 266.3 MB | TriAAN-VC decoder (FP32) |
| `triaan_vc_q8.onnx` | 76.3 MB | TriAAN-VC decoder INT8 |
| `pwg_vocoder.onnx` | 7.0 MB | ParallelWaveGAN vocoder (FP32) |
| `pwg_vocoder_q8.onnx` | 2.0 MB | PWG vocoder INT8 |

## Parity (PyTorch vs ONNX Runtime)

All measurements on random input, T = 50 frames (0.5 s).

| Component | max\_abs Δ | mean\_abs Δ | Verdict |
|---|---|---|---|
| CPC encoder | 1.06e-05 | 1.46e-07 | PASS |
| TriAAN-VC decoder | 3.76e-06 | 6.39e-07 | PASS |
| ParallelWaveGAN vocoder | 4.39e-05 | 1.19e-06 | PASS |

Max absolute difference is well below perceptual threshold for all components.

## License

MIT — upstream code and weights at [winddori2002/TriAAN-VC](https://github.com/winddori2002/TriAAN-VC).
CPC encoder from [facebookresearch/CPC\_audio](https://github.com/facebookresearch/CPC_audio) (MIT).
Vocoder derived from [kan-bayashi/ParallelWaveGAN](https://github.com/kan-bayashi/ParallelWaveGAN) (MIT).

## References

- TriAAN-VC paper: <https://arxiv.org/abs/2303.09057>
- Original code: <https://github.com/winddori2002/TriAAN-VC>
- CPC encoder: <https://github.com/facebookresearch/CPC_audio>
- ParallelWaveGAN: <https://github.com/kan-bayashi/ParallelWaveGAN>
- ONNX artifacts: <https://huggingface.co/TigreGotico/vconnx-triaan-vc>

## Troubleshooting

**`ImportError: No module named 'soundfile'`** — install extras: `pip install vconnx`.

**Output audio is silent or very short** — ensure the input is 16 000 Hz mono float32;
resample before passing to `clone_voice`.

**Very slow inference** — the TriAAN-VC decoder is 266 MB; first-run ORT graph
compilation takes a few seconds.  Subsequent calls on the same `VoiceCloner` instance
reuse the loaded session.
