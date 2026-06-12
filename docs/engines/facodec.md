# FACodec engine

FACodec (Factorized Audio Codec) is the core speech representation model from
NaturalSpeech 3 (Microsoft Research / Amphion, ICML 2024).  It disentangles
speech into four independently controllable subspaces: **content, prosody,
timbre, acoustic detail**.

Voice conversion is zero-shot: encode source and reference, extract source
content/prosody tokens and reference timbre embedding, decode the combination.
No per-speaker fine-tuning or adaptation is required.

## Architecture

| Component | Description |
|---|---|
| Encoder (V2) | Convolutional downsampler (2×4×5×5 = 200× hop, 16 kHz → 80 Hz) |
| Timbre extractor | 4-layer Transformer encoder → mean-pool over time → 256-d speaker embedding |
| Quantizer | Hierarchical factorised VQ: prosody (1 codebook) + content (2) + residual (3) = 6 total |
| Decoder (V2) | Upsampling convolutional synthesizer with AdaIN timbre conditioning |

## VC recipe (V2 path)

```
1. enc_feats_src = encoder(wav_src)       # (1, 256, T_src)
2. enc_feats_ref = encoder(wav_ref)       # (1, 256, T_ref)
3. mel_src       = prosody_mel(wav_src)   # (1, 20, T_src) — numpy, see below
4. vq_ids_src    = quantize(enc_feats_src, mel_src)   # (6, 1, T_src) int64
5. spk_embs_ref  = timbre(enc_feats_ref)  # (1, 256)
6. wav_out       = decode(vq_ids_src, spk_embs_ref)   # prosody+content from src, timbre from ref
```

The prosody mel (step 3) is computed in pure numpy (no ONNX component): standard
STFT mel spectrogram with n_fft=1024, hop=200, win=800, n_mels=80, sr=16000,
fmin=0, fmax=8000, log-compressed; the adapter uses the first 20 bins.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `quantized` | `False` | Use INT8 quantized models |

## Parity results (fp32 torch vs ORT, 1 s dummy input)

| Component | max_abs Δ | mean_abs Δ | Verdict |
|---|---|---|---|
| facodec_encoder | 1.62e-05 | 2.36e-06 | PASS |
| facodec_timbre | 1.43e-06 | 6.40e-08 | PASS |
| facodec_quantize | exact int64 match | — | PASS |
| facodec_decoder | 7.50e-09 | 1.46e-09 | PASS |

## Model sizes

| File | Size |
|---|---|
| `facodec_encoder.onnx` (fp32) | 16.5 MB |
| `facodec_encoder_q8.onnx` (INT8) | 4.7 MB |
| `facodec_timbre.onnx` (fp32) | 33.0 MB |
| `facodec_timbre_q8.onnx` (INT8) | 12.1 MB |
| `facodec_quantize.onnx` (fp32) | 33.4 MB |
| `facodec_quantize_q8.onnx` (INT8) | 12.5 MB |
| `facodec_decoder.onnx` (fp32) | 66.2 MB |
| `facodec_decoder_q8.onnx` (INT8) | 36.8 MB |

## Intelligibility (WER gate)

| Reference voice | WER | Gate |
|---|---|---|
| en-US-AriaNeural | 0% | PASS |
| en-GB-SoniaNeural | 0% | PASS |

## ONNX components

| File | I/O | Description |
|---|---|---|
| `facodec_encoder.onnx` | wav(1,1,N) → enc_feats(1,256,T) | Convolutional encoder |
| `facodec_timbre.onnx` | enc_feats(1,256,T) → spk_embs(1,256) | Timbre TransformerEncoder |
| `facodec_quantize.onnx` | (enc_feats,mel_20) → vq_ids(6,1,T) | Hierarchical VQ (6 codebooks) |
| `facodec_decoder.onnx` | (vq_ids,spk_embs) → wav(1,1,N) | vq2emb + AdaIN + conv decoder |

HF repo: [TigreGotico/vconnx-facodec](https://huggingface.co/TigreGotico/vconnx-facodec)

## License

Weights: **Apache-2.0** (verified on [amphion/naturalspeech3_facodec](https://huggingface.co/amphion/naturalspeech3_facodec) HF card).  
Code: Amphion (open-mmlab/Amphion) — Apache-2.0 (repository header) / MIT (per-module header).  
ONNX artifacts inherit the upstream Apache-2.0 license.

## Export notes

See [`conversion/export_facodec.py`](../../conversion/export_facodec.py) for the
full export script.

The export clones Amphion from GitHub (sparse checkout of `models/codec/ns3_codec`)
and installs the `einops` dependency.  The four components export cleanly with
the legacy TorchScript ONNX exporter (opset 14, `dynamo=False`).

The quantizer's `CNNLSTM` modules (used for F0/phone predictors during training)
are not traced — the quantize wrapper traces only the VQ code-assignment path
(nearest-neighbour lookup), which is fully convolutional.

## References

- https://arxiv.org/abs/2403.03100 (NaturalSpeech 3)
- https://huggingface.co/amphion/naturalspeech3_facodec
- https://github.com/open-mmlab/Amphion
