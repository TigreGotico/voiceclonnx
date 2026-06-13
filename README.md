# voiceclonnx

![PyPI](https://img.shields.io/pypi/v/voiceclonnx)
![Python](https://img.shields.io/pypi/pyversions/voiceclonnx)
![License](https://img.shields.io/pypi/l/voiceclonnx)

**Pure-ONNX voice conversion. 14 engines. Zero PyTorch at runtime.**

Audio-to-audio only — voiceclonnx converts the voice in an existing speech file
to sound like a reference speaker. Text-driven synthesis (text → cloned audio) is
a TTS concern and is out of scope.

---

## Why voiceclonnx

- **Zero PyTorch at runtime.** Every engine runs on `onnxruntime`, `numpy`,
  `soundfile`, and `huggingface_hub` only. No torch, no CUDA driver required
  for inference.
- **One install, every engine.** `pip install voiceclonnx` activates all 14
  engines immediately — no per-engine extras, no optional groups for inference.
- **Widest pure-ONNX VC collection available.** 14 distinct architectures in a
  single unified API: kNN feature-swap, factorized codec, flow-matching,
  RVQ token-swap, and AR codec-LM families.
- **Every engine STT-verified.** Each demo clip is transcribed with
  faster-whisper and scored against the source text. WER is published and
  gated — no engine ships without a passing intelligibility score.
- **INT8 quantization with measured tradeoffs.** Most engines ship `*_q8.onnx`
  variants: 45–75% smaller, faster on CPU, with documented WER cost per engine.
- **Documented conversion toolchain.** A step-by-step guide covers
  export → parity → quantize → push → adapter for anyone adding a new engine.

---

## Listen first, install later

**[demo/README.md](demo/README.md)** — every engine converts the same sentence
to two reference voices (Aria and Sonia). GitHub renders the audio players inline.
Compare all 14 engines by ear, zero code required.

---

## Install

```bash
pip install voiceclonnx
```

Core dependencies: `onnxruntime`, `numpy`, `soundfile`, `huggingface_hub`.
ONNX models are downloaded on first use from Hugging Face Hub.

For model conversion / export tooling:

```bash
pip install "voiceclonnx[convert]"   # torch, onnx, transformers, librosa (export only)
pip install "voiceclonnx[test]"      # pytest, faster-whisper, edge-tts (test suite)
```

---

## Quick start

### Python

```python
from voiceclonnx import VoiceCloner

cloner = VoiceCloner(engine="facodec")
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
print(cloner.sample_rate)   # 16000
```

### CLI

```bash
# Convert a WAV file
voiceclonnx clone --engine facodec \
             --audio source.wav \
             --voice reference.wav \
             --out converted.wav

# List all registered engines
voiceclonnx list
```

---

## Engine comparison

All engines are included in `pip install voiceclonnx` — no per-engine extras.
WER is measured with faster-whisper `base.en` against the source transcript
(lower is better; 0% = perfectly intelligible). Full data: [demo/VERIFICATION.md](demo/VERIFICATION.md).

| Engine | Family | Sample rate | WER | INT8 | Model | Best for |
|--------|--------|-------------|-----|------|-------|----------|
| `facodec` | Factorized codec | 16 kHz | **0%** | ✅ | [TigreGotico/voiceclonnx-facodec](https://huggingface.co/TigreGotico/voiceclonnx-facodec) | Best overall quality |
| `mimi` | RVQ token-swap | 24 kHz | **0%** | ✅ | [TigreGotico/voiceclonnx-mimi](https://huggingface.co/TigreGotico/voiceclonnx-mimi) | 24 kHz, zero WER |
| `openvoice` | Tone-color transfer | 22 kHz | **0%** | ✅ | [TigreGotico/voiceclonnx-openvoice-v2](https://huggingface.co/TigreGotico/voiceclonnx-openvoice-v2) | Broadest style range |
| `quickvc` | HuBERT-soft + VITS | 16 kHz | **0%** | ✅ | [TigreGotico/voiceclonnx-quickvc](https://huggingface.co/TigreGotico/voiceclonnx-quickvc) | Fastest CPU (0.14× RTF) |
| `chatterbox` | AR codec-LM | 24 kHz | 4–8% | ✅ (8% WER) | [TigreGotico/voiceclonnx-chatterbox](https://huggingface.co/TigreGotico/voiceclonnx-chatterbox) | Natural prosody, expressive style |
| `triaan` | Triple-AAN | 16 kHz | 4% | ✅ | [TigreGotico/voiceclonnx-triaan-vc](https://huggingface.co/TigreGotico/voiceclonnx-triaan-vc) | Good quality, small footprint |
| `speechtokenizer` | RVQ token-swap | 16 kHz | 4–12% | ✅ | [TigreGotico/voiceclonnx-speechtokenizer](https://huggingface.co/TigreGotico/voiceclonnx-speechtokenizer) | HuBERT-distilled content fidelity |
| `cosyvoice` | Flow-matching | 22 kHz | 8% | ⚠ int8 degrades | [TigreGotico/voiceclonnx-cosyvoice](https://huggingface.co/TigreGotico/voiceclonnx-cosyvoice) | Cross-lingual conversion |
| `linacodec` | Codec + Transformer | **48 kHz** | 8–15% | ⚠ int8 degrades | [TigreGotico/voiceclonnx-linacodec](https://huggingface.co/TigreGotico/voiceclonnx-linacodec) | Highest sample rate (48 kHz) |
| `bicodec` | Semantic + global tokens | 16 kHz | 12% | ✅ | [TigreGotico/voiceclonnx-bicodec](https://huggingface.co/TigreGotico/voiceclonnx-bicodec) | SparkTTS zero-shot VC |
| `freevc` | WavLM + VITS | 16 kHz | 12% | ⚠ int8 degrades | [TigreGotico/voiceclonnx-freevc](https://huggingface.co/TigreGotico/voiceclonnx-freevc) | No text annotations needed |
| `knnvc` | kNN feature-swap | 16 kHz | 12–15% | ✅ | [TigreGotico/voiceclonnx-knn-vc](https://huggingface.co/TigreGotico/voiceclonnx-knn-vc) | Lightweight (123 MB int8) |
| `focalcodec` | kNN feature-swap | 16 kHz | 15–19% | ⚠ int8 degrades | [TigreGotico/voiceclonnx-focalcodec](https://huggingface.co/TigreGotico/voiceclonnx-focalcodec) | NeurIPS 2025 architecture |
| `rvc` | ContentVec + VITS | 40/48 kHz | 38%† | ✅ (base only) | [TigreGotico/voiceclonnx-rvc](https://huggingface.co/TigreGotico/voiceclonnx-rvc) | Any-to-ONE, community voices |

> †`rvc` WER reflects a sample community model. Any-to-ONE semantics differ from
> all other engines — see [Choosing an engine](#choosing-an-engine).

---

## Choosing an engine

**Best intelligibility (0% WER):** `facodec`, `mimi`, `openvoice`, `quickvc` —
start here unless you have a specific constraint.

**Fastest CPU inference:** `quickvc` at ~0.14× RTF — the clear choice for
latency-sensitive or embedded use.

**Highest output sample rate:** `linacodec` at 48 kHz — for downstream
processing that requires full-bandwidth audio.

**Natural prosody / expressive style:** `chatterbox` — AR codec-LM that
transfers speaking style along with voice timbre.

**Smallest INT8 footprint:** `knnvc` at ~123 MB; `quickvc` at ~130 MB.

**Any-to-ONE voice models (RVC ecosystem):** `rvc` uses a voice model rather than
a reference audio clip. `reference_voice` is a path to an `.onnx` RVC model
(local file or HF repo ID). Thousands of community-trained voices exist on HF.

```python
# rvc: reference_voice = path to an RVC .onnx model, NOT an audio file
cloner = VoiceCloner(engine="rvc")
out = cloner.clone_voice("source.wav", "/path/to/myvoice.onnx", "out.wav")
```

**Non-commercial only:** `bicodec` weights are CC BY-NC-SA 4.0 — verify before
deploying commercially.

---

## Quantized models

All engines except `chatterbox` support `quantized=True`, which loads `*_q8.onnx`
INT8 variants: 45–75% smaller on disk and faster on CPU at a measured quality cost.

```python
cloner = VoiceCloner(engine="knnvc", quantized=True)
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
```

Some engines degrade significantly in INT8: `freevc`, `focalcodec`, `cosyvoice`,
and `linacodec` should be used in fp32 for production.

`chatterbox` INT8 matches fp32 quality (8% WER, 57% smaller) — we quantize
and host it at `TigreGotico/voiceclonnx-chatterbox` since upstream ships fp32 only.

See [docs/QUANTS.md](docs/QUANTS.md) for the full WER and size comparison.

---

## Adding an engine

1. Subclass `VoiceClonerBase` from `voiceclonnx.engines.base`.
2. Implement `clone_voice(audio, reference_voice, out_path) -> str`.
3. Call `register_engine(EngineEntry(alias=..., adapter_class=...))`.
4. Add the auto-import to `voiceclonnx/__init__.py`.

See [docs/converting.md](docs/converting.md) for the full export → parity →
quantize → push → adapter workflow, and [CONTRIBUTING.md](CONTRIBUTING.md) for
the contribution checklist.

---

## Documentation

- [demo/README.md](demo/README.md) — listen to every engine, no install
- [docs/index.md](docs/index.md) — engine families, install matrix, navigation
- [docs/QUANTS.md](docs/QUANTS.md) — fp32 vs INT8 WER and size comparison
- [docs/api.md](docs/api.md) — VoiceCloner, VoiceClonerBase, registry
- [docs/engines/](docs/engines/) — per-engine guides (config, model, WER, CLI)
- [docs/converting.md](docs/converting.md) — ONNX export / parity / quantize / push toolchain
- [examples/](examples/) — Python and shell examples

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

Model weights are governed by their upstream licenses (MIT, Apache-2.0, CC BY 4.0,
CC BY-NC-SA 4.0 for bicodec). See [docs/converting.md](docs/converting.md) for
the weight-license policy (distributable vs local-only).
