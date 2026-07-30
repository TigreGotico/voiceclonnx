# voiceclonnx

![PyPI](https://img.shields.io/pypi/v/voiceclonnx)
![Python](https://img.shields.io/pypi/pyversions/voiceclonnx)
![License](https://img.shields.io/pypi/l/voiceclonnx)

voiceclonnx converts the voice in an existing speech file to sound like a
reference speaker. It runs 10 engines, each on pure ONNX, with no PyTorch
needed at runtime.

voiceclonnx works on audio only. It converts speech to speech. Text-driven
synthesis (text to cloned audio) is a TTS concern and is out of scope.

---

## Why voiceclonnx

- **No PyTorch at runtime.** Every engine runs on `onnxruntime`, `numpy`,
  `soundfile`, and `huggingface_hub` only. Inference needs no torch and no
  CUDA driver.
- **One install, every engine.** `pip install voiceclonnx` activates all 10
  engines. There are no per-engine extras and no optional groups for
  inference.
- **10 architectures behind one API.** The engines cover kNN feature-swap,
  factorized codec, flow-matching, tone-color transfer, AR codec-LM,
  speaker-decoupled codec, and any-to-ONE conversion. Each transfers the
  target voice, not just the words.
- **STT- and speaker-verified.** Each demo clip is transcribed with
  faster-whisper to measure WER and intelligibility, and scored for speaker
  similarity to the target voice. Both results are published in the
  [speaker-similarity benchmark](demo/SPEAKER_SIMILARITY.md).
- **INT8 quantization with measured tradeoffs.** Most engines ship `*_q8.onnx`
  variants that are 45-75% smaller and faster on CPU, with the WER cost
  documented per engine.
- **Documented conversion toolchain.** A step-by-step guide covers export,
  parity check, quantization, push, and adapter setup for anyone adding a new
  engine.

---

## Listen first, install later

[demo/README.md](demo/README.md) converts the same sentence with every engine
to two reference voices (Aria and Sonia). GitHub renders the audio players
inline, so you can compare all 10 engines by ear without writing code.

---

## Install

```bash
pip install voiceclonnx
```

Core dependencies: `onnxruntime`, `numpy`, `soundfile`, `huggingface_hub`.
voiceclonnx downloads ONNX models from Hugging Face Hub on first use.

For model conversion and export tooling:

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

`pip install voiceclonnx` includes all engines, with no per-engine extras.
The ONNX models live in the
[voiceclonnx HF collection](https://huggingface.co/collections/TigreGotico/voiceclonnx-pure-onnx-voice-conversion).
WER is measured with faster-whisper `base.en` against the source transcript
(lower is better. 0% means the output is fully intelligible). Full data:
[demo/VERIFICATION.md](demo/VERIFICATION.md).

| Engine | Family | Sample rate | WER | INT8 | Model | Best for |
|--------|--------|-------------|-----|------|-------|----------|
| `facodec` | Factorized codec | 16 kHz | **0%** | ✅ | [TigreGotico/voiceclonnx-facodec](https://huggingface.co/TigreGotico/voiceclonnx-facodec) | Best overall quality (0% WER + strong timbre) |
| `openvoice` | Tone-color transfer | 22 kHz | **0%** | ✅ | [TigreGotico/voiceclonnx-openvoice-v2](https://huggingface.co/TigreGotico/voiceclonnx-openvoice-v2) | Broadest style range, 0% WER |
| `chatterbox` | AR codec-LM | 24 kHz | 4-8% | ✅ (8% WER) | [TigreGotico/voiceclonnx-chatterbox](https://huggingface.co/TigreGotico/voiceclonnx-chatterbox) | Natural prosody, strongest source-to-target shift |
| `triaan` | Triple-AAN | 16 kHz | 4% | ✅ | [TigreGotico/voiceclonnx-triaan-vc](https://huggingface.co/TigreGotico/voiceclonnx-triaan-vc) | Good quality, small footprint |
| `cosyvoice` | Flow-matching | 22 kHz | 8% | ⚠ int8 degrades | [TigreGotico/voiceclonnx-cosyvoice](https://huggingface.co/TigreGotico/voiceclonnx-cosyvoice) | Cross-lingual conversion |
| `bicodec` | Semantic + global tokens | 16 kHz | 12% | ✅ | [TigreGotico/voiceclonnx-bicodec](https://huggingface.co/TigreGotico/voiceclonnx-bicodec) | SparkTTS zero-shot VC |
| `knnvc` | kNN feature-swap | 16 kHz | 12-15% | ✅ | [TigreGotico/voiceclonnx-knn-vc](https://huggingface.co/TigreGotico/voiceclonnx-knn-vc) | Lightweight (123 MB int8), strong timbre |
| `focalcodec` | kNN feature-swap | 16 kHz | 15-19% | ⚠ int8 degrades | [TigreGotico/voiceclonnx-focalcodec](https://huggingface.co/TigreGotico/voiceclonnx-focalcodec) | Best timbre similarity (NeurIPS 2025) |
| `lscodec` | Speaker-decoupled codec | 24 kHz | ~35% | ✅ | [TigreGotico/voiceclonnx-lscodec](https://huggingface.co/TigreGotico/voiceclonnx-lscodec) | **Best timbre transfer**, trades some WER (Interspeech 2025) |
| `rvc` | ContentVec + VITS | 40/48 kHz | 38%† | ✅ (base only) | [TigreGotico/voiceclonnx-rvc](https://huggingface.co/TigreGotico/voiceclonnx-rvc) | Any-to-ONE, community voices |



> †`rvc` WER reflects a sample community model. Any-to-ONE semantics differ from
> all other engines. See [Choosing an engine](#choosing-an-engine).
>
> WER measures intelligibility, not voice similarity. Every engine is also scored
> for how closely its output matches the target speaker. See the
> [speaker-similarity benchmark](demo/SPEAKER_SIMILARITY.md).

---

## Choosing an engine

**Best all-rounders (0% WER + strong timbre):** `facodec`, `openvoice`: start
here unless you have a specific constraint.

**Best target-voice fidelity (speaker similarity):** `focalcodec`, `lscodec`,
`chatterbox`, `facodec`, `knnvc`, `openvoice`: see the ranked
[speaker-similarity benchmark](demo/SPEAKER_SIMILARITY.md). `lscodec` gives the
strongest timbre transfer in the codec family, at a cost of about 35% WER.
Pick it when voice identity matters more than exact transcription.

**Highest output sample rate:** `rvc` at up to 48 kHz (any-to-ONE). `chatterbox` runs
at 24 kHz for any-to-any.

**Natural prosody and expressive style:** `chatterbox`, an AR codec-LM that
transfers speaking style along with voice timbre.

**Smallest INT8 footprint:** `knnvc` at about 123 MB.

**Any-to-ONE voice models (RVC ecosystem):** `rvc` uses a voice model instead
of a reference audio clip. `reference_voice` is a path to an `.onnx` RVC model
(local file or HF repo ID). Thousands of community-trained voices exist on HF.

```python
# rvc: reference_voice = path to an RVC .onnx model, NOT an audio file
cloner = VoiceCloner(engine="rvc")
out = cloner.clone_voice("source.wav", "/path/to/myvoice.onnx", "out.wav")
```

**Non-commercial only:** `bicodec` weights are CC BY-NC-SA 4.0. Verify the
license before deploying it commercially.

---

## Quantized models

All engines except `chatterbox` support `quantized=True`, which loads
`*_q8.onnx` INT8 variants. These are 45-75% smaller on disk and faster on
CPU, at a measured quality cost.

```python
cloner = VoiceCloner(engine="knnvc", quantized=True)
out = cloner.clone_voice("source.wav", "reference.wav", "out.wav")
```

Some engines degrade significantly in INT8: use `focalcodec` and `cosyvoice`
in fp32 for production.

`chatterbox` INT8 matches fp32 quality (8% WER, 57% smaller). This project
quantizes and hosts it at `TigreGotico/voiceclonnx-chatterbox` because
upstream ships fp32 only.

See [docs/QUANTS.md](docs/QUANTS.md) for the full WER and size comparison.

---

## Adding an engine

1. Subclass `VoiceClonerBase` from `voiceclonnx.engines.base`.
2. Implement `clone_voice(audio, reference_voice, out_path) -> str`.
3. Call `register_engine(EngineEntry(alias=..., adapter_class=...))`.
4. Add the auto-import to `voiceclonnx/__init__.py`.

See [docs/converting.md](docs/converting.md) for the full export, parity
check, quantization, push, and adapter workflow, and
[CONTRIBUTING.md](CONTRIBUTING.md) for the contribution checklist.

---

## Documentation

- [demo/README.md](demo/README.md): listen to every engine, no install
- [docs/index.md](docs/index.md): engine families, install matrix, navigation
- [docs/QUANTS.md](docs/QUANTS.md): fp32 vs INT8 WER and size comparison
- [docs/api.md](docs/api.md): VoiceCloner, VoiceClonerBase, registry
- [docs/engines/](docs/engines/): per-engine guides (config, model, WER, CLI)
- [docs/converting.md](docs/converting.md): ONNX export / parity / quantize / push toolchain
- [examples/](examples/): Python and shell examples

---

## License

Apache 2.0. See [LICENSE](LICENSE).

Model weights are governed by their upstream licenses (MIT, Apache-2.0, CC BY 4.0,
CC BY-NC-SA 4.0 for bicodec). See [docs/converting.md](docs/converting.md) for
the weight-license policy (distributable vs local-only).
