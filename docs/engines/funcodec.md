# Engine: FunCodec — Investigation Findings (BLOCKED)

**Status:** Investigation complete — engine cannot be implemented as specified.
See [issue #38](https://github.com/TigreGotico/voiceclonnx/issues/38) for the blocking report.

---

## Premise vs reality

The issue premised that FunCodec includes "semantic variants" with an SSL-distilled first
RVQ layer — the same paradigm as SpeechTokenizer — enabling zero-shot VC by swapping RVQ-1
content tokens from source with RVQ-2..N acoustic tokens from the reference.

After full investigation of the FunCodec codebase (`modelscope/FunCodec`, commit `HEAD` 2026-06)
and all published HF checkpoints, this premise does not hold.

---

## What FunCodec actually offers

### Acoustic codecs (all published checkpoints)

All six checkpoints on `alibaba-damo/*` HuggingFace are **pure EnCodec-style reconstruction
codecs** — the quantizer is a flat `costume_quantizer` (residual VQ) with no distillation
objective on any layer.

| Checkpoint | Architecture | Semantic layer? |
|---|---|---|
| `audio_codec-encodec-en-libritts-16k-nq32ds320-pytorch` | EnCodec | No |
| `audio_codec-encodec-en-libritts-16k-nq32ds640-pytorch` | EnCodec | No |
| `audio_codec-encodec-zh_en-general-16k-nq32ds320-pytorch` | EnCodec | No |
| `audio_codec-encodec-zh_en-general-16k-nq32ds640-pytorch` | EnCodec | No |
| `audio_codec-freqcodec_magphase-en-libritts-16k-gr1nq32ds320-pytorch` | FreqCodec | No |
| `audio_codec-freqcodec_magphase-en-libritts-16k-gr8nq32ds320-pytorch` | FreqCodec | No |

These are reconstruction codecs. Using any of them for VC by swapping RVQ-1 tokens would
replicate the acoustic-only RVQ trap: the first layer encodes low-frequency signal shape,
not phonetic content. The result is unintelligible (high WER) output.

### `codec_semantic_aug` — PPG-conditioned, not SSL-distilled

The `CodecSemanticAug` model class (`funcodec/models/codec_semantic_aug.py`) is the only
"semantic" variant in the codebase. It is **not** the SSL-distilled paradigm:

- It requires PPG (phonetic posteriorgrams) as an **external inference-time input**
  (`speech, ppg → encode → quantize → decode`).
- The PPG comes from an upstream ASR model run separately before codec inference.
- No published checkpoint exists for this variant on HuggingFace or ModelScope.
- The paradigm is **conditional** codec synthesis (TTS/voice conversion with ASR
  features as conditioning), not a self-contained content-vs-timbre split.

`SpeechTokenizer.encode(audio)` returns disentangled tokens in a single pass.
`CodecSemanticAug.inference(speech, ppg)` requires a separately extracted PPG tensor —
it cannot be used for zero-shot VC from audio alone.

### `funcodec_en_libritts-16k-semantic` — does not exist

The specific checkpoint name mentioned in the issue
(`funcodec_en_libritts-16k-semantic`) does not appear in the FunCodec GitHub
repository, the `alibaba-damo` HuggingFace namespace, or ModelScope. It was likely
inferred from the paper abstract rather than verified against the published artifacts.

---

## Why this blocks the engine

voiceclonnx requires:

1. A **self-contained** encoder: `audio → tokens` (no external conditioning).
2. An RVQ structure where **layer 1** is content-dominant by design (SSL distillation or
   equivalent factorization objective).
3. A **decoder** that takes the mixed token stream and produces a plausible voice conversion.

FunCodec does not provide any checkpoint that satisfies criteria 1 and 2 simultaneously.

Using the acoustic-only checkpoints would produce a reconstruction-only codec. The RVQ
swap would fail the WER gate (expected >60% WER based on the similar acoustic-codec path
rejected for SpeechTokenizer before distillation was verified).

---

## Possible future paths

If a FunCodec semantic checkpoint is published in the future — specifically one trained with
an SSL (HuBERT/WavLM) distillation objective on RVQ-1 and available without PPG
conditioning at inference — the implementation would follow the SpeechTokenizer recipe
exactly (see `voiceclonnx/engines/speechtokenizer.py`):

1. Export `encoder.onnx` (audio → features) and `decoder.onnx` (features → audio).
2. Export `codebooks.npy` for the quantizer.
3. Implement `_rvq_encode`, `_rvq_decode`, `_swap_rvq_tokens` (identical to SpeechTokenizer).
4. Register adapter with `content_layers=1` default.

The `conversion/export_funcodec.py` stub in this branch documents the export recipe for
when such a checkpoint becomes available.
