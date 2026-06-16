# Changelog

## Unreleased

- **Added `lscodec`** ([LSCodec](https://github.com/X-LANCE/LSCodec-Inference),
  Interspeech 2025) — a speaker-decoupled discrete codec. It passed the
  speaker-similarity gate (target-sim 0.54, the strongest timbre transfer in the
  codec family) with a moderate ~35% WER tradeoff. The 3-model ONNX export
  (encoder + WavLM-prompt + CTXVEC2WAV vocoder) was validated against the
  upstream torch model (ONNX↔torch cosine 0.97). Weights: MIT.

- **Curated the engine roster to 9** (`facodec`, `openvoice`, `chatterbox`,
  `triaan`, `cosyvoice`, `bicodec`, `knnvc`, `focalcodec`, `rvc`). Removed
  `freevc`, `speechtokenizer`, `mimi`, `linacodec`, `quickvc`, `vec2wav`, and
  `seedvc`: a speakeronnx speaker-similarity audit found they sat at the
  no-conversion floor (kept the *source* voice), were neural codecs repurposed
  for VC, or were unbenchmarked/garbled ports. Export-parity gates confirmed the
  weakness was the model/recipe, not the ONNX export. See
  [demo/SPEAKER_SIMILARITY.md](demo/SPEAKER_SIMILARITY.md).
- Added `demo/speaker_similarity.py` — ranks engines by speaker similarity to the
  target voice (not just WER).

## 0.0.1a1

- Initial release: engine registry, `VoiceCloner` facade, Chatterbox ONNX adapter
- CLI: `voiceclonnx clone`, `voiceclonnx list`
- Audio-to-audio only; no TTS surface
