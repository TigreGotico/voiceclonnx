# Changelog

## Unreleased

- **Added `lscodec`** ([LSCodec](https://github.com/X-LANCE/LSCodec-Inference),
  Interspeech 2025) — a speaker-decoupled discrete codec with the strongest
  timbre transfer of the codec family (speaker similarity 0.54) at a moderate
  ~35% WER. Pure-ONNX: encoder + WavLM prompt + CTXVEC2WAV vocoder, 24 kHz.
  Weights: MIT.
- Added `demo/speaker_similarity.py` and the
  [speaker-similarity benchmark](demo/SPEAKER_SIMILARITY.md) — ranks engines by
  how closely the output matches the target voice, not just WER.

## 0.0.1a1

- Initial release: engine registry, `VoiceCloner` facade, Chatterbox ONNX adapter
- CLI: `voiceclonnx clone`, `voiceclonnx list`
- Audio-to-audio only; no TTS surface
