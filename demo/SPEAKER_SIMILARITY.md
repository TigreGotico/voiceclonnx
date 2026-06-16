# Speaker-similarity benchmark

Word Error Rate measures whether the words are intelligible. It does not measure
whether the converted voice actually sounds like the *target speaker*. This page
ranks every engine by **speaker similarity to the target voice**, the metric that
matters most for voice conversion.

Reproduce:

```bash
pip install voiceclonnx speakeronnx soxr
python demo/speaker_similarity.py                    # 3-model ensemble
python demo/speaker_similarity.py wespeaker-resnet34 # single, most discriminative
```

## Method

- For each engine output `<engine>__{aria,sonia}.wav`, take its speaker embedding
  with [speakeronnx](https://github.com/TigreGotico/speakeronnx) and compute the
  **cosine similarity to the target reference** (`reference_aria.wav` /
  `reference_sonia.wav`). Higher means closer to the intended voice.
- Also measure **source similarity** (cosine to `source.wav`). High source
  similarity with low target similarity means the output leaked the source voice.
- Primary model: `wespeaker-resnet34`. The source→target baseline — the score an
  untouched copy of the source would get against the target — is only **0.09**,
  so anything well above that is performing real conversion. Cross-checked with a
  three-model ensemble (`+ titanet-large + campplus`).

## Ranking (wespeaker-resnet34, target similarity)

`rvc` is any-to-ONE and converts to a different community voice, so it is not
directly comparable on the aria/sonia targets and is omitted here.

| Rank | Engine | Target sim | WER | Notes |
|---|---|---|---|---|
| 1 | `focalcodec` | **0.61** | 15–19% | Strongest timbre transfer |
| 2 | `lscodec` | **0.54** | ~35% | Strong timbre; trades intelligibility |
| 3 | `chatterbox` | **0.54** | 4–8% | Strong timbre **and** intelligibility |
| 4 | `knnvc` | 0.49 | 12–15% | Strong timbre, lightweight |
| 5 | `facodec` | 0.44 | **0%** | Best balance (0% WER + good timbre) |
| 6 | `openvoice` | 0.37 | **0%** | Good balance, broad style range |
| 7 | `bicodec` | 0.29 | 12% | Moderate |
| 8 | `triaan` | 0.29 | 4% | Moderate, small footprint |
| 9 | `cosyvoice` | 0.21 | 8% | Modest timbre, cross-lingual |

## Reading the table

WER and speaker similarity are independent axes. `facodec` and `openvoice` reach
0% WER with good timbre — the safe all-round choices. `focalcodec` and `lscodec`
push timbre hardest and accept higher WER for it. Pick by which axis matters more
for your application: intelligibility, or how closely the output matches the
target voice.

## Validation

Every engine's ONNX pipeline is checked against its original PyTorch
implementation: the same source and reference are run through both, and the
speaker embeddings of the two outputs are compared. Engines ship with that
agreement at cosine ≥ ~0.9 (for example, `lscodec` ONNX↔torch ≈ 0.97), so the
behaviour you hear is the model's, faithfully reproduced in ONNX.
