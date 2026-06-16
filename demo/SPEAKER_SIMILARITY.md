# Speaker-similarity audit & engine curation

WER tells you whether the words are intelligible. It says **nothing** about
whether the converted voice actually sounds like the *target speaker*. This
audit ranks engines by **speaker similarity to the target voice** using
[speakeronnx](https://github.com/TigreGotico/speakeronnx), then validates the
weak ones against their **original pre-export PyTorch model** to separate "the
model/recipe is weak" from "the ONNX export is broken".

**It is also the rationale for the curated roster.** Seven engines that scored at
the no-conversion floor (they kept the *source* voice), were codecs repurposed
for VC, or were unbenchmarked/garbled ports were **removed**. The measurements
below are why.

Reproduce the ranking:

```bash
pip install voiceclonnx speakeronnx soxr
python demo/speaker_similarity.py                    # 3-model ensemble
python demo/speaker_similarity.py wespeaker-resnet34 # single, most discriminative
```

(The per-engine torch parity scripts were removed along with the engines they
analysed; their results are recorded below.)

## How it's measured

- For each engine output `<engine>__{aria,sonia}.wav` we take its speaker
  embedding and compute **cosine similarity to the target reference**
  (`reference_aria.wav` / `reference_sonia.wav`). Higher = closer to the
  intended voice.
- We also measure **source similarity** (cosine to `source.wav`). High
  source-similarity with low target-similarity means the engine barely changed
  the speaker — it leaked the source voice through.
- Primary model: `wespeaker-resnet34` (clearest separation — the source→target
  "no-conversion floor" is only **0.090**, so anything well above that is doing
  real conversion). Cross-checked with a 3-model ensemble
  (`+ titanet-large + campplus`); the top and bottom groups are stable.

## Ranking that drove the curation (wespeaker-resnet34, target-similarity)

`rvc` is **kept** but excluded from this ranking — it is any-to-ONE and its demo
targets a different community voice (`woman1`), not aria/sonia.

| Rank | Engine | Target-sim | WER | Roster | Read |
|---|---|---|---|---|---|
| 1 | `focalcodec` | **0.606** | 15–19% | ✅ keep | Best timbre transfer |
| 2 | `chatterbox` | **0.538** | 4–8% | ✅ keep | Best all-rounder (also moves *away* from source) |
| 3 | `knnvc` | 0.490 | 12–15% | ✅ keep | Strong timbre |
| 4 | `facodec` | 0.444 | **0%** | ✅ keep | Excellent balance (0% WER + good timbre) |
| 5 | `openvoice` | 0.372 | **0%** | ✅ keep | Good balance |
| 6 | `vec2wav` | 0.322 | 119–127% | ✂ **removed** | Garbled — destroys content (content path bypassed) |
| 7 | `bicodec` | 0.290 | 12% | ✅ keep | Moderate but real transfer |
| 8 | `triaan` | 0.286 | 4% | ✅ keep | Moderate but real transfer |
| 9 | `cosyvoice` | 0.210 | 8% | ✅ keep | Weak-ish but 2× floor, cross-lingual |
| 10 | `freevc` | 0.105 | 12% | ✂ **removed** | At the floor — leans toward source |
| 11 | `speechtokenizer` | 0.092 | 4–12% | ✂ **removed** | At the no-conversion floor |
| 12 | `mimi` | 0.054 | **0%** | ✂ **removed** | Perfect words, **wrong voice** (≈ source) |
| 13 | `linacodec` | 0.032 | 8–15% | ✂ **removed** | Near-floor; codec, not a VC model |
| 14 | `quickvc` | 0.011 | **0%** | ✂ **removed** | Perfect words, almost no conversion |

(`seedvc` was also removed — it shipped unbenchmarked with its core content path
bypassed, so it had no demo clips to score.)

### The WER paradox (why we don't rank by WER)

`mimi`, `quickvc`, `facodec`, `openvoice` all scored **0% WER**, but split into
two opposite groups:

- `facodec` / `openvoice` — 0% WER **and** good timbre transfer. **Kept.**
- `mimi` / `quickvc` — 0% WER but the output still sounds like the **source**
  speaker (`mimi` source-similarity ≈ 0.89). Perfect words, wrong voice. A
  WER-only table makes these look top-tier; on the actual VC task they were at
  the bottom. **Removed.**

## Was it the model or the export? (parity gates)

Before removing the weak engines we confirmed the failure was the model/recipe,
**not** our ONNX export — by running the original PyTorch model through the same
recipe and comparing to the shipped ONNX output. If torch ≈ ONNX but **both**
miss the target, the export is fine.

| Engine | torch ↔ onnx agreement | torch → target | Verdict |
|---|---|---|---|
| `mimi` | decoder corr **0.99**, full-VC corr **0.96**, roundtrip 0.95 | low (≈ source) | **Export faithful.** RVQ stream-swap keeps the source's acoustic streams (1–31), so the source *timbre* is retained by design. Recipe limit. |
| `speechtokenizer` | embedding **0.89**, wav-corr 0.73 | low (0.08–0.13) | **Export faithful.** The original torch model *also* misses the target — content RVQ layers (kept from source) carry speaker identity. Recipe/model limit. |
| `freevc` | embedding 0.72–0.77, wav-corr 0.49 | low (−0.00 to 0.12, leans source) | **Model-limited.** Original torch FreeVC also leans toward the source on this pair. Export mostly faithful (slightly more numeric drift than the codec engines). |

**None of the removed engines were bad because of the ONNX export.** They were
removed because the underlying model/recipe doesn't transfer the target voice.

### quickvc deep-dive

`quickvc` is a *dedicated* VC model that still scored ~0 timbre transfer, so we
probed its speaker path directly on the published ONNX (the upstream generator
checkpoint `G_1200000.pth` is non-public, so a torch parity gate isn't possible):

- **Weakly-discriminative speaker encoder.** Two different target voices produce
  nearly the same d-vector: `cosine(g_aria, g_sonia) = 0.829`. Not degenerate
  (`g_aria` vs source = 0.55), just weak on unseen voices.
- **`g` is wired but direction barely matters.** Zeroing `g` changes the output
  (`cosine(out_aria, out_zero) = 0.455`), so conditioning is connected — yet
  swapping the target aria→sonia moves the output by only ~0.16
  (`cosine(out_aria, out_sonia) = 0.844`), and neither output resembles its
  target (≤0.08).
- **Diagnosis:** the speaker encoder collapses distinct references to
  near-identical d-vectors — consistent with QuickVC being **any-to-many**
  (trained for a fixed target set), so it generalises poorly to arbitrary
  references. A model/usage-fit limitation, not an export bug.

## Upstream provenance — why some were never VC models

Several removed engines are not voice-conversion models at all — they are neural
codecs repurposed for VC:

| Engine | What it is upstream | Claims VC? | Why removed |
|---|---|---|---|
| `mimi` | Kyutai **audio-compression codec** for Moshi/speech-LLMs | No | Not a VC model; stream-swap keeps source timbre |
| `linacodec` | "Highly compressive **neural audio codec**"; VC is an *indirect* side-use | Secondary | Codec-first; near-floor timbre transfer |
| `speechtokenizer` | EnCodec-style **RVQ codec** (ACL 2024); paper demos zero-shot VC | Downstream demo | Modest transfer is the ceiling; sat at the floor here |
| `quickvc` | Dedicated **any-to-many** VC model | Yes | Generalises poorly to arbitrary references (see deep-dive) |
| `freevc` | Dedicated **zero-shot any-to-any** VC (ICASSP 2023) | Yes | Original torch model also leans source on this eval |
| `vec2wav` | **VC-native** discrete-token vocoder (Interspeech 2024) | Yes | The port is garbled (content path bypassed) |
| `seedvc` | Flow-matching VC (Seed-VC) | Yes | Shipped unbenchmarked with the content path bypassed |

The curated roster keeps only engines that measurably move the converted voice
toward the target: `focalcodec`, `chatterbox`, `knnvc`, `facodec`, `openvoice`,
`bicodec`, `triaan`, `cosyvoice`, `lscodec`, and `rvc` (any-to-ONE).

## Added: lscodec (the gate working as intended)

`lscodec` ([LSCodec](https://github.com/X-LANCE/LSCodec-Inference), Interspeech
2025) was the one backlog engine that passed this gate and was **added**. It is a
*speaker-decoupled* codec — built for VC — and it shows: target-similarity
**0.54** (would rank #3 here), source-similarity only 0.19. The ONNX export was
validated against the upstream torch model the same way (ONNX↔torch cosine
**0.97**) before shipping. It is the inverse tradeoff from the removed engines —
strongest timbre transfer in the codec family, but a moderate ~35% WER. Every
other open porting request was blocked (non-commercial weights, unreleased, or
no license) — see the issue tracker.
