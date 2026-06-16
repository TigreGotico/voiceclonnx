"""Rank engines by speaker similarity to the target voice using speakeronnx.

For each engine output we compute cosine similarity between the output's speaker
embedding and the *target* reference embedding (aria / sonia). Higher = the
converted voice sounds closer to the intended target speaker.

We also compute similarity to the *source* utterance: a high source-similarity
with low target-similarity means the engine barely converted (passed the source
speaker through).
"""
import glob
import os
import sys

import numpy as np
from speakeronnx import SpeakerEmbedder, cosine

DEMO = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DEMO, "outputs")

REF = {
    "aria": os.path.join(DEMO, "reference_aria.wav"),
    "sonia": os.path.join(DEMO, "reference_sonia.wav"),
}
SOURCE = os.path.join(DEMO, "source.wav")

MODELS = sys.argv[1:] or ["wespeaker-resnet34", "titanet-large", "campplus"]


def target_of(fname):
    """Map an output filename to its target voice key, or None if N/A."""
    base = os.path.basename(fname)
    if "aria" in base:
        return "aria"
    if "sonia" in base:
        return "sonia"
    return None  # e.g. rvc__woman1 — any-to-ONE, different target


def engine_of(fname):
    base = os.path.basename(fname).replace(".wav", "")
    return base.split("__")[0]


def main():
    files = sorted(glob.glob(os.path.join(OUT, "*.wav")))
    # per-model cache of embeddings, then per (engine,target) similarity
    # results[engine][target] = list of target-sim across models
    results = {}
    src_sim = {}
    ref_floor = {}  # source-vs-reference baseline per model/target

    for model in MODELS:
        print(f"\n=== model: {model} ===", file=sys.stderr)
        emb = SpeakerEmbedder(model=model)
        ref_emb = {k: emb.embed(v) for k, v in REF.items()}
        src_emb = emb.embed(SOURCE)
        for tgt in REF:
            ref_floor.setdefault(tgt, []).append(cosine(src_emb, ref_emb[tgt]))
        for f in files:
            tgt = target_of(f)
            if tgt is None:
                continue
            e = engine_of(f)
            v = emb.embed(f)
            results.setdefault(e, {}).setdefault(tgt, []).append(cosine(v, ref_emb[tgt]))
            src_sim.setdefault(e, {}).setdefault(tgt, []).append(cosine(v, src_emb))

    # Per-engine aggregates.
    #  tgt_sim   : mean cosine(output, target) across targets, per model
    #  margin    : mean (cosine(output,target) - cosine(output,source)), scale-invariant
    # We rank engines within each model by tgt_sim, then average ranks (Borda)
    # so models with different cosine scales contribute equally.
    n_models = len(MODELS)
    engines = list(results.keys())

    # per-model mean target sim and margin for each engine
    per_model_tgt = {e: [] for e in engines}   # [model] -> mean tgt sim
    per_model_margin = {e: [] for e in engines}
    for mi in range(n_models):
        for e in engines:
            tvals = [results[e][t][mi] for t in results[e]]
            svals = [src_sim[e][t][mi] for t in src_sim[e]]
            per_model_tgt[e].append(float(np.mean(tvals)))
            per_model_margin[e].append(float(np.mean(tvals) - np.mean(svals)))

    # average rank across models (1 = best per model)
    mean_rank = {e: 0.0 for e in engines}
    for mi in range(n_models):
        order = sorted(engines, key=lambda e: per_model_tgt[e][mi], reverse=True)
        for r, e in enumerate(order, 1):
            mean_rank[e] += r / n_models

    rows = []
    for e in engines:
        rows.append((
            e,
            float(np.mean(per_model_tgt[e])),     # ensemble target sim
            float(np.mean(per_model_margin[e])),  # ensemble conversion margin
            mean_rank[e],
        ))
    rows.sort(key=lambda r: r[3])  # by mean rank (lower = better)

    floor = {t: float(np.mean(v)) for t, v in ref_floor.items()}
    floor_overall = float(np.mean(list(floor.values())))
    print(f"\nModels: {', '.join(MODELS)}")
    print(f"Source->target baseline (no-conversion floor): mean={floor_overall:.3f}")
    print("target-sim = cosine(output, target voice)  |  margin = target-sim minus source-sim")
    print("margin > 0 means the output sounds more like the TARGET than the SOURCE.\n")
    print(f"{'rank':<5}{'engine':<16}{'tgt-sim':<10}{'margin':<10}{'mean-rank':<10}")
    print("-" * 51)
    for i, (e, tsim, margin, mr) in enumerate(rows, 1):
        print(f"{i:<5}{e:<16}{tsim:<10.3f}{margin:<+10.3f}{mr:<10.2f}")


if __name__ == "__main__":
    main()
