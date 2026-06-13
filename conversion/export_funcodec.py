"""Export stub for FunCodec semantic codec engine.

BLOCKED — see docs/engines/funcodec.md for the full investigation findings.

Summary
-------
FunCodec (modelscope/FunCodec, MIT) does not provide a publicly available
checkpoint that satisfies the semantic RVQ paradigm required for zero-shot
voice conversion:

1. All published HF checkpoints (``alibaba-damo/audio_codec-encodec-*``) are
   pure EnCodec-style reconstruction codecs with no SSL distillation objective
   on any RVQ layer.  Using them for VC via RVQ-1 token swap would fail the
   WER gate.

2. The ``CodecSemanticAug`` model class (``funcodec/models/codec_semantic_aug.py``)
   requires PPG (phonetic posteriorgrams) as an **external inference-time input** —
   it is NOT a self-contained codec where RVQ-1 autonomously captures content.
   No public checkpoint exists for this variant.

3. The checkpoint name ``funcodec_en_libritts-16k-semantic`` cited in issue #38
   does not exist on HuggingFace, ModelScope, or the FunCodec GitHub repository.

This stub is retained so the export recipe is documented for when a qualifying
checkpoint (SSL-distilled RVQ-1, self-contained, publicly available) is released.
If such a checkpoint appears, the implementation follows SpeechTokenizer exactly
(see ``conversion/export_speechtokenizer.py``).

Recipe (for future implementer)
---------------------------------
When a semantic FunCodec checkpoint is available:

1. Load the checkpoint.  Verify that ``encode(audio)`` (no PPG argument) returns
   integer code indices where layer 0 tracks phonetic content.
2. Build encoder wrapper: ``audio (1, 1, N) → codes (Q, 1, T) int64``.
3. Build decoder wrapper: ``codes (Q, 1, T) int64 → waveform (1, 1, N) float32``.
4. Export both with opset 14, dynamic axes on N and T.
5. Extract codebooks: ``(Q, codebook_size, D) float32`` → ``codebooks.npy``.
6. Run parity against upstream ``model.encode`` + ``model.decode`` (never vs
   a reconstruction — that is the SpeechTokenizer-parity-trap lesson).
7. Quantize, write manifest/provenance, push to TigreGotico/voiceclonnx-funcodec.
8. Implement ``voiceclonnx/engines/funcodec.py`` (reuse SpeechTokenizer helpers).
"""

from __future__ import annotations

import sys

def main():
    print(
        "[export_funcodec] BLOCKED: no qualifying FunCodec semantic checkpoint found.\n"
        "See docs/engines/funcodec.md for the investigation details and future recipe.\n"
        "Issue #38 is blocked pending a published SSL-distilled FunCodec checkpoint."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
