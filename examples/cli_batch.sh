#!/usr/bin/env bash
# cli_batch.sh — batch voice-convert a folder of WAV files using the vconnx CLI.
#
# Usage:
#   ./examples/cli_batch.sh <input_dir> <reference.wav> <output_dir> [engine]
#
# Arguments:
#   input_dir     Directory containing source WAV files.
#   reference.wav Path to the reference speaker WAV.
#   output_dir    Directory where converted WAVs will be written.
#   engine        Engine alias (default: chatterbox).
#
# Example:
#   ./examples/cli_batch.sh ./wavs ref_speaker.wav ./out chatterbox
#
# Requirements:
#   pip install "vconnx[chatterbox]"   # or whichever engine you choose

set -euo pipefail

INPUT_DIR="${1:?Usage: $0 <input_dir> <reference.wav> <output_dir> [engine]}"
REFERENCE="${2:?Reference WAV required}"
OUTPUT_DIR="${3:?Output directory required}"
ENGINE="${4:-chatterbox}"

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "ERROR: input_dir '$INPUT_DIR' is not a directory." >&2
    exit 1
fi

if [[ ! -f "$REFERENCE" ]]; then
    echo "ERROR: reference '$REFERENCE' not found." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

total=0
ok=0
fail=0

for wav in "$INPUT_DIR"/*.wav; do
    [[ -f "$wav" ]] || continue
    total=$((total + 1))
    base="$(basename "$wav" .wav)"
    out="$OUTPUT_DIR/${base}_converted.wav"

    echo "  [$total] $wav → $out"
    if vconnx clone \
            --engine "$ENGINE" \
            --audio "$wav" \
            --voice "$REFERENCE" \
            --out "$out"; then
        ok=$((ok + 1))
    else
        echo "  FAILED: $wav" >&2
        fail=$((fail + 1))
    fi
done

echo ""
echo "Batch complete: $ok/$total converted, $fail failed."
echo "Output: $OUTPUT_DIR"
