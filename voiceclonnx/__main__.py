"""CLI entry point: ``voiceclonnx``."""

from __future__ import annotations

import argparse


def _cmd_clone(args: argparse.Namespace) -> None:
    from voiceclonnx import VoiceCloner

    extra = {}
    if args.exaggeration is not None:
        extra["exaggeration"] = args.exaggeration
    if args.max_new_tokens is not None:
        extra["max_new_tokens"] = args.max_new_tokens

    cloner = VoiceCloner(engine=args.engine, **extra)
    out = cloner.clone_voice(args.audio, args.voice, args.out)
    print(f"Saved: {out}")


def _cmd_list(args: argparse.Namespace) -> None:
    from voiceclonnx.engines.base import ENGINE_REGISTRY

    if not ENGINE_REGISTRY:
        print("(no engines registered)")
        return
    for alias, entry in sorted(ENGINE_REGISTRY.items()):
        print(f"{alias}")
        print(f"  {entry.description}")
        if entry.extras:
            print(f"  Install : pip install voiceclonnx[{entry.extras}]")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="voiceclonnx",
        description="Pure-ONNX multi-engine voice-cloning CLI (audio-to-audio)",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    # --- clone ---
    p_clone = sub.add_parser("clone", help="Voice-convert a WAV file")
    p_clone.add_argument("--engine", default="chatterbox", help="Engine alias")
    p_clone.add_argument("--audio", required=True, help="Source audio WAV path")
    p_clone.add_argument("--voice", required=True, help="Reference voice WAV path")
    p_clone.add_argument("--out", required=True, help="Output WAV path")
    p_clone.add_argument("--exaggeration", type=float, default=None,
                         help="Voice exaggeration (engine-specific, e.g. 0.6)")
    p_clone.add_argument("--max-new-tokens", dest="max_new_tokens", type=int,
                         default=None, help="Max speech tokens (engine-specific)")

    # --- list ---
    sub.add_parser("list", help="List available engines")

    args = parser.parse_args()
    dispatch = {"clone": _cmd_clone, "list": _cmd_list}
    dispatch[args.cmd](args)


if __name__ == "__main__":
    main()
