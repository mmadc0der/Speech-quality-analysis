from __future__ import annotations

import argparse
from pathlib import Path

from pronunciation_backend.training.mmap_dataset import pack_jsonl_split_to_mmap


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pack JSONL feature shards into an mmap-backed training dataset.")
    parser.add_argument("--features-dir", required=True, help="Path to a split directory containing part-*.jsonl shards.")
    parser.add_argument("--output-dir", help="Optional output directory. Defaults to <features-dir>/mmap.")
    parser.add_argument("--acoustic-dtype", choices=["float16", "float32"], default="float16")
    parser.add_argument("--progress-every", type=int, default=250_000)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    pack_jsonl_split_to_mmap(
        Path(args.features_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        overwrite=args.overwrite,
        acoustic_dtype=args.acoustic_dtype,
        progress_every=args.progress_every,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
