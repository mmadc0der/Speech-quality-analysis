from __future__ import annotations

import argparse
from pathlib import Path

from pronunciation_backend.training.parquet_dataset import bake_mmap_to_parquet, resolve_mmap_dir_for_bake


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize mmap-backed .npy feature tables into a dense Parquet file under <split>/parquet/."
    )
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--features-dir", type=Path, help="Feature split directory (mmap resolved automatically).")
    g.add_argument("--mmap-dir", type=Path, help="Explicit mmap directory containing manifest.json.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for words.parquet (default: <mmap-parent>/parquet).",
    )
    parser.add_argument(
        "--row-group-utterances",
        type=int,
        default=4096,
        help="Utterances per Parquet row group / write batch.",
    )
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--compression",
        choices=["snappy", "zstd", "gzip", "none"],
        default="snappy",
        help="Parquet compression (none = larger files, often faster CPU decode).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    mmap_dir = args.mmap_dir if args.mmap_dir is not None else resolve_mmap_dir_for_bake(args.features_dir)
    bake_mmap_to_parquet(
        mmap_dir,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
        row_group_utterances=args.row_group_utterances,
        progress_every=args.progress_every,
        compression=args.compression,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
