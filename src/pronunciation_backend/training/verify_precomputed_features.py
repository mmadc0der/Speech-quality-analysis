from __future__ import annotations

import argparse
import json
from pathlib import Path

from pronunciation_backend.training.schemas import PhoneEmbeddingArtifact, TrainingUtteranceArtifact


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Verify precomputed phone feature shards against schema and optional aligned artifacts.")
    parser.add_argument("--features-dir", required=True, help="Directory containing part-*.jsonl feature shards.")
    parser.add_argument("--aligned-path", help="Optional aligned manifest to compare expected word/phone counts.")
    parser.add_argument("--report-path", help="Optional JSON output path.")
    parser.add_argument("--max-rows", type=int, help="Optional limit for quick smoke checks.")
    return parser


def _iter_nonempty_lines(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line:
                yield line


def _summarize_aligned(path: Path) -> dict[str, int]:
    expected_utterances = 0
    expected_rows = 0
    for line in _iter_nonempty_lines(path):
        artifact = TrainingUtteranceArtifact.model_validate_json(line)
        expected_utterances += 1
        expected_rows += len(artifact.phone_labels)
    return {
        "aligned_utterances": expected_utterances,
        "aligned_feature_rows": expected_rows,
    }


def _verify_feature_rows(features_dir: Path, *, max_rows: int | None) -> dict[str, object]:
    shard_paths = sorted(features_dir.glob("part-*.jsonl"))
    if not shard_paths:
        raise FileNotFoundError(f"No feature shards found in {features_dir}")

    total_rows = 0
    utterance_groups = 0
    previous_utterance_id: str | None = None
    embedding_dim_histogram: dict[int, int] = {}
    frame_count_min: int | None = None
    frame_count_max: int | None = None
    duration_ms_min: int | None = None
    duration_ms_max: int | None = None
    split_counts: dict[str, int] = {}
    errors: list[str] = []
    sample_row: dict[str, object] | None = None

    for shard_path in shard_paths:
        for line in _iter_nonempty_lines(shard_path):
            artifact = PhoneEmbeddingArtifact.model_validate_json(line)
            total_rows += 1
            if sample_row is None:
                sample_row = artifact.model_dump(mode="json")

            if artifact.utterance_id != previous_utterance_id:
                utterance_groups += 1
                previous_utterance_id = artifact.utterance_id

            embedding_dim = len(artifact.mean_embedding)
            embedding_dim_histogram[embedding_dim] = embedding_dim_histogram.get(embedding_dim, 0) + 1

            frame_count_min = artifact.frame_count if frame_count_min is None else min(frame_count_min, artifact.frame_count)
            frame_count_max = artifact.frame_count if frame_count_max is None else max(frame_count_max, artifact.frame_count)
            duration_ms_min = artifact.duration_ms if duration_ms_min is None else min(duration_ms_min, artifact.duration_ms)
            duration_ms_max = artifact.duration_ms if duration_ms_max is None else max(duration_ms_max, artifact.duration_ms)
            split_counts[artifact.split] = split_counts.get(artifact.split, 0) + 1

            if max_rows is not None and total_rows >= max_rows:
                break
        if max_rows is not None and total_rows >= max_rows:
            break

    return {
        "shards": len(shard_paths),
        "rows": total_rows,
        "utterance_groups": utterance_groups,
        "embedding_dim_histogram": embedding_dim_histogram,
        "frame_count_min": frame_count_min,
        "frame_count_max": frame_count_max,
        "duration_ms_min": duration_ms_min,
        "duration_ms_max": duration_ms_max,
        "split_counts": split_counts,
        "sample_row": sample_row,
        "errors": errors,
    }


def main() -> int:
    args = _build_parser().parse_args()
    features_dir = Path(args.features_dir)
    if not features_dir.exists():
        print(f"missing features dir: {features_dir}")
        return 1

    summary = {
        "features_dir": str(features_dir),
        "feature_summary": _verify_feature_rows(features_dir, max_rows=args.max_rows),
    }

    if args.aligned_path:
        aligned_path = Path(args.aligned_path)
        if not aligned_path.exists():
            print(f"missing aligned path: {aligned_path}")
            return 1
        aligned_summary = _summarize_aligned(aligned_path)
        summary["aligned_summary"] = aligned_summary
        feature_summary = summary["feature_summary"]
        summary["comparison"] = {
            "rows_match": feature_summary["rows"] == aligned_summary["aligned_feature_rows"],
            "utterance_groups_match": feature_summary["utterance_groups"] == aligned_summary["aligned_utterances"],
            "row_delta": feature_summary["rows"] - aligned_summary["aligned_feature_rows"],
            "utterance_group_delta": feature_summary["utterance_groups"] - aligned_summary["aligned_utterances"],
        }

    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
