from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path

from pronunciation_backend.training.schemas import PreparedUtteranceArtifact
from pronunciation_backend.training.speechocean_utils import (
    load_scores,
    read_kaldi_mapping,
    read_wav_scp,
    resolve_speechocean_raw_root,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare persistent SpeechOcean762 utterance manifests from a preloaded dataset root."
    )
    parser.add_argument("--dataset-root", required=True, help="Dataset root under /cold/pronunciation/datasets/speechocean762.")
    parser.add_argument("--output-dir", help="Output directory for prepared manifests. Defaults to <dataset-root>/prepared.")
    parser.add_argument(
        "--val-speaker-fraction",
        type=float,
        default=0.1,
        help="Fraction of official-train speakers reserved for validation.",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=1337,
        help="Seed used for deterministic speaker-disjoint train/val splitting.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _assign_official_train_splits(
    speakers: set[str],
    *,
    val_speaker_fraction: float,
    split_seed: int,
) -> dict[str, str]:
    if not 0.0 <= val_speaker_fraction < 1.0:
        raise ValueError("val_speaker_fraction must be in [0.0, 1.0).")

    ordered = sorted(speakers)
    if len(ordered) <= 1 or val_speaker_fraction == 0.0:
        return {speaker_id: "train" for speaker_id in ordered}

    rng = random.Random(split_seed)
    rng.shuffle(ordered)
    val_count = int(math.ceil(len(ordered) * val_speaker_fraction))
    val_count = max(1, min(len(ordered) - 1, val_count))
    val_speakers = set(ordered[:val_count])
    return {
        speaker_id: ("val" if speaker_id in val_speakers else "train")
        for speaker_id in ordered
    }


def _write_jsonl(path: Path, items: list[PreparedUtteranceArtifact], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {path}")
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(item.model_dump_json() + "\n")


def _scan_split(
    split_dir: Path,
    *,
    dataset_root: Path,
    raw_root: Path,
    effective_split: str,
    scores: dict[str, dict],
) -> list[PreparedUtteranceArtifact]:
    utt2spk = read_kaldi_mapping(split_dir / "utt2spk")
    wav_scp = read_wav_scp(split_dir / "wav.scp", raw_root=raw_root, dataset_root=dataset_root)

    rows: list[PreparedUtteranceArtifact] = []
    for utterance_id, speaker_id in sorted(utt2spk.items()):
        score_entry = scores.get(utterance_id)
        if score_entry is None:
            continue
        audio_path = wav_scp.get(utterance_id)
        if audio_path is None:
            continue
        text = str(score_entry["text"]).strip()
        if not text:
            continue
        rows.append(
            PreparedUtteranceArtifact(
                utterance_id=utterance_id,
                speaker_id=speaker_id,
                dataset="speechocean762",
                split=effective_split,  # type: ignore[arg-type]
                text=text,
                normalized_text=text.lower(),
                audio_path=audio_path,
                transcript_path=None,
            )
        )
    return rows


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"missing dataset root: {dataset_root}")
        return 1

    raw_root = resolve_speechocean_raw_root(dataset_root)
    output_dir = Path(args.output_dir) if args.output_dir else dataset_root / "prepared"
    scores = load_scores(raw_root / "scores.json")

    official_train_dir = raw_root / "train"
    official_test_dir = raw_root / "test"
    official_train_speakers = set(read_kaldi_mapping(official_train_dir / "utt2spk").values())
    train_val_assignment = _assign_official_train_splits(
        official_train_speakers,
        val_speaker_fraction=args.val_speaker_fraction,
        split_seed=args.split_seed,
    )

    rows_by_split: dict[str, list[PreparedUtteranceArtifact]] = {"train": [], "val": [], "test": []}
    for row in _scan_split(
        official_test_dir,
        dataset_root=dataset_root,
        raw_root=raw_root,
        effective_split="test",
        scores=scores,
    ):
        rows_by_split["test"].append(row)

    for row in _scan_split(
        official_train_dir,
        dataset_root=dataset_root,
        raw_root=raw_root,
        effective_split="train",
        scores=scores,
    ):
        effective_split = train_val_assignment.get(row.speaker_id, "train")
        rows_by_split[effective_split].append(row.model_copy(update={"split": effective_split}))

    for split, rows in rows_by_split.items():
        rows.sort(key=lambda item: (item.speaker_id, item.utterance_id))
        target = output_dir / f"{split}.jsonl"
        _write_jsonl(target, rows, overwrite=args.overwrite)
        print(f"wrote split={split} utterances={len(rows)} path={target}")

    summary = {
        "dataset": "speechocean762",
        "dataset_root": str(dataset_root),
        "raw_root": str(raw_root),
        "output_dir": str(output_dir),
        "val_speaker_fraction": args.val_speaker_fraction,
        "split_seed": args.split_seed,
        "counts": {split: len(rows) for split, rows in rows_by_split.items()},
        "speakers": {
            split: len(Counter(row.speaker_id for row in rows))
            for split, rows in rows_by_split.items()
        },
        "official_train_speakers": len(official_train_speakers),
        "scores_entries": len(scores),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
