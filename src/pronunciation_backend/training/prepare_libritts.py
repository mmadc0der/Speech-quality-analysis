from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from pronunciation_backend.training.schemas import PreparedUtteranceArtifact


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare persistent LibriTTS utterance manifests from a preloaded dataset root.")
    parser.add_argument("--dataset-root", required=True, help="Root directory containing LibriTTS subset folders or files.")
    parser.add_argument("--output-dir", help="Output directory for prepared manifests. Defaults to <dataset-root>/prepared.")
    parser.add_argument("--audio-extensions", nargs="+", default=[".wav", ".flac"])
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _split_from_subset(path: Path) -> str:
    subset = path.parts[0].lower() if path.parts else ""
    if subset.startswith("train"):
        return "train"
    if subset.startswith("dev"):
        return "val"
    if subset.startswith("test"):
        return "test"
    return "train"


def _find_transcript(audio_path: Path) -> Path | None:
    normalized = audio_path.with_suffix(".normalized.txt")
    if normalized.exists():
        return normalized
    original = audio_path.with_suffix(".original.txt")
    if original.exists():
        return original
    plain = audio_path.with_suffix(".txt")
    if plain.exists():
        return plain
    return None


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _relative_str(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _candidate_audio_files(dataset_root: Path, audio_extensions: set[str]) -> list[Path]:
    audio_files: list[Path] = []
    for candidate in dataset_root.rglob("*"):
        if not candidate.is_file():
            continue
        if "prepared" in candidate.parts or "aligned" in candidate.parts:
            continue
        if candidate.suffix.lower() in audio_extensions:
            audio_files.append(candidate)
    return sorted(audio_files)


def _scan_dataset(dataset_root: Path, audio_extensions: set[str]) -> tuple[dict[str, list[PreparedUtteranceArtifact]], dict[str, int]]:
    artifacts: dict[str, list[PreparedUtteranceArtifact]] = {"train": [], "val": [], "test": []}
    stats = {
        "audio_candidates": 0,
        "missing_transcript": 0,
        "empty_transcript": 0,
        "prepared_rows": 0,
    }

    for audio_path in _candidate_audio_files(dataset_root, audio_extensions):
        stats["audio_candidates"] += 1
        transcript_path = _find_transcript(audio_path)
        if transcript_path is None:
            stats["missing_transcript"] += 1
            continue

        relative_audio = audio_path.relative_to(dataset_root)
        split = _split_from_subset(relative_audio)
        text = _normalize_text(transcript_path.read_text(encoding="utf-8"))
        if not text:
            stats["empty_transcript"] += 1
            continue

        artifacts[split].append(
            PreparedUtteranceArtifact(
                utterance_id=audio_path.stem,
                speaker_id=audio_path.parent.parent.name if len(audio_path.parts) >= 3 else "unknown",
                dataset="libritts",
                split=split,
                text=text,
                normalized_text=text.lower(),
                audio_path=_relative_str(audio_path, dataset_root),
                transcript_path=_relative_str(transcript_path, dataset_root),
            )
        )
        stats["prepared_rows"] += 1

    return artifacts, stats


def _write_jsonl(path: Path, items: list[PreparedUtteranceArtifact], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing manifest: {path}")
    with path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(item.model_dump_json() + "\n")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print(f"missing dataset root: {dataset_root}")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else dataset_root / "prepared"
    audio_extensions = {extension.lower() if extension.startswith(".") else f".{extension.lower()}" for extension in args.audio_extensions}
    artifacts, stats = _scan_dataset(dataset_root, audio_extensions)

    for split, rows in artifacts.items():
        target = output_dir / f"{split}.jsonl"
        _write_jsonl(target, rows, overwrite=args.overwrite)
        print(f"wrote split={split} utterances={len(rows)} path={target}")

    summary = {
        "dataset": "libritts",
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "audio_extensions": sorted(audio_extensions),
        "scan_stats": stats,
        "counts": {split: len(rows) for split, rows in artifacts.items()},
        "speakers": {
            split: len(Counter(row.speaker_id for row in rows))
            for split, rows in artifacts.items()
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        "scan stats: "
        f"audio_candidates={stats['audio_candidates']} "
        f"missing_transcript={stats['missing_transcript']} "
        f"empty_transcript={stats['empty_transcript']} "
        f"prepared_rows={stats['prepared_rows']}"
    )
    print(f"wrote summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
