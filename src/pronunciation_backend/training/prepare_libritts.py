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
    parser.add_argument("--audio-extension", default=".wav")
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


def _scan_dataset(dataset_root: Path, audio_extension: str) -> dict[str, list[PreparedUtteranceArtifact]]:
    artifacts: dict[str, list[PreparedUtteranceArtifact]] = {"train": [], "val": [], "test": []}
    pattern = f"*{audio_extension}"

    for audio_path in sorted(dataset_root.rglob(pattern)):
        if "prepared" in audio_path.parts or "aligned" in audio_path.parts:
            continue

        transcript_path = _find_transcript(audio_path)
        if transcript_path is None:
            continue

        relative_audio = audio_path.relative_to(dataset_root)
        split = _split_from_subset(relative_audio)
        text = _normalize_text(transcript_path.read_text(encoding="utf-8"))
        if not text:
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

    return artifacts


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
    artifacts = _scan_dataset(dataset_root, args.audio_extension)

    for split, rows in artifacts.items():
        target = output_dir / f"{split}.jsonl"
        _write_jsonl(target, rows, overwrite=args.overwrite)
        print(f"wrote split={split} utterances={len(rows)} path={target}")

    summary = {
        "dataset": "libritts",
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "counts": {split: len(rows) for split, rows in artifacts.items()},
        "speakers": {
            split: len(Counter(row.speaker_id for row in rows))
            for split, rows in artifacts.items()
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
