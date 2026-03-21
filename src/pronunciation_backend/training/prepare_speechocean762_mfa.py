from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from pronunciation_backend.training.schemas import PreparedUtteranceArtifact


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize an MFA-ready SpeechOcean762 corpus with mirrored audio paths and .lab transcripts."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--prepared-dir", help="Defaults to <dataset-root>/prepared.")
    parser.add_argument("--output-dir", help="Defaults to <dataset-root>/mfa_corpus.")
    parser.add_argument(
        "--link-mode",
        choices=("auto", "symlink", "hardlink", "copy"),
        default="auto",
        help="How to materialize audio files into the MFA corpus.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _read_prepared(path: Path) -> list[PreparedUtteranceArtifact]:
    rows: list[PreparedUtteranceArtifact] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            rows.append(PreparedUtteranceArtifact.model_validate_json(line))
    return rows


def _safe_unlink(path: Path) -> None:
    if path.exists() or path.is_symlink():
        path.unlink()


def _materialize_audio(source: Path, target: Path, *, link_mode: str, overwrite: bool) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        raise FileNotFoundError(f"Missing source audio for MFA corpus: {source}")

    if target.exists() or target.is_symlink():
        if not overwrite:
            return "reused"
        _safe_unlink(target)

    modes = [link_mode] if link_mode != "auto" else ["symlink", "hardlink", "copy"]
    last_error: OSError | None = None
    for mode in modes:
        try:
            if mode == "symlink":
                target.symlink_to(source)
            elif mode == "hardlink":
                os.link(source, target)
            elif mode == "copy":
                shutil.copy2(source, target)
            else:
                raise ValueError(f"Unsupported link mode: {mode}")
            return mode
        except OSError as exc:
            last_error = exc

    raise OSError(f"Could not materialize audio at {target}: {last_error}")


def _write_lab(path: Path, text: str, *, overwrite: bool) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return "reused"
    normalized = " ".join(text.strip().split())
    path.write_text(normalized + "\n", encoding="utf-8")
    return "written"


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    prepared_dir = Path(args.prepared_dir) if args.prepared_dir else dataset_root / "prepared"
    output_dir = Path(args.output_dir) if args.output_dir else dataset_root / "mfa_corpus"

    if not dataset_root.exists():
        print(f"missing dataset root: {dataset_root}")
        return 1
    if not prepared_dir.exists():
        print(f"missing prepared dir: {prepared_dir}")
        return 1

    summary: dict[str, object] = {
        "dataset": "speechocean762",
        "dataset_root": str(dataset_root),
        "prepared_dir": str(prepared_dir),
        "output_dir": str(output_dir),
        "link_mode": args.link_mode,
        "counts": {},
    }

    total_rows = 0
    total_audio_created = 0
    total_audio_reused = 0
    total_lab_written = 0
    total_lab_reused = 0

    for split in ("train", "val", "test"):
        prepared_path = prepared_dir / f"{split}.jsonl"
        rows = _read_prepared(prepared_path) if prepared_path.exists() else []

        split_audio_created = 0
        split_audio_reused = 0
        split_lab_written = 0
        split_lab_reused = 0

        for row in rows:
            source_audio = dataset_root / row.audio_path
            target_audio = output_dir / row.audio_path
            target_lab = target_audio.with_suffix(".lab")

            audio_result = _materialize_audio(
                source_audio,
                target_audio,
                link_mode=args.link_mode,
                overwrite=args.overwrite,
            )
            if audio_result == "reused":
                split_audio_reused += 1
            else:
                split_audio_created += 1

            lab_result = _write_lab(target_lab, row.text, overwrite=args.overwrite)
            if lab_result == "reused":
                split_lab_reused += 1
            else:
                split_lab_written += 1

        total_rows += len(rows)
        total_audio_created += split_audio_created
        total_audio_reused += split_audio_reused
        total_lab_written += split_lab_written
        total_lab_reused += split_lab_reused

        summary["counts"][split] = {
            "utterances": len(rows),
            "audio_created": split_audio_created,
            "audio_reused": split_audio_reused,
            "lab_written": split_lab_written,
            "lab_reused": split_lab_reused,
        }
        print(
            f"prepared split={split} utterances={len(rows)} "
            f"audio_created={split_audio_created} audio_reused={split_audio_reused} "
            f"lab_written={split_lab_written} lab_reused={split_lab_reused}"
        )

    summary["totals"] = {
        "utterances": total_rows,
        "audio_created": total_audio_created,
        "audio_reused": total_audio_reused,
        "lab_written": total_lab_written,
        "lab_reused": total_lab_reused,
    }

    summary_path = output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
