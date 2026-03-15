from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from pronunciation_backend.training.cmudict_utils import arpabet_to_ipa, load_cmudict, normalize_word_token, strip_phone_stress
from pronunciation_backend.training.schemas import PreparedUtteranceArtifact, TrainingPhoneLabel, TrainingUtteranceArtifact
from pronunciation_backend.training.textgrid_utils import Interval, parse_textgrid

SKIP_WORDS = {"", "sp", "sil", "spn", "<eps>"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build LibriTTS word-level aligned artifacts from prepared manifests and MFA TextGrids.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--prepared-dir", help="Defaults to <dataset-root>/prepared")
    parser.add_argument("--output-dir", help="Defaults to <dataset-root>/aligned")
    parser.add_argument("--textgrid-root", required=True, help="Root containing MFA TextGrid outputs mirrored to audio paths.")
    parser.add_argument("--cmudict-path", required=True)
    parser.add_argument("--sample-rate", type=int, default=24_000)
    parser.add_argument("--word-tier", default="words")
    parser.add_argument("--phone-tier", default="phones")
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--max-utterances", type=int)
    parser.add_argument("--min-word-ms", type=int, default=80)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _read_prepared(path: Path, *, max_utterances: int | None) -> list[PreparedUtteranceArtifact]:
    items: list[PreparedUtteranceArtifact] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            items.append(PreparedUtteranceArtifact.model_validate_json(line))
            if max_utterances is not None and len(items) >= max_utterances:
                break
    return items


def _write_jsonl(path: Path, rows: list[TrainingUtteranceArtifact], *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing aligned manifest: {path}")
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(row.model_dump_json() + "\n")


def _resolve_textgrid(textgrid_root: Path, audio_path: str) -> Path:
    relative_audio = Path(audio_path)
    return (textgrid_root / relative_audio).with_suffix(".TextGrid")


def _select_tier(tiers: dict[str, object], preferred: str, fallbacks: tuple[str, ...]) -> list[Interval]:
    candidates = (preferred, *fallbacks)
    for name in candidates:
        tier = tiers.get(name)
        if tier is not None:
            return tier.intervals  # type: ignore[return-value]
    lowered = {key.lower(): value for key, value in tiers.items()}
    for name in candidates:
        tier = lowered.get(name.lower())
        if tier is not None:
            return tier.intervals  # type: ignore[return-value]
    raise KeyError(f"Could not find tier {preferred!r} in TextGrid.")


def _interval_ms(interval: Interval) -> tuple[int, int]:
    start_ms = int(round(interval.xmin * 1000.0))
    end_ms = int(round(interval.xmax * 1000.0))
    return start_ms, max(start_ms + 1, end_ms)


def _phones_in_word(phone_intervals: list[Interval], word_interval: Interval) -> list[Interval]:
    selected: list[Interval] = []
    for phone in phone_intervals:
        label = strip_phone_stress(phone.text)
        if label.lower() in SKIP_WORDS:
            continue
        if phone.xmin >= word_interval.xmin and phone.xmax <= word_interval.xmax:
            selected.append(phone)
    return selected


def _build_word_artifact(
    prepared: PreparedUtteranceArtifact,
    word_interval: Interval,
    phone_intervals: list[Interval],
    *,
    cmudict: dict[str, list[str]],
    sample_rate: int,
    min_word_ms: int,
) -> tuple[TrainingUtteranceArtifact | None, str | None]:
    raw_word = word_interval.text.strip()
    normalized_word = normalize_word_token(raw_word)
    if normalized_word in SKIP_WORDS or not normalized_word:
        return None, "skip_word_label"

    canonical_phones = cmudict.get(normalized_word)
    if canonical_phones is None:
        return None, "missing_cmudict"

    word_start_ms, word_end_ms = _interval_ms(word_interval)
    if word_end_ms - word_start_ms < min_word_ms:
        return None, "too_short"

    aligned_phones = _phones_in_word(phone_intervals, word_interval)
    observed_phones = [strip_phone_stress(interval.text) for interval in aligned_phones]
    if len(observed_phones) != len(canonical_phones):
        return None, "phone_count_mismatch"
    if observed_phones != canonical_phones:
        return None, "phone_identity_mismatch"

    phone_labels: list[TrainingPhoneLabel] = []
    for index, (canonical_phone, phone_interval) in enumerate(zip(canonical_phones, aligned_phones)):
        start_ms, end_ms = _interval_ms(phone_interval)
        phone_labels.append(
            TrainingPhoneLabel(
                phoneme=canonical_phone,
                index=index,
                start_ms=start_ms,
                end_ms=end_ms,
                pronunciation_class="correct",
                human_score=2.0,
                omission_label=False,
                pronounced_phone=canonical_phone,
            )
        )

    artifact = TrainingUtteranceArtifact(
        utterance_id=f"{prepared.utterance_id}__{normalized_word}__{word_start_ms}",
        speaker_id=prepared.speaker_id,
        dataset="libritts",
        split=prepared.split,
        target_word=normalized_word,
        canonical_phones=canonical_phones,
        ipa=arpabet_to_ipa(canonical_phones),
        audio_path=prepared.audio_path,
        sample_rate=sample_rate,
        duration_ms=word_end_ms - word_start_ms,
        audio_quality={"status": "ok", "source": "native_libritts"},
        alignment_source="mfa",
        phone_labels=phone_labels,
    )
    return artifact, None


def _print_progress(*, split: str, processed: int, total: int, emitted: int, started_at: float) -> None:
    elapsed = max(1e-6, time.monotonic() - started_at)
    rate = processed / elapsed
    remaining = max(0, total - processed)
    eta_seconds = int(round(remaining / rate)) if rate > 0 else 0
    print(f"split={split} progress={processed}/{total} emitted={emitted} utt_per_s={rate:.2f} eta_s={eta_seconds}")


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    prepared_dir = Path(args.prepared_dir) if args.prepared_dir else dataset_root / "prepared"
    output_dir = Path(args.output_dir) if args.output_dir else dataset_root / "aligned"
    textgrid_root = Path(args.textgrid_root)
    cmudict = load_cmudict(Path(args.cmudict_path))

    if not prepared_dir.exists():
        print(f"missing prepared dir: {prepared_dir}")
        return 1
    if not textgrid_root.exists():
        print(f"missing textgrid root: {textgrid_root}")
        return 1

    summary: dict[str, object] = {
        "dataset": "libritts",
        "prepared_dir": str(prepared_dir),
        "output_dir": str(output_dir),
        "textgrid_root": str(textgrid_root),
        "cmudict_path": args.cmudict_path,
        "counts": {},
        "skip_reasons": {},
    }

    for split in ("train", "val", "test"):
        prepared_path = prepared_dir / f"{split}.jsonl"
        rows = _read_prepared(prepared_path, max_utterances=args.max_utterances) if prepared_path.exists() else []
        emitted: list[TrainingUtteranceArtifact] = []
        skip_reasons: Counter[str] = Counter()
        started_at = time.monotonic()

        for index, prepared in enumerate(rows, start=1):
            textgrid_path = _resolve_textgrid(textgrid_root, prepared.audio_path)
            if not textgrid_path.exists():
                skip_reasons["missing_textgrid"] += 1
                if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(rows)):
                    _print_progress(split=split, processed=index, total=len(rows), emitted=len(emitted), started_at=started_at)
                continue

            try:
                tiers = parse_textgrid(textgrid_path)
                word_intervals = _select_tier(tiers, args.word_tier, ("word",))
                phone_intervals = _select_tier(tiers, args.phone_tier, ("phone",))
            except KeyError:
                skip_reasons["missing_tier"] += 1
                if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(rows)):
                    _print_progress(split=split, processed=index, total=len(rows), emitted=len(emitted), started_at=started_at)
                continue

            for word_interval in word_intervals:
                artifact, reason = _build_word_artifact(
                    prepared,
                    word_interval,
                    phone_intervals,
                    cmudict=cmudict,
                    sample_rate=args.sample_rate,
                    min_word_ms=args.min_word_ms,
                )
                if artifact is not None:
                    emitted.append(artifact)
                elif reason is not None:
                    skip_reasons[reason] += 1

            if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(rows)):
                _print_progress(split=split, processed=index, total=len(rows), emitted=len(emitted), started_at=started_at)

        target = output_dir / f"{split}.jsonl"
        _write_jsonl(target, emitted, overwrite=args.overwrite)
        summary["counts"][split] = len(emitted)  # type: ignore[index]
        summary["skip_reasons"][split] = dict(skip_reasons)  # type: ignore[index]
        print(f"wrote split={split} aligned_rows={len(emitted)} path={target}")

    summary_path = output_dir / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
