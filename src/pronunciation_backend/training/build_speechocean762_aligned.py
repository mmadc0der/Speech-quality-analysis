from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

from pronunciation_backend.training.cmudict_utils import arpabet_to_ipa, normalize_word_token
from pronunciation_backend.training.schemas import PreparedUtteranceArtifact, TrainingPhoneLabel, TrainingUtteranceArtifact
from pronunciation_backend.training.speechocean_utils import (
    canonical_phones_from_word,
    is_omission_pronunciation,
    load_scores,
    mispronunciations_by_index,
    normalize_score_word_text,
    phone_scores_from_word,
    pronunciation_class_from_score,
    resolve_speechocean_raw_root,
    resolve_scores_path,
)
from pronunciation_backend.training.textgrid_utils import Interval, parse_textgrid

SKIP_WORDS = {"", "sp", "sil", "spn", "<eps>"}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build SpeechOcean762 word-level aligned artifacts from prepared manifests, scores, and MFA TextGrids."
    )
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--prepared-dir", help="Defaults to <dataset-root>/prepared")
    parser.add_argument("--output-dir", help="Defaults to <dataset-root>/aligned")
    parser.add_argument("--textgrid-root", required=True, help="Root containing MFA TextGrid outputs mirrored to audio paths.")
    parser.add_argument("--sample-rate", type=int, default=16_000)
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
    return (textgrid_root / Path(audio_path)).with_suffix(".TextGrid")


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
        label = normalize_word_token(phone.text)
        if label in SKIP_WORDS:
            continue
        if phone.xmin >= word_interval.xmin and phone.xmax <= word_interval.xmax:
            selected.append(phone)
    return selected


def _scored_words(entry: dict[str, Any]) -> list[dict[str, Any]]:
    words = entry.get("words")
    if not isinstance(words, list):
        raise ValueError(f"Expected score entry words list: {entry}")
    return [word for word in words if normalize_score_word_text(str(word.get("text", "")))]


def _textgrid_words(word_intervals: list[Interval]) -> list[Interval]:
    return [
        interval
        for interval in word_intervals
        if normalize_score_word_text(interval.text) not in SKIP_WORDS
    ]


def _build_word_artifact(
    prepared: PreparedUtteranceArtifact,
    score_word: dict[str, Any],
    word_interval: Interval,
    phone_intervals: list[Interval],
    *,
    sample_rate: int,
    min_word_ms: int,
) -> tuple[TrainingUtteranceArtifact | None, str | None]:
    normalized_word = normalize_score_word_text(str(score_word["text"]))
    if normalized_word in SKIP_WORDS or not normalized_word:
        return None, "skip_word_label"

    canonical_phones = canonical_phones_from_word(score_word)
    phone_scores = phone_scores_from_word(score_word)
    if len(canonical_phones) != len(phone_scores):
        return None, "phone_score_length_mismatch"

    word_start_ms, word_end_ms = _interval_ms(word_interval)
    if word_end_ms - word_start_ms < min_word_ms:
        return None, "too_short"

    aligned_phones = _phones_in_word(phone_intervals, word_interval)
    if len(aligned_phones) != len(canonical_phones):
        return None, "phone_count_mismatch"

    mispronunciations = mispronunciations_by_index(score_word)
    phone_labels: list[TrainingPhoneLabel] = []
    for index, (canonical_phone, phone_interval, human_score) in enumerate(
        zip(canonical_phones, aligned_phones, phone_scores)
    ):
        start_ms, end_ms = _interval_ms(phone_interval)
        pronounced_phone = mispronunciations.get(index)
        phone_labels.append(
            TrainingPhoneLabel(
                phoneme=canonical_phone,
                index=index,
                start_ms=start_ms,
                end_ms=end_ms,
                pronunciation_class=pronunciation_class_from_score(human_score),  # type: ignore[arg-type]
                human_score=human_score,
                omission_label=is_omission_pronunciation(pronounced_phone),
                pronounced_phone=pronounced_phone,
            )
        )

    artifact = TrainingUtteranceArtifact(
        utterance_id=f"{prepared.utterance_id}__{normalized_word}__{word_start_ms}",
        speaker_id=prepared.speaker_id,
        dataset="speechocean762",
        split=prepared.split,
        target_word=normalized_word,
        canonical_phones=canonical_phones,
        ipa=arpabet_to_ipa(canonical_phones),
        audio_path=prepared.audio_path,
        sample_rate=sample_rate,
        duration_ms=word_end_ms - word_start_ms,
        audio_quality={"status": "ok", "source": "speechocean762"},
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
    raw_root = resolve_speechocean_raw_root(dataset_root)
    scores = load_scores(resolve_scores_path(raw_root))

    if not prepared_dir.exists():
        print(f"missing prepared dir: {prepared_dir}")
        return 1
    if not textgrid_root.exists():
        print(f"missing textgrid root: {textgrid_root}")
        return 1

    summary: dict[str, object] = {
        "dataset": "speechocean762",
        "prepared_dir": str(prepared_dir),
        "output_dir": str(output_dir),
        "textgrid_root": str(textgrid_root),
        "raw_root": str(raw_root),
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
            score_entry = scores.get(prepared.utterance_id)
            if score_entry is None:
                skip_reasons["missing_score_entry"] += 1
                continue

            textgrid_path = _resolve_textgrid(textgrid_root, prepared.audio_path)
            if not textgrid_path.exists():
                skip_reasons["missing_textgrid"] += 1
                if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(rows)):
                    _print_progress(split=split, processed=index, total=len(rows), emitted=len(emitted), started_at=started_at)
                continue

            try:
                tiers = parse_textgrid(textgrid_path)
                word_intervals = _textgrid_words(_select_tier(tiers, args.word_tier, ("word",)))
                phone_intervals = _select_tier(tiers, args.phone_tier, ("phone",))
            except KeyError:
                skip_reasons["missing_tier"] += 1
                if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(rows)):
                    _print_progress(split=split, processed=index, total=len(rows), emitted=len(emitted), started_at=started_at)
                continue

            score_words = _scored_words(score_entry)
            if len(score_words) != len(word_intervals):
                skip_reasons["word_count_mismatch"] += 1
                if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(rows)):
                    _print_progress(split=split, processed=index, total=len(rows), emitted=len(emitted), started_at=started_at)
                continue

            normalized_score_words = [normalize_score_word_text(str(word["text"])) for word in score_words]
            normalized_intervals = [normalize_score_word_text(interval.text) for interval in word_intervals]
            if normalized_score_words != normalized_intervals:
                skip_reasons["word_identity_mismatch"] += 1
                if args.progress_every > 0 and (index % args.progress_every == 0 or index == len(rows)):
                    _print_progress(split=split, processed=index, total=len(rows), emitted=len(emitted), started_at=started_at)
                continue

            for score_word, word_interval in zip(score_words, word_intervals):
                artifact, reason = _build_word_artifact(
                    prepared,
                    score_word,
                    word_interval,
                    phone_intervals,
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
