from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import replace
from pathlib import Path

from pronunciation_backend.config import settings
from pronunciation_backend.models import EncodedFrames, PhoneSpan, PreparedAudio
from pronunciation_backend.services.aligner import PhoneFeatureBuilder, phone_duration_weight
from pronunciation_backend.services.audio_prep import AudioPrepService
from pronunciation_backend.services.feature_encoder import SSLFeatureEncoder
from pronunciation_backend.training.feature_store import (
    FeaturePrecomputeSpec,
    FeatureStoreLayout,
    FeatureStoreState,
    plan_feature_store,
    verify_feature_store,
)
from pronunciation_backend.training.schemas import PhoneEmbeddingArtifact, TrainingPhoneLabel, TrainingUtteranceArtifact


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Precompute phone-level training features into the hashed feature store.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--backbone-id", required=True)
    parser.add_argument("--backbone-revision", default="main")
    parser.add_argument("--adapter-id")
    parser.add_argument("--embedding-source", required=True, choices=["hubert", "wav2vec2", "fallback"])
    parser.add_argument("--alignment-source", default="mfa", choices=["mfa", "custom_ctc", "manual"])
    parser.add_argument("--pooling-version", default="phone_mean_v1")
    parser.add_argument("--artifact-schema-version", default="phone_embedding_artifact_v1")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--device", default=settings.device)
    parser.add_argument("--aligned-dir", default="aligned")
    parser.add_argument("--shard-size", type=int, default=2_000)
    parser.add_argument("--max-utterances", type=int)
    parser.add_argument("--min-audio-ms", type=int, default=100)
    parser.add_argument("--max-audio-ms", type=int, default=30_000)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--auto-plan", action="store_true", default=True)
    parser.add_argument("--no-auto-plan", action="store_false", dest="auto_plan")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _spec_from_args(args: argparse.Namespace) -> FeaturePrecomputeSpec:
    return FeaturePrecomputeSpec(
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        splits=args.splits,
        backbone_id=args.backbone_id,
        backbone_revision=args.backbone_revision,
        adapter_id=args.adapter_id,
        embedding_source=args.embedding_source,
        alignment_source=args.alignment_source,
        pooling_version=args.pooling_version,
        artifact_schema_version=args.artifact_schema_version,
        sample_rate=args.sample_rate,
    )


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_artifacts(path: Path, *, max_utterances: int | None) -> list[TrainingUtteranceArtifact]:
    if not path.exists():
        raise FileNotFoundError(f"Aligned artifact file not found: {path}")

    artifacts: list[TrainingUtteranceArtifact] = []
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            artifacts.append(TrainingUtteranceArtifact.model_validate_json(line))
            if max_utterances is not None and len(artifacts) >= max_utterances:
                break
    return artifacts


def _resolve_audio_path(dataset_root: Path, audio_path: str) -> Path:
    candidate = Path(audio_path)
    if candidate.is_absolute():
        return candidate
    return dataset_root / candidate


def _load_audio(audio_prep: AudioPrepService, audio_path: Path) -> PreparedAudio:
    return audio_prep.decode(audio_path.read_bytes())


def _alignment_confidence(source: str) -> float:
    if source == "manual":
        return 0.98
    if source == "mfa":
        return 0.92
    return 0.85


def _spans_from_labels(labels: list[TrainingPhoneLabel], encoded: EncodedFrames, phones: list[str], alignment_source: str) -> list[PhoneSpan]:
    frame_count = max(1, len(encoded.embeddings))
    frame_ms = max(encoded.frame_ms, 1e-6)
    expected_weights = [phone_duration_weight(phone) for phone in phones]
    expected_total = max(sum(expected_weights), 1e-6)

    spans: list[PhoneSpan] = []
    for index, label in enumerate(labels):
        start_frame = max(0, min(frame_count - 1, int(math.floor(label.start_ms / frame_ms))))
        end_frame = max(start_frame + 1, int(math.ceil(label.end_ms / frame_ms)))
        end_frame = min(frame_count, end_frame)
        observed_frames = max(1, end_frame - start_frame)
        expected_frames = max(1.0, frame_count * (expected_weights[index] / expected_total))
        duration_z = (observed_frames - expected_frames) / max(1.0, expected_frames * 0.35)
        spans.append(
            PhoneSpan(
                phoneme=label.phoneme,
                start_frame=start_frame,
                end_frame=end_frame,
                start_ms=label.start_ms,
                end_ms=max(label.start_ms + 1, label.end_ms),
                alignment_confidence=_alignment_confidence(alignment_source),
                duration_z_score=round(float(duration_z), 3),
            )
        )

    return spans


def _regression_target_from_human_score(human_score: float) -> float:
    mapping = {0.0: 15.0, 1.0: 60.0, 2.0: 92.0}
    rounded = float(round(human_score))
    if rounded in mapping and abs(human_score - rounded) < 1e-6:
        return mapping[rounded]
    return max(0.0, min(100.0, 15.0 + (human_score / 2.0) * 77.0))


def _artifact_rows(
    utterance: TrainingUtteranceArtifact,
    phone_features: list,
    spans: list[PhoneSpan],
    *,
    backbone_id: str,
    embedding_source: str,
) -> list[PhoneEmbeddingArtifact]:
    rows: list[PhoneEmbeddingArtifact] = []
    labels = utterance.phone_labels
    if not (len(phone_features) == len(labels) == len(spans)):
        raise ValueError(f"Feature/label/span length mismatch for utterance {utterance.utterance_id}")

    for index, (features, label, span) in enumerate(zip(phone_features, labels, spans)):
        prev_phoneme = labels[index - 1].phoneme if index > 0 else None
        next_phoneme = labels[index + 1].phoneme if index + 1 < len(labels) else None
        rows.append(
            PhoneEmbeddingArtifact(
                utterance_id=utterance.utterance_id,
                speaker_id=utterance.speaker_id,
                dataset=utterance.dataset,
                split=utterance.split,
                target_word=utterance.target_word,
                accent_target=utterance.accent_target,
                phoneme=label.phoneme,
                index=label.index,
                prev_phoneme=prev_phoneme,
                next_phoneme=next_phoneme,
                frame_count=max(1, span.end_frame - span.start_frame),
                backbone_id=backbone_id,
                embedding_source=embedding_source,
                mean_embedding=features.mean_embedding,
                variance=features.variance,
                duration_ms=features.duration_ms,
                duration_z_score=features.duration_z_score,
                alignment_confidence=features.alignment_confidence,
                energy_mean=features.energy_mean,
                pronunciation_class=label.pronunciation_class,
                human_score=label.human_score,
                regression_target=_regression_target_from_human_score(label.human_score),
                omission_target=int(label.omission_label),
            )
        )
    return rows


def _write_jsonl_shards(split_dir: Path, rows: list[PhoneEmbeddingArtifact], *, shard_size: int, overwrite: bool) -> int:
    split_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for existing in split_dir.glob("part-*.jsonl"):
            existing.unlink()
    elif any(split_dir.glob("part-*.jsonl")):
        raise FileExistsError(f"Refusing to overwrite existing shard files in {split_dir}. Use --overwrite to replace them.")

    shard_index = 0
    written_rows = 0
    for start in range(0, len(rows), shard_size):
        shard_rows = rows[start : start + shard_size]
        target = split_dir / f"part-{shard_index:04d}.jsonl"
        with target.open("w", encoding="utf-8") as handle:
            for row in shard_rows:
                handle.write(row.model_dump_json() + "\n")
        shard_index += 1
        written_rows += len(shard_rows)
    return written_rows


def _print_progress(
    *,
    split: str,
    processed: int,
    total: int,
    split_rows: int,
    started_at: float,
) -> None:
    elapsed = max(1e-6, time.monotonic() - started_at)
    rate = processed / elapsed
    remaining = max(0, total - processed)
    eta_seconds = int(round(remaining / rate)) if rate > 0 else 0
    print(
        f"split={split} progress={processed}/{total} "
        f"rows={split_rows} utt_per_s={rate:.2f} eta_s={eta_seconds}"
    )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    spec = _spec_from_args(args)
    if args.auto_plan:
        feature_key, feature_paths = plan_feature_store(spec, create=True, active_settings=settings)
    else:
        layout = FeatureStoreLayout(settings)
        feature_key = layout.compute_feature_key(spec)
        feature_paths = layout.expected_manifest_paths(spec.dataset, feature_key)

    ok, messages = verify_feature_store(spec, active_settings=settings)
    if not ok:
        for message in messages:
            print(message)
        print("Feature store verification failed.")
        return 1

    dataset_root = Path(spec.dataset_root)
    state_path = feature_paths["state"]

    model_settings = replace(
        settings,
        sample_rate=spec.sample_rate,
        min_audio_ms=args.min_audio_ms,
        max_audio_ms=args.max_audio_ms,
        use_hf_encoder=spec.embedding_source != "fallback",
        backbone_id=spec.backbone_id,
        device=args.device,
    )
    audio_prep = AudioPrepService(model_settings)
    encoder = SSLFeatureEncoder(model_settings)
    feature_builder = PhoneFeatureBuilder()

    state_payload = _load_json(state_path)
    state = FeatureStoreState.model_validate(state_payload)
    state.status = "running"
    state.output_format = "jsonl"
    _write_json(state_path, state.model_dump(mode="json"))

    aligned_dir = dataset_root / args.aligned_dir
    total_rows = 0
    total_utterances = 0
    try:
        for split in spec.splits:
            artifact_path = aligned_dir / f"{split}.jsonl"
            utterances = _read_artifacts(artifact_path, max_utterances=args.max_utterances)
            split_rows: list[PhoneEmbeddingArtifact] = []
            split_started_at = time.monotonic()
            total_utterances_in_split = len(utterances)
            for index, utterance in enumerate(utterances, start=1):
                if utterance.split != split:
                    raise ValueError(f"Utterance {utterance.utterance_id} declares split={utterance.split}, expected {split}")
                audio_path = _resolve_audio_path(dataset_root, utterance.audio_path)
                prepared = _load_audio(audio_prep, audio_path)
                encoded = encoder.encode(prepared)
                spans = _spans_from_labels(utterance.phone_labels, encoded, utterance.canonical_phones, spec.alignment_source)
                phone_features = feature_builder.build(encoded, spans)
                split_rows.extend(
                    _artifact_rows(
                        utterance,
                        phone_features,
                        spans,
                        backbone_id=spec.backbone_id,
                        embedding_source=spec.embedding_source,
                    )
                )
                if args.progress_every > 0 and (index % args.progress_every == 0 or index == total_utterances_in_split):
                    _print_progress(
                        split=split,
                        processed=index,
                        total=total_utterances_in_split,
                        split_rows=len(split_rows),
                        started_at=split_started_at,
                    )

            split_dir = feature_paths["split_root"] / split
            written_rows = _write_jsonl_shards(split_dir, split_rows, shard_size=args.shard_size, overwrite=args.overwrite)
            state.split_counts[split] = written_rows
            state.utterance_counts[split] = len(utterances)
            _write_json(state_path, state.model_dump(mode="json"))

            total_rows += written_rows
            total_utterances += len(utterances)
            print(f"split={split} utterances={len(utterances)} rows={written_rows} output_dir={split_dir}")
    except FileNotFoundError as exc:
        print(str(exc))
        print("Expected aligned artifacts under: <dataset-root>/aligned/<split>.jsonl")
        print("Generate aligned training artifacts before feature precompute.")
        return 1
    except FileExistsError as exc:
        print(str(exc))
        return 1

    state.status = "complete"
    _write_json(state_path, state.model_dump(mode="json"))
    print(f"completed dataset={spec.dataset} feature_key={feature_key} utterances={total_utterances} rows={total_rows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
