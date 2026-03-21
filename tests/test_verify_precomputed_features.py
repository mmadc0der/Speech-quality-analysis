from __future__ import annotations

import json
import sys
from pathlib import Path

from pronunciation_backend.training.schemas import PhoneEmbeddingArtifact, TrainingPhoneLabel, TrainingUtteranceArtifact
from pronunciation_backend.training.verify_precomputed_features import main as verify_precomputed_features_main


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_verify_precomputed_features_matches_aligned_counts(tmp_path: Path, monkeypatch) -> None:
    features_dir = tmp_path / "features"
    aligned_path = tmp_path / "aligned" / "test.jsonl"

    aligned = TrainingUtteranceArtifact(
        utterance_id="utt__bear__0",
        speaker_id="spk1",
        dataset="speechocean762",
        split="test",
        target_word="bear",
        canonical_phones=["B", "EH", "R"],
        ipa="ber",
        audio_path="audio.wav",
        duration_ms=600,
        audio_quality={"status": "ok", "source": "speechocean762"},
        alignment_source="mfa",
        phone_labels=[
            TrainingPhoneLabel(phoneme="B", index=0, start_ms=0, end_ms=100, pronunciation_class="correct", human_score=2.0),
            TrainingPhoneLabel(phoneme="EH", index=1, start_ms=100, end_ms=300, pronunciation_class="accented", human_score=1.0),
            TrainingPhoneLabel(phoneme="R", index=2, start_ms=300, end_ms=600, pronunciation_class="wrong_or_missed", human_score=0.0),
        ],
    )
    _write_text(aligned_path, aligned.model_dump_json() + "\n")

    rows = [
        PhoneEmbeddingArtifact(
            utterance_id="utt__bear__0",
            speaker_id="spk1",
            dataset="speechocean762",
            split="test",
            target_word="bear",
            phoneme="B",
            index=0,
            frame_count=2,
            backbone_id="facebook/hubert-base-ls960",
            embedding_source="hubert",
            mean_embedding=[0.0, 1.0],
            variance=0.1,
            duration_ms=100,
            duration_z_score=0.0,
            alignment_confidence=0.92,
            energy_mean=0.2,
            pronunciation_class="correct",
            human_score=2.0,
            regression_target=92.0,
            omission_target=0,
        ),
        PhoneEmbeddingArtifact(
            utterance_id="utt__bear__0",
            speaker_id="spk1",
            dataset="speechocean762",
            split="test",
            target_word="bear",
            phoneme="EH",
            index=1,
            frame_count=3,
            backbone_id="facebook/hubert-base-ls960",
            embedding_source="hubert",
            mean_embedding=[0.0, 1.0],
            variance=0.2,
            duration_ms=200,
            duration_z_score=0.1,
            alignment_confidence=0.92,
            energy_mean=0.2,
            pronunciation_class="accented",
            human_score=1.0,
            regression_target=60.0,
            omission_target=0,
        ),
        PhoneEmbeddingArtifact(
            utterance_id="utt__bear__0",
            speaker_id="spk1",
            dataset="speechocean762",
            split="test",
            target_word="bear",
            phoneme="R",
            index=2,
            frame_count=4,
            backbone_id="facebook/hubert-base-ls960",
            embedding_source="hubert",
            mean_embedding=[0.0, 1.0],
            variance=0.3,
            duration_ms=300,
            duration_z_score=-0.1,
            alignment_confidence=0.92,
            energy_mean=0.1,
            pronunciation_class="wrong_or_missed",
            human_score=0.0,
            regression_target=15.0,
            omission_target=1,
        ),
    ]
    _write_text(features_dir / "part-0000.jsonl", "\n".join(row.model_dump_json() for row in rows) + "\n")

    report_path = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "verify_precomputed_features",
            "--features-dir",
            str(features_dir),
            "--aligned-path",
            str(aligned_path),
            "--report-path",
            str(report_path),
        ],
    )
    assert verify_precomputed_features_main() == 0

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["comparison"]["rows_match"] is True
    assert report["comparison"]["utterance_groups_match"] is True
    assert report["feature_summary"]["rows"] == 3
    assert report["feature_summary"]["utterance_groups"] == 1
    assert report["feature_summary"]["embedding_dim_histogram"] == {"2": 3}
