from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from pronunciation_backend.training.build_libritts_aligned import main as build_libritts_aligned_main
from pronunciation_backend.training.schemas import PreparedUtteranceArtifact, TrainingUtteranceArtifact


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict) -> None:
    _write_text(path, json.dumps(payload, indent=2) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def test_build_libritts_aligned_ctc_backend(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "libritts"
    prepared_dir = dataset_root / "prepared"
    output_dir = dataset_root / "aligned"
    cmudict_path = tmp_path / "cmudict"

    # Write dummy CMUdict
    _write_text(
        cmudict_path,
        "CAT  K AE T\nBEAR  B EH R\n",
    )

    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_row = PreparedUtteranceArtifact(
        utterance_id="utt-test",
        speaker_id="19",
        dataset="libritts",
        split="test",
        text="CAT BEAR",
        normalized_text="cat bear",
        audio_path="dev-clean/19/198/utt-test.wav",
        transcript_path=None,
    )
    _write_text(prepared_dir / "test.jsonl", prepared_row.model_dump_json() + "\n")

    audio_file = dataset_root / prepared_row.audio_path
    audio_file.parent.mkdir(parents=True, exist_ok=True)
    audio_file.write_bytes(b"dummy wav bytes")

    import torch
    class MockCtcForcedAligner:
        def __init__(self, model_id, device):
            pass
        def align_audio(self, samples, sample_rate, arpabet_phones):
            from pronunciation_backend.services.ctc_aligner import AlignedPhone
            return [
                AlignedPhone(phone="K", start_ms=10, end_ms=100, score=0.9),
                AlignedPhone(phone="AE", start_ms=100, end_ms=200, score=0.8),
                AlignedPhone(phone="T", start_ms=200, end_ms=300, score=0.7),
                AlignedPhone(phone="B", start_ms=300, end_ms=400, score=0.95),
                AlignedPhone(phone="EH", start_ms=400, end_ms=500, score=0.85),
                AlignedPhone(phone="R", start_ms=500, end_ms=600, score=0.75),
            ]

    import torchaudio
    monkeypatch.setattr(torchaudio, "load", lambda path: (torch.zeros(1, 16000), 16000))
    monkeypatch.setattr("pronunciation_backend.services.ctc_aligner.CtcForcedAligner", MockCtcForcedAligner)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_libritts_aligned",
            "--dataset-root",
            str(dataset_root),
            "--prepared-dir",
            str(prepared_dir),
            "--output-dir",
            str(output_dir),
            "--cmudict-path",
            str(cmudict_path),
            "--alignment-backend",
            "ctc",
            "--overwrite",
        ],
    )

    assert build_libritts_aligned_main() == 0

    aligned_file = output_dir / "test.jsonl"
    assert aligned_file.exists()
    rows = _read_jsonl(aligned_file)
    # Since we aligned 2 words, we should get 2 word-level TrainingUtteranceArtifact rows!
    assert len(rows) == 2
    
    artifact1 = TrainingUtteranceArtifact.model_validate(rows[0])
    assert artifact1.target_word == "cat"
    assert artifact1.alignment_source == "custom_ctc"
    assert len(artifact1.phone_labels) == 3
    assert [label.phoneme for label in artifact1.phone_labels] == ["K", "AE", "T"]
    assert [label.start_ms for label in artifact1.phone_labels] == [10, 100, 200]
    assert [label.end_ms for label in artifact1.phone_labels] == [100, 200, 300]

    artifact2 = TrainingUtteranceArtifact.model_validate(rows[1])
    assert artifact2.target_word == "bear"
    assert artifact2.alignment_source == "custom_ctc"
    assert len(artifact2.phone_labels) == 3
    assert [label.phoneme for label in artifact2.phone_labels] == ["B", "EH", "R"]
    assert [label.start_ms for label in artifact2.phone_labels] == [300, 400, 500]
    assert [label.end_ms for label in artifact2.phone_labels] == [400, 500, 600]
