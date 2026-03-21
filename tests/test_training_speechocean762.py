from __future__ import annotations

import json
import sys
from pathlib import Path

from pronunciation_backend.training.build_speechocean762_aligned import main as build_speechocean762_aligned_main
from pronunciation_backend.training.prepare_speechocean762_mfa import main as prepare_speechocean762_mfa_main
from pronunciation_backend.training.prepare_speechocean762 import main as prepare_speechocean762_main
from pronunciation_backend.training.schemas import PreparedUtteranceArtifact, TrainingUtteranceArtifact
from pronunciation_backend.training.speechocean_utils import resolve_speechocean_raw_root


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


def test_prepare_speechocean762_creates_speaker_disjoint_val_split(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "speechocean762"
    raw_root = dataset_root / "raw" / "speechocean762"

    _write_json(
        raw_root / "resource" / "scores.json",
        {
            "utt-a": {"text": "WE CALL", "words": []},
            "utt-b": {"text": "IT BEAR", "words": []},
            "utt-c": {"text": "MARK", "words": []},
        },
    )
    _write_text(raw_root / "train" / "utt2spk", "utt-a spk-a\nutt-b spk-b\n")
    _write_text(raw_root / "test" / "utt2spk", "utt-c spk-c\n")
    _write_text(raw_root / "train" / "wav.scp", "utt-a WAVE/SPEAKERA/utt-a.WAV\nutt-b WAVE/SPEAKERB/utt-b.WAV\n")
    _write_text(raw_root / "test" / "wav.scp", "utt-c WAVE/SPEAKERC/utt-c.WAV\n")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_speechocean762",
            "--dataset-root",
            str(dataset_root),
            "--val-speaker-fraction",
            "0.5",
            "--split-seed",
            "7",
            "--overwrite",
        ],
    )
    assert prepare_speechocean762_main() == 0

    train_rows = [PreparedUtteranceArtifact.model_validate(row) for row in _read_jsonl(dataset_root / "prepared" / "train.jsonl")]
    val_rows = [PreparedUtteranceArtifact.model_validate(row) for row in _read_jsonl(dataset_root / "prepared" / "val.jsonl")]
    test_rows = [PreparedUtteranceArtifact.model_validate(row) for row in _read_jsonl(dataset_root / "prepared" / "test.jsonl")]

    assert len(train_rows) == 1
    assert len(val_rows) == 1
    assert len(test_rows) == 1
    assert {row.speaker_id for row in train_rows}.isdisjoint({row.speaker_id for row in val_rows})
    assert {row.speaker_id for row in train_rows} | {row.speaker_id for row in val_rows} == {"spk-a", "spk-b"}
    assert test_rows[0].speaker_id == "spk-c"
    assert train_rows[0].audio_path.startswith("raw/speechocean762/WAVE/")


def test_resolve_speechocean_raw_root_finds_deeply_nested_unpacked_root(tmp_path: Path) -> None:
    dataset_root = tmp_path / "speechocean762"
    raw_root = dataset_root / "unpacked" / "speechocean762" / "speechocean762"
    _write_json(raw_root / "resource" / "scores.json", {})
    _write_text(raw_root / "train" / "wav.scp", "")
    _write_text(raw_root / "test" / "wav.scp", "")

    assert resolve_speechocean_raw_root(dataset_root) == raw_root


def test_build_speechocean762_aligned_emits_phone_labels_from_scores(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "speechocean762"
    raw_root = dataset_root / "raw" / "speechocean762"
    prepared_dir = dataset_root / "prepared"
    textgrid_root = dataset_root / "textgrids"

    _write_json(
        raw_root / "resource" / "scores.json",
        {
            "utt-test": {
                "text": "BEAR",
                "accuracy": 6,
                "words": [
                    {
                        "text": "BEAR",
                        "phones": "B EH0 R",
                        "phones-accuracy": [2.0, 1.0, 0.2],
                        "mispronunciations": [
                            {
                                "canonical-phone": "R",
                                "index": 2,
                                "pronounced-phone": "<unk>",
                            }
                        ],
                    }
                ],
            }
        },
    )
    _write_text(raw_root / "train" / "wav.scp", "")
    _write_text(raw_root / "test" / "wav.scp", "utt-test WAVE/SPEAKER0003/utt-test.WAV\n")

    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_row = PreparedUtteranceArtifact(
        utterance_id="utt-test",
        speaker_id="0003",
        dataset="speechocean762",
        split="test",
        text="BEAR",
        normalized_text="bear",
        audio_path="raw/speechocean762/WAVE/SPEAKER0003/utt-test.WAV",
        transcript_path=None,
    )
    _write_text(prepared_dir / "test.jsonl", prepared_row.model_dump_json() + "\n")

    _write_text(
        textgrid_root / "raw" / "speechocean762" / "WAVE" / "SPEAKER0003" / "utt-test.TextGrid",
        """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 0.60
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 0.60
        intervals: size = 1
        intervals [1]:
            xmin = 0.00
            xmax = 0.60
            text = "BEAR"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 0.60
        intervals: size = 3
        intervals [1]:
            xmin = 0.00
            xmax = 0.20
            text = "B"
        intervals [2]:
            xmin = 0.20
            xmax = 0.40
            text = "EH0"
        intervals [3]:
            xmin = 0.40
            xmax = 0.60
            text = "R"
""",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_speechocean762_aligned",
            "--dataset-root",
            str(dataset_root),
            "--textgrid-root",
            str(textgrid_root),
            "--overwrite",
        ],
    )
    assert build_speechocean762_aligned_main() == 0

    rows = [TrainingUtteranceArtifact.model_validate(row) for row in _read_jsonl(dataset_root / "aligned" / "test.jsonl")]
    assert len(rows) == 1

    artifact = rows[0]
    assert artifact.target_word == "bear"
    assert artifact.canonical_phones == ["B", "EH", "R"]
    assert [label.pronunciation_class for label in artifact.phone_labels] == [
        "correct",
        "accented",
        "wrong_or_missed",
    ]
    assert [label.human_score for label in artifact.phone_labels] == [2.0, 1.0, 0.2]
    assert artifact.phone_labels[2].omission_label is True
    assert artifact.phone_labels[2].pronounced_phone == "<unk>"


def test_prepare_speechocean762_mfa_materializes_mirrored_audio_and_lab(tmp_path: Path, monkeypatch) -> None:
    dataset_root = tmp_path / "speechocean762"
    prepared_dir = dataset_root / "prepared"
    source_audio = dataset_root / "unpacked" / "speechocean762" / "WAVE" / "SPEAKER0001" / "utt-a.WAV"
    output_dir = dataset_root / "mfa_corpus"

    _write_text(source_audio, "fake wav bytes")
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_row = PreparedUtteranceArtifact(
        utterance_id="utt-a",
        speaker_id="0001",
        dataset="speechocean762",
        split="train",
        text="WE CALL IT BEAR",
        normalized_text="we call it bear",
        audio_path="unpacked/speechocean762/WAVE/SPEAKER0001/utt-a.WAV",
        transcript_path=None,
    )
    _write_text(prepared_dir / "train.jsonl", prepared_row.model_dump_json() + "\n")
    _write_text(prepared_dir / "val.jsonl", "")
    _write_text(prepared_dir / "test.jsonl", "")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_speechocean762_mfa",
            "--dataset-root",
            str(dataset_root),
            "--output-dir",
            str(output_dir),
            "--link-mode",
            "copy",
            "--overwrite",
        ],
    )
    assert prepare_speechocean762_mfa_main() == 0

    mirrored_audio = output_dir / "unpacked" / "speechocean762" / "WAVE" / "SPEAKER0001" / "utt-a.WAV"
    mirrored_lab = mirrored_audio.with_suffix(".lab")
    assert mirrored_audio.exists()
    assert mirrored_audio.read_text(encoding="utf-8") == "fake wav bytes"
    assert mirrored_lab.read_text(encoding="utf-8") == "WE CALL IT BEAR\n"
