from __future__ import annotations

import json
from pathlib import Path

from pronunciation_backend.training.dataset import collate_word_batches
from pronunciation_backend.training.mmap_dataset import (
    WordMemmapDataset,
    pack_jsonl_split_to_mmap,
    resolve_mmap_dataset_dir,
)


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _row(utterance_id: str, phoneme: str, index: int, regression_target: float, omission_target: int) -> dict:
    return {
        "utterance_id": utterance_id,
        "speaker_id": "speaker-1",
        "dataset": "libritts",
        "split": "train",
        "target_word": "hello",
        "accent_target": "en-US",
        "phoneme": phoneme,
        "index": index,
        "prev_phoneme": None,
        "next_phoneme": None,
        "frame_count": 4,
        "backbone_id": "test-backbone",
        "embedding_source": "hubert",
        "mean_embedding": [float(index + 1)] * 768,
        "variance": 0.25 + index,
        "duration_ms": 120,
        "duration_z_score": -0.5 + index,
        "alignment_confidence": 0.9,
        "energy_mean": 0.75 + index,
        "pronunciation_class": "correct",
        "human_score": 2.0,
        "regression_target": regression_target,
        "omission_target": omission_target,
    }


def test_pack_and_load_mmap_dataset(tmp_path: Path) -> None:
    split_dir = tmp_path / "train"
    split_dir.mkdir()
    _write_rows(
        split_dir / "part-0000.jsonl",
        [
            _row("utt-1", "HH", 0, 92.0, 0),
            _row("utt-1", "AH", 1, 88.0, 0),
            _row("utt-2", "L", 0, 77.0, 1),
        ],
    )

    mmap_dir = pack_jsonl_split_to_mmap(split_dir, acoustic_dtype="float32")
    assert resolve_mmap_dataset_dir(split_dir) == mmap_dir

    dataset = WordMemmapDataset(mmap_dir)
    assert len(dataset) == 2

    first = dataset[0]
    second = dataset[1]
    assert first["acoustic_features"].shape == (2, 771)
    assert second["acoustic_features"].shape == (1, 771)
    assert first["match_targets"].tolist() == [92.0, 88.0]
    assert second["presence_targets"].tolist() == [0.0]

    batch = collate_word_batches([first, second])
    assert batch["acoustic_features"].shape == (2, 2, 771)
    assert batch["phoneme_ids"].shape == (2, 2)
    assert batch["attention_mask"].tolist() == [[True, True], [True, False]]
