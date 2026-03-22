import torch

from pronunciation_backend.training.scoring_targets import (
    ACCENTED_THRESHOLD,
    CORRECT_THRESHOLD,
    class_index_from_target_score,
    class_name_from_index,
)


def test_class_index_from_target_score_uses_dataset_aligned_thresholds() -> None:
    assert class_index_from_target_score(15.0) == 0
    assert class_index_from_target_score(ACCENTED_THRESHOLD - 1e-6) == 0
    assert class_index_from_target_score(ACCENTED_THRESHOLD) == 1
    assert class_index_from_target_score(CORRECT_THRESHOLD - 1e-6) == 1
    assert class_index_from_target_score(CORRECT_THRESHOLD) == 2


def test_class_name_from_index_round_trips() -> None:
    assert class_name_from_index(0) == "wrong_or_missed"
    assert class_name_from_index(1) == "accented"
    assert class_name_from_index(2) == "correct"
