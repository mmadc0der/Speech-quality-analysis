from __future__ import annotations

from typing import Final

import torch

CLASS_ORDER: Final[tuple[str, str, str]] = ("wrong_or_missed", "accented", "correct")
CLASS_TO_INDEX: Final[dict[str, int]] = {label: index for index, label in enumerate(CLASS_ORDER)}
INDEX_TO_CLASS: Final[dict[int, str]] = {index: label for label, index in CLASS_TO_INDEX.items()}
CLASS_TARGET_SCORES: Final[tuple[float, float, float]] = (15.0, 60.0, 92.0)
CLASS_TARGET_HUMAN_SCORES: Final[tuple[float, float, float]] = (0.0, 1.0, 2.0)
ACCENTED_THRESHOLD: Final[float] = (CLASS_TARGET_SCORES[0] + CLASS_TARGET_SCORES[1]) / 2.0
CORRECT_THRESHOLD: Final[float] = (CLASS_TARGET_SCORES[1] + CLASS_TARGET_SCORES[2]) / 2.0


def class_target_score_tensor(*, device: torch.device | None = None) -> torch.Tensor:
    return torch.tensor(CLASS_TARGET_SCORES, dtype=torch.float32, device=device)


def class_target_human_score_tensor(*, device: torch.device | None = None) -> torch.Tensor:
    return torch.tensor(CLASS_TARGET_HUMAN_SCORES, dtype=torch.float32, device=device)


def expected_score_from_probs(class_probs: torch.Tensor) -> torch.Tensor:
    if class_probs.size(-1) != len(CLASS_ORDER):
        raise ValueError(
            f"class_probs last dim must be {len(CLASS_ORDER)}, got {tuple(class_probs.shape)}"
        )
    target_scores = class_target_score_tensor(device=class_probs.device).to(dtype=class_probs.dtype)
    return class_probs @ target_scores


def expected_human_score_from_probs(class_probs: torch.Tensor) -> torch.Tensor:
    if class_probs.size(-1) != len(CLASS_ORDER):
        raise ValueError(
            f"class_probs last dim must be {len(CLASS_ORDER)}, got {tuple(class_probs.shape)}"
        )
    target_scores = class_target_human_score_tensor(device=class_probs.device).to(dtype=class_probs.dtype)
    return class_probs @ target_scores


def class_index_from_name(name: str) -> int:
    try:
        return CLASS_TO_INDEX[name]
    except KeyError as exc:
        raise ValueError(f"Unknown pronunciation class: {name}") from exc


def class_name_from_index(index: int) -> str:
    try:
        return INDEX_TO_CLASS[index]
    except KeyError as exc:
        raise ValueError(f"Unknown pronunciation class index: {index}") from exc


def class_index_from_target_score(target: float) -> int:
    if target >= CORRECT_THRESHOLD:
        return CLASS_TO_INDEX["correct"]
    if target >= ACCENTED_THRESHOLD:
        return CLASS_TO_INDEX["accented"]
    return CLASS_TO_INDEX["wrong_or_missed"]
