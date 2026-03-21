from __future__ import annotations

import torch

from pronunciation_backend.training.train_scorer import apply_negative_sampling


def test_apply_negative_sampling_is_deterministic_for_fixed_seed() -> None:
    acoustic_features = torch.ones((2, 3, 4), dtype=torch.float32)
    phoneme_ids = torch.tensor([[2, 3, 4], [5, 6, 7]], dtype=torch.long)
    match_targets = torch.full((2, 3), 92.0, dtype=torch.float32)
    presence_targets = torch.ones((2, 3), dtype=torch.float32)
    attention_mask = torch.tensor([[True, True, False], [True, True, True]])

    rng_a = torch.Generator(device="cpu")
    rng_a.manual_seed(123)
    first = apply_negative_sampling(
        acoustic_features,
        phoneme_ids,
        match_targets,
        presence_targets,
        attention_mask,
        prob=0.5,
        rng=rng_a,
    )

    rng_b = torch.Generator(device="cpu")
    rng_b.manual_seed(123)
    second = apply_negative_sampling(
        acoustic_features,
        phoneme_ids,
        match_targets,
        presence_targets,
        attention_mask,
        prob=0.5,
        rng=rng_b,
    )

    for left, right in zip(first, second):
        assert torch.equal(left, right)
