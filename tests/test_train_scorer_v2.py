import torch

from pronunciation_backend.training.train_scorer_v2 import _move_batch_to_device


def test_move_batch_to_device_builds_v2_targets() -> None:
    batch = {
        "acoustic_features": torch.randn(1, 3, 771),
        "phoneme_ids": torch.tensor([[2, 3, 4]], dtype=torch.long),
        "match_targets": torch.tensor([[15.0, 60.0, 92.0]], dtype=torch.float32),
        "duration_targets": torch.tensor([[15.0, 60.0, 92.0]], dtype=torch.float32),
        "presence_targets": torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32),
        "attention_mask": torch.tensor([[True, True, True]]),
    }

    moved = _move_batch_to_device(batch, torch.device("cpu"))

    assert moved["acoustic_embeddings"].shape == (1, 3, 768)
    assert torch.equal(moved["class_targets"], torch.tensor([[0, 1, 2]], dtype=torch.long))
    assert torch.equal(moved["omission_targets"], torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32))
