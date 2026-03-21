import torch

from pronunciation_backend.training.pretrain_acoustic_encoder_v2 import (
    AcousticEncoderPretrainModel,
    _masked_reconstruction_loss,
    _partition_muon_param_groups,
)


def test_partition_muon_param_groups_covers_all_trainable_params() -> None:
    model = AcousticEncoderPretrainModel(num_layers=1)

    param_groups = _partition_muon_param_groups(model)
    grouped_ids = {
        id(param)
        for group in param_groups
        for param in group["params"]  # type: ignore[index]
    }
    all_ids = {id(param) for param in model.parameters() if param.requires_grad}

    assert len(param_groups) == 2
    assert grouped_ids == all_ids


def test_partition_muon_param_groups_routes_encoder_matrices_to_muon_group() -> None:
    model = AcousticEncoderPretrainModel(num_layers=1)

    muon_group, aux_group = _partition_muon_param_groups(model)

    assert all(param.ndim >= 2 for param in muon_group["params"])  # type: ignore[index]
    assert any(param.ndim < 2 for param in aux_group["params"])  # type: ignore[index]


def test_masked_reconstruction_loss_uses_only_masked_positions() -> None:
    prediction = torch.tensor(
        [
            [[1.0, 1.0], [10.0, 10.0]],
            [[2.0, 2.0], [20.0, 20.0]],
        ]
    )
    target = torch.tensor(
        [
            [[1.0, 1.0], [4.0, 4.0]],
            [[2.0, 2.0], [8.0, 8.0]],
        ]
    )
    mask_positions = torch.tensor(
        [
            [False, True],
            [False, False],
        ]
    )

    loss = _masked_reconstruction_loss(
        prediction,
        target,
        mask_positions=mask_positions,
    )

    assert loss.item() == 36.0
