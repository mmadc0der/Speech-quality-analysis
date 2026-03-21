import torch

from pronunciation_backend.training.acoustic_encoder_v2 import (
    AcousticEncoderV2,
    sample_mask_positions,
)


def test_encoder_preserves_shape_and_zeros_padding() -> None:
    model = AcousticEncoderV2(num_layers=2)
    inputs = torch.randn(2, 5, 768)
    attention_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, True],
        ]
    )

    outputs = model(inputs, attention_mask)

    assert outputs.shape == (2, 5, 384)
    assert torch.allclose(outputs[0, 3:], torch.zeros_like(outputs[0, 3:]))


def test_encoder_accepts_mask_positions() -> None:
    torch.manual_seed(0)
    model = AcousticEncoderV2(num_layers=1)
    model.eval()
    inputs = torch.randn(1, 4, 768)
    attention_mask = torch.tensor([[True, True, True, True]])
    mask_positions = torch.tensor([[False, True, False, False]])

    unmasked = model(inputs, attention_mask)
    masked = model(inputs, attention_mask, mask_positions=mask_positions)

    assert masked.shape == unmasked.shape
    assert not torch.allclose(masked[:, 1], unmasked[:, 1])


def test_encoder_uses_bias_free_linear_layers() -> None:
    model = AcousticEncoderV2(num_layers=1)

    linear_layers = [module for module in model.modules() if isinstance(module, torch.nn.Linear)]

    assert linear_layers
    assert all(layer.bias is None for layer in linear_layers)


def test_sample_mask_positions_respects_padding_and_block_size() -> None:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)
    attention_mask = torch.tensor(
        [
            [True, True, True, True, False, False],
            [True, True, True, False, False, False],
        ]
    )

    sampled = sample_mask_positions(
        attention_mask,
        mask_ratio=0.5,
        block_size=2,
        generator=generator,
    )

    assert sampled.shape == attention_mask.shape
    assert not torch.any(sampled & ~attention_mask)
    assert sampled[0].sum().item() >= 2
    assert sampled[1].sum().item() >= 2
