import torch

from pronunciation_backend.training.acoustic_encoder_v2 import AcousticEncoderV2
from pronunciation_backend.training.scorer_model_v2 import PhonemeScorerModelV2
from pronunciation_backend.training.scoring_targets import (
    CLASS_TARGET_HUMAN_SCORES,
    CLASS_TARGET_SCORES,
    expected_human_score_from_probs,
    expected_score_from_probs,
)


def test_expected_score_from_probs_matches_dataset_targets() -> None:
    class_probs = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.25, 0.50, 0.25],
        ],
        dtype=torch.float32,
    )

    expected_score = expected_score_from_probs(class_probs)
    expected_human = expected_human_score_from_probs(class_probs)

    assert torch.allclose(expected_score[:3], torch.tensor(CLASS_TARGET_SCORES))
    assert torch.allclose(expected_human[:3], torch.tensor(CLASS_TARGET_HUMAN_SCORES))
    assert torch.isclose(expected_score[3], torch.tensor(56.75))
    assert torch.isclose(expected_human[3], torch.tensor(1.0))


def test_scorer_model_v2_forward_shapes() -> None:
    model = PhonemeScorerModelV2(acoustic_layers=2, scorer_layers=1)
    acoustic_embeddings = torch.randn(2, 5, 768)
    phoneme_ids = torch.tensor(
        [
            [2, 3, 4, 0, 0],
            [5, 6, 7, 8, 9],
        ],
        dtype=torch.long,
    )
    attention_mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, True],
        ]
    )

    outputs = model(acoustic_embeddings, phoneme_ids, attention_mask)

    assert outputs["quality_logits"].shape == (2, 5, 3)
    assert outputs["omission_logit"].shape == (2, 5)
    assert outputs["class_probs"].shape == (2, 5, 3)
    assert outputs["expected_score"].shape == (2, 5)
    assert outputs["expected_human_score"].shape == (2, 5)
    assert torch.allclose(outputs["expected_score"][0, 3:], torch.zeros_like(outputs["expected_score"][0, 3:]))


def test_scorer_model_v2_can_load_pretrained_encoder_weights() -> None:
    pretrained_encoder = AcousticEncoderV2(num_layers=1)
    checkpoint = {
        "model_state_dict": {
            f"encoder.{key}": value.clone()
            for key, value in pretrained_encoder.state_dict().items()
        }
    }
    model = PhonemeScorerModelV2(acoustic_layers=1, scorer_layers=1)

    model.load_pretrained_acoustic_encoder(checkpoint)

    loaded_state = model.acoustic_encoder.state_dict()
    for key, value in pretrained_encoder.state_dict().items():
        assert torch.equal(loaded_state[key], value)


def test_scorer_model_v2_can_freeze_and_unfreeze_encoder() -> None:
    model = PhonemeScorerModelV2(acoustic_layers=1, scorer_layers=1)

    model.set_acoustic_encoder_trainable(False)
    assert all(not param.requires_grad for param in model.acoustic_encoder.parameters())

    model.set_acoustic_encoder_trainable(True)
    assert all(param.requires_grad for param in model.acoustic_encoder.parameters())
