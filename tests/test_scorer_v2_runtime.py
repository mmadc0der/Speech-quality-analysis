from __future__ import annotations

from pathlib import Path

import torch

from pronunciation_backend.models import PhoneFeatures
from pronunciation_backend.services.scorer_v2_runtime import ScorerV2Runtime
from pronunciation_backend.training.scorer_model_v2 import PhonemeScorerModelV2


def _phone_features() -> list[PhoneFeatures]:
    base_embedding = [0.01] * 768
    return [
        PhoneFeatures(
            phoneme="TH",
            start_ms=0,
            end_ms=100,
            mean_embedding=base_embedding,
            variance=0.1,
            duration_ms=100,
            duration_z_score=0.0,
            alignment_confidence=0.92,
            energy_mean=0.12,
            starts_late=False,
        ),
        PhoneFeatures(
            phoneme="AO",
            start_ms=100,
            end_ms=240,
            mean_embedding=[0.02] * 768,
            variance=0.08,
            duration_ms=140,
            duration_z_score=0.1,
            alignment_confidence=0.95,
            energy_mean=0.14,
            starts_late=False,
        ),
    ]


def test_scorer_v2_runtime_loads_checkpoint_and_scores(tmp_path: Path) -> None:
    model = PhonemeScorerModelV2()
    checkpoint_path = tmp_path / "scorer_v2_test.pt"
    torch.save({"model_state_dict": model.state_dict(), "config": {}}, checkpoint_path)

    runtime = ScorerV2Runtime(
        checkpoint_path=checkpoint_path,
        backbone_id="facebook/hubert-base-ls960",
        device="cpu",
    )
    result = runtime.score(_phone_features())

    assert len(result.phone_predictions) == 2
    assert result.model_info.runtime_backend == "scorer_v2"
    assert result.model_info.checkpoint_name == checkpoint_path.name
    assert set(result.phone_predictions[0].quality_class_probs) == {"wrong_or_missed", "accented", "correct"}
    assert 0.0 <= result.phone_predictions[0].omission_probability <= 1.0
