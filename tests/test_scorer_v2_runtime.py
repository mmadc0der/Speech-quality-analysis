from __future__ import annotations

from pathlib import Path

import torch

from pronunciation_backend.models import PhoneFeatures
from pronunciation_backend.services.scorer_v2_runtime import ScorerV2Runtime
from pronunciation_backend.services.tensor_mapper import PhoneFeatureTensorMapper
from pronunciation_backend.training.scorer_model_v2 import PhonemeScorerModelV2
from pronunciation_backend.training.v3_architecture import (
    V3_ACOUSTIC_LAYERS,
    V3_D_MODEL,
    V3_FFN_DIM,
    V3_NUM_HEADS,
    V3_SSL_FEATURE_FACTOR,
)


def _phone_features(acoustic_dim: int = 768) -> list[PhoneFeatures]:
    base_embedding = [0.01] * acoustic_dim
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
            mean_embedding=[0.02] * acoustic_dim,
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
    assert result.model_info.model_version == "v2"
    assert result.model_info.checkpoint_name == checkpoint_path.name
    assert set(result.phone_predictions[0].quality_class_probs) == {"wrong_or_missed", "accented", "correct"}
    assert 0.0 <= result.phone_predictions[0].omission_probability <= 1.0
    assert runtime.feature_spec().acoustic_input_dim == 768


def test_scorer_v2_runtime_loads_v3_checkpoint(tmp_path: Path) -> None:
    model = PhonemeScorerModelV2(
        acoustic_input_dim=1536,
        d_model=V3_D_MODEL,
        num_heads=V3_NUM_HEADS,
        acoustic_layers=V3_ACOUSTIC_LAYERS,
        ffn_dim=V3_FFN_DIM,
        architecture_version="v3",
    )
    checkpoint_path = tmp_path / "scorer_v3_test.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": {
                "architecture_version": "v3",
                "acoustic_input_dim": 1536,
                "ssl_feature_factor": V3_SSL_FEATURE_FACTOR,
                "pooling_mode": "subspan_end_concat",
                "ssl_base_dim": 768,
                "d_model": V3_D_MODEL,
                "num_heads": V3_NUM_HEADS,
                "acoustic_layers": V3_ACOUSTIC_LAYERS,
                "ffn_dim": V3_FFN_DIM,
            },
        },
        checkpoint_path,
    )

    runtime = ScorerV2Runtime(
        checkpoint_path=checkpoint_path,
        backbone_id="facebook/hubert-base-ls960",
        device="cpu",
    )
    result = runtime.score(_phone_features(acoustic_dim=1536))

    assert result.model_info.model_version == "v3"
    assert runtime.feature_spec().acoustic_input_dim == 1536
    assert runtime.feature_spec().ssl_feature_factor == 2
    assert runtime.feature_spec().pooling_mode == "subspan_end_concat"


def test_tensor_mapper_accepts_1536_dim_features() -> None:
    mapper = PhoneFeatureTensorMapper(acoustic_dim=1536)
    inputs = mapper.build_inputs(_phone_features(acoustic_dim=1536))
    assert inputs["acoustic_embeddings"].shape == (1, 2, 1536)
