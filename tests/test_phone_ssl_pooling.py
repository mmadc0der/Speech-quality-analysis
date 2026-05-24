from __future__ import annotations

import numpy as np
import pytest
import torch

from pronunciation_backend.services.aligner import PhoneFeatureBuilder
from pronunciation_backend.services.phone_ssl_pooling import (
    chunk_end_frame_indices,
    parse_pooling_version,
    pool_phone_ssl_features_numpy,
    pool_phone_ssl_features_torch,
    pooling_version_for,
    resolved_acoustic_input_dim,
)
from pronunciation_backend.training.v3_architecture import (
    V3_ACOUSTIC_LAYERS,
    V3_D_MODEL,
    V3_FFN_DIM,
    V3_NUM_HEADS,
    V3_SSL_FEATURE_FACTOR,
    apply_v3_training_defaults,
)


def test_chunk_end_frame_indices_for_factor_two() -> None:
    assert chunk_end_frame_indices(4, 2) == [1, 3]
    assert chunk_end_frame_indices(5, 2) == [1, 4]


def test_chunk_end_frame_indices_duplicates_single_frame() -> None:
    assert chunk_end_frame_indices(1, 2) == [0, 0]


def test_subspan_end_concat_pooling_numpy() -> None:
    segment = np.arange(12, dtype=np.float32).reshape(4, 3)
    pooled = pool_phone_ssl_features_numpy(
        segment,
        pooling_mode="subspan_end_concat",
        ssl_feature_factor=2,
        ssl_base_dim=3,
    )
    assert pooled.tolist() == [3.0, 4.0, 5.0, 9.0, 10.0, 11.0]


def test_subspan_end_concat_pooling_torch_matches_numpy() -> None:
    segment = np.arange(12, dtype=np.float32).reshape(4, 3)
    numpy_pooled = pool_phone_ssl_features_numpy(
        segment,
        pooling_mode="subspan_end_concat",
        ssl_feature_factor=2,
        ssl_base_dim=3,
    )
    torch_pooled = pool_phone_ssl_features_torch(
        torch.from_numpy(segment),
        pooling_mode="subspan_end_concat",
        ssl_feature_factor=2,
        ssl_base_dim=3,
    )
    assert torch_pooled.detach().cpu().numpy().tolist() == numpy_pooled.tolist()


def test_pooling_version_round_trip() -> None:
    version = pooling_version_for(pooling_mode="subspan_end_concat", ssl_feature_factor=2)
    assert version == "phone_subspan_end_concat_v1_factor2"
    assert parse_pooling_version(version) == ("subspan_end_concat", 2)


def test_phone_feature_builder_supports_factor_two() -> None:
    from pronunciation_backend.models import EncodedFrames, PhoneSpan

    embeddings = np.stack(
        [
            np.full(768, 0.0, dtype=np.float32),
            np.full(768, 1.0, dtype=np.float32),
            np.full(768, 2.0, dtype=np.float32),
            np.full(768, 3.0, dtype=np.float32),
        ],
        axis=0,
    )
    encoded = EncodedFrames(
        embeddings=embeddings,
        frame_ms=10.0,
        energy=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
    )
    spans = [
        PhoneSpan(
            phoneme="AA",
            start_frame=0,
            end_frame=4,
            start_ms=0,
            end_ms=40,
            alignment_confidence=1.0,
            duration_z_score=0.0,
        )
    ]
    builder = PhoneFeatureBuilder(pooling_mode="subspan_end_concat", ssl_feature_factor=2)
    features = builder.build(encoded, spans)

    assert len(features[0].mean_embedding) == resolved_acoustic_input_dim(ssl_feature_factor=2)
    assert features[0].mean_embedding[:768] == pytest.approx([1.0] * 768)
    assert features[0].mean_embedding[768:] == pytest.approx([3.0] * 768)


def test_apply_v3_training_defaults_sets_size_preset() -> None:
    class Args:
        architecture_version = "v3"
        ssl_feature_factor = None
        pooling_mode = None
        acoustic_input_dim = 768
        d_model = 384
        num_heads = 6
        acoustic_layers = 6
        scorer_layers = 2
        ffn_dim = 1536

    args = Args()
    apply_v3_training_defaults(args)

    assert args.ssl_feature_factor == V3_SSL_FEATURE_FACTOR
    assert args.pooling_mode == "subspan_end_concat"
    assert args.acoustic_input_dim == 1536
    assert args.d_model == V3_D_MODEL
    assert args.num_heads == V3_NUM_HEADS
    assert args.acoustic_layers == V3_ACOUSTIC_LAYERS
    assert args.ffn_dim == V3_FFN_DIM
