from __future__ import annotations

import re
from typing import Literal

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional runtime
    torch = None

SSL_BASE_DIM = 768
SCALAR_FEATURE_DIM = 3
DEFAULT_ACOUSTIC_FEATURE_DIM = SSL_BASE_DIM + SCALAR_FEATURE_DIM

PoolingMode = Literal["mean", "subspan_end_concat"]

V3_POOLING_VERSION = "phone_subspan_end_concat_v1_factor2"
V3_SSL_FEATURE_FACTOR = 2


def validate_ssl_feature_factor(factor: int) -> int:
    if factor < 1:
        raise ValueError(f"ssl_feature_factor must be >= 1, got {factor}")
    return factor


def resolved_acoustic_input_dim(*, ssl_feature_factor: int, ssl_base_dim: int = SSL_BASE_DIM) -> int:
    validate_ssl_feature_factor(ssl_feature_factor)
    return ssl_base_dim * ssl_feature_factor


def acoustic_feature_storage_dim(*, ssl_feature_factor: int, ssl_base_dim: int = SSL_BASE_DIM) -> int:
    return resolved_acoustic_input_dim(
        ssl_feature_factor=ssl_feature_factor,
        ssl_base_dim=ssl_base_dim,
    ) + SCALAR_FEATURE_DIM


def pooling_version_for(*, pooling_mode: PoolingMode, ssl_feature_factor: int) -> str:
    if pooling_mode == "mean":
        if ssl_feature_factor != 1:
            raise ValueError("mean pooling requires ssl_feature_factor=1")
        return "phone_mean_v1"
    if pooling_mode == "subspan_end_concat":
        factor = validate_ssl_feature_factor(ssl_feature_factor)
        return f"phone_subspan_end_concat_v1_factor{factor}"
    raise ValueError(f"Unsupported pooling_mode={pooling_mode!r}")


def parse_pooling_version(pooling_version: str) -> tuple[PoolingMode, int]:
    if pooling_version == "phone_mean_v1":
        return "mean", 1
    match = re.fullmatch(r"phone_subspan_end_concat_v1_factor(\d+)", pooling_version)
    if match is not None:
        return "subspan_end_concat", int(match.group(1))
    raise ValueError(f"Unsupported pooling_version={pooling_version!r}")


def chunk_end_frame_indices(num_frames: int, factor: int) -> list[int]:
    factor = validate_ssl_feature_factor(factor)
    if num_frames <= 0:
        return [0] * factor
    if factor == 1:
        return [num_frames - 1]

    indices: list[int] = []
    for chunk_index in range(factor):
        start = (chunk_index * num_frames) // factor
        end = ((chunk_index + 1) * num_frames) // factor
        if end <= start:
            end = min(start + 1, num_frames)
        end_frame = min(max(start, end - 1), num_frames - 1)
        indices.append(end_frame)
    return indices


def pool_phone_ssl_features_numpy(
    segment: np.ndarray,
    *,
    pooling_mode: PoolingMode = "mean",
    ssl_feature_factor: int = 1,
    ssl_base_dim: int = SSL_BASE_DIM,
) -> np.ndarray:
    frame_array = np.asarray(segment, dtype=np.float32)
    if frame_array.ndim == 1:
        frame_array = frame_array.reshape(1, -1)
    if frame_array.size == 0:
        frame_array = np.zeros((1, ssl_base_dim), dtype=np.float32)

    if pooling_mode == "mean":
        if ssl_feature_factor != 1:
            raise ValueError("mean pooling requires ssl_feature_factor=1")
        return frame_array.mean(axis=0).astype(np.float32)

    if pooling_mode == "subspan_end_concat":
        factor = validate_ssl_feature_factor(ssl_feature_factor)
        indices = chunk_end_frame_indices(frame_array.shape[0], factor)
        vectors = [frame_array[index] for index in indices]
        pooled = np.concatenate(vectors, axis=0).astype(np.float32)
        expected_dim = resolved_acoustic_input_dim(
            ssl_feature_factor=factor,
            ssl_base_dim=ssl_base_dim,
        )
        if pooled.shape[0] != expected_dim:
            raise ValueError(
                f"Expected pooled vector dim {expected_dim}, got {pooled.shape[0]} "
                f"for factor={factor} and hidden_dim={frame_array.shape[-1]}"
            )
        return pooled

    raise ValueError(f"Unsupported pooling_mode={pooling_mode!r}")


def pool_phone_ssl_features_torch(
    segment: object,
    *,
    pooling_mode: PoolingMode = "mean",
    ssl_feature_factor: int = 1,
    ssl_base_dim: int = SSL_BASE_DIM,
) -> object:
    if torch is None:  # pragma: no cover - optional runtime
        raise RuntimeError("torch is required for GPU phone SSL pooling")

    hidden_tensor = segment.float()
    if hidden_tensor.ndim == 1:
        hidden_tensor = hidden_tensor.unsqueeze(0)
    if hidden_tensor.numel() == 0:
        hidden_tensor = torch.zeros((1, ssl_base_dim), device=hidden_tensor.device, dtype=torch.float32)

    if pooling_mode == "mean":
        if ssl_feature_factor != 1:
            raise ValueError("mean pooling requires ssl_feature_factor=1")
        return hidden_tensor.mean(dim=0)

    if pooling_mode == "subspan_end_concat":
        factor = validate_ssl_feature_factor(ssl_feature_factor)
        indices = chunk_end_frame_indices(int(hidden_tensor.shape[0]), factor)
        vectors = [hidden_tensor[index] for index in indices]
        pooled = torch.cat(vectors, dim=0)
        expected_dim = resolved_acoustic_input_dim(
            ssl_feature_factor=factor,
            ssl_base_dim=ssl_base_dim,
        )
        if int(pooled.shape[0]) != expected_dim:
            raise ValueError(
                f"Expected pooled vector dim {expected_dim}, got {int(pooled.shape[0])} "
                f"for factor={factor} and hidden_dim={int(hidden_tensor.shape[-1])}"
            )
        return pooled

    raise ValueError(f"Unsupported pooling_mode={pooling_mode!r}")
