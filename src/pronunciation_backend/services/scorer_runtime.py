from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pronunciation_backend.models import PhoneFeatures
from pronunciation_backend.services.phone_ssl_pooling import PoolingMode


@dataclass(frozen=True)
class ScorerFeatureSpec:
    acoustic_input_dim: int
    ssl_feature_factor: int
    pooling_mode: PoolingMode
    ssl_base_dim: int = 768


@dataclass(frozen=True)
class ScorerModelInfo:
    runtime_backend: str
    model_version: str
    checkpoint_name: str
    backbone_id: str
    device: str
    class_labels: tuple[str, ...]


@dataclass(frozen=True)
class ScorerPhonePrediction:
    phoneme: str
    start_ms: int
    end_ms: int
    expected_score: float
    expected_human_score: float
    omission_probability: float
    predicted_class: str
    quality_class_probs: dict[str, float]
    alignment_confidence: float


@dataclass(frozen=True)
class ScorerRuntimeResult:
    phone_predictions: list[ScorerPhonePrediction]
    model_info: ScorerModelInfo


class ScorerRuntime(Protocol):
    def score(self, phone_features: list[PhoneFeatures]) -> ScorerRuntimeResult:
        ...

    def model_info(self) -> ScorerModelInfo:
        ...

    def feature_spec(self) -> ScorerFeatureSpec:
        ...
