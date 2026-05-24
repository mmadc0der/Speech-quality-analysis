from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover - optional runtime
    torch = None

from pronunciation_backend.models import PhoneFeatures
from pronunciation_backend.services.phone_ssl_pooling import SSL_BASE_DIM, resolved_acoustic_input_dim
from pronunciation_backend.services.scorer_runtime import (
    ScorerFeatureSpec,
    ScorerModelInfo,
    ScorerPhonePrediction,
    ScorerRuntimeResult,
)
from pronunciation_backend.services.tensor_mapper import PhoneFeatureTensorMapper
from pronunciation_backend.training.scorer_model_v2 import (
    PhonemeScorerModelV2,
    scorer_model_kwargs_from_config,
)
from pronunciation_backend.training.scoring_targets import CLASS_ORDER, class_name_from_index


def _checkpoint_config_value(payload: dict[str, object], key: str, default: object) -> object:
    config = payload.get("config")
    if isinstance(config, dict) and key in config:
        return config[key]
    return default


@dataclass
class ScorerV2Runtime:
    checkpoint_path: Path
    backbone_id: str
    device: str = "cpu"
    strict_load: bool = True
    _model: PhonemeScorerModelV2 = field(init=False, repr=False)
    _model_info: ScorerModelInfo = field(init=False, repr=False)
    _feature_spec: ScorerFeatureSpec = field(init=False, repr=False)
    _tensor_mapper: PhoneFeatureTensorMapper = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if torch is None:
            raise RuntimeError("torch is required to serve the v2 scorer")
        device = torch.device(self.device)
        payload = torch.load(self.checkpoint_path, map_location=device)
        if not isinstance(payload, dict):
            raise ValueError(f"Unsupported checkpoint payload type: {type(payload)!r}")

        config = payload.get("config") if isinstance(payload.get("config"), dict) else None
        model_kwargs = scorer_model_kwargs_from_config(config)
        model = PhonemeScorerModelV2(**model_kwargs).to(device)
        state_dict = payload["model_state_dict"] if "model_state_dict" in payload else payload
        if not isinstance(state_dict, dict):
            raise ValueError("checkpoint must contain a model_state_dict dictionary")
        model.load_state_dict(state_dict, strict=self.strict_load)
        model.eval()
        self._model = model

        ssl_base_dim = int(_checkpoint_config_value(payload, "ssl_base_dim", SSL_BASE_DIM))
        config_acoustic_dim = _checkpoint_config_value(payload, "acoustic_input_dim", None)
        ssl_feature_factor_raw = _checkpoint_config_value(payload, "ssl_feature_factor", None)
        if ssl_feature_factor_raw is not None:
            ssl_feature_factor = int(ssl_feature_factor_raw)
        elif config_acoustic_dim is not None:
            ssl_feature_factor = max(1, int(config_acoustic_dim) // ssl_base_dim)
        else:
            ssl_feature_factor = 1

        pooling_mode = str(
            _checkpoint_config_value(
                payload,
                "pooling_mode",
                "subspan_end_concat" if ssl_feature_factor > 1 else "mean",
            )
        )
        acoustic_input_dim = int(
            config_acoustic_dim
            if config_acoustic_dim is not None
            else resolved_acoustic_input_dim(
                ssl_feature_factor=ssl_feature_factor,
                ssl_base_dim=ssl_base_dim,
            )
        )
        expected_input = resolved_acoustic_input_dim(
            ssl_feature_factor=ssl_feature_factor,
            ssl_base_dim=ssl_base_dim,
        )
        if acoustic_input_dim != expected_input:
            raise ValueError(
                f"Checkpoint acoustic_input_dim={acoustic_input_dim} does not match "
                f"768 * ssl_feature_factor={ssl_feature_factor} ({expected_input})."
            )

        self._feature_spec = ScorerFeatureSpec(
            acoustic_input_dim=acoustic_input_dim,
            ssl_feature_factor=ssl_feature_factor,
            pooling_mode=pooling_mode,  # type: ignore[arg-type]
            ssl_base_dim=ssl_base_dim,
        )
        self._tensor_mapper = PhoneFeatureTensorMapper(acoustic_dim=acoustic_input_dim)

        architecture_version = str(_checkpoint_config_value(payload, "architecture_version", "v2_compat"))
        model_version = "v3" if architecture_version == "v3" else "v2"
        self._model_info = ScorerModelInfo(
            runtime_backend="scorer_v2",
            model_version=model_version,
            checkpoint_name=self.checkpoint_path.name,
            backbone_id=self.backbone_id,
            device=str(device),
            class_labels=CLASS_ORDER,
        )

    def model_info(self) -> ScorerModelInfo:
        return self._model_info

    def feature_spec(self) -> ScorerFeatureSpec:
        return self._feature_spec

    def score(self, phone_features: list[PhoneFeatures]) -> ScorerRuntimeResult:
        model_inputs = self._tensor_mapper.build_inputs(phone_features)
        if torch is None:
            raise RuntimeError("torch is required to serve the v2 scorer")

        device = next(self._model.parameters()).device
        model_inputs = {
            key: value.to(device=device, non_blocking=True)
            for key, value in model_inputs.items()
        }
        with torch.inference_mode():
            outputs = self._model(
                acoustic_embeddings=model_inputs["acoustic_embeddings"],
                phoneme_ids=model_inputs["phoneme_ids"],
                attention_mask=model_inputs["attention_mask"],
            )

        mask = model_inputs["attention_mask"][0]
        class_probs = outputs["class_probs"][0][mask].detach().cpu()
        expected_scores = outputs["expected_score"][0][mask].detach().cpu().tolist()
        expected_human_scores = outputs["expected_human_score"][0][mask].detach().cpu().tolist()
        omission_probabilities = torch.sigmoid(outputs["omission_logit"][0][mask]).detach().cpu().tolist()
        class_indices = outputs["quality_logits"][0][mask].argmax(dim=-1).detach().cpu().tolist()

        predictions: list[ScorerPhonePrediction] = []
        for features, probs_tensor, predicted_index, expected_score, expected_human_score, omission_probability in zip(
            phone_features,
            class_probs,
            class_indices,
            expected_scores,
            expected_human_scores,
            omission_probabilities,
        ):
            probs = {
                label: float(prob)
                for label, prob in zip(CLASS_ORDER, probs_tensor.tolist())
            }
            predictions.append(
                ScorerPhonePrediction(
                    phoneme=features.phoneme,
                    start_ms=features.start_ms,
                    end_ms=features.end_ms,
                    expected_score=float(expected_score),
                    expected_human_score=float(expected_human_score),
                    omission_probability=float(omission_probability),
                    predicted_class=class_name_from_index(int(predicted_index)),
                    quality_class_probs=probs,
                    alignment_confidence=float(features.alignment_confidence),
                )
            )

        return ScorerRuntimeResult(
            phone_predictions=predictions,
            model_info=self._model_info,
        )
