from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import logging

try:
    import torch
except ImportError:  # pragma: no cover - optional runtime
    torch = None

from pronunciation_backend.models import PhoneFeatures
from pronunciation_backend.services.scorer_runtime import (
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


logger = logging.getLogger(__name__)


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
    compile_model: bool = False
    compile_mode: str = "reduce-overhead"
    tensor_mapper: PhoneFeatureTensorMapper = field(default_factory=PhoneFeatureTensorMapper)
    _model: PhonemeScorerModelV2 = field(init=False, repr=False)
    _model_info: ScorerModelInfo = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if torch is None:
            raise RuntimeError("torch is required to serve the v2 scorer")
        device = torch.device(self.device)
        payload = torch.load(self.checkpoint_path, map_location=device)
        if not isinstance(payload, dict):
            raise ValueError(f"Unsupported checkpoint payload type: {type(payload)!r}")

        config = payload.get("config") if isinstance(payload.get("config"), dict) else None
        model = PhonemeScorerModelV2(**scorer_model_kwargs_from_config(config)).to(device)
        state_dict = payload["model_state_dict"] if "model_state_dict" in payload else payload
        if not isinstance(state_dict, dict):
            raise ValueError("checkpoint must contain a model_state_dict dictionary")
        model.load_state_dict(state_dict, strict=self.strict_load)
        model.eval()
        self._model = self._maybe_compile(model)
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

    def score(self, phone_features: list[PhoneFeatures]) -> ScorerRuntimeResult:
        model_inputs = self.tensor_mapper.build_inputs(phone_features)
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

    def warmup(self) -> None:
        feature_dim = self.tensor_mapper.acoustic_dim
        dummy_features = [
            PhoneFeatures(
                phoneme="AA",
                start_ms=0,
                end_ms=40,
                mean_embedding=[0.0] * feature_dim,
                variance=0.0,
                duration_ms=40,
                duration_z_score=0.0,
                alignment_confidence=1.0,
                energy_mean=0.0,
                starts_late=False,
            )
        ]
        self.score(dummy_features)

    def _maybe_compile(self, model: PhonemeScorerModelV2) -> PhonemeScorerModelV2:
        if not self.compile_model:
            return model
        if torch is None or not hasattr(torch, "compile"):
            logger.warning("torch.compile is unavailable; keeping scorer in eager mode")
            return model
        try:
            compiled_model = torch.compile(model, mode=self.compile_mode)
            logger.info("Compiled scorer model with torch.compile", extra={"compile_mode": self.compile_mode})
            return compiled_model
        except Exception:  # pragma: no cover - compile failures are environment specific
            logger.exception("torch.compile failed for scorer model; falling back to eager mode")
            return model
