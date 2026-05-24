from __future__ import annotations

from dataclasses import dataclass

from pronunciation_backend.models import (
    AudioQualityPayload,
    ModelInfoPayload,
    PreparedAudio,
    PrimaryIssuePayload,
    PronunciationAssessmentResponse,
    PronunciationPhonePayload,
    QualityClassProbabilitiesPayload,
    ReferencePayload,
)
from pronunciation_backend.services.scorer_runtime import ScorerPhonePrediction, ScorerRuntimeResult


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


@dataclass
class ResponseMapper:
    def build_audio_quality(self, prepared_audio: PreparedAudio) -> AudioQualityPayload:
        return AudioQualityPayload(
            status=prepared_audio.quality_status if prepared_audio.quality_status in {"ok", "low_confidence", "rejected"} else "low_confidence",
            snr_estimate=prepared_audio.snr_estimate,
            duration_ms=prepared_audio.duration_ms,
            rms=prepared_audio.rms,
            clipping_ratio=prepared_audio.clipping_ratio,
            silence_ratio=prepared_audio.silence_ratio,
            original_duration_ms=prepared_audio.original_duration_ms,
            trim_start_ms=prepared_audio.trim_start_ms,
            trim_end_ms=prepared_audio.trim_end_ms,
            trim_applied=prepared_audio.trim_applied,
        )

    def build_response(
        self,
        *,
        word: str,
        ipa: str,
        prepared_audio: PreparedAudio,
        runtime_result: ScorerRuntimeResult,
        reference: ReferencePayload | None,
    ) -> PronunciationAssessmentResponse:
        audio_quality = self.build_audio_quality(prepared_audio)
        confidence_multiplier = {"ok": 1.0, "low_confidence": 0.9, "rejected": 0.75}[audio_quality.status]
        phone_payloads = [
            self._build_phone_payload(
                prediction,
                trim_start_ms=prepared_audio.trim_start_ms,
                confidence_multiplier=confidence_multiplier,
            )
            for prediction in runtime_result.phone_predictions
        ]
        overall_score = self._overall_score(runtime_result.phone_predictions)
        overall_confidence = self._overall_confidence(phone_payloads, confidence_multiplier=confidence_multiplier)
        primary_issue = self._primary_issue(phone_payloads)
        model_info = runtime_result.model_info
        return PronunciationAssessmentResponse(
            word=word,
            ipa=ipa,
            overall_score=overall_score,
            confidence=overall_confidence,
            audio_quality=audio_quality,
            phonemes=phone_payloads,
            primary_issue=primary_issue,
            reference=reference,
            model_info=ModelInfoPayload(
                runtime_backend=model_info.runtime_backend,
                model_version=model_info.model_version,
                checkpoint_name=model_info.checkpoint_name,
                backbone_id=model_info.backbone_id,
                device=model_info.device,
                class_labels=list(model_info.class_labels),
            ),
        )

    def _build_phone_payload(
        self,
        prediction: ScorerPhonePrediction,
        *,
        trim_start_ms: int,
        confidence_multiplier: float,
    ) -> PronunciationPhonePayload:
        class_confidence = prediction.quality_class_probs[prediction.predicted_class]
        confidence = _clamp(
            ((0.7 * class_confidence) + (0.3 * prediction.alignment_confidence)) * confidence_multiplier,
            0.0,
            1.0,
        )
        return PronunciationPhonePayload(
            phoneme=prediction.phoneme,
            start_ms=prediction.start_ms + trim_start_ms,
            end_ms=prediction.end_ms + trim_start_ms,
            expected_score=round(prediction.expected_score, 3),
            expected_human_score=round(prediction.expected_human_score, 3),
            omission_probability=round(prediction.omission_probability, 6),
            confidence=round(confidence, 3),
            alignment_confidence=round(prediction.alignment_confidence, 3),
            predicted_class=prediction.predicted_class,
            quality_class_probs=QualityClassProbabilitiesPayload(
                wrong_or_missed=round(prediction.quality_class_probs["wrong_or_missed"], 6),
                accented=round(prediction.quality_class_probs["accented"], 6),
                correct=round(prediction.quality_class_probs["correct"], 6),
            ),
        )

    def _overall_score(self, predictions: list[ScorerPhonePrediction]) -> float:
        if not predictions:
            return 0.0
        weighted_sum = 0.0
        total_weight = 0.0
        for prediction in predictions:
            duration_weight = max(1.0, float(prediction.end_ms - prediction.start_ms))
            alignment_weight = 0.5 + (0.5 * prediction.alignment_confidence)
            weight = duration_weight * alignment_weight
            weighted_sum += prediction.expected_score * weight
            total_weight += weight
        return round(weighted_sum / max(total_weight, 1.0), 3)

    def _overall_confidence(
        self,
        phone_payloads: list[PronunciationPhonePayload],
        *,
        confidence_multiplier: float,
    ) -> float:
        if not phone_payloads:
            return 0.0
        del confidence_multiplier
        average_confidence = sum(phone.confidence for phone in phone_payloads) / len(phone_payloads)
        return round(_clamp(average_confidence, 0.0, 1.0), 3)

    def _primary_issue(self, phone_payloads: list[PronunciationPhonePayload]) -> PrimaryIssuePayload:
        if not phone_payloads:
            return PrimaryIssuePayload(phoneme="", type="no_signal", message="no phoneme segments available")

        worst = max(phone_payloads, key=self._severity)
        issue_type = self._issue_type(worst)
        if issue_type == "none":
            message = f"all phonemes are currently above the intervention threshold; weakest segment is {worst.phoneme}"
        elif issue_type == "possibly_missing":
            message = f"phoneme {worst.phoneme} may be omitted or heavily reduced"
        else:
            message = f"phoneme {worst.phoneme} is the main candidate for correction"
        return PrimaryIssuePayload(
            phoneme=worst.phoneme,
            type=issue_type,
            message=message,
        )

    def _issue_type(self, phone: PronunciationPhonePayload) -> str:
        if phone.omission_probability >= 0.5:
            return "possibly_missing"
        if phone.predicted_class != "correct":
            return phone.predicted_class
        if phone.confidence < 0.6:
            return "low_confidence"
        return "none"

    def _severity(self, phone: PronunciationPhonePayload) -> float:
        predicted_class_penalty = {
            "correct": 0.0,
            "accented": 12.0,
            "wrong_or_missed": 24.0,
        }[phone.predicted_class]
        return (
            (100.0 - phone.expected_score)
            + (phone.omission_probability * 35.0)
            + ((1.0 - phone.confidence) * 25.0)
            + predicted_class_penalty
        )
