from __future__ import annotations

from dataclasses import dataclass
from math import exp

from pronunciation_backend.models import (
    AudioQualityPayload,
    PhoneFeatures,
    PhonemeAssessmentPayload,
    PrimaryIssuePayload,
)


def _clamp_score(value: float) -> int:
    return max(0, min(100, int(round(value))))


@dataclass
class PhoneScoringHead:
    """MVP scorer with hooks for later replacement by a trained head."""

    def score(self, phone_features: list[PhoneFeatures]) -> list[PhonemeAssessmentPayload]:
        assessments: list[PhonemeAssessmentPayload] = []
        for features in phone_features:
            variance_penalty = min(28.0, features.variance * 120.0)
            energy_bonus = min(12.0, features.energy_mean * 22.0)
            duration_penalty = min(40.0, abs(features.duration_z_score) * 35.0)
            presence_penalty = 0.0 if features.energy_mean > 0.08 else 35.0

            match_score = _clamp_score(82.0 - variance_penalty + energy_bonus)
            duration_score = _clamp_score(92.0 - duration_penalty)
            presence_score = _clamp_score(96.0 - presence_penalty - max(0.0, -features.duration_z_score * 12.0))
            confidence = max(
                0.35,
                min(
                    0.98,
                    round(
                        0.55
                        + (features.alignment_confidence * 0.35)
                        + (min(features.energy_mean, 0.2) * 0.5)
                        - (min(features.variance, 0.2) * 0.25),
                        3,
                    ),
                ),
            )

            status = self._derive_status(match_score, duration_score, presence_score, features.starts_late, features.duration_z_score)
            assessments.append(
                PhonemeAssessmentPayload(
                    phoneme=features.phoneme,
                    start_ms=features.start_ms,
                    end_ms=features.end_ms,
                    match_score=match_score,
                    duration_score=duration_score,
                    presence_score=presence_score,
                    confidence=confidence,
                    status=status,
                )
            )
        return assessments

    def _derive_status(
        self,
        match_score: int,
        duration_score: int,
        presence_score: int,
        starts_late: bool,
        duration_z_score: float,
    ) -> str:
        if presence_score < 45:
            return "possibly_missing"
        if match_score < 58:
            return "low_match"
        if starts_late:
            return "late_start"
        if duration_score < 58 and duration_z_score < 0:
            return "too_short"
        if duration_score < 58 and duration_z_score > 0:
            return "too_long"
        if presence_score < 70 or match_score < 70:
            return "weak"
        return "ok"


@dataclass
class CalibrationAndIssueService:
    def build_audio_quality(self, quality_status: str, snr_estimate: float, duration_ms: int, rms: float, clipping_ratio: float, silence_ratio: float) -> AudioQualityPayload:
        return AudioQualityPayload(
            status=quality_status if quality_status in {"ok", "low_confidence", "rejected"} else "low_confidence",
            snr_estimate=snr_estimate,
            duration_ms=duration_ms,
            rms=rms,
            clipping_ratio=clipping_ratio,
            silence_ratio=silence_ratio,
        )

    def overall_score(self, phonemes: list[PhonemeAssessmentPayload], audio_quality: AudioQualityPayload) -> tuple[int, float]:
        if not phonemes:
            return 0, 0.0

        weighted_sum = 0.0
        total_weight = 0.0
        confidences: list[float] = []
        for item in phonemes:
            phone_score = (0.5 * item.match_score) + (0.25 * item.duration_score) + (0.25 * item.presence_score)
            weight = 1.0 + (0.5 if len(item.phoneme) > 1 else 0.0)
            weighted_sum += phone_score * weight
            total_weight += weight
            confidences.append(item.confidence)

        score = weighted_sum / max(1.0, total_weight)
        quality_multiplier = {"ok": 1.0, "low_confidence": 0.92, "rejected": 0.75}[audio_quality.status]
        confidence = (sum(confidences) / len(confidences)) * quality_multiplier
        return _clamp_score(score * quality_multiplier), round(max(0.0, min(1.0, confidence)), 3)

    def primary_issue(self, phonemes: list[PhonemeAssessmentPayload], overall_confidence: float) -> PrimaryIssuePayload:
        if not phonemes:
            return PrimaryIssuePayload(phoneme="", type="no_signal", message="no phoneme segments available")

        ranked = sorted(phonemes, key=lambda phone: self._severity(phone, overall_confidence), reverse=True)
        worst = ranked[0]
        issue_type = worst.status if worst.status != "ok" else "low_match"
        return PrimaryIssuePayload(
            phoneme=worst.phoneme,
            type=issue_type,
            message=f"phoneme {worst.phoneme} has the largest deviation",
        )

    def _severity(self, phone: PhonemeAssessmentPayload, overall_confidence: float) -> float:
        low_match = 100 - phone.match_score
        low_presence = 100 - phone.presence_score
        duration_penalty = 100 - phone.duration_score
        confidence_scale = 0.65 + (overall_confidence * 0.35)
        return confidence_scale * ((1.0 * low_match) + (0.75 * low_presence) + (0.45 * duration_penalty))
