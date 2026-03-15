from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TrainingPhoneLabel(BaseModel):
    phoneme: str
    index: int = Field(ge=0)
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    pronunciation_class: Literal["correct", "accented", "wrong_or_missed"]
    human_score: float = Field(ge=0, le=2)
    omission_label: bool = False
    pronounced_phone: str | None = None


class TrainingUtteranceArtifact(BaseModel):
    utterance_id: str
    speaker_id: str
    dataset: str
    accent_target: Literal["en-US"] = "en-US"
    target_word: str
    canonical_phones: list[str]
    ipa: str
    audio_path: str
    sample_rate: int = Field(default=16_000, ge=1)
    duration_ms: int = Field(ge=0)
    audio_quality: dict[str, float | str]
    alignment_source: Literal["mfa", "custom_ctc", "manual"]
    phone_labels: list[TrainingPhoneLabel]


class PhoneEmbeddingArtifact(BaseModel):
    utterance_id: str
    phoneme: str
    index: int = Field(ge=0)
    mean_embedding: list[float]
    variance: float = Field(ge=0)
    duration_ms: int = Field(ge=0)
    duration_z_score: float
    alignment_confidence: float = Field(ge=0, le=1)
    energy_mean: float = Field(ge=0)
    pronunciation_class: Literal["correct", "accented", "wrong_or_missed"]
    regression_target: float = Field(ge=0, le=100)
    omission_target: int = Field(ge=0, le=1)


class PhoneScorePrediction(BaseModel):
    phoneme: str
    index: int = Field(ge=0)
    match_score: float = Field(ge=0, le=100)
    duration_score: float = Field(ge=0, le=100)
    presence_score: float = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    predicted_class: Literal["correct", "accented", "wrong_or_missed"]
