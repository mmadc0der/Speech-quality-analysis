from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field


class AudioQualityPayload(BaseModel):
    status: Literal["ok", "low_confidence", "rejected"]
    snr_estimate: float = Field(ge=0)
    duration_ms: int = Field(ge=0)
    rms: float = Field(ge=0)
    clipping_ratio: float = Field(ge=0, le=1)
    silence_ratio: float = Field(ge=0, le=1)


class PhonemeAssessmentPayload(BaseModel):
    phoneme: str
    start_ms: int = Field(ge=0)
    end_ms: int = Field(ge=0)
    match_score: int = Field(ge=0, le=100)
    duration_score: int = Field(ge=0, le=100)
    presence_score: int = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    status: Literal[
        "ok",
        "low_match",
        "too_short",
        "too_long",
        "weak",
        "possibly_missing",
        "late_start",
    ]


class PrimaryIssuePayload(BaseModel):
    phoneme: str
    type: str
    message: str


class ReferencePayload(BaseModel):
    ipa: str
    audio_id: str
    asset_path: str | None = None


class PronunciationAssessmentResponse(BaseModel):
    word: str
    accent_target: Literal["en-US"] = "en-US"
    ipa: str
    overall_score: int = Field(ge=0, le=100)
    confidence: float = Field(ge=0, le=1)
    audio_quality: AudioQualityPayload
    phonemes: list[PhonemeAssessmentPayload]
    primary_issue: PrimaryIssuePayload
    reference: ReferencePayload


@dataclass(frozen=True)
class LexiconEntry:
    word: str
    phones: list[str]
    ipa: str
    reference_audio_id: str
    syllables: list[list[str]] = field(default_factory=list)
    stress_pattern: str | None = None


@dataclass(frozen=True)
class PreparedAudio:
    samples: list[float]
    sample_rate: int
    duration_ms: int
    rms: float
    clipping_ratio: float
    silence_ratio: float
    snr_estimate: float
    quality_status: str


@dataclass(frozen=True)
class EncodedFrames:
    embeddings: list[list[float]]
    frame_ms: float
    energy: list[float]


@dataclass(frozen=True)
class PhoneSpan:
    phoneme: str
    start_frame: int
    end_frame: int
    start_ms: int
    end_ms: int
    alignment_confidence: float
    duration_z_score: float


@dataclass(frozen=True)
class PhoneFeatures:
    phoneme: str
    start_ms: int
    end_ms: int
    mean_embedding: list[float]
    variance: float
    duration_ms: int
    duration_z_score: float
    alignment_confidence: float
    energy_mean: float
    starts_late: bool
