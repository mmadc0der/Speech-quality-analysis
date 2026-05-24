from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from pronunciation_backend.models import EncodedFrames, LexiconEntry, PhoneSpan, PreparedAudio
from pronunciation_backend.services.aligner import phone_duration_weight
from pronunciation_backend.services.mfa_aligner import AlignmentUnavailableError


PHONE_CTC_ALIGNMENT_CONFIDENCE_FLOOR = 0.35


@dataclass(frozen=True)
class PhoneVocabulary:
    """Stable mapping between canonical phone labels and model emission columns."""

    labels: tuple[str, ...]
    blank_token: str = "<blank>"

    def __post_init__(self) -> None:
        if self.blank_token not in self.labels:
            raise ValueError(f"Phone vocabulary must include blank token {self.blank_token!r}")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("Phone vocabulary labels must be unique")

    @classmethod
    def from_phones(cls, phones: list[str] | tuple[str, ...], *, blank_token: str = "<blank>") -> "PhoneVocabulary":
        ordered = [blank_token]
        for phone in phones:
            if phone != blank_token and phone not in ordered:
                ordered.append(phone)
        return cls(labels=tuple(ordered), blank_token=blank_token)

    @property
    def blank_id(self) -> int:
        return self.labels.index(self.blank_token)

    def id_for_phone(self, phone: str) -> int:
        try:
            return self.labels.index(phone)
        except ValueError as exc:
            raise KeyError(f"Phone {phone!r} is missing from the phone CTC vocabulary") from exc


@dataclass(frozen=True)
class FrameEmissions:
    """Frame-level phone probabilities or log-probabilities."""

    values: np.ndarray
    log_probabilities: bool = False

    def log_probs(self) -> np.ndarray:
        scores = np.asarray(self.values, dtype=np.float32)
        if scores.ndim != 2:
            raise ValueError(f"Expected frame emissions with shape [frames, labels], got {scores.shape}")
        if self.log_probabilities:
            return scores
        return np.log(np.clip(scores, 1e-8, 1.0))


@dataclass(frozen=True)
class MonotonicPhoneSegment:
    phoneme: str
    start_frame: int
    end_frame: int
    confidence: float


class PhoneEmissionModel(Protocol):
    def infer(self, prepared: PreparedAudio, encoded: EncodedFrames) -> FrameEmissions:
        ...


def decode_monotonic_phone_segments(
    emissions: FrameEmissions,
    target_phones: list[str] | tuple[str, ...],
    vocabulary: PhoneVocabulary,
) -> list[MonotonicPhoneSegment]:
    """Constrained Viterbi over the expected phone sequence.

    This is intentionally narrow: every target phone receives at least one frame,
    and transitions can only stay on the same phone or advance by one phone.
    Blank is reserved for future CTC collapse/insertion work but is not emitted
    into spans yet.
    """

    phones = tuple(target_phones)
    if not phones:
        return []

    log_probs = emissions.log_probs()
    frame_count = log_probs.shape[0]
    phone_count = len(phones)
    if frame_count < phone_count:
        raise ValueError(f"Need at least one frame per phone: frames={frame_count} phones={phone_count}")

    phone_ids = [vocabulary.id_for_phone(phone) for phone in phones]
    phone_scores = log_probs[:, phone_ids]

    neg_inf = -np.inf
    best = np.full((frame_count, phone_count), neg_inf, dtype=np.float32)
    backpointers = np.full((frame_count, phone_count), -1, dtype=np.int32)
    best[0, 0] = phone_scores[0, 0]
    backpointers[0, 0] = 0

    for frame in range(1, frame_count):
        max_phone = min(frame, phone_count - 1)
        for phone_index in range(max_phone + 1):
            stay_score = best[frame - 1, phone_index]
            advance_score = best[frame - 1, phone_index - 1] if phone_index > 0 else neg_inf
            if advance_score > stay_score:
                previous_phone = phone_index - 1
                previous_score = advance_score
            else:
                previous_phone = phone_index
                previous_score = stay_score
            best[frame, phone_index] = previous_score + phone_scores[frame, phone_index]
            backpointers[frame, phone_index] = previous_phone

    if not np.isfinite(best[-1, -1]):
        raise ValueError("Unable to decode a monotonic phone path from emissions")

    path = np.zeros((frame_count,), dtype=np.int32)
    phone_index = phone_count - 1
    for frame in range(frame_count - 1, -1, -1):
        path[frame] = phone_index
        phone_index = int(backpointers[frame, phone_index])
        if phone_index < 0:
            break

    segments: list[MonotonicPhoneSegment] = []
    start = 0
    for frame in range(1, frame_count + 1):
        if frame == frame_count or path[frame] != path[start]:
            current_phone_index = int(path[start])
            segment_scores = phone_scores[start:frame, current_phone_index]
            confidence = float(np.exp(np.mean(segment_scores)))
            segments.append(
                MonotonicPhoneSegment(
                    phoneme=phones[current_phone_index],
                    start_frame=start,
                    end_frame=frame,
                    confidence=max(PHONE_CTC_ALIGNMENT_CONFIDENCE_FLOOR, min(0.99, confidence)),
                )
            )
            start = frame

    return segments


def segments_to_phone_spans(
    segments: list[MonotonicPhoneSegment],
    encoded: EncodedFrames,
) -> list[PhoneSpan]:
    frame_count = max(1, len(encoded.embeddings))
    expected_weights = [phone_duration_weight(segment.phoneme) for segment in segments]
    expected_total = max(sum(expected_weights), 1e-6)

    spans: list[PhoneSpan] = []
    for index, segment in enumerate(segments):
        start_frame = max(0, min(frame_count - 1, segment.start_frame))
        end_frame = max(start_frame + 1, min(frame_count, segment.end_frame))
        observed_frames = max(1, end_frame - start_frame)
        expected_frames = max(1.0, frame_count * (expected_weights[index] / expected_total))
        duration_z = (observed_frames - expected_frames) / max(1.0, expected_frames * 0.35)
        spans.append(
            PhoneSpan(
                phoneme=segment.phoneme,
                start_frame=start_frame,
                end_frame=end_frame,
                start_ms=int(round(start_frame * encoded.frame_ms)),
                end_ms=int(round(end_frame * encoded.frame_ms)),
                alignment_confidence=round(segment.confidence, 3),
                duration_z_score=round(float(duration_z), 3),
            )
        )
    return spans


def decode_phone_ctc_spans(
    emissions: FrameEmissions,
    entry: LexiconEntry,
    encoded: EncodedFrames,
    vocabulary: PhoneVocabulary,
) -> list[PhoneSpan]:
    segments = decode_monotonic_phone_segments(emissions, entry.phones, vocabulary)
    return segments_to_phone_spans(segments, encoded)


@dataclass
class PhoneCtcAligner:
    """Experimental in-process phone aligner boundary.

    The request-time contract is ready, but model inference intentionally remains
    behind an explicit checkpoint-backed implementation.
    """

    checkpoint_path: Path | None
    vocabulary: PhoneVocabulary
    emission_model: PhoneEmissionModel | None = None

    def align(self, entry: LexiconEntry, prepared: PreparedAudio, encoded: EncodedFrames) -> list[PhoneSpan]:
        emissions = self._infer_emissions(entry, prepared, encoded)
        return decode_phone_ctc_spans(emissions, entry, encoded, self.vocabulary)

    def _infer_emissions(
        self,
        entry: LexiconEntry,
        prepared: PreparedAudio,
        encoded: EncodedFrames,
    ) -> FrameEmissions:
        del entry
        if self.checkpoint_path is None:
            raise AlignmentUnavailableError(
                "Experimental phone CTC aligner requires PRONUNCIATION_PHONE_CTC_CHECKPOINT_PATH."
            )
        if self.emission_model is None:
            raise NotImplementedError(
                "Phone CTC inference is not implemented yet. Train a phone-emission head, load it from "
                "PRONUNCIATION_PHONE_CTC_CHECKPOINT_PATH, and provide a PhoneEmissionModel implementation."
            )
        return self.emission_model.infer(prepared, encoded)
