from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pronunciation_backend.models import EncodedFrames, LexiconEntry, PhoneFeatures, PhoneSpan

VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}
FRICATIVES = {"DH", "F", "S", "SH", "TH", "V", "Z", "ZH", "HH"}
STOPS = {"B", "D", "G", "K", "P", "T"}
AFFRICATES = {"CH", "JH"}
NASALS = {"M", "N", "NG"}
LIQUIDS = {"L", "R"}
GLIDES = {"W", "Y"}


def phone_duration_weight(phone: str) -> float:
    if phone in VOWELS:
        return 1.8
    if phone in FRICATIVES:
        return 1.35
    if phone in AFFRICATES:
        return 1.25
    if phone in STOPS:
        return 0.85
    if phone in NASALS:
        return 1.1
    if phone in LIQUIDS or phone in GLIDES:
        return 1.0
    return 1.0


@dataclass
class ConstrainedPhonemeAligner:
    """Heuristic aligner that partitions frames over a known phone sequence."""

    def align(self, entry: LexiconEntry, encoded: EncodedFrames) -> list[PhoneSpan]:
        frame_count = max(1, len(encoded.embeddings))
        phones = entry.phones
        weights = np.array([phone_duration_weight(phone) for phone in phones], dtype=np.float32)
        proportions = weights / weights.sum()
        raw_lengths = np.maximum(1, np.floor(proportions * frame_count).astype(int))

        delta = frame_count - int(raw_lengths.sum())
        index = 0
        while delta != 0:
            target = index % len(raw_lengths)
            if delta > 0:
                raw_lengths[target] += 1
                delta -= 1
            elif raw_lengths[target] > 1:
                raw_lengths[target] -= 1
                delta += 1
            index += 1

        spans: list[PhoneSpan] = []
        cursor = 0
        mean_energy = float(np.mean(encoded.energy)) if encoded.energy.size > 0 else 0.0
        for phone, length in zip(phones, raw_lengths):
            start_frame = cursor
            end_frame = min(frame_count, cursor + int(length))
            if end_frame <= start_frame:
                end_frame = min(frame_count, start_frame + 1)

            segment_energy = encoded.energy[start_frame:end_frame]
            if segment_energy.size == 0:
                segment_energy = np.asarray([mean_energy], dtype=np.float32)
            energy_ratio = float(np.mean(segment_energy) / (mean_energy + 1e-6)) if mean_energy else 1.0
            expected_weight = phone_duration_weight(phone)
            observed_weight = max(0.5, (end_frame - start_frame) / max(1.0, frame_count / len(phones)))
            duration_z = float((observed_weight - expected_weight) / max(0.35, expected_weight))
            confidence = max(0.4, min(0.98, 0.72 + 0.18 * min(energy_ratio, 1.2)))

            spans.append(
                PhoneSpan(
                    phoneme=phone,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_ms=int(round(start_frame * encoded.frame_ms)),
                    end_ms=int(round(end_frame * encoded.frame_ms)),
                    alignment_confidence=round(confidence, 3),
                    duration_z_score=round(duration_z, 3),
                )
            )
            cursor = end_frame

        if spans:
            last = spans[-1]
            spans[-1] = PhoneSpan(
                phoneme=last.phoneme,
                start_frame=last.start_frame,
                end_frame=frame_count,
                start_ms=last.start_ms,
                end_ms=int(round(frame_count * encoded.frame_ms)),
                alignment_confidence=last.alignment_confidence,
                duration_z_score=last.duration_z_score,
            )

        return spans


@dataclass
class PhoneFeatureBuilder:
    def build(self, encoded: EncodedFrames, spans: list[PhoneSpan]) -> list[PhoneFeatures]:
        frame_array = np.asarray(encoded.embeddings, dtype=np.float32)
        energy_array = np.asarray(encoded.energy, dtype=np.float32) if encoded.energy.size > 0 else np.zeros((1,), dtype=np.float32)
        late_threshold_ms = encoded.frame_ms * 1.5

        features: list[PhoneFeatures] = []
        for index, span in enumerate(spans):
            segment = frame_array[span.start_frame : span.end_frame]
            if segment.size == 0:
                segment = np.zeros((1, frame_array.shape[1]), dtype=np.float32)
            segment_energy = energy_array[span.start_frame : span.end_frame]
            if segment_energy.size == 0:
                segment_energy = np.zeros((1,), dtype=np.float32)

            features.append(
                PhoneFeatures(
                    phoneme=span.phoneme,
                    start_ms=span.start_ms,
                    end_ms=span.end_ms,
                    mean_embedding=segment.mean(axis=0).astype(np.float32).tolist(),
                    variance=float(segment.var()),
                    duration_ms=max(1, span.end_ms - span.start_ms),
                    duration_z_score=span.duration_z_score,
                    alignment_confidence=span.alignment_confidence,
                    energy_mean=float(segment_energy.mean()),
                    starts_late=index > 0 and span.start_ms - spans[index - 1].end_ms > late_threshold_ms,
                )
            )
        return features
