from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from pronunciation_backend.config import Settings
from pronunciation_backend.models import PreparedAudio


class AudioValidationError(ValueError):
    """Raised when uploaded audio cannot be decoded or is unusable."""


@dataclass
class AudioPrepService:
    settings: Settings

    def decode(self, audio_bytes: bytes, *, enable_trim: bool = True) -> PreparedAudio:
        if not audio_bytes:
            raise AudioValidationError("Empty audio payload.")

        try:
            samples, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        except RuntimeError as exc:
            raise AudioValidationError("Unsupported or invalid audio file.") from exc

        return self._prepare_audio(samples, sample_rate, enable_trim=enable_trim)

    def decode_path(self, path: str | Path, *, enable_trim: bool = True) -> PreparedAudio:
        try:
            samples, sample_rate = sf.read(Path(path), dtype="float32", always_2d=False)
        except RuntimeError as exc:
            raise AudioValidationError("Unsupported or invalid audio file.") from exc
        except OSError as exc:
            raise AudioValidationError(f"Unable to read audio file: {path}") from exc

        return self._prepare_audio(samples, sample_rate, enable_trim=enable_trim)

    def _prepare_audio(self, samples: np.ndarray, sample_rate: int, *, enable_trim: bool) -> PreparedAudio:
        mono = self._to_mono(samples)
        resampled = self._resample(mono, sample_rate, self.settings.sample_rate)
        normalized = self._normalize(resampled)

        original_duration_ms = int(round((len(normalized) / self.settings.sample_rate) * 1000))
        if original_duration_ms < self.settings.min_audio_ms:
            raise AudioValidationError("Audio is too short for pronunciation scoring.")
        if original_duration_ms > self.settings.max_audio_ms:
            raise AudioValidationError("Audio is too long for word-level pronunciation scoring.")

        prepared_samples = normalized
        trim_start_ms = 0
        trim_end_ms = original_duration_ms
        trim_applied = False
        if enable_trim:
            trim_range = self._detect_word_region(normalized)
            if trim_range is not None:
                trim_start_sample, trim_end_sample = trim_range
                prepared_samples = normalized[trim_start_sample:trim_end_sample]
                trim_start_ms = self._sample_to_ms(trim_start_sample)
                trim_end_ms = self._sample_to_ms(trim_end_sample)
                trim_applied = trim_start_sample > 0 or trim_end_sample < len(normalized)

        duration_ms = int(round((len(prepared_samples) / self.settings.sample_rate) * 1000))
        if duration_ms < self.settings.min_audio_ms:
            prepared_samples = normalized
            duration_ms = original_duration_ms
            trim_start_ms = 0
            trim_end_ms = original_duration_ms
            trim_applied = False

        rms = float(np.sqrt(np.mean(np.square(prepared_samples))) if len(prepared_samples) else 0.0)
        clipping_ratio = float(np.mean(np.abs(prepared_samples) >= self.settings.clipping_threshold))
        silence_ratio = float(np.mean(np.abs(prepared_samples) < self.settings.silence_threshold))
        snr_estimate = self._estimate_snr(prepared_samples)
        quality_status = self._quality_status(rms, clipping_ratio, silence_ratio)

        return PreparedAudio(
            samples=prepared_samples.astype(np.float32),
            sample_rate=self.settings.sample_rate,
            duration_ms=duration_ms,
            rms=rms,
            clipping_ratio=clipping_ratio,
            silence_ratio=silence_ratio,
            snr_estimate=snr_estimate,
            quality_status=quality_status,
            original_duration_ms=original_duration_ms,
            trim_start_ms=trim_start_ms,
            trim_end_ms=trim_end_ms,
            trim_applied=trim_applied,
        )

    def _to_mono(self, samples: np.ndarray) -> np.ndarray:
        if samples.ndim == 1:
            return samples
        return samples.mean(axis=1)

    def _resample(self, samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
        if source_rate == target_rate:
            return samples
        duration = len(samples) / float(source_rate)
        source_points = np.linspace(0, duration, num=len(samples), endpoint=False)
        target_length = max(1, int(round(duration * target_rate)))
        target_points = np.linspace(0, duration, num=target_length, endpoint=False)
        return np.interp(target_points, source_points, samples).astype(np.float32)

    def _normalize(self, samples: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(samples))) if len(samples) else 0.0
        if peak == 0:
            return samples.astype(np.float32)
        return (samples / peak).astype(np.float32)

    def _detect_word_region(self, samples: np.ndarray) -> tuple[int, int] | None:
        if len(samples) == 0:
            return None

        frame_size = max(1, int(round(self.settings.sample_rate * 0.02)))
        hop_size = max(1, int(round(self.settings.sample_rate * 0.01)))
        seconds_per_hop = hop_size / self.settings.sample_rate
        pad_frames = max(1, int(round(0.18 / seconds_per_hop)))
        merge_gap_frames = max(1, int(round(0.18 / seconds_per_hop)))
        min_active_frames = max(1, int(round(0.04 / seconds_per_hop)))
        min_keep_ms = 180

        rms_values: list[float] = []
        frame_starts: list[int] = []
        for start in range(0, len(samples), hop_size):
            window = samples[start : start + frame_size]
            if len(window) == 0:
                continue
            rms_values.append(float(np.sqrt(np.mean(np.square(window)))))
            frame_starts.append(start)

        if not rms_values:
            return None

        rms_array = np.asarray(rms_values, dtype=np.float32)
        peak_rms = float(rms_array.max())
        if peak_rms <= self.settings.silence_threshold:
            return None

        smoothed_rms = self._moving_average(rms_array, window_size=max(3, int(round(0.05 / seconds_per_hop))))
        low_threshold = max(self.settings.silence_threshold * 1.25, peak_rms * 0.08)
        strong_threshold = max(self.settings.silence_threshold * 2.0, peak_rms * 0.18)
        active = smoothed_rms >= low_threshold
        if not np.any(active):
            return None

        active_ranges = self._collect_active_ranges(
            active_mask=active,
            merge_gap_frames=merge_gap_frames,
            min_active_frames=min_active_frames,
        )
        if not active_ranges:
            return None

        range_scores = [
            float(np.sum(smoothed_rms[start:end]) * max(1, end - start))
            for start, end in active_ranges
        ]
        max_range_score = max(range_scores)

        significant_ranges: list[tuple[int, int]] = []
        for index, (start, end) in enumerate(active_ranges):
            range_peak = float(smoothed_rms[start:end].max()) if end > start else 0.0
            range_score = range_scores[index]
            if range_peak >= strong_threshold or range_score >= max_range_score * 0.2:
                significant_ranges.append((start, end))

        if not significant_ranges:
            significant_ranges = [active_ranges[int(np.argmax(range_scores))]]

        padded_ranges = [
            (max(0, start - pad_frames), min(len(frame_starts), end + pad_frames))
            for start, end in significant_ranges
        ]
        merged_ranges = self._merge_ranges(padded_ranges, merge_gap_frames=merge_gap_frames)
        if not merged_ranges:
            return None

        best_start = min(start for start, _ in merged_ranges)
        best_end = max(end for _, end in merged_ranges)

        start_sample = frame_starts[best_start]
        end_frame_start = frame_starts[max(0, best_end - 1)]
        end_sample = min(len(samples), end_frame_start + frame_size)
        if end_sample <= start_sample:
            return None
        trimmed_duration_ms = self._sample_to_ms(end_sample - start_sample)
        removed_ms = self._sample_to_ms(start_sample) + self._sample_to_ms(len(samples) - end_sample)
        if trimmed_duration_ms < min_keep_ms:
            return None
        if removed_ms < max(120, int(round(self._sample_to_ms(len(samples)) * 0.12))):
            return None
        return start_sample, end_sample

    def _moving_average(self, values: np.ndarray, *, window_size: int) -> np.ndarray:
        if len(values) == 0 or window_size <= 1:
            return values
        kernel = np.ones(window_size, dtype=np.float32) / float(window_size)
        return np.convolve(values, kernel, mode="same")

    def _collect_active_ranges(
        self,
        *,
        active_mask: np.ndarray,
        merge_gap_frames: int,
        min_active_frames: int,
    ) -> list[tuple[int, int]]:
        active_ranges: list[tuple[int, int]] = []
        start_index: int | None = None
        gap = 0
        for index, is_active in enumerate(active_mask.tolist()):
            if is_active:
                if start_index is None:
                    start_index = index
                gap = 0
                continue
            if start_index is None:
                continue
            gap += 1
            if gap > merge_gap_frames:
                end_index = index - gap + 1
                if end_index - start_index >= min_active_frames:
                    active_ranges.append((start_index, end_index))
                start_index = None
                gap = 0
        if start_index is not None:
            end_index = len(active_mask)
            if end_index - start_index >= min_active_frames:
                active_ranges.append((start_index, end_index))
        return active_ranges

    def _merge_ranges(self, ranges: list[tuple[int, int]], *, merge_gap_frames: int) -> list[tuple[int, int]]:
        if not ranges:
            return []
        merged: list[tuple[int, int]] = []
        current_start, current_end = sorted(ranges)[0]
        for start, end in sorted(ranges)[1:]:
            if start - current_end <= merge_gap_frames:
                current_end = max(current_end, end)
                continue
            merged.append((current_start, current_end))
            current_start, current_end = start, end
        merged.append((current_start, current_end))
        return merged

    def _sample_to_ms(self, sample_index: int) -> int:
        return int(round((sample_index / self.settings.sample_rate) * 1000))

    def _estimate_snr(self, samples: np.ndarray) -> float:
        if len(samples) < 2:
            return 0.0
        signal_power = float(np.mean(np.square(samples)))
        diff = np.diff(samples, prepend=samples[:1])
        noise_power = float(np.mean(np.square(diff))) + 1e-8
        snr = 10.0 * np.log10((signal_power + 1e-8) / noise_power)
        return float(max(0.0, round(snr, 2)))

    def _quality_status(self, rms: float, clipping_ratio: float, silence_ratio: float) -> str:
        if rms < 0.04 or silence_ratio > 0.75:
            return "rejected"
        if clipping_ratio > 0.08 or silence_ratio > 0.45:
            return "low_confidence"
        return "ok"
