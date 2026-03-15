from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import soundfile as sf

from pronunciation_backend.config import Settings
from pronunciation_backend.models import PreparedAudio


class AudioValidationError(ValueError):
    """Raised when uploaded audio cannot be decoded or is unusable."""


@dataclass
class AudioPrepService:
    settings: Settings

    def decode(self, audio_bytes: bytes) -> PreparedAudio:
        if not audio_bytes:
            raise AudioValidationError("Empty audio payload.")

        try:
            samples, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        except RuntimeError as exc:
            raise AudioValidationError("Unsupported or invalid audio file.") from exc

        mono = self._to_mono(samples)
        resampled = self._resample(mono, sample_rate, self.settings.sample_rate)
        normalized = self._normalize(resampled)

        duration_ms = int(round((len(normalized) / self.settings.sample_rate) * 1000))
        if duration_ms < self.settings.min_audio_ms:
            raise AudioValidationError("Audio is too short for pronunciation scoring.")
        if duration_ms > self.settings.max_audio_ms:
            raise AudioValidationError("Audio is too long for word-level pronunciation scoring.")

        rms = float(np.sqrt(np.mean(np.square(normalized))) if len(normalized) else 0.0)
        clipping_ratio = float(np.mean(np.abs(normalized) >= self.settings.clipping_threshold))
        silence_ratio = float(np.mean(np.abs(normalized) < self.settings.silence_threshold))
        snr_estimate = self._estimate_snr(normalized)
        quality_status = self._quality_status(rms, clipping_ratio, silence_ratio)

        return PreparedAudio(
            samples=normalized.astype(np.float32).tolist(),
            sample_rate=self.settings.sample_rate,
            duration_ms=duration_ms,
            rms=rms,
            clipping_ratio=clipping_ratio,
            silence_ratio=silence_ratio,
            snr_estimate=snr_estimate,
            quality_status=quality_status,
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
