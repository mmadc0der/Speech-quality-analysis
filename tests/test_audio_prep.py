from __future__ import annotations

import io
import wave

import numpy as np

from pronunciation_backend.config import Settings
from pronunciation_backend.services.audio_prep import AudioPrepService


def _wav_bytes(samples: np.ndarray, sample_rate: int = 16_000) -> bytes:
    pcm = np.int16(np.clip(samples, -1.0, 1.0) * 32767)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def _sine(duration_s: float, *, amplitude: float = 0.3, frequency_hz: float = 220.0, sample_rate: int = 16_000) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * frequency_hz * t)).astype(np.float32)


def _noise(duration_s: float, *, amplitude: float = 0.08, sample_rate: int = 16_000) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(0.0, amplitude, size=int(sample_rate * duration_s)).astype(np.float32)


def test_audio_prep_trims_to_detected_word_region() -> None:
    sample_rate = 16_000
    silence_a = np.zeros((sample_rate,), dtype=np.float32)
    speech = _sine(0.4, sample_rate=sample_rate)
    silence_b = np.zeros((int(sample_rate * 0.6),), dtype=np.float32)
    audio_bytes = _wav_bytes(np.concatenate([silence_a, speech, silence_b]))

    prepared = AudioPrepService(Settings()).decode(audio_bytes, enable_trim=True)

    assert prepared.trim_applied is True
    assert prepared.original_duration_ms == 2000
    assert 760 <= prepared.trim_start_ms <= 1100
    assert 1300 <= prepared.trim_end_ms <= 1650
    assert prepared.duration_ms < prepared.original_duration_ms


def test_audio_prep_can_skip_trimming() -> None:
    sample_rate = 16_000
    samples = np.concatenate(
        [
            np.zeros((sample_rate,), dtype=np.float32),
            _sine(0.4, sample_rate=sample_rate),
            np.zeros((int(sample_rate * 0.6),), dtype=np.float32),
        ]
    )

    prepared = AudioPrepService(Settings()).decode(_wav_bytes(samples), enable_trim=False)

    assert prepared.trim_applied is False
    assert prepared.trim_start_ms == 0
    assert prepared.trim_end_ms == 2000
    assert prepared.duration_ms == 2000


def test_audio_prep_trims_long_leading_silence_before_late_word() -> None:
    settings = Settings(max_audio_ms=4000)
    samples = np.concatenate(
        [
            np.zeros((int(settings.sample_rate * 2.8),), dtype=np.float32),
            _sine(0.35, amplitude=0.32, sample_rate=settings.sample_rate),
            np.zeros((int(settings.sample_rate * 0.45),), dtype=np.float32),
        ]
    )

    prepared = AudioPrepService(settings).decode(_wav_bytes(samples, settings.sample_rate), enable_trim=True)

    assert prepared.trim_applied is True
    assert prepared.original_duration_ms == 3600
    assert prepared.trim_start_ms >= 2400
    assert prepared.trim_end_ms <= 3600
    assert prepared.duration_ms < 1200


def test_audio_prep_preserves_weak_trailing_burst() -> None:
    settings = Settings()
    samples = np.concatenate(
        [
            np.zeros((int(settings.sample_rate * 0.35),), dtype=np.float32),
            _sine(0.24, amplitude=0.28, sample_rate=settings.sample_rate),
            np.zeros((int(settings.sample_rate * 0.08),), dtype=np.float32),
            _noise(0.035, amplitude=0.045, sample_rate=settings.sample_rate),
            np.zeros((int(settings.sample_rate * 0.45),), dtype=np.float32),
        ]
    )

    prepared = AudioPrepService(settings).decode(_wav_bytes(samples, settings.sample_rate), enable_trim=True)

    assert prepared.trim_applied is True
    assert prepared.trim_start_ms <= 420
    assert prepared.trim_end_ms >= 840
    assert prepared.duration_ms >= 420


def test_audio_prep_merges_close_speech_islands() -> None:
    settings = Settings()
    samples = np.concatenate(
        [
            np.zeros((int(settings.sample_rate * 0.3),), dtype=np.float32),
            _sine(0.1, amplitude=0.18, frequency_hz=180.0, sample_rate=settings.sample_rate),
            np.zeros((int(settings.sample_rate * 0.12),), dtype=np.float32),
            _sine(0.18, amplitude=0.32, frequency_hz=260.0, sample_rate=settings.sample_rate),
            np.zeros((int(settings.sample_rate * 0.6),), dtype=np.float32),
        ]
    )

    prepared = AudioPrepService(settings).decode(_wav_bytes(samples, settings.sample_rate), enable_trim=True)

    assert prepared.trim_applied is True
    assert prepared.trim_start_ms <= 320
    assert prepared.trim_end_ms >= 700
    assert prepared.duration_ms < prepared.original_duration_ms


def test_audio_prep_falls_back_when_trimmed_window_becomes_too_short() -> None:
    settings = Settings(min_audio_ms=250)
    samples = np.concatenate(
        [
            np.zeros((int(settings.sample_rate * 0.11),), dtype=np.float32),
            _sine(0.03, amplitude=0.4, sample_rate=settings.sample_rate),
            np.zeros((int(settings.sample_rate * 0.12),), dtype=np.float32),
        ]
    )

    prepared = AudioPrepService(settings).decode(_wav_bytes(samples, settings.sample_rate), enable_trim=True)

    assert prepared.original_duration_ms == 260
    assert prepared.trim_applied is False
    assert prepared.trim_start_ms == 0
    assert prepared.trim_end_ms == 260
    assert prepared.duration_ms == 260
