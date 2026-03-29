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


def test_audio_prep_trims_to_detected_word_region() -> None:
    sample_rate = 16_000
    silence_a = np.zeros((sample_rate,), dtype=np.float32)
    t = np.linspace(0, 0.4, int(sample_rate * 0.4), endpoint=False)
    speech = 0.3 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    silence_b = np.zeros((int(sample_rate * 0.6),), dtype=np.float32)
    audio_bytes = _wav_bytes(np.concatenate([silence_a, speech, silence_b]))

    prepared = AudioPrepService(Settings()).decode(audio_bytes, enable_trim=True)

    assert prepared.trim_applied is True
    assert prepared.original_duration_ms == 2000
    assert 800 <= prepared.trim_start_ms <= 1100
    assert 1300 <= prepared.trim_end_ms <= 1600
    assert prepared.duration_ms < prepared.original_duration_ms


def test_audio_prep_can_skip_trimming() -> None:
    sample_rate = 16_000
    samples = np.concatenate(
        [
            np.zeros((sample_rate,), dtype=np.float32),
            0.3 * np.sin(2 * np.pi * 220 * np.linspace(0, 0.4, int(sample_rate * 0.4), endpoint=False)).astype(np.float32),
            np.zeros((int(sample_rate * 0.6),), dtype=np.float32),
        ]
    )

    prepared = AudioPrepService(Settings()).decode(_wav_bytes(samples), enable_trim=False)

    assert prepared.trim_applied is False
    assert prepared.trim_start_ms == 0
    assert prepared.trim_end_ms == 2000
    assert prepared.duration_ms == 2000
