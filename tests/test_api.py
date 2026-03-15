from __future__ import annotations

import io
import wave

import numpy as np
from fastapi.testclient import TestClient

from pronunciation_backend.main import app


client = TestClient(app)


def _sine_wave_bytes(duration_ms: int = 700, sample_rate: int = 16_000) -> bytes:
    t = np.linspace(0, duration_ms / 1000.0, int(sample_rate * duration_ms / 1000.0), endpoint=False)
    signal = 0.2 * np.sin(2 * np.pi * 220 * t)
    pcm = np.int16(signal * 32767)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_supported_words() -> None:
    response = client.get("/v1/words")
    assert response.status_code == 200
    assert "thought" in response.json()["words"]


def test_score_pronunciation() -> None:
    response = client.post(
        "/v1/pronunciation/score",
        data={"word": "thought"},
        files={"audio": ("sample.wav", _sine_wave_bytes(), "audio/wav")},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["word"] == "thought"
    assert payload["accent_target"] == "en-US"
    assert payload["primary_issue"]["phoneme"]
    assert len(payload["phonemes"]) == 3
