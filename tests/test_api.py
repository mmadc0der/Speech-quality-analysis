from __future__ import annotations

import io
import wave
from dataclasses import dataclass

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pronunciation_backend.main import create_app
from pronunciation_backend.models import (
    AudioQualityPayload,
    ModelInfoPayload,
    PrimaryIssuePayload,
    PronunciationAssessmentResponse,
    PronunciationPhonePayload,
    QualityClassProbabilitiesPayload,
    ReferencePayload,
)
from pronunciation_backend.services.scorer_runtime import ScorerModelInfo


@dataclass
class _FakeLexiconService:
    def all_words(self) -> list[str]:
        return ["thought", "through"]


@dataclass
class _FakePipeline:
    lexicon_service: _FakeLexiconService
    last_no_trim: bool | None = None

    def model_info(self) -> ScorerModelInfo:
        return ScorerModelInfo(
            runtime_backend="scorer_v2",
            model_version="v2",
            checkpoint_name="fake.pt",
            backbone_id="facebook/hubert-base-ls960",
            device="cpu",
            class_labels=("wrong_or_missed", "accented", "correct"),
        )

    def assess_word(self, word: str, audio_bytes: bytes, *, no_trim: bool = False) -> PronunciationAssessmentResponse:
        assert audio_bytes
        self.last_no_trim = no_trim
        return PronunciationAssessmentResponse(
            word=word,
            ipa="θɔt",
            overall_score=81.25,
            confidence=0.91,
            audio_quality=AudioQualityPayload(
                status="ok",
                snr_estimate=24.0,
                duration_ms=700,
                rms=0.2,
                clipping_ratio=0.0,
                silence_ratio=0.1,
                original_duration_ms=700,
                trim_start_ms=0,
                trim_end_ms=700,
                trim_applied=False,
            ),
            phonemes=[
                PronunciationPhonePayload(
                    phoneme="TH",
                    start_ms=0,
                    end_ms=120,
                    expected_score=62.5,
                    expected_human_score=1.1,
                    omission_probability=0.02,
                    confidence=0.88,
                    alignment_confidence=0.93,
                    predicted_class="accented",
                    quality_class_probs=QualityClassProbabilitiesPayload(
                        wrong_or_missed=0.11,
                        accented=0.72,
                        correct=0.17,
                    ),
                ),
                PronunciationPhonePayload(
                    phoneme="AO",
                    start_ms=120,
                    end_ms=340,
                    expected_score=88.0,
                    expected_human_score=1.9,
                    omission_probability=0.01,
                    confidence=0.93,
                    alignment_confidence=0.96,
                    predicted_class="correct",
                    quality_class_probs=QualityClassProbabilitiesPayload(
                        wrong_or_missed=0.02,
                        accented=0.08,
                        correct=0.9,
                    ),
                ),
                PronunciationPhonePayload(
                    phoneme="T",
                    start_ms=340,
                    end_ms=520,
                    expected_score=84.1,
                    expected_human_score=1.8,
                    omission_probability=0.03,
                    confidence=0.92,
                    alignment_confidence=0.94,
                    predicted_class="correct",
                    quality_class_probs=QualityClassProbabilitiesPayload(
                        wrong_or_missed=0.03,
                        accented=0.11,
                        correct=0.86,
                    ),
                ),
            ],
            primary_issue=PrimaryIssuePayload(
                phoneme="TH",
                type="accented",
                message="phoneme TH is the main candidate for correction",
            ),
            reference=ReferencePayload(
                ipa="θɔt",
                audio_id="thought_en_us_01",
                asset_path="assets/reference_audio/thought_en_us_01.wav",
            ),
            model_info=ModelInfoPayload(
                runtime_backend="scorer_v2",
                model_version="v2",
                checkpoint_name="fake.pt",
                backbone_id="facebook/hubert-base-ls960",
                device="cpu",
                class_labels=["wrong_or_missed", "accented", "correct"],
            ),
        )


@pytest.fixture
def client() -> TestClient:
    with TestClient(create_app(pipeline_override=_FakePipeline(lexicon_service=_FakeLexiconService()))) as test_client:
        yield test_client


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


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_ready"] is True
    assert response.json()["runtime_backend"] == "scorer_v2"


def test_supported_words(client: TestClient) -> None:
    response = client.get("/v1/words")
    assert response.status_code == 200
    assert "thought" in response.json()["words"]


def test_score_pronunciation(client: TestClient) -> None:
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
    assert payload["phonemes"][0]["predicted_class"] == "accented"
    assert payload["model_info"]["runtime_backend"] == "scorer_v2"
    assert payload["audio_quality"]["trim_applied"] is False


def test_score_pronunciation_forwards_no_trim(client: TestClient) -> None:
    pipeline = client.app.state.pipeline
    response = client.post(
        "/v1/pronunciation/score",
        data={"word": "thought", "noTrim": "true"},
        files={"audio": ("sample.wav", _sine_wave_bytes(), "audio/wav")},
    )
    assert response.status_code == 200
    assert pipeline.last_no_trim is True
