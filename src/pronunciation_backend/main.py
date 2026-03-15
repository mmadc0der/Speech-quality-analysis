from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from pronunciation_backend.config import settings
from pronunciation_backend.models import PronunciationAssessmentResponse
from pronunciation_backend.services.aligner import ConstrainedPhonemeAligner, PhoneFeatureBuilder
from pronunciation_backend.services.audio_prep import AudioPrepService, AudioValidationError
from pronunciation_backend.services.feature_encoder import SSLFeatureEncoder
from pronunciation_backend.services.lexicon import LexiconService, UnknownWordError
from pronunciation_backend.services.pipeline import PronunciationPipeline
from pronunciation_backend.services.reference import ReferenceAudioService
from pronunciation_backend.services.scoring import CalibrationAndIssueService, PhoneScoringHead


@lru_cache(maxsize=1)
def get_pipeline() -> PronunciationPipeline:
    return PronunciationPipeline(
        lexicon_service=LexiconService(settings.lexicon_path),
        reference_audio_service=ReferenceAudioService(settings.reference_manifest_path),
        audio_prep_service=AudioPrepService(settings),
        feature_encoder=SSLFeatureEncoder(settings),
        aligner=ConstrainedPhonemeAligner(),
        feature_builder=PhoneFeatureBuilder(),
        scoring_head=PhoneScoringHead(),
        calibration_service=CalibrationAndIssueService(),
    )


app = FastAPI(
    title="Pronunciation Backend MVP",
    version="0.1.0",
    description="Word-level American English pronunciation assessment backend.",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/words")
def supported_words() -> dict[str, list[str]]:
    return {"words": get_pipeline().lexicon_service.all_words()}


@app.post("/v1/pronunciation/score", response_model=PronunciationAssessmentResponse)
async def score_pronunciation(
    word: str = Form(...),
    audio: UploadFile = File(...),
    speaker_id: str | None = Form(default=None),
) -> PronunciationAssessmentResponse:
    del speaker_id  # reserved for future personalization
    try:
        audio_bytes = await audio.read()
        return get_pipeline().assess_word(word=word, audio_bytes=audio_bytes)
    except UnknownWordError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AudioValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
