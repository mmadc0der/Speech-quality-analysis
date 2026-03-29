from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

from pronunciation_backend.config import Settings, settings
from pronunciation_backend.models import PronunciationAssessmentResponse
from pronunciation_backend.services.aligner import ConstrainedPhonemeAligner, PhoneFeatureBuilder
from pronunciation_backend.services.audio_prep import AudioPrepService, AudioValidationError
from pronunciation_backend.services.feature_encoder import SSLFeatureEncoder
from pronunciation_backend.services.lexicon import LexiconService, UnknownWordError
from pronunciation_backend.services.pipeline import PronunciationPipeline
from pronunciation_backend.services.reference import ReferenceAudioService
from pronunciation_backend.services.response_mapper import ResponseMapper
from pronunciation_backend.services.scorer_v2_runtime import ScorerV2Runtime


def build_pipeline(active_settings: Settings) -> PronunciationPipeline:
    active_settings.validate_runtime()
    scorer_runtime = ScorerV2Runtime(
        checkpoint_path=active_settings.scorer_checkpoint_path,
        backbone_id=active_settings.backbone_id,
        device=active_settings.scorer_device,
        strict_load=active_settings.scorer_strict_load,
    )
    return PronunciationPipeline(
        lexicon_service=LexiconService(active_settings.lexicon_path),
        reference_audio_service=ReferenceAudioService(active_settings.reference_manifest_path),
        audio_prep_service=AudioPrepService(active_settings),
        feature_encoder=SSLFeatureEncoder(active_settings),
        aligner=ConstrainedPhonemeAligner(),
        feature_builder=PhoneFeatureBuilder(),
        scorer_runtime=scorer_runtime,
        response_mapper=ResponseMapper(),
    )


def create_app(
    *,
    settings_override: Settings | None = None,
    pipeline_override: PronunciationPipeline | None = None,
) -> FastAPI:
    active_settings = settings_override or settings

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline_override or build_pipeline(active_settings)
        yield

    app = FastAPI(
        title="Pronunciation Backend MVP",
        version="0.2.0",
        description="Word-level American English pronunciation assessment backend.",
        lifespan=lifespan,
    )

    def get_pipeline_from_request(request: Request) -> PronunciationPipeline:
        pipeline = getattr(request.app.state, "pipeline", None)
        if pipeline is None:
            raise RuntimeError("Pronunciation pipeline is not initialized")
        return pipeline

    @app.get("/health")
    def health(request: Request) -> dict[str, object]:
        pipeline = get_pipeline_from_request(request)
        model_info = pipeline.model_info()
        return {
            "status": "ok",
            "model_ready": True,
            "runtime_backend": model_info.runtime_backend,
            "model_version": model_info.model_version,
            "backbone_id": model_info.backbone_id,
            "device": model_info.device,
        }

    @app.get("/v1/words")
    def supported_words(request: Request) -> dict[str, list[str]]:
        return {"words": get_pipeline_from_request(request).lexicon_service.all_words()}

    @app.post("/v1/pronunciation/score", response_model=PronunciationAssessmentResponse)
    async def score_pronunciation(
        request: Request,
        word: str = Form(...),
        audio: UploadFile = File(...),
        speaker_id: str | None = Form(default=None),
        no_trim: bool = Form(default=False, alias="noTrim"),
    ) -> PronunciationAssessmentResponse:
        del speaker_id  # reserved for future personalization
        try:
            audio_bytes = await audio.read()
            return get_pipeline_from_request(request).assess_word(word=word, audio_bytes=audio_bytes, no_trim=no_trim)
        except UnknownWordError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except AudioValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()
