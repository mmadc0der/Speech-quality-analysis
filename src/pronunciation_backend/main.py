from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile

from pronunciation_backend.config import Settings, settings
from pronunciation_backend.models import PronunciationAssessmentResponse
from pronunciation_backend.services.aligner import PhoneFeatureBuilder
from pronunciation_backend.services.audio_prep import AudioPrepService, AudioValidationError
from pronunciation_backend.services.feature_encoder import SSLFeatureEncoder
from pronunciation_backend.services.lexicon import LexiconService, UnknownWordError
from pronunciation_backend.services.mfa_aligner import AlignmentError, MfaForcedAligner
from pronunciation_backend.services.phone_ctc_aligner import PhoneCtcAligner, PhoneVocabulary
from pronunciation_backend.services.pipeline import PronunciationPipeline
from pronunciation_backend.services.reference import ReferenceAudioService
from pronunciation_backend.services.response_mapper import ResponseMapper
from pronunciation_backend.services.scorer_v2_runtime import ScorerV2Runtime

logger = logging.getLogger(__name__)


def configure_logging(active_settings: Settings) -> None:
    level = getattr(logging, active_settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    )
    logging.getLogger("pronunciation_backend").setLevel(level)
    logging.getLogger().setLevel(logging.INFO)
    for noisy_logger in ("urllib3", "huggingface_hub", "filelock"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    logger.info("Configured pronunciation backend logging level=%s", active_settings.log_level.upper())


def _runtime_preflight_entry(active_settings: Settings, pipeline: PronunciationPipeline):
    if active_settings.mfa_preflight_audio_path is not None and active_settings.mfa_preflight_word is not None:
        word = active_settings.mfa_preflight_word.strip()
        if word:
            entry = pipeline.lexicon_service.get_word(word)
            asset_path = active_settings.mfa_preflight_audio_path
            if asset_path.exists():
                return entry, asset_path
    for word in pipeline.lexicon_service.all_words():
        entry = pipeline.lexicon_service.get_word(word)
        if not entry.reference_audio_id:
            continue
        reference = pipeline.reference_audio_service.get_reference(entry.reference_audio_id, entry.ipa)
        asset_path = reference.asset_path
        if asset_path and Path(asset_path).exists():
            return entry, Path(asset_path)
    return None, None


def build_pipeline(active_settings: Settings) -> PronunciationPipeline:
    active_settings.validate_runtime()
    lexicon_service = LexiconService(
        active_settings.lexicon_path,
        cmudict_path=active_settings.cmudict_path,
    )
    if active_settings.mfa_runtime_dictionary_path is not None and not active_settings.mfa_runtime_dictionary_path.exists():
        lexicon_service.write_runtime_dictionary(active_settings.mfa_runtime_dictionary_path)
    scorer_runtime = ScorerV2Runtime(
        checkpoint_path=active_settings.scorer_checkpoint_path,
        backbone_id=active_settings.backbone_id,
        device=active_settings.scorer_device,
        strict_load=active_settings.scorer_strict_load,
        compile_model=active_settings.scorer_compile,
        compile_mode=active_settings.scorer_compile_mode,
    )
    return PronunciationPipeline(
        lexicon_service=lexicon_service,
        reference_audio_service=ReferenceAudioService(active_settings.reference_manifest_path),
        audio_prep_service=AudioPrepService(active_settings),
        feature_encoder=SSLFeatureEncoder(
            active_settings,
            compile_model=active_settings.hf_compile,
            compile_mode=active_settings.hf_compile_mode,
        ),
        aligner=_build_aligner(active_settings, lexicon_service),
        feature_builder=PhoneFeatureBuilder(),
        scorer_runtime=scorer_runtime,
        response_mapper=ResponseMapper(),
    )


def _build_aligner(active_settings: Settings, lexicon_service: LexiconService):
    if active_settings.aligner_backend == "mfa":
        return MfaForcedAligner(
            command=active_settings.mfa_command,
            acoustic_model=active_settings.mfa_acoustic_model,
            work_root=active_settings.mfa_work_root,
            timeout_seconds=active_settings.mfa_timeout_seconds,
            runtime_dictionary_path=active_settings.mfa_runtime_dictionary_path,
            clean=active_settings.mfa_clean,
        )
    if active_settings.aligner_backend == "phone_ctc":
        phones = [
            phone
            for word in lexicon_service.all_words()
            for phone in lexicon_service.get_word(word).phones
        ]
        return PhoneCtcAligner(
            checkpoint_path=active_settings.phone_ctc_checkpoint_path,
            vocabulary=PhoneVocabulary.from_phones(phones),
        )
    raise ValueError(f"Unsupported aligner backend: {active_settings.aligner_backend}")


def _warm_runtime_pipeline(active_settings: Settings, pipeline: PronunciationPipeline) -> None:
    pipeline.feature_encoder.warmup()
    pipeline.scorer_runtime.warmup()

    entry, asset_path = _runtime_preflight_entry(active_settings, pipeline)
    if entry is None or asset_path is None:
        logger.warning("Skipping MFA preflight because no bundled reference audio asset was found")
        return

    try:
        prepared = pipeline.audio_prep_service.decode_path(asset_path)
        encoded = pipeline.feature_encoder.encode(prepared)
        if hasattr(pipeline.aligner, "preflight"):
            timings = pipeline.aligner.preflight(entry, prepared, encoded)  # type: ignore[attr-defined]
            logger.info(
                "Completed MFA preflight word=%s audio_path=%s total_ms=%.3f subprocess_ms=%.3f "
                "parse_ms=%.3f mapping_ms=%.3f",
                entry.word,
                str(asset_path),
                timings.total_ms,
                timings.subprocess_ms,
                timings.parse_ms,
                timings.mapping_ms,
                extra={
                    "word": entry.word,
                    "audio_path": str(asset_path),
                    "total_ms": timings.total_ms,
                    "subprocess_ms": timings.subprocess_ms,
                    "parse_ms": timings.parse_ms,
                    "mapping_ms": timings.mapping_ms,
                },
            )
            return

        pipeline.aligner.align(entry, prepared, encoded)
    except Exception:
        logger.exception(
            "MFA preflight failed; continuing startup without blocking scoring requests",
            extra={"word": entry.word, "audio_path": str(asset_path)},
        )


def create_app(
    *,
    settings_override: Settings | None = None,
    pipeline_override: PronunciationPipeline | None = None,
) -> FastAPI:
    active_settings = settings_override or settings
    configure_logging(active_settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipeline = pipeline_override or build_pipeline(active_settings)
        if pipeline_override is None:
            _warm_runtime_pipeline(active_settings, app.state.pipeline)
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
            pipeline = get_pipeline_from_request(request)
            if hasattr(pipeline, "assess_word_with_timings"):
                response, timings = pipeline.assess_word_with_timings(
                    word=word,
                    audio_bytes=audio_bytes,
                    no_trim=no_trim,
                )
                logger.info(
                    "Completed pronunciation scoring word=%s total_ms=%.3f audio_prep_ms=%.3f "
                    "feature_encode_ms=%.3f alignment_ms=%.3f alignment_subprocess_ms=%s "
                    "feature_build_ms=%.3f scorer_ms=%.3f reference_ms=%.3f response_ms=%.3f",
                    word,
                    timings.total_ms,
                    timings.audio_prep_ms,
                    timings.feature_encode_ms,
                    timings.alignment_ms,
                    (
                        f"{timings.alignment_subprocess_ms:.3f}"
                        if timings.alignment_subprocess_ms is not None
                        else "none"
                    ),
                    timings.feature_build_ms,
                    timings.scorer_ms,
                    timings.reference_ms,
                    timings.response_ms,
                    extra={
                        "word": word,
                        "audio_prep_ms": round(timings.audio_prep_ms, 3),
                        "feature_encode_ms": round(timings.feature_encode_ms, 3),
                        "alignment_ms": round(timings.alignment_ms, 3),
                        "alignment_subprocess_ms": (
                            round(timings.alignment_subprocess_ms, 3)
                            if timings.alignment_subprocess_ms is not None
                            else None
                        ),
                        "feature_build_ms": round(timings.feature_build_ms, 3),
                        "scorer_ms": round(timings.scorer_ms, 3),
                        "reference_ms": round(timings.reference_ms, 3),
                        "response_ms": round(timings.response_ms, 3),
                        "total_ms": round(timings.total_ms, 3),
                    },
                )
                return response
            return pipeline.assess_word(word=word, audio_bytes=audio_bytes, no_trim=no_trim)
        except UnknownWordError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except AudioValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except AlignmentError as exc:
            logger.exception("Alignment failed during pronunciation scoring")
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


app = create_app()
