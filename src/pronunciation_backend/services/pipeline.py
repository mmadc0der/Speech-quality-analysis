from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from pronunciation_backend.models import PronunciationAssessmentResponse
from pronunciation_backend.services.aligner import PhonemeAligner, PhoneFeatureBuilder
from pronunciation_backend.services.audio_prep import AudioPrepService
from pronunciation_backend.services.feature_encoder import SSLFeatureEncoder
from pronunciation_backend.services.lexicon import LexiconService
from pronunciation_backend.services.reference import ReferenceAudioService
from pronunciation_backend.services.response_mapper import ResponseMapper
from pronunciation_backend.services.scorer_runtime import ScorerModelInfo, ScorerRuntime


@dataclass(frozen=True)
class AssessmentTimings:
    audio_prep_ms: float
    feature_encode_ms: float
    alignment_ms: float
    feature_build_ms: float
    scorer_ms: float
    reference_ms: float
    response_ms: float
    total_ms: float
    alignment_subprocess_ms: float | None = None
    alignment_parse_ms: float | None = None
    alignment_mapping_ms: float | None = None


@dataclass
class PronunciationPipeline:
    lexicon_service: LexiconService
    reference_audio_service: ReferenceAudioService
    audio_prep_service: AudioPrepService
    feature_encoder: SSLFeatureEncoder
    aligner: PhonemeAligner
    feature_builder: PhoneFeatureBuilder
    scorer_runtime: ScorerRuntime
    response_mapper: ResponseMapper

    def assess_word(self, word: str, audio_bytes: bytes, *, no_trim: bool = False) -> PronunciationAssessmentResponse:
        response, _timings = self.assess_word_with_timings(word, audio_bytes, no_trim=no_trim)
        return response

    def assess_word_with_timings(
        self,
        word: str,
        audio_bytes: bytes,
        *,
        no_trim: bool = False,
    ) -> tuple[PronunciationAssessmentResponse, AssessmentTimings]:
        total_started = perf_counter()
        entry = self.lexicon_service.get_word(word)

        audio_started = perf_counter()
        prepared = self.audio_prep_service.decode(audio_bytes, enable_trim=not no_trim)
        audio_ms = (perf_counter() - audio_started) * 1000.0

        encode_started = perf_counter()
        encoded = self.feature_encoder.encode(prepared)
        encode_ms = (perf_counter() - encode_started) * 1000.0

        align_started = perf_counter()
        if hasattr(self.aligner, "align_with_timing"):
            spans, alignment_timings = self.aligner.align_with_timing(entry, prepared, encoded)  # type: ignore[attr-defined]
            align_ms = alignment_timings.total_ms
            alignment_subprocess_ms = alignment_timings.subprocess_ms
            alignment_parse_ms = alignment_timings.parse_ms
            alignment_mapping_ms = alignment_timings.mapping_ms
        else:
            spans = self.aligner.align(entry, prepared, encoded)
            align_ms = (perf_counter() - align_started) * 1000.0
            alignment_subprocess_ms = None
            alignment_parse_ms = None
            alignment_mapping_ms = None

        feature_build_started = perf_counter()
        phone_features = self.feature_builder.build(encoded, spans)
        feature_build_ms = (perf_counter() - feature_build_started) * 1000.0

        score_started = perf_counter()
        runtime_result = self.scorer_runtime.score(phone_features)
        score_ms = (perf_counter() - score_started) * 1000.0

        reference_ms = 0.0
        reference = None
        if entry.reference_audio_id:
            reference_started = perf_counter()
            reference = self.reference_audio_service.get_reference(entry.reference_audio_id, entry.ipa)
            reference_ms = (perf_counter() - reference_started) * 1000.0

        response_started = perf_counter()
        response = self.response_mapper.build_response(
            word=entry.word,
            ipa=entry.ipa,
            prepared_audio=prepared,
            runtime_result=runtime_result,
            reference=reference,
        )
        response_ms = (perf_counter() - response_started) * 1000.0
        total_ms = (perf_counter() - total_started) * 1000.0
        return response, AssessmentTimings(
            audio_prep_ms=audio_ms,
            feature_encode_ms=encode_ms,
            alignment_ms=align_ms,
            feature_build_ms=feature_build_ms,
            scorer_ms=score_ms,
            reference_ms=reference_ms,
            response_ms=response_ms,
            total_ms=total_ms,
            alignment_subprocess_ms=alignment_subprocess_ms,
            alignment_parse_ms=alignment_parse_ms,
            alignment_mapping_ms=alignment_mapping_ms,
        )

    def model_info(self) -> ScorerModelInfo:
        return self.scorer_runtime.model_info()
