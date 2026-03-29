from __future__ import annotations

from dataclasses import dataclass

from pronunciation_backend.models import PronunciationAssessmentResponse
from pronunciation_backend.services.aligner import ConstrainedPhonemeAligner, PhoneFeatureBuilder
from pronunciation_backend.services.audio_prep import AudioPrepService
from pronunciation_backend.services.feature_encoder import SSLFeatureEncoder
from pronunciation_backend.services.lexicon import LexiconService
from pronunciation_backend.services.reference import ReferenceAudioService
from pronunciation_backend.services.response_mapper import ResponseMapper
from pronunciation_backend.services.scorer_runtime import ScorerModelInfo, ScorerRuntime


@dataclass
class PronunciationPipeline:
    lexicon_service: LexiconService
    reference_audio_service: ReferenceAudioService
    audio_prep_service: AudioPrepService
    feature_encoder: SSLFeatureEncoder
    aligner: ConstrainedPhonemeAligner
    feature_builder: PhoneFeatureBuilder
    scorer_runtime: ScorerRuntime
    response_mapper: ResponseMapper

    def assess_word(self, word: str, audio_bytes: bytes) -> PronunciationAssessmentResponse:
        entry = self.lexicon_service.get_word(word)
        prepared = self.audio_prep_service.decode(audio_bytes)
        encoded = self.feature_encoder.encode(prepared)
        spans = self.aligner.align(entry, encoded)
        phone_features = self.feature_builder.build(encoded, spans)
        runtime_result = self.scorer_runtime.score(phone_features)
        reference = self.reference_audio_service.get_reference(entry.reference_audio_id, entry.ipa)
        return self.response_mapper.build_response(
            word=entry.word,
            ipa=entry.ipa,
            prepared_audio=prepared,
            runtime_result=runtime_result,
            reference=reference,
        )

    def model_info(self) -> ScorerModelInfo:
        return self.scorer_runtime.model_info()
