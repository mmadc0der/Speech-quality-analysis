from __future__ import annotations

from dataclasses import dataclass

from pronunciation_backend.models import PronunciationAssessmentResponse
from pronunciation_backend.services.aligner import ConstrainedPhonemeAligner, PhoneFeatureBuilder
from pronunciation_backend.services.audio_prep import AudioPrepService
from pronunciation_backend.services.feature_encoder import SSLFeatureEncoder
from pronunciation_backend.services.lexicon import LexiconService
from pronunciation_backend.services.reference import ReferenceAudioService
from pronunciation_backend.services.scoring import CalibrationAndIssueService, PhoneScoringHead


@dataclass
class PronunciationPipeline:
    lexicon_service: LexiconService
    reference_audio_service: ReferenceAudioService
    audio_prep_service: AudioPrepService
    feature_encoder: SSLFeatureEncoder
    aligner: ConstrainedPhonemeAligner
    feature_builder: PhoneFeatureBuilder
    scoring_head: PhoneScoringHead
    calibration_service: CalibrationAndIssueService

    def assess_word(self, word: str, audio_bytes: bytes) -> PronunciationAssessmentResponse:
        entry = self.lexicon_service.get_word(word)
        prepared = self.audio_prep_service.decode(audio_bytes)
        encoded = self.feature_encoder.encode(prepared)
        spans = self.aligner.align(entry, encoded)
        phone_features = self.feature_builder.build(encoded, spans)
        phone_scores = self.scoring_head.score(phone_features)
        audio_quality = self.calibration_service.build_audio_quality(
            quality_status=prepared.quality_status,
            snr_estimate=prepared.snr_estimate,
            duration_ms=prepared.duration_ms,
            rms=prepared.rms,
            clipping_ratio=prepared.clipping_ratio,
            silence_ratio=prepared.silence_ratio,
        )
        overall_score, confidence = self.calibration_service.overall_score(phone_scores, audio_quality)
        primary_issue = self.calibration_service.primary_issue(phone_scores, confidence)
        reference = self.reference_audio_service.get_reference(entry.reference_audio_id, entry.ipa)

        return PronunciationAssessmentResponse(
            word=entry.word,
            ipa=entry.ipa,
            overall_score=overall_score,
            confidence=confidence,
            audio_quality=audio_quality,
            phonemes=phone_scores,
            primary_issue=primary_issue,
            reference=reference,
        )
