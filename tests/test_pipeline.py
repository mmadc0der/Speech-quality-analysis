from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pronunciation_backend.models import EncodedFrames, LexiconEntry, PhoneFeatures, PhoneSpan, PreparedAudio, ReferencePayload
from pronunciation_backend.services.pipeline import PronunciationPipeline
from pronunciation_backend.services.response_mapper import ResponseMapper
from pronunciation_backend.services.scorer_runtime import ScorerModelInfo, ScorerPhonePrediction, ScorerRuntimeResult


@dataclass
class _FakeLexiconService:
    def get_word(self, word: str) -> LexiconEntry:
        return LexiconEntry(
            word=word,
            phones=["TH", "AO", "T"],
            ipa="θɔt",
            reference_audio_id="thought_en_us_01",
        )


@dataclass
class _FakeReferenceAudioService:
    def get_reference(self, audio_id: str, ipa: str) -> ReferencePayload:
        return ReferencePayload(ipa=ipa, audio_id=audio_id, asset_path="assets/reference.wav")


@dataclass
class _FakeAudioPrepService:
    def decode(self, audio_bytes: bytes, *, enable_trim: bool = True) -> PreparedAudio:
        assert audio_bytes == b"audio"
        assert enable_trim is True
        return PreparedAudio(
            samples=np.zeros((16_000,), dtype=np.float32),
            sample_rate=16_000,
            duration_ms=500,
            rms=0.2,
            clipping_ratio=0.0,
            silence_ratio=0.1,
            snr_estimate=25.0,
            quality_status="ok",
            original_duration_ms=1000,
            trim_start_ms=500,
            trim_end_ms=1000,
            trim_applied=True,
        )


@dataclass
class _FakeFeatureEncoder:
    def encode(self, prepared: PreparedAudio) -> EncodedFrames:
        del prepared
        return EncodedFrames(
            embeddings=np.ones((10, 768), dtype=np.float32),
            frame_ms=20.0,
            energy=np.full((10,), 0.2, dtype=np.float32),
        )


@dataclass
class _FakeAligner:
    def align(self, entry: LexiconEntry, prepared: PreparedAudio, encoded: EncodedFrames) -> list[PhoneSpan]:
        del entry, prepared, encoded
        return [
            PhoneSpan("TH", 0, 3, 0, 60, 0.9, 0.0),
            PhoneSpan("AO", 3, 7, 60, 140, 0.95, 0.0),
            PhoneSpan("T", 7, 10, 140, 200, 0.92, 0.0),
        ]


@dataclass
class _FakeFeatureBuilder:
    def build(self, encoded: EncodedFrames, spans: list[PhoneSpan]) -> list[PhoneFeatures]:
        del encoded
        return [
            PhoneFeatures(
                phoneme=span.phoneme,
                start_ms=span.start_ms,
                end_ms=span.end_ms,
                mean_embedding=[0.01] * 768,
                variance=0.1,
                duration_ms=span.end_ms - span.start_ms,
                duration_z_score=0.0,
                alignment_confidence=span.alignment_confidence,
                energy_mean=0.2,
                starts_late=False,
            )
            for span in spans
        ]


@dataclass
class _FakeScorerRuntime:
    seen_features: list[PhoneFeatures] | None = None

    def model_info(self) -> ScorerModelInfo:
        return ScorerModelInfo(
            runtime_backend="scorer_v2",
            model_version="v2",
            checkpoint_name="fake.pt",
            backbone_id="facebook/hubert-base-ls960",
            device="cpu",
            class_labels=("wrong_or_missed", "accented", "correct"),
        )

    def score(self, phone_features: list[PhoneFeatures]) -> ScorerRuntimeResult:
        self.seen_features = phone_features
        return ScorerRuntimeResult(
            phone_predictions=[
                ScorerPhonePrediction(
                    phoneme=feature.phoneme,
                    start_ms=feature.start_ms,
                    end_ms=feature.end_ms,
                    expected_score=50.0 + (index * 10.0),
                    expected_human_score=1.0 + (index * 0.25),
                    omission_probability=0.05,
                    predicted_class="accented" if index == 0 else "correct",
                    quality_class_probs={
                        "wrong_or_missed": 0.1,
                        "accented": 0.7 if index == 0 else 0.1,
                        "correct": 0.2 if index == 0 else 0.8,
                    },
                    alignment_confidence=feature.alignment_confidence,
                )
                for index, feature in enumerate(phone_features)
            ],
            model_info=self.model_info(),
        )


def test_pipeline_omits_reference_when_not_curated() -> None:
    @dataclass
    class _LexiconWithoutReference:
        def get_word(self, word: str) -> LexiconEntry:
            return LexiconEntry(
                word=word,
                phones=["TH", "AO", "T"],
                ipa="θɔt",
            )

    @dataclass
    class _UnusedReferenceAudioService:
        def get_reference(self, audio_id: str, ipa: str) -> ReferencePayload:
            raise AssertionError("reference lookup should be skipped when reference_audio_id is missing")

    scorer_runtime = _FakeScorerRuntime()
    pipeline = PronunciationPipeline(
        lexicon_service=_LexiconWithoutReference(),
        reference_audio_service=_UnusedReferenceAudioService(),
        audio_prep_service=_FakeAudioPrepService(),
        feature_encoder=_FakeFeatureEncoder(),
        aligner=_FakeAligner(),
        feature_builder=_FakeFeatureBuilder(),
        scorer_runtime=scorer_runtime,
        response_mapper=ResponseMapper(),
    )

    response, timings = pipeline.assess_word_with_timings(word="thought", audio_bytes=b"audio")

    assert response.reference is None
    assert timings.reference_ms == 0.0


def test_pipeline_uses_runtime_and_response_mapper() -> None:
    scorer_runtime = _FakeScorerRuntime()
    pipeline = PronunciationPipeline(
        lexicon_service=_FakeLexiconService(),
        reference_audio_service=_FakeReferenceAudioService(),
        audio_prep_service=_FakeAudioPrepService(),
        feature_encoder=_FakeFeatureEncoder(),
        aligner=_FakeAligner(),
        feature_builder=_FakeFeatureBuilder(),
        scorer_runtime=scorer_runtime,
        response_mapper=ResponseMapper(),
    )

    response, timings = pipeline.assess_word_with_timings(word="thought", audio_bytes=b"audio")

    assert response.word == "thought"
    assert response.model_info.runtime_backend == "scorer_v2"
    assert response.primary_issue.phoneme == "TH"
    assert len(response.phonemes) == 3
    assert response.audio_quality.trim_applied is True
    assert response.phonemes[0].start_ms == 500
    assert response.phonemes[0].end_ms == 560
    assert scorer_runtime.seen_features is not None
    assert [feature.phoneme for feature in scorer_runtime.seen_features] == ["TH", "AO", "T"]
    assert timings.total_ms >= timings.audio_prep_ms
    assert timings.scorer_ms >= 0.0
