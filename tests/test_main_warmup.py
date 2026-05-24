from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from fastapi.testclient import TestClient

from pronunciation_backend.config import Settings
from pronunciation_backend.main import _warm_runtime_pipeline, create_app
from pronunciation_backend.models import LexiconEntry
from pronunciation_backend.services.lexicon import LexiconService
from pronunciation_backend.services.mfa_dictionary import alignment_dictionary_phones, runtime_dictionary_line
from pronunciation_backend.services.scorer_runtime import ScorerModelInfo


def test_alignment_dictionary_phones_applies_stress_from_syllables() -> None:
    entry = LexiconEntry(
        word="banana",
        phones=["B", "AH", "N", "AE", "N", "AH"],
        ipa="bənænə",
        syllables=[["B", "AH"], ["N", "AE"], ["N", "AH"]],
        stress_pattern="010",
    )

    assert alignment_dictionary_phones(entry) == ["B", "AH0", "N", "AE1", "N", "AH0"]
    assert runtime_dictionary_line(entry) == "banana B AH0 N AE1 N AH0"


def test_lexicon_runtime_dictionary_lines_use_stressed_phones(tmp_path: Path) -> None:
    lexicon_path = tmp_path / "lexicon.json"
    lexicon_path.write_text(
        json.dumps(
            {
                "cat": {
                    "word": "cat",
                    "phones": ["K", "AE", "T"],
                    "ipa": "kæt",
                    "syllables": [["K", "AE", "T"]],
                    "stress_pattern": "1",
                }
            }
        ),
        encoding="utf-8",
    )
    service = LexiconService(lexicon_path)

    assert "cat K AE1 T" in service.runtime_dictionary_lines()

    dict_path = tmp_path / "runtime.dict"
    service.write_runtime_dictionary(dict_path)
    assert "cat K AE1 T" in dict_path.read_text(encoding="utf-8").splitlines()


@dataclass
class _WarmupPipeline:
    feature_encoder_warmed: bool = False
    scorer_warmed: bool = False

    @dataclass
    class _FeatureEncoder:
        outer: "_WarmupPipeline"

        def warmup(self) -> None:
            self.outer.feature_encoder_warmed = True

        def encode(self, prepared):  # type: ignore[no-untyped-def]
            del prepared
            raise AssertionError("encode should not run when preflight is skipped")

    @dataclass
    class _ScorerRuntime:
        outer: "_WarmupPipeline"

        def warmup(self) -> None:
            self.outer.scorer_warmed = True

        def model_info(self) -> ScorerModelInfo:
            return ScorerModelInfo(
                runtime_backend="scorer_v2",
                model_version="v2",
                checkpoint_name="fake.pt",
                backbone_id="facebook/hubert-base-ls960",
                device="cpu",
                class_labels=("wrong_or_missed", "accented", "correct"),
            )

    @dataclass
    class _LexiconService:
        def all_words(self) -> list[str]:
            return []

    @dataclass
    class _ReferenceAudioService:
        def get_reference(self, audio_id: str, ipa: str):  # type: ignore[no-untyped-def]
            del audio_id, ipa
            raise AssertionError("reference lookup should not run when no preflight asset exists")

    @dataclass
    class _Aligner:
        def preflight(self, entry, prepared, encoded):  # type: ignore[no-untyped-def]
            del entry, prepared, encoded
            raise AssertionError("preflight should not run when no asset exists")

    feature_encoder: _FeatureEncoder = field(init=False)
    scorer_runtime: _ScorerRuntime = field(init=False)
    lexicon_service: _LexiconService = field(init=False)
    reference_audio_service: _ReferenceAudioService = field(init=False)
    aligner: _Aligner = field(init=False)

    def model_info(self) -> ScorerModelInfo:
        return self.scorer_runtime.model_info()

    def __post_init__(self) -> None:
        self.feature_encoder = self._FeatureEncoder(self)
        self.scorer_runtime = self._ScorerRuntime(self)
        self.lexicon_service = self._LexiconService()
        self.reference_audio_service = self._ReferenceAudioService()
        self.aligner = self._Aligner()


def test_warm_runtime_pipeline_does_not_require_outer_scope_settings() -> None:
    pipeline = _WarmupPipeline()
    active_settings = Settings(use_hf_encoder=True)

    _warm_runtime_pipeline(active_settings, pipeline)  # type: ignore[arg-type]

    assert pipeline.feature_encoder_warmed is True
    assert pipeline.scorer_warmed is True


def test_create_app_starts_with_pipeline_override_without_warmup() -> None:
    with TestClient(create_app(pipeline_override=_WarmupPipeline())) as client:  # type: ignore[arg-type]
        response = client.get("/health")

    assert response.status_code == 200
