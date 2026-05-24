from __future__ import annotations

from pathlib import Path

import pytest

from pronunciation_backend.models import LexiconEntry
from pronunciation_backend.services.lexicon import LexiconService, UnknownWordError


@pytest.fixture
def curated_lexicon_path(tmp_path: Path) -> Path:
    path = tmp_path / "en_us_words.json"
    path.write_text(
        """
        {
          "thought": {
            "word": "thought",
            "phones": ["TH", "AO", "T"],
            "ipa": "θɔt",
            "reference_audio_id": "thought_en_us_01",
            "syllables": [["TH", "AO", "T"]],
            "stress_pattern": "1"
          }
        }
        """.strip(),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def cmudict_path(tmp_path: Path) -> Path:
    path = tmp_path / "cmudict.dict"
    path.write_text(
        "\n".join(
            [
                ";;; CMUdict test fixture",
                "WORK W ER1 K",
                "CAT K AE1 T",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_lexicon_service_returns_curated_override(
    curated_lexicon_path: Path,
    cmudict_path: Path,
) -> None:
    service = LexiconService(curated_lexicon_path, cmudict_path=cmudict_path)

    entry = service.get_word("thought")

    assert entry == LexiconEntry(
        word="thought",
        phones=["TH", "AO", "T"],
        ipa="θɔt",
        reference_audio_id="thought_en_us_01",
        syllables=[["TH", "AO", "T"]],
        stress_pattern="1",
    )


def test_lexicon_service_resolves_cmudict_only_word(
    curated_lexicon_path: Path,
    cmudict_path: Path,
) -> None:
    service = LexiconService(curated_lexicon_path, cmudict_path=cmudict_path)

    entry = service.get_word("Work")

    assert entry.word == "work"
    assert entry.phones == ["W", "ER", "K"]
    assert entry.alignment_phones == ["W", "ER1", "K"]
    assert entry.ipa == "wɝk"
    assert entry.reference_audio_id is None
    assert entry.syllables == []
    assert entry.stress_pattern is None


def test_lexicon_service_normalizes_punctuation(
    curated_lexicon_path: Path,
    cmudict_path: Path,
) -> None:
    service = LexiconService(curated_lexicon_path, cmudict_path=cmudict_path)

    entry = service.get_word("  Work!!!  ")

    assert entry.word == "work"
    assert entry.phones == ["W", "ER", "K"]


def test_lexicon_service_rejects_unknown_word(
    curated_lexicon_path: Path,
    cmudict_path: Path,
) -> None:
    service = LexiconService(curated_lexicon_path, cmudict_path=cmudict_path)

    with pytest.raises(UnknownWordError, match="was not found in CMUdict"):
        service.get_word("notawordxyz")


def test_lexicon_service_rejects_empty_token(
    curated_lexicon_path: Path,
    cmudict_path: Path,
) -> None:
    service = LexiconService(curated_lexicon_path, cmudict_path=cmudict_path)

    with pytest.raises(UnknownWordError, match="not a supported dictionary token"):
        service.get_word("!!!")
