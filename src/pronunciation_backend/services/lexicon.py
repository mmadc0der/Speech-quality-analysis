from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pronunciation_backend.models import LexiconEntry
from pronunciation_backend.training.cmudict_utils import (
    VARIANT_SUFFIX_RE,
    arpabet_to_ipa,
    load_cmudict,
    normalize_word_token,
    strip_phone_stress,
)


class UnknownWordError(ValueError):
    """Raised when a target word is not found in CMUdict or is an unsupported token."""


def _load_cmudict_entries(cmudict_path: Path | None) -> dict[str, list[str]]:
    if cmudict_path is not None:
        if not cmudict_path.exists():
            raise FileNotFoundError(f"CMUdict file does not exist: {cmudict_path}")
        return load_cmudict(cmudict_path)

    import cmudict

    raw = cmudict.dict()
    entries: dict[str, list[str]] = {}
    for word, pronunciations in raw.items():
        normalized = VARIANT_SUFFIX_RE.sub("", word).lower()
        if normalized in entries or not pronunciations:
            continue
        entries[normalized] = [strip_phone_stress(phone) for phone in pronunciations[0]]
    return entries


@dataclass
class LexiconService:
    lexicon_path: Path
    cmudict_path: Path | None = None

    def __post_init__(self) -> None:
        self._curated_entries = self._load_curated_entries()
        self._cmudict_entries = _load_cmudict_entries(self.cmudict_path)

    def _load_curated_entries(self) -> dict[str, LexiconEntry]:
        raw = json.loads(self.lexicon_path.read_text(encoding="utf-8"))
        return {
            key.lower(): LexiconEntry(
                word=value["word"],
                phones=value["phones"],
                ipa=value["ipa"],
                reference_audio_id=value.get("reference_audio_id"),
                syllables=value.get("syllables", []),
                stress_pattern=value.get("stress_pattern"),
            )
            for key, value in raw.items()
        }

    def get_word(self, word: str) -> LexiconEntry:
        normalized = normalize_word_token(word)
        if not normalized:
            raise UnknownWordError(f"Word '{word}' is not a supported dictionary token.")

        curated = self._curated_entries.get(normalized)
        if curated is not None:
            return curated

        phones = self._cmudict_entries.get(normalized)
        if phones is None:
            raise UnknownWordError(f"Word '{word}' was not found in CMUdict.")

        return LexiconEntry(
            word=normalized,
            phones=phones,
            ipa=arpabet_to_ipa(phones),
        )

    def all_words(self) -> list[str]:
        return sorted(self._curated_entries)
