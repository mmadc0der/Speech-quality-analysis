from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pronunciation_backend.models import LexiconEntry
from pronunciation_backend.training.cmudict_utils import (
    VARIANT_SUFFIX_RE,
    arpabet_to_ipa,
    normalize_word_token,
    strip_phone_stress,
)


class UnknownWordError(ValueError):
    """Raised when a target word is not found in CMUdict or is an unsupported token."""


def _load_cmudict_entries(cmudict_path: Path | None) -> dict[str, tuple[list[str], list[str]]]:
    if cmudict_path is not None:
        if not cmudict_path.exists():
            raise FileNotFoundError(f"CMUdict file does not exist: {cmudict_path}")
        entries: dict[str, tuple[list[str], list[str]]] = {}
        with cmudict_path.open("r", encoding="latin-1") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith(";;;"):
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                normalized = VARIANT_SUFFIX_RE.sub("", parts[0]).lower()
                if normalized in entries:
                    continue
                alignment_phones = [phone.upper() for phone in parts[1:]]
                entries[normalized] = (
                    [strip_phone_stress(phone) for phone in alignment_phones],
                    alignment_phones,
                )
        return entries

    import cmudict

    raw = cmudict.dict()
    entries: dict[str, tuple[list[str], list[str]]] = {}
    for word, pronunciations in raw.items():
        normalized = VARIANT_SUFFIX_RE.sub("", word).lower()
        if normalized in entries or not pronunciations:
            continue
        alignment_phones = [phone.upper() for phone in pronunciations[0]]
        entries[normalized] = (
            [strip_phone_stress(phone) for phone in alignment_phones],
            alignment_phones,
        )
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
                alignment_phones=value.get("alignment_phones", []),
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

        cmudict_entry = self._cmudict_entries.get(normalized)
        if cmudict_entry is None:
            raise UnknownWordError(f"Word '{word}' was not found in CMUdict.")
        phones, alignment_phones = cmudict_entry

        return LexiconEntry(
            word=normalized,
            phones=phones,
            ipa=arpabet_to_ipa(phones),
            alignment_phones=alignment_phones,
        )

    def all_words(self) -> list[str]:
        return sorted(self._curated_entries)
