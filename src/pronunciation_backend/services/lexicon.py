from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pronunciation_backend.models import LexiconEntry


class UnknownWordError(ValueError):
    """Raised when a target word is not in the curated MVP lexicon."""


@dataclass
class LexiconService:
    lexicon_path: Path

    def __post_init__(self) -> None:
        self._entries = self._load_entries()

    def _load_entries(self) -> dict[str, LexiconEntry]:
        raw = json.loads(self.lexicon_path.read_text(encoding="utf-8"))
        return {
            key.lower(): LexiconEntry(
                word=value["word"],
                phones=value["phones"],
                ipa=value["ipa"],
                reference_audio_id=value["reference_audio_id"],
                syllables=value.get("syllables", []),
                stress_pattern=value.get("stress_pattern"),
            )
            for key, value in raw.items()
        }

    def get_word(self, word: str) -> LexiconEntry:
        normalized = word.strip().lower()
        entry = self._entries.get(normalized)
        if entry is None:
            raise UnknownWordError(f"Word '{word}' is not supported by the MVP lexicon.")
        return entry

    def all_words(self) -> list[str]:
        return sorted(self._entries)
