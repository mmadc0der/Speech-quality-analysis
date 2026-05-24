from __future__ import annotations

from pronunciation_backend.models import LexiconEntry
from pronunciation_backend.training.cmudict_utils import normalize_word_token, strip_phone_stress

ARPABET_VOWELS = {
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "EH",
    "ER",
    "EY",
    "IH",
    "IY",
    "OW",
    "OY",
    "UH",
    "UW",
}


def alignment_dictionary_phones(entry: LexiconEntry) -> list[str]:
    if not entry.syllables or not entry.stress_pattern:
        return list(entry.phones)

    flattened = [phone for syllable in entry.syllables for phone in syllable]
    if [strip_phone_stress(phone) for phone in flattened] != [strip_phone_stress(phone) for phone in entry.phones]:
        return list(entry.phones)

    stressed: list[str] = []
    for syllable_index, syllable in enumerate(entry.syllables):
        stress_digit = _stress_digit(entry.stress_pattern, syllable_index)
        for phone in syllable:
            base_phone = strip_phone_stress(phone)
            if base_phone in ARPABET_VOWELS:
                stressed.append(f"{base_phone}{stress_digit}")
            else:
                stressed.append(base_phone)
    return stressed


def runtime_dictionary_line(entry: LexiconEntry) -> str:
    transcript_token = normalize_word_token(entry.word)
    phones = alignment_dictionary_phones(entry)
    return f"{transcript_token} {' '.join(phones)}"


def _stress_digit(stress_pattern: str, syllable_index: int) -> str:
    if syllable_index >= len(stress_pattern):
        return "0"
    digit = stress_pattern[syllable_index]
    return digit if digit in {"0", "1", "2"} else "0"
