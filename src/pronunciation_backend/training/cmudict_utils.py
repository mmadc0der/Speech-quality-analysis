from __future__ import annotations

import re
from pathlib import Path


ARPABET_TO_IPA = {
    "AA": "a",
    "AE": "ae",
    "AH": "ah",
    "AO": "o",
    "AW": "aʊ",
    "AY": "aɪ",
    "B": "b",
    "CH": "tʃ",
    "D": "d",
    "DH": "ð",
    "EH": "e",
    "ER": "ɝ",
    "EY": "eɪ",
    "F": "f",
    "G": "g",
    "HH": "h",
    "IH": "ɪ",
    "IY": "i",
    "JH": "dʒ",
    "K": "k",
    "L": "l",
    "M": "m",
    "N": "n",
    "NG": "ŋ",
    "OW": "oʊ",
    "OY": "ɔɪ",
    "P": "p",
    "R": "ɹ",
    "S": "s",
    "SH": "ʃ",
    "T": "t",
    "TH": "θ",
    "UH": "ʊ",
    "UW": "u",
    "V": "v",
    "W": "w",
    "Y": "j",
    "Z": "z",
    "ZH": "ʒ",
}

VARIANT_SUFFIX_RE = re.compile(r"\(\d+\)$")
NON_WORD_RE = re.compile(r"[^a-z']+")
STRESS_RE = re.compile(r"\d")


def normalize_word_token(token: str) -> str:
    return NON_WORD_RE.sub("", token.strip().lower())


def strip_phone_stress(phone: str) -> str:
    return STRESS_RE.sub("", phone.upper())


def arpabet_to_ipa(phones: list[str]) -> str:
    return "".join(ARPABET_TO_IPA.get(strip_phone_stress(phone), strip_phone_stress(phone).lower()) for phone in phones)


def load_cmudict(path: Path) -> dict[str, list[str]]:
    lexicon: dict[str, list[str]] = {}
    with path.open("r", encoding="latin-1") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith(";;;"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            word = VARIANT_SUFFIX_RE.sub("", parts[0]).lower()
            if word in lexicon:
                continue
            lexicon[word] = [strip_phone_stress(phone) for phone in parts[1:]]
    return lexicon
