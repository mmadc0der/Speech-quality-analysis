from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pronunciation_backend.training.cmudict_utils import normalize_word_token, strip_phone_stress

EXPECTED_KALDI_FILES = ("train/wav.scp", "test/wav.scp")
SCORES_LOCATIONS = ("scores.json", "resource/scores.json")


def resolve_speechocean_raw_root(dataset_root: Path) -> Path:
    candidates = [
        dataset_root,
        dataset_root / "raw",
        dataset_root / "speechocean762",
        dataset_root / "raw" / "speechocean762",
        dataset_root / "unpacked" / "speechocean762",
    ]
    discovered: list[Path] = []
    if dataset_root.exists():
        for scores_path in dataset_root.rglob("scores.json"):
            discovered.append(scores_path.parent)
            discovered.append(scores_path.parent.parent)
    candidates.extend(sorted(discovered, key=lambda path: (len(path.parts), str(path))))
    for candidate in candidates:
        if _looks_like_raw_root(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find SpeechOcean762 corpus files under {dataset_root}. "
        "Expected scores.json plus train/test Kaldi metadata."
    )


def relative_str(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def load_scores(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected scores payload to be a JSON object: {path}")
    return payload


def resolve_scores_path(raw_root: Path) -> Path:
    for relative in SCORES_LOCATIONS:
        candidate = raw_root / relative
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find SpeechOcean762 scores file under {raw_root}. "
        "Expected scores.json or resource/scores.json."
    )


def read_kaldi_mapping(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            key, value = line.split(maxsplit=1)
            mapping[key] = value
    return mapping


def read_wav_scp(path: Path, *, raw_root: Path, dataset_root: Path) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for utterance_id, raw_value in read_kaldi_mapping(path).items():
        audio_path = Path(raw_value)
        candidate = audio_path if audio_path.is_absolute() else raw_root / audio_path
        resolved[utterance_id] = relative_str(candidate, dataset_root)
    return resolved


def normalize_score_word_text(text: str) -> str:
    return normalize_word_token(text)


def canonical_phones_from_word(word_payload: dict[str, Any]) -> list[str]:
    raw_phones = word_payload.get("phones") or word_payload.get("ref-phones")
    if raw_phones is None:
        raise ValueError(f"Word payload is missing phones: {word_payload}")
    if isinstance(raw_phones, str):
        phones = raw_phones.split()
    else:
        phones = list(raw_phones)
    return [strip_phone_stress(str(phone)) for phone in phones]


def phone_scores_from_word(word_payload: dict[str, Any]) -> list[float]:
    raw_scores = word_payload.get("phones-accuracy")
    if raw_scores is None:
        raise ValueError(f"Word payload is missing phones-accuracy: {word_payload}")
    return [float(score) for score in raw_scores]


def pronunciation_class_from_score(score: float) -> str:
    if score >= 1.5:
        return "correct"
    if score >= 0.5:
        return "accented"
    return "wrong_or_missed"


def mispronunciations_by_index(word_payload: dict[str, Any]) -> dict[int, str]:
    result: dict[int, str] = {}
    for item in word_payload.get("mispronunciations", []):
        if not isinstance(item, dict):
            continue
        index = item.get("index")
        pronounced_phone = item.get("pronounced-phone")
        if index is None or pronounced_phone is None:
            continue
        result[int(index)] = str(pronounced_phone)
    return result


def is_omission_pronunciation(pronounced_phone: str | None) -> bool:
    if pronounced_phone is None:
        return False
    normalized = pronounced_phone.strip().lower()
    return normalized in {"<unk>", "unk", "", "\\"}


def _looks_like_raw_root(path: Path) -> bool:
    return (
        path.exists()
        and any((path / relative).exists() for relative in SCORES_LOCATIONS)
        and all((path / relative).exists() for relative in EXPECTED_KALDI_FILES)
    )
