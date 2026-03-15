from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    sample_rate: int = 16_000
    min_audio_ms: int = 250
    max_audio_ms: int = 4_000
    silence_threshold: float = 0.015
    clipping_threshold: float = 0.985
    use_hf_encoder: bool = os.getenv("PRONUNCIATION_USE_HF_ENCODER", "0") == "1"
    backbone_id: str = os.getenv("PRONUNCIATION_BACKBONE_ID", "facebook/hubert-base-ls960")
    device: str = os.getenv("PRONUNCIATION_DEVICE", "cpu")
    lexicon_path: Path = Path(__file__).resolve().parent / "resources" / "en_us_words.json"
    reference_manifest_path: Path = Path(__file__).resolve().parent / "resources" / "reference_audio_manifest.json"


settings = Settings()
