from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default) == "1"


def _env_path(name: str, default: str) -> Path:
    return Path(os.getenv(name, default))


def _optional_env_path(name: str) -> Path | None:
    value = os.getenv(name)
    return Path(value) if value else None


def _optional_env_value(name: str) -> str | None:
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


@dataclass(frozen=True)
class Settings:
    sample_rate: int = 16_000
    min_audio_ms: int = 250
    max_audio_ms: int = 4_000
    silence_threshold: float = 0.015
    clipping_threshold: float = 0.985
    use_hf_encoder: bool = field(default_factory=lambda: _env_flag("PRONUNCIATION_USE_HF_ENCODER"))
    backbone_id: str = field(default_factory=lambda: os.getenv("PRONUNCIATION_BACKBONE_ID", "facebook/hubert-base-ls960"))
    device: str = field(default_factory=lambda: os.getenv("PRONUNCIATION_DEVICE", "cpu"))
    hf_compile: bool = field(default_factory=lambda: _env_flag("PRONUNCIATION_HF_COMPILE"))
    hf_compile_mode: str = field(default_factory=lambda: os.getenv("PRONUNCIATION_HF_COMPILE_MODE", "reduce-overhead"))
    runtime_backend: str = field(default_factory=lambda: os.getenv("PRONUNCIATION_RUNTIME_BACKEND", "scorer_v2"))
    aligner_backend: str = field(default_factory=lambda: os.getenv("PRONUNCIATION_ALIGNER_BACKEND", "mfa"))
    scorer_device: str = field(default_factory=lambda: os.getenv("PRONUNCIATION_SCORER_DEVICE", os.getenv("PRONUNCIATION_DEVICE", "cpu")))
    scorer_strict_load: bool = field(default_factory=lambda: _env_flag("PRONUNCIATION_SCORER_STRICT_LOAD", "1"))
    scorer_compile: bool = field(default_factory=lambda: _env_flag("PRONUNCIATION_SCORER_COMPILE"))
    scorer_compile_mode: str = field(default_factory=lambda: os.getenv("PRONUNCIATION_SCORER_COMPILE_MODE", "reduce-overhead"))
    scorer_checkpoint_path: Path | None = field(default_factory=lambda: _optional_env_path("PRONUNCIATION_SCORER_CHECKPOINT_PATH"))
    mfa_command: str | None = field(default_factory=lambda: _optional_env_value("PRONUNCIATION_MFA_COMMAND"))
    mfa_acoustic_model: str | None = field(default_factory=lambda: _optional_env_value("PRONUNCIATION_MFA_ACOUSTIC_MODEL"))
    mfa_runtime_dictionary_path: Path | None = field(default_factory=lambda: _optional_env_path("PRONUNCIATION_MFA_RUNTIME_DICTIONARY_PATH"))
    mfa_clean: bool = field(default_factory=lambda: _env_flag("PRONUNCIATION_MFA_CLEAN", "0"))
    mfa_preflight_audio_path: Path | None = field(default_factory=lambda: _optional_env_path("PRONUNCIATION_MFA_PREFLIGHT_AUDIO_PATH"))
    mfa_preflight_word: str | None = field(default_factory=lambda: _optional_env_value("PRONUNCIATION_MFA_PREFLIGHT_WORD"))
    mfa_work_root: Path = field(
        default_factory=lambda: _env_path(
            "PRONUNCIATION_MFA_WORK_ROOT",
            os.getenv("PRONUNCIATION_STORAGE_ROOT", "/cold/pronunciation") + "/runtime/mfa",
        )
    )
    mfa_timeout_seconds: float = field(default_factory=lambda: float(os.getenv("PRONUNCIATION_MFA_TIMEOUT_SECONDS", "30")))
    storage_root: Path = field(default_factory=lambda: _env_path("PRONUNCIATION_STORAGE_ROOT", "/cold/pronunciation"))
    hf_home: Path = field(default_factory=lambda: _env_path("HF_HOME", "/cold/huggingface"))
    dataset_root: Path = field(default_factory=lambda: _env_path("PRONUNCIATION_DATASET_ROOT", os.getenv("PRONUNCIATION_STORAGE_ROOT", "/cold/pronunciation") + "/datasets"))
    feature_root: Path = field(default_factory=lambda: _env_path("PRONUNCIATION_FEATURE_ROOT", os.getenv("PRONUNCIATION_STORAGE_ROOT", "/cold/pronunciation") + "/features"))
    checkpoint_root: Path = field(default_factory=lambda: _env_path("PRONUNCIATION_CHECKPOINT_ROOT", os.getenv("PRONUNCIATION_STORAGE_ROOT", "/cold/pronunciation") + "/checkpoints"))
    report_root: Path = field(default_factory=lambda: _env_path("PRONUNCIATION_REPORT_ROOT", os.getenv("PRONUNCIATION_STORAGE_ROOT", "/cold/pronunciation") + "/reports"))
    lexicon_path: Path = Path(__file__).resolve().parent / "resources" / "en_us_words.json"
    reference_manifest_path: Path = Path(__file__).resolve().parent / "resources" / "reference_audio_manifest.json"

    def validate_runtime(self) -> None:
        if self.runtime_backend != "scorer_v2":
            raise ValueError(f"Unsupported runtime backend: {self.runtime_backend}")
        if self.aligner_backend != "mfa":
            raise ValueError(f"Unsupported aligner backend: {self.aligner_backend}")
        if self.scorer_checkpoint_path is None:
            raise ValueError("PRONUNCIATION_SCORER_CHECKPOINT_PATH must be set for scorer_v2 serving")
        if not self.scorer_checkpoint_path.exists():
            raise FileNotFoundError(f"Configured scorer checkpoint does not exist: {self.scorer_checkpoint_path}")
        if not self.use_hf_encoder:
            raise ValueError("PRONUNCIATION_USE_HF_ENCODER=1 is required for scorer_v2 serving")


settings = Settings()
