from __future__ import annotations

from pathlib import Path

import pytest

from pronunciation_backend.config import Settings


def test_validate_runtime_requires_checkpoint_path() -> None:
    settings = Settings(
        use_hf_encoder=True,
        scorer_checkpoint_path=None,
    )

    with pytest.raises(ValueError, match="PRONUNCIATION_SCORER_CHECKPOINT_PATH"):
        settings.validate_runtime()


def test_validate_runtime_rejects_missing_checkpoint(tmp_path: Path) -> None:
    settings = Settings(
        use_hf_encoder=True,
        scorer_checkpoint_path=tmp_path / "missing.pt",
    )

    with pytest.raises(FileNotFoundError, match="Configured scorer checkpoint does not exist"):
        settings.validate_runtime()


def test_validate_runtime_rejects_unknown_aligner_backend(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "scorer.pt"
    checkpoint_path.write_bytes(b"checkpoint")
    settings = Settings(
        use_hf_encoder=True,
        scorer_checkpoint_path=checkpoint_path,
        aligner_backend="heuristic",
    )

    with pytest.raises(ValueError, match="Unsupported aligner backend"):
        settings.validate_runtime()
