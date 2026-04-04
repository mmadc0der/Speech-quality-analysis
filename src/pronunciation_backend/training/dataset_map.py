from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class DatasetPartRecord(BaseModel):
    status: str = "missing"
    source: str | None = None
    source_type: str | None = None
    extracted_path: str | None = None
    markers: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    last_updated: str | None = None


class DatasetRecord(BaseModel):
    dataset: str
    dataset_root: str
    raw_root: str
    prepared_dir: str
    aligned_dir: str
    reports_dir: str
    requested_parts: list[str] = Field(default_factory=list)
    discovered_parts: list[str] = Field(default_factory=list)
    part_records: dict[str, DatasetPartRecord] = Field(default_factory=dict)
    stage_status: dict[str, str] = Field(
        default_factory=lambda: {
            "download": "missing",
            "prepare": "missing",
            "align": "missing",
            "feature_plan": "missing",
            "feature_precompute": "missing",
        }
    )
    stage_paths: dict[str, str] = Field(default_factory=dict)
    integrity: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    last_refreshed: str | None = None


class DatasetMap(BaseModel):
    version: int = 1
    workspace_root: str
    dataset_root: str
    updated_at: str | None = None
    datasets: dict[str, DatasetRecord] = Field(default_factory=dict)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def default_dataset_map_path() -> Path:
    return repo_root() / ".pronunciation_dataset_map.json"


def load_dataset_map(path: Path, *, workspace_root: Path, dataset_root: Path) -> DatasetMap:
    if not path.exists():
        return DatasetMap(
            workspace_root=str(workspace_root),
            dataset_root=str(dataset_root),
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    dataset_map = DatasetMap.model_validate(payload)
    if dataset_map.workspace_root != str(workspace_root):
        dataset_map.workspace_root = str(workspace_root)
    if dataset_map.dataset_root != str(dataset_root):
        dataset_map.dataset_root = str(dataset_root)
    return dataset_map


def save_dataset_map(dataset_map: DatasetMap, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dataset_map.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
