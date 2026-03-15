from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from pronunciation_backend.config import Settings, settings


class FeaturePrecomputeSpec(BaseModel):
    dataset: str
    dataset_root: str
    splits: list[Literal["train", "val", "test"]]
    backbone_id: str
    backbone_revision: str = "main"
    adapter_id: str | None = None
    embedding_source: Literal["hubert", "wav2vec2", "fallback"]
    alignment_source: Literal["mfa", "custom_ctc", "manual"]
    pooling_version: str = "phone_mean_v1"
    artifact_schema_version: str = "phone_embedding_artifact_v1"
    sample_rate: int = Field(default=16_000, ge=1)


class FeatureStoreState(BaseModel):
    feature_key: str
    status: Literal["planned", "running", "complete"] = "planned"
    split_counts: dict[str, int] = Field(default_factory=dict)


@dataclass(frozen=True)
class FeatureStoreLayout:
    settings: Settings

    def dataset_root(self, dataset: str) -> Path:
        return self.settings.dataset_root / dataset

    def feature_dir(self, dataset: str, feature_key: str) -> Path:
        return self.settings.feature_root / dataset / feature_key

    def compute_feature_key(self, spec: FeaturePrecomputeSpec) -> str:
        canonical = json.dumps(spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def expected_manifest_paths(self, dataset: str, feature_key: str) -> dict[str, Path]:
        base = self.feature_dir(dataset, feature_key)
        return {
            "spec": base / "spec.json",
            "state": base / "state.json",
            "logs": base / "logs",
            "split_root": base / "splits",
        }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def plan_feature_store(spec: FeaturePrecomputeSpec, *, create: bool, active_settings: Settings) -> tuple[str, dict[str, Path]]:
    layout = FeatureStoreLayout(active_settings)
    feature_key = layout.compute_feature_key(spec)
    manifest_paths = layout.expected_manifest_paths(spec.dataset, feature_key)

    if create:
        manifest_paths["logs"].mkdir(parents=True, exist_ok=True)
        manifest_paths["split_root"].mkdir(parents=True, exist_ok=True)
        for split in spec.splits:
            (manifest_paths["split_root"] / split).mkdir(parents=True, exist_ok=True)
        _write_json(manifest_paths["spec"], spec.model_dump(mode="json"))
        _write_json(manifest_paths["state"], FeatureStoreState(feature_key=feature_key).model_dump(mode="json"))

    return feature_key, manifest_paths


def verify_feature_store(spec: FeaturePrecomputeSpec, *, active_settings: Settings) -> tuple[bool, list[str]]:
    layout = FeatureStoreLayout(active_settings)
    feature_key = layout.compute_feature_key(spec)
    paths = layout.expected_manifest_paths(spec.dataset, feature_key)
    messages: list[str] = []

    dataset_root = Path(spec.dataset_root)
    if not dataset_root.exists():
        messages.append(f"missing dataset root: {dataset_root}")
    if not paths["spec"].exists():
        messages.append(f"missing spec manifest: {paths['spec']}")
    if not paths["state"].exists():
        messages.append(f"missing state manifest: {paths['state']}")

    split_root = paths["split_root"]
    for split in spec.splits:
        split_dir = split_root / split
        if not split_dir.exists():
            messages.append(f"missing split dir: {split_dir}")

    return len(messages) == 0, messages


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or verify hashed feature-store directories.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", required=True)
    common.add_argument("--dataset-root", required=True)
    common.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    common.add_argument("--backbone-id", required=True)
    common.add_argument("--backbone-revision", default="main")
    common.add_argument("--adapter-id")
    common.add_argument("--embedding-source", required=True, choices=["hubert", "wav2vec2", "fallback"])
    common.add_argument("--alignment-source", default="mfa", choices=["mfa", "custom_ctc", "manual"])
    common.add_argument("--pooling-version", default="phone_mean_v1")
    common.add_argument("--artifact-schema-version", default="phone_embedding_artifact_v1")
    common.add_argument("--sample-rate", type=int, default=16_000)

    subparsers.add_parser("plan", parents=[common], help="Create hashed feature-store manifests.")
    subparsers.add_parser("verify", parents=[common], help="Verify dataset and feature-store manifests.")
    return parser


def _spec_from_args(args: argparse.Namespace) -> FeaturePrecomputeSpec:
    return FeaturePrecomputeSpec(
        dataset=args.dataset,
        dataset_root=args.dataset_root,
        splits=args.splits,
        backbone_id=args.backbone_id,
        backbone_revision=args.backbone_revision,
        adapter_id=args.adapter_id,
        embedding_source=args.embedding_source,
        alignment_source=args.alignment_source,
        pooling_version=args.pooling_version,
        artifact_schema_version=args.artifact_schema_version,
        sample_rate=args.sample_rate,
    )


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    spec = _spec_from_args(args)

    if args.command == "plan":
        feature_key, paths = plan_feature_store(spec, create=True, active_settings=settings)
        print(f"feature_key={feature_key}")
        print(f"feature_dir={paths['split_root'].parent}")
        print(f"spec_path={paths['spec']}")
        print(f"state_path={paths['state']}")
        return 0

    ok, messages = verify_feature_store(spec, active_settings=settings)
    if ok:
        layout = FeatureStoreLayout(settings)
        feature_key = layout.compute_feature_key(spec)
        print(f"verified feature store for {spec.dataset} feature_key={feature_key}")
        return 0

    for message in messages:
        print(message)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
