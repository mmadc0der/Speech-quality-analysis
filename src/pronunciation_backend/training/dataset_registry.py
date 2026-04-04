from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DownloadSupport = Literal["direct", "manual"]
SupportLevel = Literal["full", "scaffold", "download_only"]


@dataclass(frozen=True)
class DatasetPartSpec:
    name: str
    filename: str | None
    source_url: str | None
    import_subdir: str
    expected_markers: tuple[str, ...]
    source_hint: str
    copy_target_name: str | None = None


@dataclass(frozen=True)
class DatasetSpec:
    slug: str
    description: str
    download_support: DownloadSupport
    prepare_support: SupportLevel
    align_support: SupportLevel
    parts: dict[str, DatasetPartSpec]
    normalization_roots: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def _openslr_parts(resource_id: int) -> dict[str, DatasetPartSpec]:
    part_names = (
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
    )
    return {
        name: DatasetPartSpec(
            name=name,
            filename=f"{name}.tar.gz",
            source_url=f"https://openslr.trmal.net/resources/{resource_id}/{name}.tar.gz",
            import_subdir="raw",
            expected_markers=(f"raw/{name}",),
            source_hint=f"Provide a local {name}.tar.gz archive or extracted {name}/ directory.",
            copy_target_name=name,
        )
        for name in part_names
    }


DATASET_REGISTRY: dict[str, DatasetSpec] = {
    "libritts": DatasetSpec(
        slug="libritts",
        description="English read speech at 24 kHz from OpenSLR 60.",
        download_support="direct",
        prepare_support="full",
        align_support="full",
        parts=_openslr_parts(60),
        normalization_roots=("libritts", "librittsr"),
        notes=(
            "All seven official OpenSLR splits are available for download.",
            "Prepare and aligned stages reuse the existing LibriTTS scripts.",
        ),
    ),
    "speechocean762": DatasetSpec(
        slug="speechocean762",
        description="Non-native English pronunciation-scoring corpus.",
        download_support="manual",
        prepare_support="full",
        align_support="scaffold",
        parts={
            "core": DatasetPartSpec(
                name="core",
                filename=None,
                source_url=None,
                import_subdir="raw/speechocean762",
                expected_markers=(
                    "raw/speechocean762/train/wav.scp",
                    "raw/speechocean762/test/wav.scp",
                ),
                source_hint=(
                    "Provide --source speechocean762:core=/path/to/archive_or_root "
                    "for the original corpus layout containing train/test metadata and scores.json."
                ),
                copy_target_name=None,
            )
        },
        normalization_roots=("speechocean762",),
        notes=(
            "The orchestrator normalizes the raw corpus into raw/speechocean762.",
            "Align stage prepares an MFA corpus and builds aligned artifacts when TextGrids are supplied.",
        ),
    ),
    "l2_arctic": DatasetSpec(
        slug="l2_arctic",
        description="Non-native English read-speech corpus with manual annotations.",
        download_support="manual",
        prepare_support="download_only",
        align_support="download_only",
        parts={
            "full": DatasetPartSpec(
                name="full",
                filename=None,
                source_url=None,
                import_subdir="raw",
                expected_markers=("raw/PROMPTS",),
                source_hint=(
                    "Provide --source l2_arctic:full=/path/to/archive_or_root for the full L2-ARCTIC release."
                ),
                copy_target_name=None,
            )
        },
        normalization_roots=("l2-arctic", "l2_arctic", "l2arctic"),
        notes=(
            "This first-pass adapter downloads or imports the full release and normalizes raw placement.",
            "Prepare and align stages are not implemented yet and emit next-step guidance.",
        ),
    ),
    "librispeech": DatasetSpec(
        slug="librispeech",
        description="English read speech from OpenSLR 12.",
        download_support="direct",
        prepare_support="download_only",
        align_support="download_only",
        parts=_openslr_parts(12),
        normalization_roots=("librispeech",),
        notes=(
            "All seven official OpenSLR splits are available for download.",
            "This first-pass adapter currently handles download, extraction, and normalized raw placement only.",
        ),
    ),
}


def get_dataset_spec(dataset: str) -> DatasetSpec:
    try:
        return DATASET_REGISTRY[dataset]
    except KeyError as exc:
        supported = ", ".join(sorted(DATASET_REGISTRY))
        raise KeyError(f"Unsupported dataset {dataset!r}. Supported datasets: {supported}.") from exc


def resolve_requested_parts(spec: DatasetSpec, requested: list[str] | None) -> list[str]:
    if not requested:
        return list(spec.parts)

    resolved: list[str] = []
    for part in requested:
        if part == "all":
            return list(spec.parts)
        if part not in spec.parts:
            supported = ", ".join(sorted(spec.parts))
            raise ValueError(f"Dataset {spec.slug!r} does not support part {part!r}. Supported parts: {supported}.")
        resolved.append(part)
    return resolved
