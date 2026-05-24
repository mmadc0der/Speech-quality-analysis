from __future__ import annotations

import argparse
import contextlib
import shutil
import sys
import tarfile
import time
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pronunciation_backend.config import settings
from pronunciation_backend.training.build_libritts_aligned import main as build_libritts_aligned_main
from pronunciation_backend.training.build_speechocean762_aligned import main as build_speechocean762_aligned_main
from pronunciation_backend.training.dataset_map import (
    DatasetMap,
    DatasetPartRecord,
    DatasetRecord,
    default_dataset_map_path,
    load_dataset_map,
    repo_root,
    save_dataset_map,
)
from pronunciation_backend.training.dataset_registry import DatasetPartSpec, DatasetSpec, get_dataset_spec, resolve_requested_parts
from pronunciation_backend.training.feature_store import FeaturePrecomputeSpec, plan_feature_store
from pronunciation_backend.training.prepare_libritts import main as prepare_libritts_main
from pronunciation_backend.training.prepare_speechocean762 import main as prepare_speechocean762_main
from pronunciation_backend.training.prepare_speechocean762_mfa import main as prepare_speechocean762_mfa_main
from pronunciation_backend.training.speechocean_utils import resolve_speechocean_raw_root

STAGE_ORDER = ("download", "prepare", "align", "feature-plan", "feature-precompute", "refresh-map")
DOWNLOAD_CHUNK_SIZE = 1024 * 1024
DOWNLOAD_PROGRESS_EVERY_SECONDS = 15.0
DEFAULT_DOWNLOAD_TIMEOUT_SECONDS = 60.0
DEFAULT_DOWNLOAD_RETRIES = 4
DEFAULT_DOWNLOAD_RETRY_DELAY_SECONDS = 5.0


@dataclass(frozen=True)
class DatasetPaths:
    base: Path
    raw: Path
    prepared: Path
    aligned: Path
    reports: Path


def _now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download or import supported datasets into the canonical raw/prepared/aligned/reports layout, "
            "refresh a local dataset map, and optionally dispatch existing prepare/align/feature stages."
        )
    )
    parser.add_argument("--datasets", nargs="+", required=True, help="Datasets to ingest, e.g. libritts speechocean762.")
    parser.add_argument(
        "--parts",
        nargs="*",
        default=[],
        help="Optional repeated selectors in the form dataset:part1,part2 or dataset:all.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["download"],
        choices=list(STAGE_ORDER),
        help="Stages to run. Defaults to download only.",
    )
    parser.add_argument("--dataset-root", default=str(settings.dataset_root))
    parser.add_argument("--dataset-map-path", default=str(default_dataset_map_path()))
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help="Optional repeated dataset[:part]=path overrides for local archives or extracted directories.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--download-timeout-seconds", type=float, default=DEFAULT_DOWNLOAD_TIMEOUT_SECONDS)
    parser.add_argument("--download-retries", type=int, default=DEFAULT_DOWNLOAD_RETRIES)
    parser.add_argument("--download-retry-delay-seconds", type=float, default=DEFAULT_DOWNLOAD_RETRY_DELAY_SECONDS)

    parser.add_argument("--libritts-textgrid-root", help="Required for LibriTTS aligned artifact building.")
    parser.add_argument("--libritts-cmudict-path", help="Required for LibriTTS aligned artifact building.")
    parser.add_argument("--speechocean-textgrid-root", help="Optional TextGrid root to finish SpeechOcean alignment.")
    parser.add_argument(
        "--speechocean-mfa-corpus-dir",
        help="Optional output directory for the SpeechOcean MFA corpus. Defaults to <dataset>/reports/mfa_corpus.",
    )

    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--backbone-id")
    parser.add_argument("--backbone-revision", default="main")
    parser.add_argument("--adapter-id")
    parser.add_argument("--embedding-source", choices=["hubert", "wav2vec2", "fallback"])
    parser.add_argument("--alignment-source", default="mfa", choices=["mfa", "custom_ctc", "manual"])
    parser.add_argument("--pooling-version", default="phone_mean_v1")
    parser.add_argument("--artifact-schema-version", default="phone_embedding_artifact_v1")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--device", default=settings.device)
    parser.add_argument("--shard-size", type=int, default=2_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-batch-audio-ms", type=int, default=120_000)
    parser.add_argument("--max-utterances", type=int)
    parser.add_argument("--min-audio-ms", type=int, default=100)
    parser.add_argument("--max-audio-ms", type=int, default=30_000)
    parser.add_argument("--feature-progress-every", type=int, default=100)
    return parser


def _dataset_paths(dataset_root: Path, dataset: str) -> DatasetPaths:
    base = dataset_root / dataset
    return DatasetPaths(
        base=base,
        raw=base / "raw",
        prepared=base / "prepared",
        aligned=base / "aligned",
        reports=base / "reports",
    )


def _parse_part_overrides(values: list[str]) -> dict[str, list[str]]:
    overrides: dict[str, list[str]] = {}
    for raw_value in values:
        dataset, separator, parts_raw = raw_value.partition(":")
        if not separator or not parts_raw:
            raise ValueError(f"Invalid --parts entry {raw_value!r}. Expected dataset:part1,part2 or dataset:all.")
        parts = [part.strip() for part in parts_raw.split(",") if part.strip()]
        if not parts:
            raise ValueError(f"Invalid --parts entry {raw_value!r}: no parts were specified.")
        overrides.setdefault(dataset, []).extend(parts)
    return overrides


def _parse_source_overrides(values: list[str]) -> dict[tuple[str, str | None], Path]:
    overrides: dict[tuple[str, str | None], Path] = {}
    for raw_value in values:
        left, separator, right = raw_value.partition("=")
        if not separator or not right:
            raise ValueError(f"Invalid --source entry {raw_value!r}. Expected dataset[:part]=path.")
        dataset, has_part, part = left.partition(":")
        overrides[(dataset, part or None if has_part else None)] = Path(right).expanduser()
    return overrides


def _ensure_dataset_record(dataset_map: DatasetMap, dataset: str, paths: DatasetPaths) -> DatasetRecord:
    record = dataset_map.datasets.get(dataset)
    if record is None:
        record = DatasetRecord(
            dataset=dataset,
            dataset_root=str(paths.base),
            raw_root=str(paths.raw),
            prepared_dir=str(paths.prepared),
            aligned_dir=str(paths.aligned),
            reports_dir=str(paths.reports),
        )
        dataset_map.datasets[dataset] = record
    else:
        record.dataset_root = str(paths.base)
        record.raw_root = str(paths.raw)
        record.prepared_dir = str(paths.prepared)
        record.aligned_dir = str(paths.aligned)
        record.reports_dir = str(paths.reports)
    return record


def _append_note(record: DatasetRecord, note: str) -> None:
    if note not in record.notes:
        record.notes.append(note)


def _persist_dataset_map(dataset_map: DatasetMap, dataset_map_path: Path) -> None:
    dataset_map.updated_at = _now_iso()
    save_dataset_map(dataset_map, dataset_map_path)


def _is_safe_extract_path(target_root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(target_root.resolve())
        return True
    except ValueError:
        return False


def _safe_extract_tar(archive_path: Path, destination: Path) -> None:
    with tarfile.open(archive_path, "r:*") as handle:
        for member in handle.getmembers():
            member_path = destination / member.name
            if not _is_safe_extract_path(destination, member_path):
                raise ValueError(f"Refusing to extract path outside destination: {member.name}")
        handle.extractall(destination, filter="data")


def _safe_extract_zip(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as handle:
        for member_name in handle.namelist():
            member_path = destination / member_name
            if not _is_safe_extract_path(destination, member_path):
                raise ValueError(f"Refusing to extract path outside destination: {member_name}")
        handle.extractall(destination)


def _remove_path(path: Path) -> None:
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink()


def _format_bytes(num_bytes: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
        value /= 1024.0
    return f"{num_bytes}B"


def _format_seconds(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes, secs = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


def _archive_validation_error(path: Path) -> str | None:
    if not path.exists():
        return "missing file"
    if path.stat().st_size <= 0:
        return "empty file"

    lower_name = path.name.lower()
    try:
        if lower_name.endswith((".tar.gz", ".tgz", ".tar", ".tar.bz2", ".tar.xz")):
            with tarfile.open(path, "r:*") as handle:
                handle.getmembers()
            return None
        if lower_name.endswith(".zip"):
            if not zipfile.is_zipfile(path):
                return "not a zip file"
            with zipfile.ZipFile(path) as handle:
                corrupted_member = handle.testzip()
            if corrupted_member is not None:
                return f"corrupt zip member: {corrupted_member}"
            return None
    except Exception as exc:
        return str(exc)
    return None


def _download_once(url: str, destination: Path, *, timeout_seconds: float) -> Path:
    temp_destination = destination.with_suffix(destination.suffix + ".part")
    if temp_destination.exists():
        temp_destination.unlink()

    request = urllib.request.Request(url, headers={"User-Agent": "pronunciation-ingest/1.0"})
    started_at = time.monotonic()
    last_log_at = started_at
    bytes_written = 0

    with urllib.request.urlopen(request, timeout=timeout_seconds) as response, temp_destination.open("wb") as handle:
        content_length_header = response.headers.get("Content-Length")
        total_bytes = int(content_length_header) if content_length_header and content_length_header.isdigit() else None
        size_label = _format_bytes(total_bytes) if total_bytes is not None else "unknown"
        print(f"download started url={url} size={size_label} timeout={timeout_seconds:.0f}s")

        while True:
            chunk = response.read(DOWNLOAD_CHUNK_SIZE)
            if not chunk:
                break
            handle.write(chunk)
            bytes_written += len(chunk)

            now = time.monotonic()
            if now - last_log_at >= DOWNLOAD_PROGRESS_EVERY_SECONDS:
                elapsed = max(now - started_at, 1e-6)
                rate = bytes_written / elapsed
                eta = ((total_bytes - bytes_written) / rate) if total_bytes and rate > 0 else None
                total_label = _format_bytes(total_bytes) if total_bytes is not None else "unknown"
                print(
                    "download progress "
                    f"url={url} bytes={_format_bytes(bytes_written)}/{total_label} "
                    f"rate={_format_bytes(int(rate))}/s elapsed={_format_seconds(elapsed)} eta={_format_seconds(eta)}"
                )
                last_log_at = now

    temp_destination.replace(destination)
    validation_error = _archive_validation_error(destination)
    if validation_error is not None:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f"downloaded archive validation failed: {validation_error}")
    elapsed = max(time.monotonic() - started_at, 1e-6)
    average_rate = bytes_written / elapsed
    print(
        "download complete "
        f"url={url} bytes={_format_bytes(bytes_written)} elapsed={_format_seconds(elapsed)} "
        f"avg_rate={_format_bytes(int(average_rate))}/s path={destination}"
    )
    return destination


def _download_file(
    url: str,
    destination: Path,
    *,
    overwrite: bool,
    timeout_seconds: float,
    retries: int,
    retry_delay_seconds: float,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        validation_error = _archive_validation_error(destination)
        if validation_error is None:
            print(f"reusing download: {destination}")
            return destination
        print(f"discarding cached download: {destination} reason={validation_error}")
        destination.unlink()
    print(f"downloading url={url} -> {destination}")

    attempts = max(1, retries)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            print(f"download attempt={attempt}/{attempts} url={url}")
            return _download_once(url, destination, timeout_seconds=timeout_seconds)
        except Exception as exc:
            last_error = exc
            temp_destination = destination.with_suffix(destination.suffix + ".part")
            if temp_destination.exists():
                temp_destination.unlink()
            if destination.exists() and overwrite:
                destination.unlink()
            if attempt >= attempts:
                break
            sleep_seconds = retry_delay_seconds * (2 ** (attempt - 1))
            print(
                "download retry "
                f"url={url} attempt={attempt}/{attempts} wait={_format_seconds(sleep_seconds)} error={exc}"
            )
            time.sleep(sleep_seconds)

    assert last_error is not None
    raise RuntimeError(f"Failed to download {url} after {attempts} attempts: {last_error}") from last_error


def _copy_contents(source_dir: Path, destination_dir: Path, *, overwrite: bool) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)
    for child in source_dir.iterdir():
        destination = destination_dir / child.name
        if child.is_dir():
            if overwrite and destination.exists():
                _remove_path(destination)
            shutil.copytree(child, destination, dirs_exist_ok=not overwrite)
        else:
            if destination.exists() and not overwrite:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, destination)


def _promote_single_child(root: Path, expected_names: tuple[str, ...], *, overwrite: bool) -> None:
    expected = {name.lower() for name in expected_names}
    while True:
        children = [child for child in root.iterdir() if child.is_dir()]
        if len(children) != 1:
            return
        child = children[0]
        if child.name.lower() not in expected:
            return
        for nested in child.iterdir():
            target = root / nested.name
            if overwrite and target.exists():
                _remove_path(target)
            shutil.move(str(nested), str(target))
        child.rmdir()


def _normalize_raw_layout(spec: DatasetSpec, paths: DatasetPaths, *, overwrite: bool) -> None:
    if not spec.normalization_roots:
        return
    root = paths.base / next(iter(spec.parts.values())).import_subdir
    if root.exists():
        _promote_single_child(root, spec.normalization_roots, overwrite=overwrite)


def _resolve_local_source(
    dataset: str,
    part_name: str,
    part_spec: DatasetPartSpec,
    requested_parts: list[str],
    source_overrides: dict[tuple[str, str | None], Path],
) -> Path | None:
    exact = source_overrides.get((dataset, part_name))
    if exact is not None:
        return exact

    shared = source_overrides.get((dataset, None))
    if shared is None:
        return None
    if len(requested_parts) == 1:
        return shared
    if shared.is_dir():
        if part_spec.filename and (shared / part_spec.filename).exists():
            return shared / part_spec.filename
        if part_spec.copy_target_name and (shared / part_spec.copy_target_name).exists():
            return shared / part_spec.copy_target_name
    return shared


def _import_local_source(source_path: Path, destination_root: Path, part_spec: DatasetPartSpec, *, overwrite: bool) -> None:
    if not source_path.exists():
        raise FileNotFoundError(f"Local source does not exist: {source_path}")

    if source_path.is_dir():
        destination_root.mkdir(parents=True, exist_ok=True)
        if part_spec.copy_target_name:
            target_dir = destination_root / part_spec.copy_target_name
            if overwrite and target_dir.exists():
                _remove_path(target_dir)
            shutil.copytree(source_path, target_dir, dirs_exist_ok=False)
        else:
            _copy_contents(source_path, destination_root, overwrite=overwrite)
        return

    destination_root.mkdir(parents=True, exist_ok=True)
    lower_name = source_path.name.lower()
    if lower_name.endswith((".tar.gz", ".tgz", ".tar", ".tar.bz2", ".tar.xz")):
        _safe_extract_tar(source_path, destination_root)
        return
    if lower_name.endswith(".zip"):
        _safe_extract_zip(source_path, destination_root)
        return
    raise ValueError(f"Unsupported local source type for {source_path}. Expected a directory, .zip, or tar archive.")


def _join_relative(base: Path, parts: tuple[str, ...]) -> Path:
    return base.joinpath(*parts) if parts else base


def _normalized_raw_roots(spec: DatasetSpec, dataset_base: Path) -> list[Path]:
    raw_root = dataset_base / "raw"
    if not raw_root.exists():
        return []
    children = {child.name.lower(): child for child in raw_root.iterdir() if child.is_dir()}
    return [children[name.lower()] for name in spec.normalization_roots if name.lower() in children]


def _marker_candidates(spec: DatasetSpec, dataset_base: Path, marker: str) -> list[Path]:
    marker_path = Path(marker)
    candidates = [dataset_base / marker_path]
    if marker_path.parts[:1] != ("raw",):
        return candidates

    relative_to_raw = marker_path.parts[1:]
    for normalized_root in _normalized_raw_roots(spec, dataset_base):
        candidates.append(_join_relative(normalized_root, relative_to_raw))

    if spec.slug == "speechocean762":
        try:
            raw_root = resolve_speechocean_raw_root(dataset_base)
        except FileNotFoundError:
            raw_root = None
        if raw_root is not None:
            relative_to_corpus = relative_to_raw
            if relative_to_corpus[:1] == (spec.slug,):
                relative_to_corpus = relative_to_corpus[1:]
            candidates.append(_join_relative(raw_root, relative_to_corpus))

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            deduped.append(candidate)
    return deduped


def _part_ready(spec: DatasetSpec, part_spec: DatasetPartSpec, dataset_base: Path) -> bool:
    return all(any(candidate.exists() for candidate in _marker_candidates(spec, dataset_base, marker)) for marker in part_spec.expected_markers)


def _resolved_part_path(spec: DatasetSpec, part_spec: DatasetPartSpec, dataset_base: Path) -> Path:
    if spec.slug == "speechocean762":
        try:
            return resolve_speechocean_raw_root(dataset_base)
        except FileNotFoundError:
            pass
    return dataset_base / part_spec.import_subdir


def _refresh_dataset_record(
    spec: DatasetSpec,
    paths: DatasetPaths,
    record: DatasetRecord,
    *,
    requested_parts: list[str],
    now: str,
) -> None:
    record.requested_parts = requested_parts
    record.discovered_parts = [
        part_name for part_name, part_spec in spec.parts.items() if _part_ready(spec, part_spec, paths.base)
    ]

    for part_name, part_spec in spec.parts.items():
        part_record = record.part_records.setdefault(part_name, DatasetPartRecord())
        part_record.markers = [str(paths.base / marker) for marker in part_spec.expected_markers]
        if _part_ready(spec, part_spec, paths.base):
            part_record.status = "ready"
            if part_record.extracted_path is None:
                part_record.extracted_path = str(_resolved_part_path(spec, part_spec, paths.base))
            if part_record.last_updated is None:
                part_record.last_updated = now

    prepared_manifests = {split: (paths.prepared / f"{split}.jsonl").exists() for split in ("train", "val", "test")}
    aligned_manifests = {split: (paths.aligned / f"{split}.jsonl").exists() for split in ("train", "val", "test")}

    if spec.prepare_support == "download_only":
        record.stage_status["prepare"] = "not_supported"
    elif all(prepared_manifests.values()):
        record.stage_status["prepare"] = "complete"
        record.stage_paths["prepared"] = str(paths.prepared)
    elif any(prepared_manifests.values()):
        record.stage_status["prepare"] = "partial"
    else:
        record.stage_status.setdefault("prepare", "missing")

    if spec.align_support == "download_only":
        record.stage_status["align"] = "not_supported"
    elif all(aligned_manifests.values()):
        record.stage_status["align"] = "complete"
        record.stage_paths["aligned"] = str(paths.aligned)
    elif any(aligned_manifests.values()):
        record.stage_status["align"] = "partial"
    else:
        record.stage_status.setdefault("align", "missing")

    if all(part_name in record.discovered_parts for part_name in requested_parts):
        record.stage_status["download"] = "complete"
    elif any(part_name in record.discovered_parts for part_name in requested_parts):
        record.stage_status["download"] = "partial"
    else:
        record.stage_status.setdefault("download", "missing")

    record.integrity = {
        "requested_parts": requested_parts,
        "discovered_parts": record.discovered_parts,
        "prepared_manifests": prepared_manifests,
        "aligned_manifests": aligned_manifests,
        "raw_exists": paths.raw.exists(),
    }
    record.last_refreshed = now


@contextlib.contextmanager
def _argv_context(argv: list[str]):
    previous = sys.argv[:]
    sys.argv = argv
    try:
        yield
    finally:
        sys.argv = previous


def _run_main(argv: list[str], main_fn) -> int:
    with _argv_context(argv):
        return int(main_fn())


def _run_download_stage(
    spec: DatasetSpec,
    paths: DatasetPaths,
    record: DatasetRecord,
    *,
    requested_parts: list[str],
    source_overrides: dict[tuple[str, str | None], Path],
    overwrite: bool,
    download_timeout_seconds: float,
    download_retries: int,
    download_retry_delay_seconds: float,
) -> None:
    for part_name in requested_parts:
        part_spec = spec.parts[part_name]
        part_record = record.part_records.setdefault(part_name, DatasetPartRecord())
        if _part_ready(spec, part_spec, paths.base) and not overwrite:
            part_record.status = "ready"
            part_record.extracted_path = str(_resolved_part_path(spec, part_spec, paths.base))
            part_record.last_updated = _now_iso()
            print(f"dataset={spec.slug} part={part_name} status=ready")
            continue

        destination_root = paths.base / part_spec.import_subdir
        local_source = _resolve_local_source(spec.slug, part_name, part_spec, requested_parts, source_overrides)

        if local_source is not None:
            print(f"dataset={spec.slug} part={part_name} source=local path={local_source}")
            _import_local_source(local_source, destination_root, part_spec, overwrite=overwrite)
            part_record.source = str(local_source)
            part_record.source_type = "local"
        elif part_spec.source_url is not None:
            filename = part_spec.filename or Path(urllib.parse.urlparse(part_spec.source_url).path).name
            archive_path = _download_file(
                part_spec.source_url,
                paths.reports / "downloads" / filename,
                overwrite=overwrite,
                timeout_seconds=download_timeout_seconds,
                retries=download_retries,
                retry_delay_seconds=download_retry_delay_seconds,
            )
            _import_local_source(archive_path, destination_root, part_spec, overwrite=overwrite)
            part_record.source = part_spec.source_url
            part_record.source_type = "url"
        else:
            part_record.status = "manual_required"
            part_record.notes = [part_spec.source_hint]
            part_record.last_updated = _now_iso()
            _append_note(record, f"{spec.slug}:{part_name} requires a local source. {part_spec.source_hint}")
            print(f"dataset={spec.slug} part={part_name} status=manual_required")
            continue

        _normalize_raw_layout(spec, paths, overwrite=overwrite)
        part_record.status = "ready" if _part_ready(spec, part_spec, paths.base) else "incomplete"
        part_record.extracted_path = str(_resolved_part_path(spec, part_spec, paths.base))
        part_record.notes = []
        part_record.last_updated = _now_iso()
        print(f"dataset={spec.slug} part={part_name} status={part_record.status}")


def _run_prepare_stage(spec: DatasetSpec, paths: DatasetPaths, record: DatasetRecord, *, overwrite: bool) -> None:
    if spec.slug == "libritts":
        argv = [
            "prepare_libritts",
            "--dataset-root",
            str(paths.raw),
            "--output-dir",
            str(paths.prepared),
        ]
        if overwrite:
            argv.append("--overwrite")
        exit_code = _run_main(argv, prepare_libritts_main)
        if exit_code != 0:
            raise RuntimeError(f"LibriTTS prepare stage failed with exit code {exit_code}.")
        record.stage_status["prepare"] = "complete"
        record.stage_paths["prepared"] = str(paths.prepared)
        return

    if spec.slug == "speechocean762":
        argv = [
            "prepare_speechocean762",
            "--dataset-root",
            str(paths.base),
            "--output-dir",
            str(paths.prepared),
        ]
        if overwrite:
            argv.append("--overwrite")
        exit_code = _run_main(argv, prepare_speechocean762_main)
        if exit_code != 0:
            raise RuntimeError(f"SpeechOcean762 prepare stage failed with exit code {exit_code}.")
        record.stage_status["prepare"] = "complete"
        record.stage_paths["prepared"] = str(paths.prepared)
        return

    record.stage_status["prepare"] = "not_supported"
    if spec.slug == "l2_arctic":
        _append_note(record, "L2-ARCTIC raw placement is ready, but prepared-manifest generation is not implemented yet.")
    elif spec.slug == "librispeech":
        _append_note(record, "LibriSpeech raw placement is ready, but prepared-manifest generation is not implemented yet.")


def _run_align_stage(spec: DatasetSpec, paths: DatasetPaths, record: DatasetRecord, args: argparse.Namespace, *, overwrite: bool) -> None:
    if spec.slug == "libritts":
        if not args.libritts_textgrid_root or not args.libritts_cmudict_path:
            record.stage_status["align"] = "waiting_for_prerequisites"
            _append_note(
                record,
                "LibriTTS alignment needs --libritts-textgrid-root and --libritts-cmudict-path to build aligned artifacts.",
            )
            return
        argv = [
            "build_libritts_aligned",
            "--dataset-root",
            str(paths.base),
            "--prepared-dir",
            str(paths.prepared),
            "--output-dir",
            str(paths.aligned),
            "--textgrid-root",
            str(Path(args.libritts_textgrid_root)),
            "--cmudict-path",
            str(Path(args.libritts_cmudict_path)),
        ]
        if overwrite:
            argv.append("--overwrite")
        exit_code = _run_main(argv, build_libritts_aligned_main)
        if exit_code != 0:
            raise RuntimeError(f"LibriTTS aligned stage failed with exit code {exit_code}.")
        record.stage_status["align"] = "complete"
        record.stage_paths["aligned"] = str(paths.aligned)
        return

    if spec.slug == "speechocean762":
        mfa_corpus_dir = Path(args.speechocean_mfa_corpus_dir) if args.speechocean_mfa_corpus_dir else paths.reports / "mfa_corpus"
        mfa_argv = [
            "prepare_speechocean762_mfa",
            "--dataset-root",
            str(paths.base),
            "--prepared-dir",
            str(paths.prepared),
            "--output-dir",
            str(mfa_corpus_dir),
            "--link-mode",
            "copy",
        ]
        if overwrite:
            mfa_argv.append("--overwrite")
        exit_code = _run_main(mfa_argv, prepare_speechocean762_mfa_main)
        if exit_code != 0:
            raise RuntimeError(f"SpeechOcean762 MFA scaffolding stage failed with exit code {exit_code}.")
        record.stage_paths["mfa_corpus"] = str(mfa_corpus_dir)
        if not args.speechocean_textgrid_root:
            record.stage_status["align"] = "scaffolded"
            _append_note(
                record,
                f"SpeechOcean762 MFA corpus is ready at {mfa_corpus_dir}; run MFA and rerun with --speechocean-textgrid-root to build aligned artifacts.",
            )
            return

        aligned_argv = [
            "build_speechocean762_aligned",
            "--dataset-root",
            str(paths.base),
            "--prepared-dir",
            str(paths.prepared),
            "--output-dir",
            str(paths.aligned),
            "--textgrid-root",
            str(Path(args.speechocean_textgrid_root)),
        ]
        if overwrite:
            aligned_argv.append("--overwrite")
        exit_code = _run_main(aligned_argv, build_speechocean762_aligned_main)
        if exit_code != 0:
            raise RuntimeError(f"SpeechOcean762 aligned stage failed with exit code {exit_code}.")
        record.stage_status["align"] = "complete"
        record.stage_paths["aligned"] = str(paths.aligned)
        return

    record.stage_status["align"] = "not_supported"
    if spec.slug == "l2_arctic":
        _append_note(record, "L2-ARCTIC alignment is not wired yet; raw data is staged for future adapters.")
    elif spec.slug == "librispeech":
        _append_note(record, "LibriSpeech alignment is not wired yet; raw data is staged for future adapters.")


def _require_feature_args(args: argparse.Namespace) -> None:
    if not args.backbone_id:
        raise ValueError("Feature stages require --backbone-id.")
    if not args.embedding_source:
        raise ValueError("Feature stages require --embedding-source.")


def _feature_spec(dataset: str, paths: DatasetPaths, args: argparse.Namespace) -> FeaturePrecomputeSpec:
    _require_feature_args(args)
    return FeaturePrecomputeSpec(
        dataset=dataset,
        dataset_root=str(paths.base),
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


def _run_feature_plan_stage(dataset: str, paths: DatasetPaths, record: DatasetRecord, args: argparse.Namespace) -> None:
    spec = _feature_spec(dataset, paths, args)
    feature_key, manifest_paths = plan_feature_store(spec, create=True, active_settings=settings)
    record.stage_status["feature_plan"] = "complete"
    record.stage_paths["feature_plan"] = str(manifest_paths["split_root"].parent)
    _append_note(record, f"Feature-store plan ready with key {feature_key}.")


def _run_feature_precompute_stage(dataset: str, paths: DatasetPaths, record: DatasetRecord, args: argparse.Namespace, *, overwrite: bool) -> None:
    _require_feature_args(args)
    from pronunciation_backend.training.precompute_features import main as precompute_features_main

    argv = [
        "precompute_features",
        "--dataset",
        dataset,
        "--dataset-root",
        str(paths.base),
        "--splits",
        *args.splits,
        "--backbone-id",
        args.backbone_id,
        "--backbone-revision",
        args.backbone_revision,
        "--embedding-source",
        args.embedding_source,
        "--alignment-source",
        args.alignment_source,
        "--pooling-version",
        args.pooling_version,
        "--artifact-schema-version",
        args.artifact_schema_version,
        "--sample-rate",
        str(args.sample_rate),
        "--device",
        args.device,
        "--shard-size",
        str(args.shard_size),
        "--batch-size",
        str(args.batch_size),
        "--max-batch-audio-ms",
        str(args.max_batch_audio_ms),
        "--min-audio-ms",
        str(args.min_audio_ms),
        "--max-audio-ms",
        str(args.max_audio_ms),
        "--progress-every",
        str(args.feature_progress_every),
    ]
    if args.adapter_id:
        argv.extend(["--adapter-id", args.adapter_id])
    if args.max_utterances is not None:
        argv.extend(["--max-utterances", str(args.max_utterances)])
    if overwrite:
        argv.append("--overwrite")
    exit_code = _run_main(argv, precompute_features_main)
    if exit_code != 0:
        raise RuntimeError(f"Feature precompute stage failed with exit code {exit_code}.")
    record.stage_status["feature_precompute"] = "complete"


def _execute_stage(
    stage: str,
    *,
    spec: DatasetSpec,
    paths: DatasetPaths,
    record: DatasetRecord,
    requested_parts: list[str],
    source_overrides: dict[tuple[str, str | None], Path],
    args: argparse.Namespace,
) -> None:
    if stage == "download":
        _run_download_stage(
            spec,
            paths,
            record,
            requested_parts=requested_parts,
            source_overrides=source_overrides,
            overwrite=args.overwrite,
            download_timeout_seconds=args.download_timeout_seconds,
            download_retries=args.download_retries,
            download_retry_delay_seconds=args.download_retry_delay_seconds,
        )
        return
    if stage == "prepare":
        _run_prepare_stage(spec, paths, record, overwrite=args.overwrite)
        return
    if stage == "align":
        _run_align_stage(spec, paths, record, args, overwrite=args.overwrite)
        return
    if stage == "feature-plan":
        _run_feature_plan_stage(spec.slug, paths, record, args)
        return
    if stage == "feature-precompute":
        _run_feature_precompute_stage(spec.slug, paths, record, args, overwrite=args.overwrite)
        return
    if stage == "refresh-map":
        return
    raise ValueError(f"Unsupported stage: {stage}")


def main() -> int:
    args = _build_parser().parse_args()
    workspace_root = repo_root()
    dataset_root = Path(args.dataset_root)
    dataset_map_path = Path(args.dataset_map_path)
    dataset_root.mkdir(parents=True, exist_ok=True)

    dataset_map = load_dataset_map(dataset_map_path, workspace_root=workspace_root, dataset_root=dataset_root)
    part_overrides = _parse_part_overrides(args.parts)
    source_overrides = _parse_source_overrides(args.source)
    selected_stages = [stage for stage in STAGE_ORDER if stage in set(args.stages)]

    for dataset in args.datasets:
        spec = get_dataset_spec(dataset)
        requested_parts = resolve_requested_parts(spec, part_overrides.get(dataset))
        paths = _dataset_paths(dataset_root, dataset)
        paths.base.mkdir(parents=True, exist_ok=True)
        paths.raw.mkdir(parents=True, exist_ok=True)
        paths.reports.mkdir(parents=True, exist_ok=True)

        record = _ensure_dataset_record(dataset_map, dataset, paths)
        now = _now_iso()
        _refresh_dataset_record(spec, paths, record, requested_parts=requested_parts, now=now)
        _persist_dataset_map(dataset_map, dataset_map_path)

        print(f"dataset={dataset} parts={','.join(requested_parts)} stages={','.join(selected_stages)}")
        for stage in selected_stages:
            _execute_stage(
                stage,
                spec=spec,
                paths=paths,
                record=record,
                requested_parts=requested_parts,
                source_overrides=source_overrides,
                args=args,
            )
            _refresh_dataset_record(spec, paths, record, requested_parts=requested_parts, now=_now_iso())
            _persist_dataset_map(dataset_map, dataset_map_path)
            print(f"dataset={dataset} stage={stage} status={record.stage_status.get(stage.replace('-', '_'), record.stage_status.get(stage, 'ok'))}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
