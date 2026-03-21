from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from pronunciation_backend.training.mmap_dataset import (
    ACOUSTIC_FEATURE_DIM,
    MMAP_MANIFEST,
    MmapFeatureManifest,
    resolve_mmap_dataset_dir,
)

PARQUET_SUBDIR = "parquet"
PARQUET_WORDS_NAME = "words.parquet"


def resolve_parquet_dataset_path(features_dir: Path) -> Path | None:
    candidate = features_dir / PARQUET_SUBDIR / PARQUET_WORDS_NAME
    return candidate if candidate.exists() else None


def bake_mmap_to_parquet(
    mmap_dir: Path,
    *,
    output_dir: Path | None = None,
    overwrite: bool = False,
    row_group_utterances: int = 4096,
    progress_every: int = 5000,
    compression: Literal["snappy", "zstd", "gzip", "none"] = "snappy",
) -> Path:
    """
    Materialize mmap-backed .npy tables into a single columnar Parquet file (one row per utterance).
    Training can then read one file with better sequential / OS cache behavior than many npy mmaps.
    """
    manifest_path = mmap_dir / MMAP_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"MMAP manifest not found: {manifest_path}")

    manifest = MmapFeatureManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    output_dir = output_dir or (mmap_dir.parent / PARQUET_SUBDIR)
    words_path = output_dir / PARQUET_WORDS_NAME
    out_manifest_path = output_dir / MMAP_MANIFEST

    output_dir.mkdir(parents=True, exist_ok=True)
    if words_path.exists() or out_manifest_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing parquet output in {output_dir}. Use overwrite=True."
            )
        if words_path.exists():
            words_path.unlink()
        if out_manifest_path.exists():
            out_manifest_path.unlink()

    acoustic = np.load(mmap_dir / "acoustic_features.npy", mmap_mode="r")
    phoneme_ids = np.load(mmap_dir / "phoneme_ids.npy", mmap_mode="r")
    match_targets = np.load(mmap_dir / "match_targets.npy", mmap_mode="r")
    duration_targets = np.load(mmap_dir / "duration_targets.npy", mmap_mode="r")
    presence_targets = np.load(mmap_dir / "presence_targets.npy", mmap_mode="r")
    utterance_offsets = np.load(mmap_dir / "utterance_offsets.npy", mmap_mode="r")

    num_utterances = manifest.num_utterances
    comp = None if compression == "none" else compression
    writer: pq.ParquetWriter | None = None

    def flush_batch(
        batch_acoustic: list,
        batch_phoneme: list,
        batch_match: list,
        batch_dur: list,
        batch_pres: list,
        batch_seq: list,
    ) -> None:
        nonlocal writer
        table = pa.table(
            {
                "acoustic_flat": batch_acoustic,
                "phoneme_ids": batch_phoneme,
                "match_targets": batch_match,
                "duration_targets": batch_dur,
                "presence_targets": batch_pres,
                "seq_len": batch_seq,
            }
        )
        if writer is None:
            writer = pq.ParquetWriter(words_path, table.schema, compression=comp)
        writer.write_table(table)

    batch_acoustic: list = []
    batch_phoneme: list = []
    batch_match: list = []
    batch_dur: list = []
    batch_pres: list = []
    batch_seq: list = []

    for u in range(num_utterances):
        start = int(utterance_offsets[u])
        end = int(utterance_offsets[u + 1])
        seq_len = end - start
        if seq_len <= 0:
            raise ValueError(f"utterance {u}: empty range in mmap offsets")

        ac = np.asarray(acoustic[start:end], dtype=np.float32).reshape(-1)
        if ac.size != seq_len * ACOUSTIC_FEATURE_DIM:
            raise ValueError(f"utterance {u}: bad acoustic size {ac.size} vs {seq_len * ACOUSTIC_FEATURE_DIM}")

        batch_acoustic.append(ac.tolist())
        batch_phoneme.append(np.asarray(phoneme_ids[start:end], dtype=np.int16).tolist())
        batch_match.append(np.asarray(match_targets[start:end], dtype=np.float32).tolist())
        batch_dur.append(np.asarray(duration_targets[start:end], dtype=np.float32).tolist())
        batch_pres.append(np.asarray(presence_targets[start:end], dtype=np.float32).tolist())
        batch_seq.append(np.int32(seq_len))

        if len(batch_acoustic) >= row_group_utterances:
            flush_batch(batch_acoustic, batch_phoneme, batch_match, batch_dur, batch_pres, batch_seq)
            batch_acoustic = []
            batch_phoneme = []
            batch_match = []
            batch_dur = []
            batch_pres = []
            batch_seq = []

        if progress_every > 0 and (u + 1) % progress_every == 0:
            print(f"Baked {u + 1}/{num_utterances} utterances...", flush=True)

    if batch_acoustic:
        flush_batch(batch_acoustic, batch_phoneme, batch_match, batch_dur, batch_pres, batch_seq)

    if writer is not None:
        writer.close()
    else:
        raise RuntimeError("No utterances written; dataset empty?")

    shutil.copyfile(manifest_path, out_manifest_path)
    print(f"Wrote dense Parquet dataset to {words_path}", flush=True)
    return words_path


class WordParquetDataset(Dataset):
    """
    One row per utterance (same samples as WordMemmapDataset). Optional preload copies all rows to RAM.
    """

    def __init__(self, parquet_path: Path, *, preload: bool = False) -> None:
        super().__init__()
        self.parquet_path = parquet_path
        manifest_path = parquet_path.parent / MMAP_MANIFEST
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest next to parquet not found: {manifest_path}")
        self.manifest = MmapFeatureManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        self.preload = preload
        self._rows: list[dict[str, torch.Tensor]] | None = None
        self._table: pa.Table | None = None

        if preload:
            table = pq.read_table(parquet_path, memory_map=False)
            if table.num_rows != self.manifest.num_utterances:
                raise ValueError(
                    f"Parquet rows {table.num_rows} != manifest num_utterances {self.manifest.num_utterances}"
                )
            self._rows = [_row_dict_from_table(table, i) for i in range(table.num_rows)]
        else:
            self._table = pq.read_table(parquet_path, memory_map=True)
            if self._table.num_rows != self.manifest.num_utterances:
                raise ValueError(
                    f"Parquet rows {self._table.num_rows} != manifest num_utterances {self.manifest.num_utterances}"
                )

    def __len__(self) -> int:
        return self.manifest.num_utterances

    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= len(self):
            raise IndexError(index)
        if self._rows is not None:
            return {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in self._rows[index].items()
            }
        assert self._table is not None
        return _row_dict_from_table(self._table, index)


def _row_dict_from_table(table: pa.Table, index: int) -> dict[str, torch.Tensor | int]:
    seq_len = int(table["seq_len"][index].as_py())
    ac_flat = table["acoustic_flat"][index].as_py()
    if len(ac_flat) != seq_len * ACOUSTIC_FEATURE_DIM:
        raise ValueError(f"Row {index}: acoustic_flat length mismatch")
    acoustic_features = torch.tensor(ac_flat, dtype=torch.float32).view(seq_len, ACOUSTIC_FEATURE_DIM)
    phoneme_ids = torch.tensor(table["phoneme_ids"][index].as_py(), dtype=torch.long)
    match_targets = torch.tensor(table["match_targets"][index].as_py(), dtype=torch.float32)
    duration_targets = torch.tensor(table["duration_targets"][index].as_py(), dtype=torch.float32)
    presence_targets = torch.tensor(table["presence_targets"][index].as_py(), dtype=torch.float32)
    return {
        "acoustic_features": acoustic_features,
        "phoneme_ids": phoneme_ids,
        "match_targets": match_targets,
        "duration_targets": duration_targets,
        "presence_targets": presence_targets,
        "seq_len": seq_len,
    }


def resolve_mmap_dir_for_bake(features_dir: Path) -> Path:
    mmap_dir = resolve_mmap_dataset_dir(features_dir)
    if mmap_dir is None:
        raise ValueError(f"No mmap dataset under {features_dir} (expected {MMAP_MANIFEST})")
    return mmap_dir
