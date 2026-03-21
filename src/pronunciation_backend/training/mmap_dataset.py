from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from pydantic import BaseModel, Field
from torch.utils.data import Dataset, Sampler

from pronunciation_backend.training.dataset import get_phoneme_id


MMAP_SUBDIR = "mmap"
MMAP_MANIFEST = "manifest.json"
ACOUSTIC_FEATURE_DIM = 771


class MmapFeatureManifest(BaseModel):
    version: int = 1
    num_rows: int = Field(ge=0)
    num_utterances: int = Field(ge=0)
    acoustic_dim: int = Field(default=ACOUSTIC_FEATURE_DIM, ge=1)
    acoustic_dtype: Literal["float16", "float32"] = "float16"
    phoneme_dtype: Literal["int16"] = "int16"
    target_dtype: Literal["float32"] = "float32"


def resolve_mmap_dataset_dir(features_dir: Path) -> Path | None:
    candidates = [features_dir, features_dir / MMAP_SUBDIR]
    for candidate in candidates:
        if (candidate / MMAP_MANIFEST).exists():
            return candidate
    return None


def has_mmap_dataset(features_dir: Path) -> bool:
    return resolve_mmap_dataset_dir(features_dir) is not None


def _iter_json_rows(jsonl_paths: list[Path]):
    for path in jsonl_paths:
        with path.open("r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                yield json.loads(line)


def _count_rows_and_utterances(jsonl_paths: list[Path]) -> tuple[int, int]:
    num_rows = 0
    num_utterances = 0
    current_utterance_id: str | None = None
    for row in _iter_json_rows(jsonl_paths):
        utterance_id = row["utterance_id"]
        if utterance_id != current_utterance_id:
            current_utterance_id = utterance_id
            num_utterances += 1
        num_rows += 1
    return num_rows, num_utterances


def pack_jsonl_split_to_mmap(
    features_dir: Path,
    *,
    output_dir: Path | None = None,
    overwrite: bool = False,
    acoustic_dtype: Literal["float16", "float32"] = "float16",
    progress_every: int = 250_000,
) -> Path:
    jsonl_paths = sorted(features_dir.glob("part-*.jsonl"))
    if not jsonl_paths:
        raise ValueError(f"No part-*.jsonl files found in {features_dir}")

    output_dir = output_dir or (features_dir / MMAP_SUBDIR)
    manifest_path = output_dir / MMAP_MANIFEST
    expected_files = [
        output_dir / "acoustic_features.npy",
        output_dir / "phoneme_ids.npy",
        output_dir / "match_targets.npy",
        output_dir / "duration_targets.npy",
        output_dir / "presence_targets.npy",
        output_dir / "utterance_offsets.npy",
        manifest_path,
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    if not overwrite and any(path.exists() for path in expected_files):
        raise FileExistsError(
            f"Refusing to overwrite existing mmap dataset in {output_dir}. "
            "Use --overwrite to replace it."
        )
    if overwrite:
        for path in expected_files:
            if path.exists():
                path.unlink()

    print(f"Counting JSONL rows in {features_dir}...", flush=True)
    num_rows, num_utterances = _count_rows_and_utterances(jsonl_paths)
    if num_rows == 0:
        raise ValueError(f"No feature rows found in {features_dir}")

    acoustic = np.lib.format.open_memmap(
        output_dir / "acoustic_features.npy",
        mode="w+",
        dtype=np.float16 if acoustic_dtype == "float16" else np.float32,
        shape=(num_rows, ACOUSTIC_FEATURE_DIM),
    )
    phoneme_ids = np.lib.format.open_memmap(
        output_dir / "phoneme_ids.npy",
        mode="w+",
        dtype=np.int16,
        shape=(num_rows,),
    )
    match_targets = np.lib.format.open_memmap(
        output_dir / "match_targets.npy",
        mode="w+",
        dtype=np.float32,
        shape=(num_rows,),
    )
    duration_targets = np.lib.format.open_memmap(
        output_dir / "duration_targets.npy",
        mode="w+",
        dtype=np.float32,
        shape=(num_rows,),
    )
    presence_targets = np.lib.format.open_memmap(
        output_dir / "presence_targets.npy",
        mode="w+",
        dtype=np.float32,
        shape=(num_rows,),
    )
    utterance_offsets = np.lib.format.open_memmap(
        output_dir / "utterance_offsets.npy",
        mode="w+",
        dtype=np.int64,
        shape=(num_utterances + 1,),
    )

    print(
        f"Packing {num_rows} phoneme rows across {num_utterances} utterances into {output_dir}...",
        flush=True,
    )

    utterance_offsets[0] = 0
    row_index = 0
    utterance_index = 0
    current_utterance_id: str | None = None

    for row in _iter_json_rows(jsonl_paths):
        utterance_id = row["utterance_id"]
        if current_utterance_id is None:
            current_utterance_id = utterance_id
        elif utterance_id != current_utterance_id:
            utterance_index += 1
            utterance_offsets[utterance_index] = row_index
            current_utterance_id = utterance_id

        mean_embedding = np.asarray(row["mean_embedding"], dtype=np.float32)
        if mean_embedding.shape != (ACOUSTIC_FEATURE_DIM - 3,):
            raise ValueError(
                f"Expected mean_embedding to have {ACOUSTIC_FEATURE_DIM - 3} values, "
                f"got {mean_embedding.shape} for utterance {utterance_id}."
            )

        acoustic[row_index, :-3] = mean_embedding
        acoustic[row_index, -3] = row["variance"]
        acoustic[row_index, -2] = row["duration_z_score"]
        acoustic[row_index, -1] = row["energy_mean"]

        phoneme_ids[row_index] = get_phoneme_id(row["phoneme"])
        match_targets[row_index] = row["regression_target"]
        duration_targets[row_index] = row["regression_target"]
        presence_targets[row_index] = 1.0 - row["omission_target"]
        row_index += 1

        if progress_every > 0 and row_index % progress_every == 0:
            print(f"Packed {row_index}/{num_rows} rows...", flush=True)

    utterance_offsets[num_utterances] = row_index

    acoustic.flush()
    phoneme_ids.flush()
    match_targets.flush()
    duration_targets.flush()
    presence_targets.flush()
    utterance_offsets.flush()

    manifest = MmapFeatureManifest(
        num_rows=row_index,
        num_utterances=num_utterances,
        acoustic_dtype=acoustic_dtype,
    )
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote mmap dataset manifest to {manifest_path}", flush=True)
    return output_dir


class WordMemmapDataset(Dataset):
    def __init__(self, dataset_dir: Path):
        super().__init__()
        manifest_path = dataset_dir / MMAP_MANIFEST
        if not manifest_path.exists():
            raise FileNotFoundError(f"MMAP manifest not found: {manifest_path}")

        self.dataset_dir = dataset_dir
        self.manifest = MmapFeatureManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        self._acoustic: np.ndarray | None = None
        self._phoneme_ids: np.ndarray | None = None
        self._match_targets: np.ndarray | None = None
        self._duration_targets: np.ndarray | None = None
        self._presence_targets: np.ndarray | None = None
        self._utterance_offsets: np.ndarray | None = None

    def __len__(self) -> int:
        return self.manifest.num_utterances

    def _ensure_open(self) -> None:
        if self._acoustic is not None:
            return

        self._acoustic = np.load(self.dataset_dir / "acoustic_features.npy", mmap_mode="r")
        self._phoneme_ids = np.load(self.dataset_dir / "phoneme_ids.npy", mmap_mode="r")
        self._match_targets = np.load(self.dataset_dir / "match_targets.npy", mmap_mode="r")
        self._duration_targets = np.load(self.dataset_dir / "duration_targets.npy", mmap_mode="r")
        self._presence_targets = np.load(self.dataset_dir / "presence_targets.npy", mmap_mode="r")
        self._utterance_offsets = np.load(self.dataset_dir / "utterance_offsets.npy", mmap_mode="r")

    def __getitem__(self, index: int) -> dict:
        if index < 0 or index >= len(self):
            raise IndexError(index)

        self._ensure_open()
        assert self._acoustic is not None
        assert self._phoneme_ids is not None
        assert self._match_targets is not None
        assert self._duration_targets is not None
        assert self._presence_targets is not None
        assert self._utterance_offsets is not None

        start = int(self._utterance_offsets[index])
        end = int(self._utterance_offsets[index + 1])

        return {
            "acoustic_features": torch.from_numpy(np.array(self._acoustic[start:end], copy=True)),
            "phoneme_ids": torch.from_numpy(np.array(self._phoneme_ids[start:end], copy=True)),
            "match_targets": torch.from_numpy(np.array(self._match_targets[start:end], copy=True)),
            "duration_targets": torch.from_numpy(np.array(self._duration_targets[start:end], copy=True)),
            "presence_targets": torch.from_numpy(np.array(self._presence_targets[start:end], copy=True)),
            "seq_len": end - start,
        }


class BlockShuffleBatchSampler(Sampler[list[int]]):
    """
    Keeps disk access roughly sequential by traversing contiguous index blocks,
    while shuffling sample order inside each block.
    """

    def __init__(
        self,
        dataset_size: int,
        *,
        batch_size: int,
        block_words: int,
        seed: int,
        drop_last: bool = False,
        shuffle_blocks: bool = True,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if block_words <= 0:
            raise ValueError("block_words must be positive")

        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.block_words = max(batch_size, block_words)
        self.seed = seed
        self.drop_last = drop_last
        self.shuffle_blocks = shuffle_blocks
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        total = 0
        for start in range(0, self.dataset_size, self.block_words):
            block_size = min(self.block_words, self.dataset_size - start)
            if self.drop_last:
                total += block_size // self.batch_size
            else:
                total += math.ceil(block_size / self.batch_size)
        return total

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        blocks: list[tuple[int, int]] = []
        for start in range(0, self.dataset_size, self.block_words):
            end = min(start + self.block_words, self.dataset_size)
            blocks.append((start, end))

        if self.shuffle_blocks:
            rng.shuffle(blocks)

        pending_batch: list[int] = []
        for start, end in blocks:
            block_indices = list(range(start, end))
            rng.shuffle(block_indices)
            for index in block_indices:
                pending_batch.append(index)
                if len(pending_batch) == self.batch_size:
                    yield pending_batch
                    pending_batch = []
            if pending_batch and not self.drop_last:
                yield pending_batch
                pending_batch = []
            elif pending_batch and self.drop_last:
                pending_batch = []
