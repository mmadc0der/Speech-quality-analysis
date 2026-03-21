import argparse
import builtins
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from pronunciation_backend.training.dataset import WordIterableDataset, collate_word_batches
from pronunciation_backend.training.mmap_dataset import (
    BlockShuffleBatchSampler,
    WordMemmapDataset,
    resolve_mmap_dataset_dir,
)
from pronunciation_backend.training.parquet_dataset import (
    WordParquetDataset,
    resolve_parquet_dataset_path,
)
from pronunciation_backend.training.scorer_model import PhonemeScorerModel


def _log(*args, **kwargs) -> None:
    kwargs.setdefault("flush", True)
    builtins.print(*args, **kwargs)


def apply_negative_sampling(
    acoustic_features: torch.Tensor,
    phoneme_ids: torch.Tensor,
    match_targets: torch.Tensor,
    presence_targets: torch.Tensor,
    attention_mask: torch.Tensor,
    prob: float = 0.15,
    rng: torch.Generator | None = None,
):
    """
    Applies self-supervised negative sampling to simulate mispronunciations and omissions.
    Since LibriTTS has perfect native speech, we need to artificially inject errors
    so the model learns what "bad" sounds like.
    """
    batch_size, seq_len = phoneme_ids.size()
    device = phoneme_ids.device
    
    # Random floats for deciding augmentations
    rand_tensor = torch.rand(batch_size, seq_len, generator=rng, device="cpu").to(device=device, non_blocking=True)
    
    # 1. Substitution (prob/2): We tell the model to expect the WRONG phoneme.
    # It should learn that the acoustics don't match the expected phoneme, so match_score drops.
    sub_mask = (rand_tensor < (prob / 2)) & attention_mask
    random_phonemes = torch.randint(2, 42, (batch_size, seq_len), generator=rng, device="cpu").to(device=device, non_blocking=True) # 2-41 are valid phonemes
    phoneme_ids = torch.where(sub_mask, random_phonemes, phoneme_ids)
    match_targets = torch.where(sub_mask, torch.tensor(15.0, device=device), match_targets)
    
    # 2. Omission (prob/2): We zero out the acoustic features to simulate silence.
    # It should learn to output presence=0 and match_score=0.
    omit_mask = ((rand_tensor >= (prob / 2)) & (rand_tensor < prob)) & attention_mask
    acoustic_features = acoustic_features.clone()
    acoustic_features[omit_mask] = 0.0
    presence_targets = torch.where(omit_mask, torch.tensor(0.0, device=device), presence_targets)
    match_targets = torch.where(omit_mask, torch.tensor(0.0, device=device), match_targets)
    
    return acoustic_features, phoneme_ids, match_targets, presence_targets

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", required=True, help="Path to the feature store split (e.g. /cold/.../splits/train)")
    parser.add_argument("--val-features-dir", help="Optional validation feature split. When set, validation runs after every epoch.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--prefetch-factor", type=int, default=4, help="Dataloader prefetch factor")
    parser.add_argument("--checkpoint-dir", required=True, help="Where to save model weights")
    parser.add_argument("--negative-sampling-prob", type=float, default=0.15, help="Probability of synthetic corruption during training.")
    parser.add_argument("--train-seed", type=int, default=1337, help="Base seed for train-time sampling and shuffling.")
    parser.add_argument("--val-seed", type=int, default=7331, help="Fixed seed for validation-time negative sampling.")
    parser.add_argument(
        "--train-shuffle-mode",
        choices=["none", "block"],
        default="block",
        help="Shuffle strategy for mmap-backed training data.",
    )
    parser.add_argument(
        "--shuffle-block-words",
        type=int,
        default=16_384,
        help="Number of contiguous words to read before shuffling locally inside a block.",
    )
    parser.add_argument(
        "--force-mmap",
        action="store_true",
        help="Use mmap .npy tables even if <features-dir>/parquet/words.parquet exists.",
    )
    parser.add_argument(
        "--parquet-preload",
        action="store_true",
        help="Memory-map the full Parquet file as one Arrow table (faster row access than row-group reads; use --num-workers 0 if unsure).",
    )
    return parser


def _resolve_split(
    features_dir: Path,
    *,
    split_name: str,
    force_mmap: bool,
) -> tuple[Path | None, Path | None, list[Path]]:
    """
    Returns (parquet_path, mmap_dir, jsonl_paths).
    Prefers Parquet when <features-dir>/parquet/words.parquet exists unless force_mmap is set.
    """
    if not features_dir.exists():
        raise FileNotFoundError(f"{split_name} features dir not found: {features_dir}")

    mmap_dir = resolve_mmap_dataset_dir(features_dir)
    jsonl_paths = sorted(list(features_dir.glob("part-*.jsonl")))
    parquet_path: Path | None = None
    if not force_mmap:
        parquet_path = resolve_parquet_dataset_path(features_dir)

    if parquet_path is not None:
        return parquet_path, mmap_dir, jsonl_paths
    if mmap_dir is not None:
        return None, mmap_dir, jsonl_paths
    if jsonl_paths:
        return None, None, jsonl_paths
    raise ValueError(f"No parquet, mmap dataset, or part-*.jsonl files found in {features_dir}")


def _describe_split(
    features_dir: Path,
    *,
    split_name: str,
    parquet_path: Path | None,
    mmap_dir: Path | None,
    jsonl_paths: list[Path],
) -> None:
    if parquet_path is not None:
        _log(f"{split_name}: using dense parquet dataset from {parquet_path}")
    elif mmap_dir is not None:
        _log(f"{split_name}: using mmap dataset from {mmap_dir}")
    else:
        _log(f"{split_name}: found {len(jsonl_paths)} JSONL feature shard(s) in {features_dir}")


def _build_dataloader(
    *,
    features_dir: Path,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    split_name: str,
    shuffle_mode: str,
    sampler_seed: int,
    shuffle_block_words: int,
    force_mmap: bool,
    parquet_preload: bool,
) -> tuple[DataLoader, BlockShuffleBatchSampler | None]:
    parquet_path, mmap_dir, jsonl_paths = _resolve_split(
        features_dir, split_name=split_name, force_mmap=force_mmap
    )
    _describe_split(
        features_dir,
        split_name=split_name,
        parquet_path=parquet_path,
        mmap_dir=mmap_dir,
        jsonl_paths=jsonl_paths,
    )
    if parquet_preload and num_workers > 0:
        _log("Warning: --parquet-preload with --num-workers > 0 can duplicate mmap/worker overhead; prefer --num-workers 0.")

    common_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if parquet_path is not None:
        dataset = WordParquetDataset(parquet_path, preload=parquet_preload)
        if shuffle_mode == "block":
            batch_sampler = BlockShuffleBatchSampler(
                len(dataset),
                batch_size=batch_size,
                block_words=shuffle_block_words,
                seed=sampler_seed,
            )
            return (
                DataLoader(
                    dataset,
                    batch_sampler=batch_sampler,
                    collate_fn=collate_word_batches,
                    **common_kwargs,
                ),
                batch_sampler,
            )

        return (
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_word_batches,
                **common_kwargs,
            ),
            None,
        )

    if mmap_dir is not None:
        dataset = WordMemmapDataset(mmap_dir)
        if shuffle_mode == "block":
            batch_sampler = BlockShuffleBatchSampler(
                len(dataset),
                batch_size=batch_size,
                block_words=shuffle_block_words,
                seed=sampler_seed,
            )
            return (
                DataLoader(
                    dataset,
                    batch_sampler=batch_sampler,
                    collate_fn=collate_word_batches,
                    **common_kwargs,
                ),
                batch_sampler,
            )

        return (
            DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_word_batches,
                **common_kwargs,
            ),
            None,
        )

    dataset = WordIterableDataset(jsonl_paths, batch_size=batch_size)
    if shuffle_mode != "none":
        _log(f"{split_name}: block shuffle is disabled because JSONL streaming uses IterableDataset.")
    return (
        DataLoader(
            dataset,
            batch_size=None,
            collate_fn=collate_word_batches,
            **common_kwargs,
        ),
        None,
    )


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "acoustic_features": batch["acoustic_features"].to(device, dtype=torch.float32, non_blocking=True),
        "phoneme_ids": batch["phoneme_ids"].to(device, dtype=torch.long, non_blocking=True),
        "match_targets": batch["match_targets"].to(device, dtype=torch.float32, non_blocking=True),
        "duration_targets": batch["duration_targets"].to(device, dtype=torch.float32, non_blocking=True),
        "presence_targets": batch["presence_targets"].to(device, dtype=torch.float32, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
    }


def _masked_mean(loss_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (loss_tensor * mask).sum() / max(1, mask.sum())


def _make_rng(seed: int) -> torch.Generator:
    rng = torch.Generator(device="cpu")
    rng.manual_seed(seed)
    return rng


def _cache_batches(dataloader: DataLoader, *, split_name: str) -> list[dict[str, torch.Tensor]]:
    started_at = time.time()
    cached_batches: list[dict[str, torch.Tensor]] = []
    total_words = 0
    for batch in dataloader:
        cached_batch = {key: value.detach().clone() for key, value in batch.items()}
        cached_batches.append(cached_batch)
        total_words += int(batch["attention_mask"].size(0))
    elapsed = max(time.time() - started_at, 1e-6)
    _log(
        f"{split_name}: cached {len(cached_batches)} batch(es) / {total_words} words in CPU RAM "
        f"({total_words / elapsed:.1f} words/s)"
    )
    return cached_batches


def _run_epoch(
    *,
    model: PhonemeScorerModel,
    batches: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    reg_loss_fn: nn.Module,
    bce_loss_fn: nn.Module,
    optimizer: AdamW | None,
    log_every: int,
    phase: str,
    negative_sampling_prob: float,
    rng_seed: int,
) -> dict[str, float]:
    training = optimizer is not None
    if training:
        model.train()
    else:
        model.eval()

    total_match_loss = 0.0
    total_dur_loss = 0.0
    total_pres_loss = 0.0
    total_loss = 0.0
    total_steps = 0
    total_words = 0
    words_since_log = 0
    start_time = time.time()
    rng = _make_rng(rng_seed)

    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for batch in batches:
            batch_words = int(batch["attention_mask"].size(0))
            total_words += batch_words
            words_since_log += batch_words

            moved = _move_batch_to_device(batch, device)
            acoustics = moved["acoustic_features"]
            p_ids = moved["phoneme_ids"]
            matches = moved["match_targets"]
            durations = moved["duration_targets"]
            presences = moved["presence_targets"]
            mask = moved["attention_mask"]

            if training and negative_sampling_prob > 0:
                acoustics, p_ids, matches, presences = apply_negative_sampling(
                    acoustics,
                    p_ids,
                    matches,
                    presences,
                    mask,
                    prob=negative_sampling_prob,
                    rng=rng,
                )

            if training:
                optimizer.zero_grad()

            outputs = model(
                acoustic_features=acoustics,
                phoneme_ids=p_ids,
                attention_mask=mask,
            )

            m_loss = _masked_mean(reg_loss_fn(outputs["match_score"], matches), mask)
            d_loss = _masked_mean(reg_loss_fn(outputs["duration_score"], durations), mask)
            p_loss = _masked_mean(bce_loss_fn(outputs["presence_logit"], presences), mask)
            batch_total_loss = m_loss + d_loss + (10.0 * p_loss)

            if training:
                batch_total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_match_loss += m_loss.item()
            total_dur_loss += d_loss.item()
            total_pres_loss += p_loss.item()
            total_loss += batch_total_loss.item()
            total_steps += 1

            if log_every > 0 and total_steps % log_every == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                _log(
                    f"{phase.capitalize()} Step {total_steps:05d} | "
                    f"Match L: {m_loss.item():.2f} | "
                    f"Dur L: {d_loss.item():.2f} | "
                    f"Pres L: {p_loss.item():.4f} | "
                    f"{words_since_log / elapsed:.1f} words/s"
                )
                start_time = time.time()
                words_since_log = 0

    if total_steps == 0:
        raise RuntimeError(f"{phase} dataloader produced zero batches.")

    return {
        "steps": float(total_steps),
        "words": float(total_words),
        "match_loss": total_match_loss / total_steps,
        "duration_loss": total_dur_loss / total_steps,
        "presence_loss": total_pres_loss / total_steps,
        "objective_loss": total_loss / total_steps,
    }


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: PhonemeScorerModel,
    optimizer: AdamW,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float] | None,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        },
        path,
    )

def main():
    args = build_parser().parse_args()
    device = torch.device(args.device)

    train_features_dir = Path(args.features_dir)
    val_features_dir = Path(args.val_features_dir) if args.val_features_dir else None

    model = PhonemeScorerModel().to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    reg_loss_fn = nn.SmoothL1Loss(reduction="none")
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_loader, train_batch_sampler = _build_dataloader(
        features_dir=train_features_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        split_name="train",
        shuffle_mode=args.train_shuffle_mode,
        sampler_seed=args.train_seed,
        shuffle_block_words=args.shuffle_block_words,
        force_mmap=args.force_mmap,
        parquet_preload=args.parquet_preload,
    )
    val_batches = None
    if val_features_dir is not None:
        val_loader, _ = _build_dataloader(
            features_dir=val_features_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            split_name="val",
            shuffle_mode="none",
            sampler_seed=args.val_seed,
            shuffle_block_words=args.shuffle_block_words,
            force_mmap=args.force_mmap,
            parquet_preload=args.parquet_preload,
        )
        val_batches = _cache_batches(val_loader, split_name="val")

    best_val_match_loss = float("inf")

    for epoch in range(args.epochs):
        _log(f"--- Epoch {epoch + 1}/{args.epochs} ---")
        if train_batch_sampler is not None:
            train_batch_sampler.set_epoch(epoch)
        train_metrics = _run_epoch(
            model=model,
            batches=train_loader,
            device=device,
            reg_loss_fn=reg_loss_fn,
            bce_loss_fn=bce_loss_fn,
            optimizer=optimizer,
            log_every=args.log_every,
            phase="train",
            negative_sampling_prob=args.negative_sampling_prob,
            rng_seed=args.train_seed + epoch,
        )
        _log(
            f"Train Summary | Epoch {epoch + 1} | "
            f"Steps: {int(train_metrics['steps'])} | "
            f"Words: {int(train_metrics['words'])} | "
            f"Match: {train_metrics['match_loss']:.4f} | "
            f"Dur: {train_metrics['duration_loss']:.4f} | "
            f"Pres: {train_metrics['presence_loss']:.4f} | "
            f"Objective: {train_metrics['objective_loss']:.4f}"
        )

        val_metrics = None
        if val_batches is not None:
            val_metrics = _run_epoch(
                model=model,
                batches=val_batches,
                device=device,
                reg_loss_fn=reg_loss_fn,
                bce_loss_fn=bce_loss_fn,
                optimizer=None,
                log_every=args.log_every,
                phase="val",
                negative_sampling_prob=args.negative_sampling_prob,
                rng_seed=args.val_seed,
            )
            _log(
                f"Val Summary   | Epoch {epoch + 1} | "
                f"Steps: {int(val_metrics['steps'])} | "
                f"Words: {int(val_metrics['words'])} | "
                f"Match: {val_metrics['match_loss']:.4f} | "
                f"Dur: {val_metrics['duration_loss']:.4f} | "
                f"Pres: {val_metrics['presence_loss']:.4f} | "
                f"Objective: {val_metrics['objective_loss']:.4f}"
            )

        epoch_ckpt_path = checkpoint_dir / f"scorer_epoch_{epoch + 1}.pt"
        _save_checkpoint(
            epoch_ckpt_path,
            epoch=epoch + 1,
            model=model,
            optimizer=optimizer,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
        )
        _log(f"Saved checkpoint to {epoch_ckpt_path}")

        if val_metrics is not None and val_metrics["match_loss"] < best_val_match_loss:
            best_val_match_loss = val_metrics["match_loss"]
            best_ckpt_path = checkpoint_dir / "scorer_best.pt"
            _save_checkpoint(
                best_ckpt_path,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
            )
            _log(
                f"New best validation checkpoint saved to {best_ckpt_path} "
                f"(match_loss={best_val_match_loss:.4f})"
            )

if __name__ == "__main__":
    main()
