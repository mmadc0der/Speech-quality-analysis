from __future__ import annotations

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
from pronunciation_backend.training.scorer_model_v2 import PhonemeScorerModelV2
from pronunciation_backend.training.scoring_targets import (
    ACCENTED_THRESHOLD,
    CLASS_ORDER,
    CORRECT_THRESHOLD,
)

try:
    from pronunciation_backend.training.parquet_dataset import (
        WordParquetDataset,
        resolve_parquet_dataset_path,
    )
except ModuleNotFoundError:  # pragma: no cover - optional at runtime
    WordParquetDataset = None
    resolve_parquet_dataset_path = None


def _log(*args, **kwargs) -> None:
    kwargs.setdefault("flush", True)
    builtins.print(*args, **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the v2 pronunciation scorer from feature-store splits.")
    parser.add_argument("--features-dir", required=True, help="Path to the feature store split (e.g. /cold/.../splits/train)")
    parser.add_argument("--val-features-dir", help="Optional validation feature split. When set, validation runs after every epoch.")
    parser.add_argument("--checkpoint-dir", required=True, help="Where to save model checkpoints")
    parser.add_argument("--encoder-checkpoint-path", help="Optional acoustic encoder pretrain checkpoint")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--encoder-lr-scale", type=float, default=0.2)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--freeze-encoder-epochs", type=int, default=2)
    parser.add_argument("--omission-loss-weight", type=float, default=0.25)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--train-seed", type=int, default=1337)
    parser.add_argument("--val-seed", type=int, default=7331)
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
        help="Memory-map the full Parquet file as one Arrow table (prefer --num-workers 0 if unsure).",
    )
    return parser


def _resolve_split(
    features_dir: Path,
    *,
    split_name: str,
    force_mmap: bool,
) -> tuple[Path | None, Path | None, list[Path]]:
    if not features_dir.exists():
        raise FileNotFoundError(f"{split_name} features dir not found: {features_dir}")

    mmap_dir = resolve_mmap_dataset_dir(features_dir)
    jsonl_paths = sorted(list(features_dir.glob("part-*.jsonl")))
    parquet_path: Path | None = None
    if not force_mmap and resolve_parquet_dataset_path is not None:
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

    common_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }
    if parquet_path is not None:
        if WordParquetDataset is None:
            raise ModuleNotFoundError(
                "Parquet features were found but pyarrow is unavailable. Re-run with --force-mmap or install the ml extras."
            )
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
    match_targets = batch["match_targets"]
    class_targets = torch.full_like(match_targets, fill_value=2, dtype=torch.long)
    class_targets = torch.where(
        match_targets < CORRECT_THRESHOLD,
        torch.tensor(1, dtype=torch.long),
        class_targets,
    )
    class_targets = torch.where(
        match_targets < ACCENTED_THRESHOLD,
        torch.tensor(0, dtype=torch.long),
        class_targets,
    )

    presence_targets = batch["presence_targets"]
    omission_targets = 1.0 - presence_targets
    return {
        "acoustic_embeddings": batch["acoustic_features"][..., :768].to(device, dtype=torch.float32, non_blocking=True),
        "phoneme_ids": batch["phoneme_ids"].to(device, dtype=torch.long, non_blocking=True),
        "class_targets": class_targets.to(device, dtype=torch.long, non_blocking=True),
        "score_targets": match_targets.to(device, dtype=torch.float32, non_blocking=True),
        "omission_targets": omission_targets.to(device, dtype=torch.float32, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, dtype=torch.bool, non_blocking=True),
    }


def _masked_mean(loss_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weighted = loss_tensor * mask.to(dtype=loss_tensor.dtype)
    return weighted.sum() / max(1, int(mask.sum().item()))


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


def _compute_class_weights(
    dataloader: DataLoader,
    *,
    device: torch.device,
) -> torch.Tensor:
    counts = torch.zeros(len(CLASS_ORDER), dtype=torch.long)
    for batch in dataloader:
        moved = _move_batch_to_device(batch, device=torch.device("cpu"))
        mask = moved["attention_mask"]
        targets = moved["class_targets"][mask]
        counts += torch.bincount(targets, minlength=len(CLASS_ORDER))

    counts = counts.clamp_min(1)
    total = counts.sum().to(dtype=torch.float32)
    weights = total / (len(CLASS_ORDER) * counts.to(dtype=torch.float32))
    weights = weights / weights.mean()
    return weights.to(device=device)


def _build_optimizer(
    model: PhonemeScorerModelV2,
    *,
    lr: float,
    encoder_lr_scale: float,
    weight_decay: float,
) -> AdamW:
    encoder_params = [
        param for param in model.acoustic_encoder.parameters() if param.requires_grad
    ]
    other_params = [
        param
        for name, param in model.named_parameters()
        if param.requires_grad and not name.startswith("acoustic_encoder.")
    ]
    param_groups: list[dict[str, object]] = []
    if encoder_params:
        param_groups.append(
            {
                "params": encoder_params,
                "lr": lr * encoder_lr_scale,
                "weight_decay": weight_decay,
            }
        )
    if other_params:
        param_groups.append(
            {
                "params": other_params,
                "lr": lr,
                "weight_decay": weight_decay,
            }
        )
    return AdamW(param_groups)


def _maybe_load_pretrained_encoder(
    model: PhonemeScorerModelV2,
    *,
    checkpoint_path: str | None,
    device: torch.device,
) -> None:
    if not checkpoint_path:
        return
    payload = torch.load(checkpoint_path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported encoder checkpoint payload type: {type(payload)!r}")
    model.load_pretrained_acoustic_encoder(payload)
    _log(f"Loaded pretrained acoustic encoder from {checkpoint_path}")


def _update_encoder_trainability(
    *,
    model: PhonemeScorerModelV2,
    epoch_index: int,
    freeze_encoder_epochs: int,
) -> bool:
    should_freeze = epoch_index < freeze_encoder_epochs
    currently_trainable = any(param.requires_grad for param in model.acoustic_encoder.parameters())
    desired_trainable = not should_freeze
    if currently_trainable == desired_trainable:
        return False
    model.set_acoustic_encoder_trainable(desired_trainable)
    return True


def _run_epoch(
    *,
    model: PhonemeScorerModelV2,
    batches: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    class_loss_fn: nn.Module,
    omission_loss_fn: nn.Module,
    omission_loss_weight: float,
    optimizer: AdamW | None,
    log_every: int,
    phase: str,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(mode=training)

    total_quality_loss = 0.0
    total_omission_loss = 0.0
    total_score_mae = 0.0
    total_loss = 0.0
    total_steps = 0
    total_words = 0
    total_tokens = 0
    total_correct = 0
    words_since_log = 0
    start_time = time.time()

    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for batch in batches:
            batch_words = int(batch["attention_mask"].size(0))
            total_words += batch_words
            words_since_log += batch_words

            moved = _move_batch_to_device(batch, device)
            mask = moved["attention_mask"]
            logits_mask = mask.unsqueeze(-1)

            if training:
                optimizer.zero_grad()

            outputs = model(
                acoustic_embeddings=moved["acoustic_embeddings"],
                phoneme_ids=moved["phoneme_ids"],
                attention_mask=mask,
            )

            quality_loss = class_loss_fn(
                outputs["quality_logits"].reshape(-1, len(CLASS_ORDER)),
                moved["class_targets"].reshape(-1),
            ).view_as(mask)
            omission_loss = omission_loss_fn(outputs["omission_logit"], moved["omission_targets"])
            batch_quality_loss = _masked_mean(quality_loss, mask)
            batch_omission_loss = _masked_mean(omission_loss, mask)
            score_mae = _masked_mean((outputs["expected_score"] - moved["score_targets"]).abs(), mask)
            batch_total_loss = batch_quality_loss + (omission_loss_weight * batch_omission_loss)

            if training:
                batch_total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            predictions = outputs["quality_logits"].argmax(dim=-1)
            total_correct += int((predictions[mask] == moved["class_targets"][mask]).sum().item())
            total_tokens += int(mask.sum().item())
            total_quality_loss += batch_quality_loss.item()
            total_omission_loss += batch_omission_loss.item()
            total_score_mae += score_mae.item()
            total_loss += batch_total_loss.item()
            total_steps += 1

            if log_every > 0 and total_steps % log_every == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                _log(
                    f"{phase.capitalize()} Step {total_steps:05d} | "
                    f"Quality L: {batch_quality_loss.item():.4f} | "
                    f"Omit L: {batch_omission_loss.item():.4f} | "
                    f"Score MAE: {score_mae.item():.4f} | "
                    f"{words_since_log / elapsed:.1f} words/s"
                )
                start_time = time.time()
                words_since_log = 0

    if total_steps == 0:
        raise RuntimeError(f"{phase} dataloader produced zero batches.")

    return {
        "steps": float(total_steps),
        "words": float(total_words),
        "tokens": float(total_tokens),
        "quality_loss": total_quality_loss / total_steps,
        "omission_loss": total_omission_loss / total_steps,
        "score_mae": total_score_mae / total_steps,
        "class_accuracy": total_correct / max(1, total_tokens),
        "objective_loss": total_loss / total_steps,
    }


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: PhonemeScorerModelV2,
    optimizer: AdamW,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float] | None,
    class_weights: torch.Tensor,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "class_weights": class_weights.detach().cpu(),
            "config": vars(args),
        },
        path,
    )


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = PhonemeScorerModelV2().to(device)
    _maybe_load_pretrained_encoder(
        model,
        checkpoint_path=args.encoder_checkpoint_path,
        device=device,
    )
    if args.freeze_encoder_epochs > 0:
        model.set_acoustic_encoder_trainable(False)

    train_loader, train_batch_sampler = _build_dataloader(
        features_dir=Path(args.features_dir),
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
    class_weights = _compute_class_weights(train_loader, device=device)
    _log(f"train: class weights={class_weights.detach().cpu().tolist()}")
    optimizer = _build_optimizer(
        model,
        lr=args.lr,
        encoder_lr_scale=args.encoder_lr_scale,
        weight_decay=args.weight_decay,
    )
    class_loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
    omission_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    val_batches = None
    if args.val_features_dir:
        val_loader, _ = _build_dataloader(
            features_dir=Path(args.val_features_dir),
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

    best_val_quality_loss = float("inf")
    for epoch in range(args.epochs):
        _log(f"--- Epoch {epoch + 1}/{args.epochs} ---")
        changed = _update_encoder_trainability(
            model=model,
            epoch_index=epoch,
            freeze_encoder_epochs=args.freeze_encoder_epochs,
        )
        if changed:
            optimizer = _build_optimizer(
                model,
                lr=args.lr,
                encoder_lr_scale=args.encoder_lr_scale,
                weight_decay=args.weight_decay,
            )
            _log(
                "encoder: "
                + ("trainable" if any(param.requires_grad for param in model.acoustic_encoder.parameters()) else "frozen")
            )

        if train_batch_sampler is not None:
            train_batch_sampler.set_epoch(epoch)
        train_metrics = _run_epoch(
            model=model,
            batches=train_loader,
            device=device,
            class_loss_fn=class_loss_fn,
            omission_loss_fn=omission_loss_fn,
            omission_loss_weight=args.omission_loss_weight,
            optimizer=optimizer,
            log_every=args.log_every,
            phase="train",
        )
        _log(
            f"Train Summary | Epoch {epoch + 1} | "
            f"Steps: {int(train_metrics['steps'])} | "
            f"Words: {int(train_metrics['words'])} | "
            f"Tokens: {int(train_metrics['tokens'])} | "
            f"Quality: {train_metrics['quality_loss']:.4f} | "
            f"Omit: {train_metrics['omission_loss']:.4f} | "
            f"Score MAE: {train_metrics['score_mae']:.4f} | "
            f"Class Acc: {train_metrics['class_accuracy']:.4f} | "
            f"Objective: {train_metrics['objective_loss']:.4f}"
        )

        val_metrics = None
        if val_batches is not None:
            val_metrics = _run_epoch(
                model=model,
                batches=val_batches,
                device=device,
                class_loss_fn=class_loss_fn,
                omission_loss_fn=omission_loss_fn,
                omission_loss_weight=args.omission_loss_weight,
                optimizer=None,
                log_every=args.log_every,
                phase="val",
            )
            _log(
                f"Val Summary   | Epoch {epoch + 1} | "
                f"Steps: {int(val_metrics['steps'])} | "
                f"Words: {int(val_metrics['words'])} | "
                f"Tokens: {int(val_metrics['tokens'])} | "
                f"Quality: {val_metrics['quality_loss']:.4f} | "
                f"Omit: {val_metrics['omission_loss']:.4f} | "
                f"Score MAE: {val_metrics['score_mae']:.4f} | "
                f"Class Acc: {val_metrics['class_accuracy']:.4f} | "
                f"Objective: {val_metrics['objective_loss']:.4f}"
            )

        epoch_ckpt_path = checkpoint_dir / f"scorer_v2_epoch_{epoch + 1}.pt"
        _save_checkpoint(
            epoch_ckpt_path,
            epoch=epoch + 1,
            model=model,
            optimizer=optimizer,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            class_weights=class_weights,
            args=args,
        )
        _log(f"Saved checkpoint to {epoch_ckpt_path}")

        if val_metrics is not None and val_metrics["quality_loss"] < best_val_quality_loss:
            best_val_quality_loss = val_metrics["quality_loss"]
            best_ckpt_path = checkpoint_dir / "scorer_v2_best.pt"
            _save_checkpoint(
                best_ckpt_path,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                class_weights=class_weights,
                args=args,
            )
            _log(
                f"New best validation checkpoint saved to {best_ckpt_path} "
                f"(quality_loss={best_val_quality_loss:.4f})"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
