from __future__ import annotations

import argparse
import builtins
import os
import socket
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from pronunciation_backend.training.acoustic_encoder_v2 import (
    AcousticEncoderV2,
    RMSNorm,
    sample_mask_positions,
)
from pronunciation_backend.training.dataset import WordIterableDataset, collate_word_batches
from pronunciation_backend.training.mmap_dataset import (
    BlockShuffleBatchSampler,
    WordMemmapDataset,
    resolve_mmap_dataset_dir,
)


def _log(*args, **kwargs) -> None:
    kwargs.setdefault("flush", True)
    builtins.print(*args, **kwargs)


class AcousticEncoderPretrainModel(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 768,
        d_model: int = 384,
        num_heads: int = 6,
        num_layers: int = 6,
        ffn_dim: int = 1_536,
        dropout: float = 0.05,
        rope_base: float = 10_000.0,
        use_qk_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.encoder = AcousticEncoderV2(
            input_dim=input_dim,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            ffn_dim=ffn_dim,
            dropout=dropout,
            rope_base=rope_base,
            use_qk_norm=use_qk_norm,
        )
        self.reconstruction_head = nn.Sequential(
            RMSNorm(d_model),
            nn.Linear(d_model, input_dim, bias=False),
        )

    def forward(
        self,
        acoustic_embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        mask_positions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        encoded = self.encoder(
            acoustic_embeddings=acoustic_embeddings,
            attention_mask=attention_mask,
            mask_positions=mask_positions,
        )
        reconstructed = self.reconstruction_head(encoded)
        return {
            "encoded": encoded,
            "reconstructed": reconstructed,
        }


def _partition_muon_param_groups(model: AcousticEncoderPretrainModel) -> list[dict[str, object]]:
    body_hidden_weights = [
        param
        for param in model.encoder.parameters()
        if param.requires_grad and param.ndim >= 2
    ]
    body_aux_params = [
        param
        for param in model.encoder.parameters()
        if param.requires_grad and param.ndim < 2
    ]
    head_params = [
        param
        for param in model.reconstruction_head.parameters()
        if param.requires_grad
    ]
    return [
        {"params": body_hidden_weights, "use_muon": True},
        {"params": body_aux_params + head_params, "use_muon": False},
    ]


def _reserve_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _prepare_single_process_dist_env() -> dict[str, str]:
    env_updates: dict[str, str] = {}
    if not os.environ.get("MASTER_ADDR"):
        env_updates["MASTER_ADDR"] = "127.0.0.1"
    if not os.environ.get("MASTER_PORT"):
        env_updates["MASTER_PORT"] = str(_reserve_free_port())
    if not os.environ.get("RANK"):
        env_updates["RANK"] = "0"
    if not os.environ.get("WORLD_SIZE"):
        env_updates["WORLD_SIZE"] = "1"
    if not os.environ.get("LOCAL_RANK"):
        env_updates["LOCAL_RANK"] = "0"
    os.environ.update(env_updates)
    return env_updates


def _maybe_init_distributed_for_muon(device: torch.device) -> bool:
    if not torch.distributed.is_available():
        raise RuntimeError("torch.distributed is unavailable, but Muon requires it.")
    if torch.distributed.is_initialized():
        return False

    _prepare_single_process_dist_env()
    backend = "nccl" if device.type == "cuda" else "gloo"
    torch.distributed.init_process_group(backend=backend, init_method="env://")
    return True


def _build_optimizer(
    model: AcousticEncoderPretrainModel,
    *,
    device: torch.device,
    muon_lr: float,
    aux_lr: float,
    weight_decay: float,
    betas: tuple[float, float],
) -> tuple[object, bool]:
    try:
        from muon import MuonWithAuxAdam
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on optional install
        raise ModuleNotFoundError(
            "Muon optimizer is not installed. Install it with "
            "`pip install git+https://github.com/KellerJordan/Muon`."
        ) from exc

    initialized_dist = _maybe_init_distributed_for_muon(device)
    param_groups = _partition_muon_param_groups(model)
    param_groups[0]["lr"] = muon_lr
    param_groups[0]["weight_decay"] = weight_decay
    param_groups[1]["lr"] = aux_lr
    param_groups[1]["betas"] = betas
    param_groups[1]["weight_decay"] = weight_decay
    return MuonWithAuxAdam(param_groups), initialized_dist


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pretrain the v2 acoustic encoder with masked phone reconstruction.")
    parser.add_argument("--features-dir", required=True, help="Path to clean-speech feature split, e.g. /cold/.../libritts/.../splits/train")
    parser.add_argument("--val-features-dir", help="Optional held-out clean-speech validation split.")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--muon-lr", type=float, default=0.02)
    parser.add_argument("--aux-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--mask-ratio", type=float, default=0.20)
    parser.add_argument("--mask-block-size", type=int, default=2)
    parser.add_argument("--min-masks", type=int, default=1)
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
    parser.add_argument("--force-mmap", action="store_true")
    parser.add_argument("--parquet-preload", action="store_true")
    parser.add_argument("--max-batches", type=int)
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
    if not force_mmap:
        try:
            from pronunciation_backend.training.parquet_dataset import resolve_parquet_dataset_path
        except ModuleNotFoundError:
            resolve_parquet_dataset_path = None
        if resolve_parquet_dataset_path is not None:
            parquet_path = resolve_parquet_dataset_path(features_dir)

    if parquet_path is not None:
        return parquet_path, mmap_dir, jsonl_paths
    if mmap_dir is not None:
        return None, mmap_dir, jsonl_paths
    if jsonl_paths:
        return None, None, jsonl_paths
    raise ValueError(f"No parquet, mmap dataset, or part-*.jsonl files found in {features_dir}")


def _describe_split(
    *,
    split_name: str,
    features_dir: Path,
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
        split_name=split_name,
        features_dir=features_dir,
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
        try:
            from pronunciation_backend.training.parquet_dataset import WordParquetDataset
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Parquet features were found but pyarrow is unavailable. Re-run with --force-mmap or install the ml extras."
            ) from exc
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
    acoustic = batch["acoustic_features"][..., :768]
    return {
        "acoustic_embeddings": acoustic.to(device, dtype=torch.float32, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, dtype=torch.bool, non_blocking=True),
    }


def _masked_reconstruction_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    mask_positions: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have identical shapes, got {tuple(prediction.shape)} vs {tuple(target.shape)}"
        )
    if mask_positions.ndim != 2:
        raise ValueError(f"mask_positions must have shape [batch, seq], got {tuple(mask_positions.shape)}")

    if not torch.any(mask_positions):
        raise ValueError("At least one position must be masked to compute reconstruction loss.")

    diff = prediction - target
    per_token = diff.pow(2).mean(dim=-1)
    masked = per_token[mask_positions]
    return masked.mean()


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
    model: AcousticEncoderPretrainModel,
    batches: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    optimizer,
    log_every: int,
    phase: str,
    mask_ratio: float,
    mask_block_size: int,
    min_masks: int,
    rng_seed: int,
    max_batches: int | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(mode=training)

    total_loss = 0.0
    total_steps = 0
    total_words = 0
    total_masked_tokens = 0
    words_since_log = 0
    start_time = time.time()
    rng = _make_rng(rng_seed)

    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for batch_index, batch in enumerate(batches):
            if max_batches is not None and batch_index >= max_batches:
                break

            batch_words = int(batch["attention_mask"].size(0))
            total_words += batch_words
            words_since_log += batch_words

            moved = _move_batch_to_device(batch, device)
            mask_positions = sample_mask_positions(
                moved["attention_mask"],
                mask_ratio=mask_ratio,
                block_size=mask_block_size,
                min_masks=min_masks,
                generator=rng,
            )
            total_masked_tokens += int(mask_positions.sum().item())

            if training:
                optimizer.zero_grad()

            outputs = model(
                moved["acoustic_embeddings"],
                moved["attention_mask"],
                mask_positions=mask_positions,
            )
            loss = _masked_reconstruction_loss(
                outputs["reconstructed"],
                moved["acoustic_embeddings"],
                mask_positions=mask_positions,
            )

            if training:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            total_steps += 1

            if log_every > 0 and total_steps % log_every == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                _log(
                    f"{phase.capitalize()} Step {total_steps:05d} | "
                    f"Reconstruction L: {loss.item():.6f} | "
                    f"Masked tokens: {int(mask_positions.sum().item())} | "
                    f"{words_since_log / elapsed:.1f} words/s"
                )
                start_time = time.time()
                words_since_log = 0

    if total_steps == 0:
        raise RuntimeError(f"{phase} dataloader produced zero batches.")

    return {
        "steps": float(total_steps),
        "words": float(total_words),
        "masked_tokens": float(total_masked_tokens),
        "reconstruction_loss": total_loss / total_steps,
    }


def _save_checkpoint(
    path: Path,
    *,
    epoch: int,
    model: AcousticEncoderPretrainModel,
    optimizer,
    train_metrics: dict[str, float],
    val_metrics: dict[str, float] | None,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "config": vars(args),
        },
        path,
    )


def main() -> int:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = AcousticEncoderPretrainModel(dropout=args.dropout).to(device)
    optimizer, initialized_dist_for_muon = _build_optimizer(
        model,
        device=device,
        muon_lr=args.muon_lr,
        aux_lr=args.aux_lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2),
    )

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

    best_val_loss = float("inf")
    try:
        for epoch in range(args.epochs):
            _log(f"--- Epoch {epoch + 1}/{args.epochs} ---")
            if train_batch_sampler is not None:
                train_batch_sampler.set_epoch(epoch)

            train_metrics = _run_epoch(
                model=model,
                batches=train_loader,
                device=device,
                optimizer=optimizer,
                log_every=args.log_every,
                phase="train",
                mask_ratio=args.mask_ratio,
                mask_block_size=args.mask_block_size,
                min_masks=args.min_masks,
                rng_seed=args.train_seed + epoch,
                max_batches=args.max_batches,
            )
            _log(
                f"Train Summary | Epoch {epoch + 1} | "
                f"Steps: {int(train_metrics['steps'])} | "
                f"Words: {int(train_metrics['words'])} | "
                f"Masked tokens: {int(train_metrics['masked_tokens'])} | "
                f"Recon: {train_metrics['reconstruction_loss']:.6f}"
            )

            val_metrics = None
            if val_batches is not None:
                val_metrics = _run_epoch(
                    model=model,
                    batches=val_batches,
                    device=device,
                    optimizer=None,
                    log_every=args.log_every,
                    phase="val",
                    mask_ratio=args.mask_ratio,
                    mask_block_size=args.mask_block_size,
                    min_masks=args.min_masks,
                    rng_seed=args.val_seed,
                    max_batches=args.max_batches,
                )
                _log(
                    f"Val Summary   | Epoch {epoch + 1} | "
                    f"Steps: {int(val_metrics['steps'])} | "
                    f"Words: {int(val_metrics['words'])} | "
                    f"Masked tokens: {int(val_metrics['masked_tokens'])} | "
                    f"Recon: {val_metrics['reconstruction_loss']:.6f}"
                )

            epoch_ckpt_path = checkpoint_dir / f"acoustic_encoder_v2_epoch_{epoch + 1}.pt"
            _save_checkpoint(
                epoch_ckpt_path,
                epoch=epoch + 1,
                model=model,
                optimizer=optimizer,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                args=args,
            )
            _log(f"Saved checkpoint to {epoch_ckpt_path}")

            if val_metrics is not None and val_metrics["reconstruction_loss"] < best_val_loss:
                best_val_loss = val_metrics["reconstruction_loss"]
                best_ckpt_path = checkpoint_dir / "acoustic_encoder_v2_best.pt"
                _save_checkpoint(
                    best_ckpt_path,
                    epoch=epoch + 1,
                    model=model,
                    optimizer=optimizer,
                    train_metrics=train_metrics,
                    val_metrics=val_metrics,
                    args=args,
                )
                _log(
                    f"New best validation checkpoint saved to {best_ckpt_path} "
                    f"(reconstruction_loss={best_val_loss:.6f})"
                )
    finally:
        if initialized_dist_for_muon and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
