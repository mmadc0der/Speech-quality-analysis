from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from pronunciation_backend.training.dataset import WordIterableDataset, collate_word_batches
from pronunciation_backend.training.mmap_dataset import WordMemmapDataset, resolve_mmap_dataset_dir
from pronunciation_backend.training.scorer_model import PhonemeScorerModel

CLASS_ORDER = ("wrong_or_missed", "accented", "correct")
ACCENTED_THRESHOLD = 34.25
CORRECT_THRESHOLD = 72.75


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained scorer checkpoint on a feature split.")
    parser.add_argument("--features-dir", required=True, help="Feature split directory, e.g. /cold/.../splits/test")
    parser.add_argument("--checkpoint-path", required=True, help="Path to scorer checkpoint, e.g. scorer_best.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--force-mmap", action="store_true")
    parser.add_argument("--parquet-preload", action="store_true")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--report-path", help="Optional JSON summary output path.")
    return parser


def _class_from_regression_target(target: float) -> str:
    if target >= CORRECT_THRESHOLD:
        return "correct"
    if target >= ACCENTED_THRESHOLD:
        return "accented"
    return "wrong_or_missed"


def _predicted_class(match_score: float, presence_prob: float) -> str:
    if presence_prob < 0.5:
        return "wrong_or_missed"
    if match_score >= CORRECT_THRESHOLD:
        return "correct"
    if match_score >= ACCENTED_THRESHOLD:
        return "accented"
    return "wrong_or_missed"


def _safe_mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return float("nan")
    return numerator / denominator


def _pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    x_std = x.std()
    y_std = y.std()
    if x_std == 0.0 or y_std == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    array = np.asarray(values, dtype=np.float64)
    points = (1, 5, 25, 50, 75, 95, 99)
    return {f"p{point}": float(np.percentile(array, point)) for point in points}


def _bucket_histogram(values: list[float], *, bucket_size: int = 10, upper_bound: int = 100) -> dict[str, int]:
    if not values:
        return {}
    histogram: dict[str, int] = {}
    capped_upper = max(bucket_size, upper_bound)
    for start in range(0, capped_upper, bucket_size):
        end = min(capped_upper - 1, start + bucket_size - 1)
        label = f"{start:02d}-{end:02d}"
        histogram[label] = 0
    for value in values:
        clipped = max(0.0, min(float(value), capped_upper - 1e-6))
        start = int(clipped // bucket_size) * bucket_size
        end = min(capped_upper - 1, start + bucket_size - 1)
        histogram[f"{start:02d}-{end:02d}"] += 1
    return histogram


def _load_checkpoint(path: Path, *, device: torch.device) -> PhonemeScorerModel:
    payload = torch.load(path, map_location=device)
    state_dict = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload
    model = PhonemeScorerModel().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "acoustic_features": batch["acoustic_features"].to(device, dtype=torch.float32, non_blocking=True),
        "phoneme_ids": batch["phoneme_ids"].to(device, dtype=torch.long, non_blocking=True),
        "match_targets": batch["match_targets"].to(device, dtype=torch.float32, non_blocking=True),
        "duration_targets": batch["duration_targets"].to(device, dtype=torch.float32, non_blocking=True),
        "presence_targets": batch["presence_targets"].to(device, dtype=torch.float32, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
    }


def _build_eval_dataloader(
    *,
    features_dir: Path,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    force_mmap: bool,
    parquet_preload: bool,
) -> DataLoader:
    common_kwargs = {
        "num_workers": num_workers,
        "prefetch_factor": prefetch_factor if num_workers > 0 else None,
        "pin_memory": True,
        "persistent_workers": num_workers > 0,
    }

    parquet_path = None
    if not force_mmap:
        try:
            from pronunciation_backend.training.parquet_dataset import WordParquetDataset, resolve_parquet_dataset_path
        except ModuleNotFoundError:
            WordParquetDataset = None
            resolve_parquet_dataset_path = None
        if resolve_parquet_dataset_path is not None:
            parquet_path = resolve_parquet_dataset_path(features_dir)
            if parquet_path is not None and WordParquetDataset is not None:
                dataset = WordParquetDataset(parquet_path, preload=parquet_preload)
                return DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    collate_fn=collate_word_batches,
                    **common_kwargs,
                )

    mmap_dir = resolve_mmap_dataset_dir(features_dir)
    if mmap_dir is not None:
        dataset = WordMemmapDataset(mmap_dir)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_word_batches,
            **common_kwargs,
        )

    jsonl_paths = sorted(features_dir.glob("part-*.jsonl"))
    if jsonl_paths:
        dataset = WordIterableDataset(jsonl_paths, batch_size=batch_size)
        return DataLoader(
            dataset,
            batch_size=None,
            collate_fn=collate_word_batches,
            **common_kwargs,
        )

    raise ValueError(f"No parquet, mmap dataset, or part-*.jsonl files found in {features_dir}")


def _summarize_predictions(
    *,
    match_pred: list[float],
    duration_pred: list[float],
    presence_prob: list[float],
    match_target: list[float],
    duration_target: list[float],
    presence_target: list[float],
) -> dict[str, object]:
    total_phones = len(match_pred)
    if not (
        len(duration_pred) == len(presence_prob) == len(match_target) == len(duration_target) == len(presence_target) == total_phones
    ):
        raise ValueError("Prediction/target arrays must have equal length.")
    if total_phones == 0:
        raise ValueError("No phones were evaluated.")

    match_errors = [abs(pred - target) for pred, target in zip(match_pred, match_target)]
    duration_errors = [abs(pred - target) for pred, target in zip(duration_pred, duration_target)]
    match_sq_errors = [(pred - target) ** 2 for pred, target in zip(match_pred, match_target)]
    duration_sq_errors = [(pred - target) ** 2 for pred, target in zip(duration_pred, duration_target)]
    presence_predictions = [1.0 if prob >= 0.5 else 0.0 for prob in presence_prob]
    presence_correct = sum(int(pred == target) for pred, target in zip(presence_predictions, presence_target))

    confusion: dict[str, dict[str, int]] = {true_class: {pred_class: 0 for pred_class in CLASS_ORDER} for true_class in CLASS_ORDER}
    predicted_class_counts: dict[str, int] = {class_name: 0 for class_name in CLASS_ORDER}
    target_class_counts: dict[str, int] = {class_name: 0 for class_name in CLASS_ORDER}
    true_class_stats: dict[str, dict[str, list[float] | int]] = {
        class_name: {
            "count": 0,
            "match_scores": [],
            "duration_scores": [],
            "presence_probs": [],
        }
        for class_name in CLASS_ORDER
    }
    omitted_presence_probs: list[float] = []
    present_presence_probs: list[float] = []
    high_score_error_counts: dict[str, int] = {class_name: 0 for class_name in CLASS_ORDER}

    for pred_match, pred_duration, pred_presence, target_match, target_presence in zip(
        match_pred,
        duration_pred,
        presence_prob,
        match_target,
        presence_target,
    ):
        true_class = _class_from_regression_target(target_match)
        pred_class = _predicted_class(pred_match, pred_presence)
        confusion[true_class][pred_class] += 1
        predicted_class_counts[pred_class] += 1
        target_class_counts[true_class] += 1

        stats = true_class_stats[true_class]
        stats["count"] = int(stats["count"]) + 1
        stats["match_scores"].append(pred_match)  # type: ignore[union-attr]
        stats["duration_scores"].append(pred_duration)  # type: ignore[union-attr]
        stats["presence_probs"].append(pred_presence)  # type: ignore[union-attr]
        if pred_match >= CORRECT_THRESHOLD:
            high_score_error_counts[true_class] += 1

        if target_presence < 0.5:
            omitted_presence_probs.append(pred_presence)
        else:
            present_presence_probs.append(pred_presence)

    summarized_class_stats: dict[str, dict[str, float | int]] = {}
    for class_name, stats in true_class_stats.items():
        count = int(stats["count"])
        match_scores = stats["match_scores"]  # type: ignore[assignment]
        duration_scores = stats["duration_scores"]  # type: ignore[assignment]
        presence_probs = stats["presence_probs"]  # type: ignore[assignment]
        summarized_class_stats[class_name] = {
            "count": count,
            "mean_match_score": _safe_mean(match_scores),
            "mean_duration_score": _safe_mean(duration_scores),
            "mean_presence_prob": _safe_mean(presence_probs),
            "match_percentiles": _percentiles(match_scores),
            "duration_percentiles": _percentiles(duration_scores),
            "presence_percentiles": _percentiles(presence_probs),
            "match_histogram_10pt": _bucket_histogram(match_scores),
            "predicted_correct_rate": _safe_rate(confusion[class_name]["correct"], count),
            "high_match_score_rate_gte_correct_threshold": _safe_rate(high_score_error_counts[class_name], count),
        }

    confusion_rates: dict[str, dict[str, float]] = {}
    for class_name, row in confusion.items():
        row_total = max(1, sum(row.values()))
        confusion_rates[class_name] = {
            pred_class: row[pred_class] / row_total
            for pred_class in CLASS_ORDER
        }

    predicted_class_rates = {
        class_name: predicted_class_counts[class_name] / total_phones
        for class_name in CLASS_ORDER
    }
    target_class_rates = {
        class_name: target_class_counts[class_name] / total_phones
        for class_name in CLASS_ORDER
    }

    mean_match_by_class = {
        class_name: float(summarized_class_stats[class_name]["mean_match_score"])
        for class_name in CLASS_ORDER
    }
    separation = {
        "correct_minus_accented": mean_match_by_class["correct"] - mean_match_by_class["accented"],
        "accented_minus_wrong_or_missed": mean_match_by_class["accented"] - mean_match_by_class["wrong_or_missed"],
        "correct_minus_wrong_or_missed": mean_match_by_class["correct"] - mean_match_by_class["wrong_or_missed"],
    }

    diagnostics: dict[str, object] = {
        "target_class_counts": target_class_counts,
        "target_class_rates": target_class_rates,
        "predicted_class_counts": predicted_class_counts,
        "predicted_class_rates": predicted_class_rates,
        "present_phone_count": len(present_presence_probs),
        "omitted_phone_count": len(omitted_presence_probs),
        "mean_match_separation": separation,
        "degenerate_all_correct_predictions": predicted_class_counts["correct"] == total_phones,
        "weak_rank_correlation": _pearson(match_pred, match_target) < 0.3,
        "collapsed_match_separation": separation["correct_minus_wrong_or_missed"] < 5.0,
        "presence_metric_not_informative": len(omitted_presence_probs) == 0,
    }

    interpretation: list[str] = []
    if diagnostics["degenerate_all_correct_predictions"]:
        interpretation.append(
            "The checkpoint predicts every phone as 'correct'; confusion is fully collapsed into one class."
        )
    if diagnostics["collapsed_match_separation"]:
        interpretation.append(
            "Mean match scores for true classes are nearly identical, so the model is not separating good and bad phones."
        )
    if diagnostics["weak_rank_correlation"]:
        interpretation.append(
            "Match-score correlation with mapped human targets is weak, indicating poor ranking quality on this split."
        )
    if diagnostics["presence_metric_not_informative"]:
        interpretation.append(
            "Presence accuracy is not informative here because the evaluated split contains no omitted-phone positives."
        )
    if not interpretation:
        interpretation.append(
            "No obvious collapse pattern detected from the aggregate diagnostics; inspect confusion, percentiles, and sample errors next."
        )

    return {
        "phones": total_phones,
        "match_mae": _safe_mean(match_errors),
        "match_rmse": math.sqrt(_safe_mean(match_sq_errors)),
        "match_pearson": _pearson(match_pred, match_target),
        "duration_mae": _safe_mean(duration_errors),
        "duration_rmse": math.sqrt(_safe_mean(duration_sq_errors)),
        "presence_accuracy": presence_correct / total_phones,
        "mean_presence_prob_present": _safe_mean(present_presence_probs),
        "mean_presence_prob_omitted": _safe_mean(omitted_presence_probs),
        "true_class_stats": summarized_class_stats,
        "class_confusion_counts": confusion,
        "class_confusion_rates": confusion_rates,
        "diagnostics": diagnostics,
        "interpretation": interpretation,
        "notes": {
            "class_thresholds": {
                "wrong_or_missed_lt": ACCENTED_THRESHOLD,
                "accented_lt": CORRECT_THRESHOLD,
                "correct_gte": CORRECT_THRESHOLD,
            },
            "duration_target_warning": "duration_targets currently mirror regression_target in the feature pipeline.",
            "presence_threshold": 0.5,
        },
    }


def main() -> int:
    args = _build_parser().parse_args()
    device = torch.device(args.device)
    features_dir = Path(args.features_dir)
    checkpoint_path = Path(args.checkpoint_path)

    model = _load_checkpoint(checkpoint_path, device=device)
    loader = _build_eval_dataloader(
        features_dir=features_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        force_mmap=args.force_mmap,
        parquet_preload=args.parquet_preload,
    )

    match_pred: list[float] = []
    duration_pred: list[float] = []
    presence_prob: list[float] = []
    match_target: list[float] = []
    duration_target: list[float] = []
    presence_target: list[float] = []

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            moved = _move_batch_to_device(batch, device)
            outputs = model(
                acoustic_features=moved["acoustic_features"],
                phoneme_ids=moved["phoneme_ids"],
                attention_mask=moved["attention_mask"],
            )
            mask = moved["attention_mask"]

            match_pred.extend(outputs["match_score"][mask].detach().cpu().tolist())
            duration_pred.extend(outputs["duration_score"][mask].detach().cpu().tolist())
            presence_prob.extend(torch.sigmoid(outputs["presence_logit"][mask]).detach().cpu().tolist())
            match_target.extend(moved["match_targets"][mask].detach().cpu().tolist())
            duration_target.extend(moved["duration_targets"][mask].detach().cpu().tolist())
            presence_target.extend(moved["presence_targets"][mask].detach().cpu().tolist())

    summary = _summarize_predictions(
        match_pred=match_pred,
        duration_pred=duration_pred,
        presence_prob=presence_prob,
        match_target=match_target,
        duration_target=duration_target,
        presence_target=presence_target,
    )
    summary["checkpoint_path"] = str(checkpoint_path)
    summary["features_dir"] = str(features_dir)
    summary["device"] = str(device)
    summary["batches_evaluated"] = args.max_batches if args.max_batches is not None else "all"

    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.report_path:
        report_path = Path(args.report_path)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(rendered + "\n", encoding="utf-8")
        print(f"wrote report={report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
