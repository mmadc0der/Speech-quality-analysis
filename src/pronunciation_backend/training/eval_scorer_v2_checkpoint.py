from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from pronunciation_backend.training.scorer_model_v2 import PhonemeScorerModelV2
from pronunciation_backend.training.scoring_targets import (
    CLASS_ORDER,
    CORRECT_THRESHOLD,
    class_name_from_index,
)
from pronunciation_backend.training.train_scorer_v2 import _build_dataloader, _move_batch_to_device


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained v2 scorer checkpoint on a feature split.")
    parser.add_argument("--features-dir", required=True, help="Feature split directory, e.g. /cold/.../splits/test")
    parser.add_argument("--checkpoint-path", required=True, help="Path to v2 scorer checkpoint, e.g. scorer_v2_best.pt")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--force-mmap", action="store_true")
    parser.add_argument("--parquet-preload", action="store_true")
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--report-path", help="Optional JSON summary output path.")
    return parser


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


def _load_checkpoint(path: Path, *, device: torch.device) -> PhonemeScorerModelV2:
    payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    state_dict = payload["model_state_dict"] if "model_state_dict" in payload else payload
    model = PhonemeScorerModelV2().to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _summarize_predictions(
    *,
    class_pred: list[int],
    omission_prob: list[float],
    expected_score: list[float],
    class_target: list[int],
    score_target: list[float],
    omission_target: list[float],
) -> dict[str, object]:
    total_phones = len(class_pred)
    if not (
        len(omission_prob) == len(expected_score) == len(class_target) == len(score_target) == len(omission_target) == total_phones
    ):
        raise ValueError("Prediction/target arrays must have equal length.")
    if total_phones == 0:
        raise ValueError("No phones were evaluated.")

    score_errors = [abs(pred - target) for pred, target in zip(expected_score, score_target)]
    score_sq_errors = [(pred - target) ** 2 for pred, target in zip(expected_score, score_target)]
    omission_predictions = [1.0 if prob >= 0.5 else 0.0 for prob in omission_prob]
    omission_correct = sum(int(pred == target) for pred, target in zip(omission_predictions, omission_target))

    confusion: dict[str, dict[str, int]] = {true_class: {pred_class: 0 for pred_class in CLASS_ORDER} for true_class in CLASS_ORDER}
    predicted_class_counts: dict[str, int] = {class_name: 0 for class_name in CLASS_ORDER}
    target_class_counts: dict[str, int] = {class_name: 0 for class_name in CLASS_ORDER}
    true_class_stats: dict[str, dict[str, list[float] | int]] = {
        class_name: {
            "count": 0,
            "expected_scores": [],
            "omission_probs": [],
        }
        for class_name in CLASS_ORDER
    }
    omitted_probs: list[float] = []
    present_probs: list[float] = []
    high_score_error_counts: dict[str, int] = {class_name: 0 for class_name in CLASS_ORDER}

    for pred_index, pred_omit, pred_score, target_index, target_score, target_omit in zip(
        class_pred,
        omission_prob,
        expected_score,
        class_target,
        score_target,
        omission_target,
    ):
        true_class = class_name_from_index(target_index)
        pred_class = class_name_from_index(pred_index)
        confusion[true_class][pred_class] += 1
        predicted_class_counts[pred_class] += 1
        target_class_counts[true_class] += 1

        stats = true_class_stats[true_class]
        stats["count"] = int(stats["count"]) + 1
        stats["expected_scores"].append(pred_score)  # type: ignore[union-attr]
        stats["omission_probs"].append(pred_omit)  # type: ignore[union-attr]
        if pred_score >= CORRECT_THRESHOLD:
            high_score_error_counts[true_class] += 1

        if target_omit >= 0.5:
            omitted_probs.append(pred_omit)
        else:
            present_probs.append(pred_omit)

    summarized_class_stats: dict[str, dict[str, float | int]] = {}
    for class_name, stats in true_class_stats.items():
        count = int(stats["count"])
        scores = stats["expected_scores"]  # type: ignore[assignment]
        omission_probs_for_class = stats["omission_probs"]  # type: ignore[assignment]
        summarized_class_stats[class_name] = {
            "count": count,
            "mean_expected_score": _safe_mean(scores),
            "mean_omission_prob": _safe_mean(omission_probs_for_class),
            "score_percentiles": _percentiles(scores),
            "omission_percentiles": _percentiles(omission_probs_for_class),
            "score_histogram_10pt": _bucket_histogram(scores),
            "predicted_correct_rate": _safe_rate(confusion[class_name]["correct"], count),
            "high_score_rate_gte_correct_threshold": _safe_rate(high_score_error_counts[class_name], count),
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

    mean_score_by_class = {
        class_name: float(summarized_class_stats[class_name]["mean_expected_score"])
        for class_name in CLASS_ORDER
    }
    separation = {
        "correct_minus_accented": mean_score_by_class["correct"] - mean_score_by_class["accented"],
        "accented_minus_wrong_or_missed": mean_score_by_class["accented"] - mean_score_by_class["wrong_or_missed"],
        "correct_minus_wrong_or_missed": mean_score_by_class["correct"] - mean_score_by_class["wrong_or_missed"],
    }

    diagnostics: dict[str, object] = {
        "target_class_counts": target_class_counts,
        "target_class_rates": target_class_rates,
        "predicted_class_counts": predicted_class_counts,
        "predicted_class_rates": predicted_class_rates,
        "present_phone_count": len(present_probs),
        "omitted_phone_count": len(omitted_probs),
        "mean_score_separation": separation,
        "degenerate_all_correct_predictions": predicted_class_counts["correct"] == total_phones,
        "weak_rank_correlation": _pearson(expected_score, score_target) < 0.3,
        "collapsed_score_separation": separation["correct_minus_wrong_or_missed"] < 5.0,
        "omission_metric_not_informative": len(omitted_probs) == 0,
    }

    interpretation: list[str] = []
    if diagnostics["degenerate_all_correct_predictions"]:
        interpretation.append(
            "The checkpoint predicts every phone as 'correct'; class confusion is fully collapsed into one class."
        )
    if diagnostics["collapsed_score_separation"]:
        interpretation.append(
            "Expected scores for true classes are nearly identical, so the model is not separating good and bad phones."
        )
    if diagnostics["weak_rank_correlation"]:
        interpretation.append(
            "Expected-score correlation with mapped dataset targets is weak, indicating poor ranking quality on this split."
        )
    if diagnostics["omission_metric_not_informative"]:
        interpretation.append(
            "Omission accuracy is not informative here because the evaluated split contains no omitted-phone positives."
        )
    if not interpretation:
        interpretation.append(
            "No obvious collapse pattern detected from the aggregate diagnostics; inspect confusion, percentiles, and sample errors next."
        )

    return {
        "phones": total_phones,
        "score_mae": _safe_mean(score_errors),
        "score_rmse": math.sqrt(_safe_mean(score_sq_errors)),
        "score_pearson": _pearson(expected_score, score_target),
        "class_accuracy": sum(int(pred == target) for pred, target in zip(class_pred, class_target)) / total_phones,
        "omission_accuracy": omission_correct / total_phones,
        "mean_omission_prob_present": _safe_mean(present_probs),
        "mean_omission_prob_omitted": _safe_mean(omitted_probs),
        "true_class_stats": summarized_class_stats,
        "class_confusion_counts": confusion,
        "class_confusion_rates": confusion_rates,
        "diagnostics": diagnostics,
        "interpretation": interpretation,
        "notes": {
            "class_order": list(CLASS_ORDER),
            "class_target_scores": [15.0, 60.0, 92.0],
            "omission_threshold": 0.5,
            "expected_score_note": "expected_score is derived directly from class probabilities and dataset-aligned target scores.",
        },
    }


def main() -> int:
    args = _build_parser().parse_args()
    device = torch.device(args.device)
    features_dir = Path(args.features_dir)
    checkpoint_path = Path(args.checkpoint_path)

    model = _load_checkpoint(checkpoint_path, device=device)
    loader, _ = _build_dataloader(
        features_dir=features_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        split_name="eval",
        shuffle_mode="none",
        sampler_seed=0,
        shuffle_block_words=max(args.batch_size, 1024),
        force_mmap=args.force_mmap,
        parquet_preload=args.parquet_preload,
    )

    class_pred: list[int] = []
    omission_prob: list[float] = []
    expected_score: list[float] = []
    class_target: list[int] = []
    score_target: list[float] = []
    omission_target: list[float] = []

    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if args.max_batches is not None and batch_index >= args.max_batches:
                break
            moved = _move_batch_to_device(batch, device)
            outputs = model(
                acoustic_embeddings=moved["acoustic_embeddings"],
                phoneme_ids=moved["phoneme_ids"],
                attention_mask=moved["attention_mask"],
            )
            mask = moved["attention_mask"]

            class_pred.extend(outputs["quality_logits"].argmax(dim=-1)[mask].detach().cpu().tolist())
            omission_prob.extend(torch.sigmoid(outputs["omission_logit"][mask]).detach().cpu().tolist())
            expected_score.extend(outputs["expected_score"][mask].detach().cpu().tolist())
            class_target.extend(moved["class_targets"][mask].detach().cpu().tolist())
            score_target.extend(moved["score_targets"][mask].detach().cpu().tolist())
            omission_target.extend(moved["omission_targets"][mask].detach().cpu().tolist())

    summary = _summarize_predictions(
        class_pred=class_pred,
        omission_prob=omission_prob,
        expected_score=expected_score,
        class_target=class_target,
        score_target=score_target,
        omission_target=omission_target,
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
