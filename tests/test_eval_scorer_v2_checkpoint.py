from pronunciation_backend.training.eval_scorer_v2_checkpoint import _summarize_predictions


def test_summarize_predictions_reports_expected_v2_confusion() -> None:
    summary = _summarize_predictions(
        class_pred=[2, 1, 0, 0],
        omission_prob=[0.05, 0.10, 0.15, 0.80],
        expected_score=[92.0, 60.0, 15.0, 12.0],
        class_target=[2, 1, 0, 0],
        score_target=[92.0, 60.0, 15.0, 15.0],
        omission_target=[0.0, 0.0, 0.0, 1.0],
    )

    assert summary["phones"] == 4
    assert summary["class_accuracy"] == 1.0
    assert summary["omission_accuracy"] == 1.0
    assert summary["class_confusion_counts"]["correct"]["correct"] == 1
    assert summary["class_confusion_counts"]["accented"]["accented"] == 1
    assert summary["class_confusion_counts"]["wrong_or_missed"]["wrong_or_missed"] == 2
    assert summary["true_class_stats"]["correct"]["mean_expected_score"] == 92.0
    assert summary["diagnostics"]["predicted_class_counts"]["correct"] == 1
    assert summary["diagnostics"]["target_class_counts"]["wrong_or_missed"] == 2


def test_summarize_predictions_flags_collapse_when_everything_is_correct() -> None:
    summary = _summarize_predictions(
        class_pred=[2, 2, 2],
        omission_prob=[0.01, 0.02, 0.03],
        expected_score=[92.8, 92.9, 93.0],
        class_target=[2, 1, 0],
        score_target=[92.0, 60.0, 15.0],
        omission_target=[0.0, 0.0, 0.0],
    )

    assert summary["diagnostics"]["degenerate_all_correct_predictions"] is True
    assert summary["diagnostics"]["collapsed_score_separation"] is True
    assert summary["diagnostics"]["omission_metric_not_informative"] is True
    assert any("predicts every phone as 'correct'" in message for message in summary["interpretation"])
