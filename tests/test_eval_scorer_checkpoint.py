from pronunciation_backend.training.eval_scorer_checkpoint import (
    _class_from_regression_target,
    _predicted_class,
    _summarize_predictions,
)


def test_class_from_regression_target_uses_mapper_thresholds() -> None:
    assert _class_from_regression_target(15.0) == "wrong_or_missed"
    assert _class_from_regression_target(34.24) == "wrong_or_missed"
    assert _class_from_regression_target(34.25) == "accented"
    assert _class_from_regression_target(72.74) == "accented"
    assert _class_from_regression_target(72.75) == "correct"


def test_predicted_class_uses_presence_override() -> None:
    assert _predicted_class(95.0, 0.1) == "wrong_or_missed"
    assert _predicted_class(80.0, 0.8) == "correct"
    assert _predicted_class(50.0, 0.8) == "accented"
    assert _predicted_class(20.0, 0.8) == "wrong_or_missed"


def test_summarize_predictions_reports_expected_confusion() -> None:
    summary = _summarize_predictions(
        match_pred=[90.0, 55.0, 20.0, 10.0],
        duration_pred=[88.0, 50.0, 18.0, 12.0],
        presence_prob=[0.95, 0.90, 0.80, 0.10],
        match_target=[92.0, 60.0, 15.0, 15.0],
        duration_target=[92.0, 60.0, 15.0, 15.0],
        presence_target=[1.0, 1.0, 1.0, 0.0],
    )

    assert summary["phones"] == 4
    assert summary["presence_accuracy"] == 1.0
    assert summary["class_confusion_counts"]["correct"]["correct"] == 1
    assert summary["class_confusion_counts"]["accented"]["accented"] == 1
    assert summary["class_confusion_counts"]["wrong_or_missed"]["wrong_or_missed"] == 2
    assert summary["true_class_stats"]["correct"]["mean_match_score"] == 90.0
    assert summary["diagnostics"]["predicted_class_counts"]["correct"] == 1
    assert summary["diagnostics"]["target_class_counts"]["wrong_or_missed"] == 2
    assert summary["true_class_stats"]["correct"]["match_percentiles"]["p50"] == 90.0
    assert "degenerate_all_correct_predictions" in summary["diagnostics"]


def test_summarize_predictions_flags_collapse_when_everything_is_correct() -> None:
    summary = _summarize_predictions(
        match_pred=[93.0, 93.2, 93.1],
        duration_pred=[94.0, 94.0, 94.0],
        presence_prob=[0.99, 0.99, 0.99],
        match_target=[92.0, 60.0, 15.0],
        duration_target=[92.0, 60.0, 15.0],
        presence_target=[1.0, 1.0, 1.0],
    )

    assert summary["diagnostics"]["degenerate_all_correct_predictions"] is True
    assert summary["diagnostics"]["collapsed_match_separation"] is True
    assert summary["diagnostics"]["presence_metric_not_informative"] is True
    assert any("predicts every phone as 'correct'" in message for message in summary["interpretation"])
