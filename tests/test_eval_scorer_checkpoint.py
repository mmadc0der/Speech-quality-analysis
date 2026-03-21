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
