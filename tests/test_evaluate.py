import math

import numpy as np
import pytest

from src.evaluate import compute_metrics, confusion_frame, safe_macro_auc

LABELS = ["A", "B", "C", "D"]


def proba_for(y_true, labels=LABELS, seed=7):
    """Random but valid probability matrix, one row per observation."""
    rng = np.random.default_rng(seed)
    return rng.dirichlet(np.ones(len(labels)), size=len(y_true))


def test_perfect_prediction_scores_one():
    y = ["A", "B", "C", "D", "A", "B"]
    proba = np.zeros((len(y), 4))
    for row, label in enumerate(y):
        proba[row, LABELS.index(label)] = 1.0

    result = compute_metrics(y, y, proba, LABELS)

    assert result["accuracy"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["f1"] == 1.0
    assert result["mcc"] == 1.0
    assert result["auc"] == 1.0
    assert result["auc_partial"] is False


def test_accuracy_matches_hand_count():
    # 3 of 4 correct.
    y_true = ["A", "A", "B", "B"]
    y_pred = ["A", "A", "B", "A"]
    result = compute_metrics(y_true, y_pred, proba_for(y_true), LABELS)
    assert result["accuracy"] == pytest.approx(0.75)


def test_macro_precision_is_hand_checked():
    # Class A: predicted 3 times, 2 correct  -> precision 2/3
    # Class B: predicted 1 time,  1 correct  -> precision 1/1
    # Classes C, D: never predicted          -> precision 0 (zero_division=0)
    # Macro over all four labels = (2/3 + 1 + 0 + 0) / 4
    y_true = ["A", "A", "B", "B"]
    y_pred = ["A", "A", "B", "A"]
    expected = (2 / 3 + 1.0 + 0.0 + 0.0) / 4
    result = compute_metrics(y_true, y_pred, proba_for(y_true), LABELS)
    assert result["precision"] == pytest.approx(expected)


def test_all_metric_keys_present():
    y = ["A", "B", "C", "D"]
    result = compute_metrics(y, y, proba_for(y), LABELS)
    for key in ("accuracy", "auc", "precision", "recall", "f1", "mcc"):
        assert key in result
        assert isinstance(result[key], float)


# --- label safety ----------------------------------------------------------


def test_auc_with_all_classes_present_is_not_partial():
    y = ["A", "B", "C", "D"] * 5
    value, partial = safe_macro_auc(y, proba_for(y), LABELS)
    assert partial is False
    assert 0.0 <= value <= 1.0


def test_auc_with_three_of_four_classes_does_not_raise():
    y = ["A", "B", "C"] * 6
    value, partial = safe_macro_auc(y, proba_for(y), LABELS)
    assert partial is True
    assert not math.isnan(value)
    assert 0.0 <= value <= 1.0


def test_auc_with_exactly_two_classes_does_not_raise():
    """Regression: sklearn routes 2 classes to its binary path, which
    rejects a 2-column probability matrix."""
    y = ["A", "B"] * 9
    value, partial = safe_macro_auc(y, proba_for(y), LABELS)
    assert partial is True
    assert not math.isnan(value)
    assert 0.0 <= value <= 1.0


def test_auc_with_single_class_is_nan_not_an_exception():
    y = ["A"] * 8
    value, partial = safe_macro_auc(y, proba_for(y), LABELS)
    assert partial is True
    assert math.isnan(value)


def test_compute_metrics_survives_partial_classes():
    y_true = ["A", "B"] * 9
    y_pred = ["A", "A"] * 9
    result = compute_metrics(y_true, y_pred, proba_for(y_true), LABELS)
    assert result["auc_partial"] is True
    assert 0.0 <= result["accuracy"] <= 1.0


# --- confusion matrix ------------------------------------------------------


def test_confusion_frame_is_always_full_size():
    """Regression: without labels=, sklearn returns a smaller matrix while
    the heatmap still draws four ticks, silently misaligning the cells."""
    y_true = ["A", "B"] * 5
    y_pred = ["A", "A"] * 5
    frame = confusion_frame(y_true, y_pred, LABELS)
    assert frame.shape == (4, 4)
    assert list(frame.index) == LABELS
    assert list(frame.columns) == LABELS


def test_confusion_frame_counts_are_correct():
    y_true = ["A", "A", "B", "C"]
    y_pred = ["A", "B", "B", "C"]
    frame = confusion_frame(y_true, y_pred, LABELS)
    assert frame.loc["A", "A"] == 1
    assert frame.loc["A", "B"] == 1
    assert frame.loc["B", "B"] == 1
    assert frame.loc["C", "C"] == 1
    assert frame.loc["D"].sum() == 0
