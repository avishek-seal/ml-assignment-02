"""Metric computation. Pure functions: no I/O, no model objects.

Every public function takes an explicit `labels` sequence carrying the full
set of trained classes. This is mandatory, not defensive: when uploaded data
covers only some classes, sklearn either raises or silently returns a
wrongly-shaped result.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)


def safe_macro_auc(
    y_true: Sequence, y_proba: np.ndarray, labels: Sequence
) -> tuple[float, bool]:
    """Macro one-vs-rest AUC that tolerates missing classes.

    Returns (auc, is_partial). `is_partial` is True when y_true does not
    cover every trained class, meaning the figure is not comparable with a
    full-coverage run. Returns (nan, True) when fewer than two classes are
    present, where ROC is genuinely undefined.
    """
    labels = list(labels)
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba, dtype=float)

    present = [label for label in labels if (y_true == label).any()]

    if len(present) < 2:
        return float("nan"), True

    if len(present) == len(labels):
        value = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro", labels=labels
        )
        return float(value), False

    # Restrict to the classes actually present, then renormalise so each row
    # sums to 1 — roc_auc_score validates that.
    columns = [labels.index(label) for label in present]
    restricted = y_proba[:, columns]
    restricted = restricted / np.clip(
        restricted.sum(axis=1, keepdims=True), 1e-12, None
    )

    if len(present) == 2:
        # sklearn routes two classes to its binary path, which requires a
        # 1-D score array. Passing the 2-column matrix raises
        # "y should be a 1d array".
        positive = present[1]
        value = roc_auc_score(
            (y_true == positive).astype(int), restricted[:, 1]
        )
        return float(value), True

    value = roc_auc_score(
        y_true, restricted, multi_class="ovr", average="macro", labels=present
    )
    return float(value), True


def confusion_frame(
    y_true: Sequence, y_pred: Sequence, labels: Sequence
) -> pd.DataFrame:
    """Confusion matrix as a labelled frame, always len(labels) square.

    Rows are true classes, columns are predicted. Absent classes appear as
    all-zero rows rather than being dropped.
    """
    labels = list(labels)
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def compute_metrics(
    y_true: Sequence, y_pred: Sequence, y_proba: np.ndarray, labels: Sequence
) -> dict:
    """All six required metrics for one model.

    Precision, recall and F1 use macro averaging so each class counts
    equally. Under this dataset's 6.8:1 imbalance, weighted averaging would
    let DERMASON's 3,546 instances drown out BOMBAY's 522.
    """
    labels = list(labels)
    shared = {"labels": labels, "average": "macro", "zero_division": 0}

    auc, auc_partial = safe_macro_auc(y_true, y_proba, labels)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc": auc,
        "precision": float(precision_score(y_true, y_pred, **shared)),
        "recall": float(recall_score(y_true, y_pred, **shared)),
        "f1": float(f1_score(y_true, y_pred, **shared)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "auc_partial": auc_partial,
    }
