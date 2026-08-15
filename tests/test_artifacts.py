import json

import joblib
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    FEATURE_COLUMNS,
    METRIC_KEYS,
    METRICS_JSON,
    MODEL_DIR,
    MODEL_SLUGS,
    TARGET_COLUMN,
    TEST_CSV,
)


@pytest.fixture(scope="module")
def metrics():
    return json.loads(METRICS_JSON.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name,slug", list(MODEL_SLUGS.items()))
def test_artifact_loads_and_predicts(name, slug):
    pipe = joblib.load(MODEL_DIR / f"{slug}.pkl")
    sample = pd.read_csv(TEST_CSV, nrows=1)[FEATURE_COLUMNS]

    assert len(pipe.predict(sample)) == 1
    assert pipe.predict_proba(sample).shape == (1, 7)


@pytest.mark.parametrize("name,slug", list(MODEL_SLUGS.items()))
def test_artifact_is_a_pipeline_with_a_fitted_scaler(name, slug):
    """Regression guard for the bare-estimator mistake."""
    pipe = joblib.load(MODEL_DIR / f"{slug}.pkl")
    assert isinstance(pipe, Pipeline)
    scaler = pipe.named_steps["scaler"]
    assert isinstance(scaler, StandardScaler)
    assert len(scaler.mean_) == len(FEATURE_COLUMNS)


def test_metrics_json_covers_all_models(metrics):
    assert set(metrics["models"]) == set(MODEL_SLUGS)
    assert metrics["n_test_rows"] == 2709
    assert len(metrics["classes"]) == 7


@pytest.mark.parametrize("name", list(MODEL_SLUGS))
def test_metrics_are_in_range(metrics, name):
    scores = metrics["models"][name]
    for key in METRIC_KEYS:
        assert key in scores
        low = -1.0 if key == "mcc" else 0.0
        assert low <= scores[key] <= 1.0, f"{name}.{key}={scores[key]}"
    assert scores["auc_partial"] is False


def test_test_data_csv_is_wellformed():
    frame = pd.read_csv(TEST_CSV)
    assert len(frame) == 2709
    assert list(frame.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    assert frame[TARGET_COLUMN].nunique() == 7
