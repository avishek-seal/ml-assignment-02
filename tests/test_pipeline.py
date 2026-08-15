import numpy as np
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import MODEL_SLUGS
from src.pipeline import TUNING_GRIDS, build_models, tune


def test_builds_all_six_models():
    models = build_models()
    assert len(models) == 6
    assert list(models.keys()) == list(MODEL_SLUGS.keys())


def test_every_model_is_a_pipeline_with_a_scaler():
    """Regression guard: a bare estimator would let the app predict on
    unscaled input."""
    for name, pipe in build_models().items():
        assert isinstance(pipe, Pipeline), name
        assert isinstance(pipe.named_steps["scaler"], StandardScaler), name
        assert "model" in pipe.named_steps, name


def test_every_model_supports_predict_proba():
    """AUC requires probabilities."""
    for name, pipe in build_models().items():
        assert hasattr(pipe.named_steps["model"], "predict_proba"), name


def test_tuning_grids_address_pipeline_steps():
    """Params must be step-prefixed or GridSearchCV cannot find them."""
    for name, grid in TUNING_GRIDS.items():
        assert name in MODEL_SLUGS
        for key in grid:
            assert key.startswith("model__"), f"{name}: {key}"


def test_tune_returns_a_fitted_pipeline_for_tuned_models(small_dataset):
    X, y = small_dataset
    models = build_models()
    name = "Decision Tree"
    fitted, params = tune(name, models[name], X, y)
    assert isinstance(fitted, Pipeline)
    assert "model__max_depth" in params
    assert len(fitted.predict(X)) == len(y)


def test_tune_passes_through_untuned_models(small_dataset):
    X, y = small_dataset
    models = build_models()
    name = "Naive Bayes"
    fitted, params = tune(name, models[name], X, y)
    assert params == {}
    assert len(fitted.predict(X)) == len(y)


@pytest.fixture
def small_dataset():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(90, 16))
    y = np.array(["A", "B", "C"] * 30)
    return X, y
