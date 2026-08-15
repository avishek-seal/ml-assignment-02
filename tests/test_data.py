import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS, TARGET_COLUMN
from src.data import deduplicate, load_raw, split_data, write_test_csv


@pytest.fixture(scope="module")
def raw():
    return load_raw()


@pytest.fixture(scope="module")
def clean(raw):
    return deduplicate(raw)


def test_load_raw_shape(raw):
    assert raw.shape == (13611, 17)
    assert raw.isna().sum().sum() == 0


def test_load_raw_columns(raw):
    assert list(raw.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]


def test_deduplicate_removes_exactly_68_rows(raw, clean):
    assert len(raw) - len(clean) == 68
    assert len(clean) == 13543
    assert clean.duplicated().sum() == 0


def test_split_sizes(clean):
    X_train, X_test, y_train, y_test = split_data(clean)
    assert len(X_train) == 10834
    assert len(X_test) == 2709
    assert len(y_train) == 10834
    assert len(y_test) == 2709


def test_split_is_stratified(clean):
    _, _, y_train, y_test = split_data(clean)
    full = clean[TARGET_COLUMN].value_counts(normalize=True)
    for split in (y_train, y_test):
        proportions = split.value_counts(normalize=True)
        for label in full.index:
            assert abs(proportions[label] - full[label]) < 0.01


def test_split_has_no_leakage(clean):
    X_train, X_test, _, _ = split_data(clean)
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_split_is_deterministic(clean):
    first = split_data(clean)[1]
    second = split_data(clean)[1]
    assert list(first.index) == list(second.index)


def test_split_features_exclude_target(clean):
    X_train, _, _, _ = split_data(clean)
    assert list(X_train.columns) == FEATURE_COLUMNS
    assert TARGET_COLUMN not in X_train.columns


def test_write_test_csv_roundtrip(clean, tmp_path):
    _, X_test, _, y_test = split_data(clean)
    target = tmp_path / "test_data.csv"
    write_test_csv(X_test, y_test, target)

    written = pd.read_csv(target)
    assert len(written) == 2709
    assert list(written.columns) == FEATURE_COLUMNS + [TARGET_COLUMN]
    assert written[TARGET_COLUMN].nunique() == 7
