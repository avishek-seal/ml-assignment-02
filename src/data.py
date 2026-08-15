"""Row-level operations: loading, deduplication, splitting.

This module owns every decision about which rows go where. It knows
nothing about models or metrics.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    FEATURE_COLUMNS,
    RANDOM_STATE,
    RAW_CSV,
    TARGET_COLUMN,
    TEST_CSV,
    TEST_SIZE,
)


def load_raw(path: Path = RAW_CSV) -> pd.DataFrame:
    """Load the committed dataset CSV."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python scripts/fetch_data.py"
        )
    return pd.read_csv(path)


def deduplicate(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate rows.

    Sixteen continuous geometric measurements matching to full precision
    indicates a data-preparation artifact, not two distinct beans. Left in
    place, duplicates straddle the train/test split and inflate every
    reported metric.
    """
    return frame.drop_duplicates().reset_index(drop=True)


def split_data(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split. Returns (X_train, X_test, y_train, y_test)."""
    features = frame[FEATURE_COLUMNS]
    target = frame[TARGET_COLUMN]
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )


def write_test_csv(
    X_test: pd.DataFrame, y_test: pd.Series, path: Path = TEST_CSV
) -> None:
    """Write the held-out split as the app's uploadable test file."""
    combined = X_test.copy()
    combined[TARGET_COLUMN] = y_test
    combined.to_csv(path, index=False)
