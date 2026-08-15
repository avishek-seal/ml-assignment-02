"""Project-wide constants. No logic lives here."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"

RAW_CSV = DATA_DIR / "dry_bean.csv"
TEST_CSV = PROJECT_ROOT / "test_data.csv"
METRICS_JSON = MODEL_DIR / "metrics.json"

# The deadline date. Deliberately not 42: on a public dataset the default seed
# is the most fingerprintable part of a submission.
RANDOM_STATE = 18082026
TEST_SIZE = 0.2

TARGET_COLUMN = "Class"

FEATURE_COLUMNS = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4",
]

# Display name -> artifact filename stem. Insertion order is the display order
# used by the dropdown, the comparison table and the README.
MODEL_SLUGS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "k-Nearest Neighbors": "knn",
    "Naive Bayes": "naive_bayes",
    "Random Forest": "random_forest",
    "Gradient Boosting": "gradient_boosting",
}

METRIC_KEYS = ["accuracy", "auc", "precision", "recall", "f1", "mcc"]

METRIC_LABELS = {
    "accuracy": "Accuracy",
    "auc": "AUC",
    "precision": "Precision",
    "recall": "Recall",
    "f1": "F1",
    "mcc": "MCC",
}
