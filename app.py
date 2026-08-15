"""Streamlit front end for the Dry Bean classifiers.

Loads pre-trained pipelines and scores uploaded data. Never trains.
"""

import json

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.config import (
    FEATURE_COLUMNS,
    METRIC_KEYS,
    METRIC_LABELS,
    METRICS_JSON,
    MODEL_DIR,
    MODEL_SLUGS,
    TARGET_COLUMN,
    TEST_CSV,
)
from src.evaluate import classification_frame, compute_metrics, confusion_frame

st.set_page_config(
    page_title="Dry Bean Classifier", page_icon="🫘", layout="wide"
)


@st.cache_resource
def load_models() -> dict:
    return {
        name: joblib.load(MODEL_DIR / f"{slug}.pkl")
        for name, slug in MODEL_SLUGS.items()
    }


@st.cache_resource
def load_reference_metrics() -> dict:
    return json.loads(METRICS_JSON.read_text(encoding="utf-8"))


@st.cache_data
def load_frame(source) -> pd.DataFrame:
    return pd.read_csv(source)


def validate_frame(frame: pd.DataFrame) -> list[str]:
    """Return the names of any required feature columns that are missing."""
    return [column for column in FEATURE_COLUMNS if column not in frame.columns]


def score(pipe, frame: pd.DataFrame) -> dict:
    """Predict and, when labels are present, compute metrics."""
    features = frame[FEATURE_COLUMNS]
    classes = list(pipe.named_steps["model"].classes_)

    y_pred = pipe.predict(features)
    y_proba = pipe.predict_proba(features)

    if TARGET_COLUMN not in frame.columns:
        return {"y_pred": y_pred, "classes": classes, "metrics": None}

    y_true = frame[TARGET_COLUMN]
    return {
        "y_pred": y_pred,
        "classes": classes,
        "y_true": y_true,
        "metrics": compute_metrics(y_true, y_pred, y_proba, classes),
    }


def draw_confusion(frame: pd.DataFrame) -> None:
    figure, axes = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(
        frame,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        cbar=False,
        linewidths=0.5,
        ax=axes,
    )
    axes.set_xlabel("Predicted")
    axes.set_ylabel("Actual")
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)


# --- data selection --------------------------------------------------------

st.title("🫘 Dry Bean Variety Classifier")
st.caption(
    "Six classifiers trained on the UCI Dry Bean dataset — "
    "13,543 samples, 16 geometric features, 7 varieties."
)

with st.sidebar:
    st.header("Test data")
    upload = st.file_uploader("Upload a CSV", type="csv")
    use_bundled = st.button("Use bundled test_data.csv", width="stretch")
    st.caption(
        f"Needs the {len(FEATURE_COLUMNS)} feature columns. "
        f"Include `{TARGET_COLUMN}` to see metrics."
    )

if "source" not in st.session_state:
    st.session_state.source = None

# The button is checked first: once a file is uploaded the widget keeps
# returning it on every rerun, so an upload-first ordering would make the
# bundled button permanently unreachable.
if use_bundled:
    st.session_state.source = TEST_CSV
elif upload is not None:
    st.session_state.source = upload

if st.session_state.source is None:
    st.info(
        "Upload a CSV or click **Use bundled test_data.csv** in the sidebar "
        "to begin."
    )
    st.stop()

data = load_frame(st.session_state.source)
missing = validate_frame(data)

if missing:
    st.error(f"Missing required columns: {', '.join(missing)}")
    st.stop()

has_labels = TARGET_COLUMN in data.columns
st.success(f"Loaded {len(data):,} rows.")
if not has_labels:
    st.warning(
        f"No `{TARGET_COLUMN}` column — showing predictions only. "
        "Metrics and a confusion matrix need ground-truth labels."
    )

models = load_models()

evaluate_tab, compare_tab, dataset_tab = st.tabs(
    ["Evaluate", "Compare", "Dataset"]
)

# --- tab 1: evaluate -------------------------------------------------------

with evaluate_tab:
    choice = st.selectbox("Model", list(models))
    result = score(models[choice], data)

    if result["metrics"] is None:
        predictions = data.copy()
        predictions["Predicted"] = result["y_pred"]
        st.dataframe(predictions.head(200), width="stretch")
        st.bar_chart(pd.Series(result["y_pred"]).value_counts())
    else:
        metrics = result["metrics"]
        columns = st.columns(len(METRIC_KEYS))
        for column, key in zip(columns, METRIC_KEYS):
            value = metrics[key]
            column.metric(
                METRIC_LABELS[key],
                "n/a" if pd.isna(value) else f"{value:.4f}",
            )

        if metrics["auc_partial"]:
            st.warning(
                "This data does not cover all 7 varieties, so AUC is computed "
                "over the classes present and is not comparable with the "
                "full-test-set figure."
            )

        left, right = st.columns([1, 1])
        with left:
            st.subheader("Confusion matrix")
            draw_confusion(
                confusion_frame(
                    result["y_true"], result["y_pred"], result["classes"]
                )
            )
        with right:
            st.subheader("Per-class report")
            report = classification_frame(
                result["y_true"], result["y_pred"], result["classes"]
            )
            st.dataframe(report, width="stretch")
