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
def load_frame(source) -> pd.DataFrame | None:
    """Parse an uploaded CSV, returning None if it cannot be read.

    A file that is not a readable CSV must produce a friendly message
    rather than a traceback in the UI.
    """
    try:
        return pd.read_csv(source)
    except (
        pd.errors.ParserError,
        pd.errors.EmptyDataError,
        UnicodeDecodeError,
        ValueError,
        OSError,
    ):
        return None


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

if data is None:
    st.error(
        "That file could not be read as a CSV. Upload a comma-separated file "
        f"with the {len(FEATURE_COLUMNS)} feature columns."
    )
    st.stop()

missing = validate_frame(data)

if missing:
    st.error(f"Missing required columns: {', '.join(missing)}")
    st.stop()

if len(data) == 0:
    st.error("That file has a header but no data rows.")
    st.stop()

models = load_models()
trained_classes = set(
    next(iter(models.values())).named_steps["model"].classes_
)

if TARGET_COLUMN in data.columns:
    unknown = sorted(set(data[TARGET_COLUMN].astype(str)) - trained_classes)
    if unknown:
        st.error(
            f"Unrecognised `{TARGET_COLUMN}` values: {', '.join(unknown)}. "
            f"Expected one of: {', '.join(sorted(trained_classes))}."
        )
        st.stop()

has_labels = TARGET_COLUMN in data.columns
st.success(f"Loaded {len(data):,} rows.")
if not has_labels:
    st.warning(
        f"No `{TARGET_COLUMN}` column — showing predictions only. "
        "Metrics and a confusion matrix need ground-truth labels."
    )

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

# --- tab 2: compare --------------------------------------------------------

with compare_tab:
    if not has_labels:
        st.info(
            f"Add a `{TARGET_COLUMN}` column to compare models — "
            "comparison needs ground-truth labels."
        )
    else:
        st.caption(
            "All six models scored live on the currently loaded data, so this "
            "reflects your upload rather than a stored result."
        )

        rows = []
        any_partial = False
        progress = st.progress(0.0, text="Scoring models…")
        for index, (name, pipe) in enumerate(models.items(), start=1):
            metrics = score(pipe, data)["metrics"]
            any_partial = any_partial or metrics["auc_partial"]
            rows.append({"Model": name, **{
                METRIC_LABELS[key]: metrics[key] for key in METRIC_KEYS
            }})
            progress.progress(index / len(models), text=f"Scored {name}")
        progress.empty()

        if any_partial:
            st.warning(
                "This data does not cover all 7 varieties, so the AUC column "
                "is computed over the classes present and is not comparable "
                "with the full-test-set figures below."
            )

        table = pd.DataFrame(rows).set_index("Model")
        st.dataframe(
            table.style.format("{:.4f}").highlight_max(axis=0, color="#b7e4c7"),
            width="stretch",
        )

        st.subheader("Metric comparison")
        st.bar_chart(table)

        best = table["MCC"].idxmax()
        st.success(
            f"Strongest on this data by MCC: **{best}** "
            f"({table.loc[best, 'MCC']:.4f})"
        )

        with st.expander("Compare against the original test-set run"):
            reference = load_reference_metrics()
            st.caption(
                f"Recorded at training time on {reference['n_test_rows']:,} "
                "held-out rows. Loading the bundled test_data.csv should "
                "reproduce these figures exactly."
            )
            st.dataframe(
                pd.DataFrame(
                    {
                        name: {
                            METRIC_LABELS[key]: scores[key]
                            for key in METRIC_KEYS
                        }
                        for name, scores in reference["models"].items()
                    }
                ).transpose().round(4),
                width="stretch",
            )

# --- tab 3: dataset --------------------------------------------------------

with dataset_tab:
    st.subheader("Loaded data")
    left, middle, right = st.columns(3)
    left.metric("Rows", f"{len(data):,}")
    middle.metric("Features", len(FEATURE_COLUMNS))
    right.metric(
        "Varieties", data[TARGET_COLUMN].nunique() if has_labels else "—"
    )

    if has_labels:
        st.subheader("Class distribution")
        counts = data[TARGET_COLUMN].value_counts()
        st.bar_chart(counts)
        ratio = counts.max() / counts.min()
        st.caption(
            f"Largest class is {ratio:.1f}× the smallest. This imbalance is "
            "why the metrics above are macro-averaged: weighted averaging "
            "would let the largest variety dominate the score."
        )

    st.subheader("Feature summary")
    st.dataframe(
        data[FEATURE_COLUMNS].describe().transpose().round(3), width="stretch"
    )

    st.subheader("Feature correlation")
    figure, axes = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        data[FEATURE_COLUMNS].corr(),
        cmap="coolwarm",
        center=0,
        square=True,
        linewidths=0.4,
        cbar_kws={"shrink": 0.7},
        ax=axes,
    )
    figure.tight_layout()
    st.pyplot(figure)
    plt.close(figure)
    st.caption(
        "Several size features (Area, Perimeter, ConvexArea, EquivDiameter) "
        "are near-perfectly correlated, since all measure bean size. This is "
        "why the distance-based and linear models benefit from scaling."
    )
