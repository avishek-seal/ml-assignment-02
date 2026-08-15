"""Train all six models, write artifacts, print the comparison table.

    python train.py

This is the script to run on BITS Virtual Lab for the required screenshot.
"""

import json
import time

import joblib

from src.config import (
    METRIC_KEYS,
    METRIC_LABELS,
    METRICS_JSON,
    MODEL_DIR,
    MODEL_SLUGS,
    TEST_CSV,
)
from src.data import deduplicate, load_raw, split_data, write_test_csv
from src.evaluate import compute_metrics
from src.pipeline import build_models, tune


def print_table(results: dict) -> None:
    """Console comparison table — this is what the screenshot captures."""
    headers = [METRIC_LABELS[key] for key in METRIC_KEYS]
    line = f"{'Model':<24}" + "".join(f"{h:>11}" for h in headers)
    print("\n" + line)
    print("-" * len(line))
    for name, scores in results.items():
        row = f"{name:<24}"
        row += "".join(f"{scores[key]:>11.4f}" for key in METRIC_KEYS)
        print(row)
    print("-" * len(line))

    winner = max(results, key=lambda name: results[name]["mcc"])
    print(f"Best by MCC: {winner} ({results[winner]['mcc']:.4f})\n")


def main() -> None:
    started = time.time()

    raw = load_raw()
    clean = deduplicate(raw)
    print(f"Loaded {len(raw)} rows, {len(raw) - len(clean)} duplicates removed")

    X_train, X_test, y_train, y_test = split_data(clean)
    print(f"Train {len(X_train)} rows / test {len(X_test)} rows")

    write_test_csv(X_test, y_test)
    print(f"Wrote {TEST_CSV}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    results = {}
    classes = None

    for name, pipe in build_models().items():
        began = time.time()
        fitted, params = tune(name, pipe, X_train, y_train)

        classes = list(fitted.named_steps["model"].classes_)
        y_pred = fitted.predict(X_test)
        y_proba = fitted.predict_proba(X_test)

        scores = compute_metrics(y_test, y_pred, y_proba, classes)
        scores["best_params"] = params
        results[name] = scores

        path = MODEL_DIR / f"{MODEL_SLUGS[name]}.pkl"
        joblib.dump(fitted, path, compress=3)

        size_mb = path.stat().st_size / 1024 / 1024
        detail = f" {params}" if params else ""
        print(
            f"  {name:<24} {time.time() - began:6.1f}s  "
            f"{size_mb:5.2f} MB{detail}"
        )

    METRICS_JSON.write_text(
        json.dumps(
            {
                "n_test_rows": len(X_test),
                "classes": classes,
                "models": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print_table(results)
    print(f"Artifacts in {MODEL_DIR}")
    print(f"Total {time.time() - started:.1f}s")


if __name__ == "__main__":
    main()
