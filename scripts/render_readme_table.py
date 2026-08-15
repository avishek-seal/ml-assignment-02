"""Print the README comparison table as markdown, from metrics.json.

    python scripts/render_readme_table.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import METRIC_KEYS, METRIC_LABELS, METRICS_JSON  # noqa: E402


def main() -> None:
    payload = json.loads(METRICS_JSON.read_text(encoding="utf-8"))
    models = payload["models"]

    headers = ["ML Model Name"] + [METRIC_LABELS[key] for key in METRIC_KEYS]
    print("| " + " | ".join(headers) + " |")
    print("|" + "---|" * len(headers))

    for name, scores in models.items():
        cells = [f"{scores[key]:.4f}" for key in METRIC_KEYS]
        print(f"| {name} | " + " | ".join(cells) + " |")

    winner = max(models, key=lambda name: models[name]["mcc"])
    print(f"\nBest by MCC: {winner}")


if __name__ == "__main__":
    main()
