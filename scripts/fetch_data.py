"""One-off: download the UCI Dry Bean dataset and write it as CSV.

Run once; data/dry_bean.csv is committed so train.py never needs network
access (BITS Virtual Lab may restrict egress).

    python scripts/fetch_data.py
"""

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
from scipy.io import arff

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import DATA_DIR, RAW_CSV, TARGET_COLUMN  # noqa: E402

URL = "https://archive.ics.uci.edu/static/public/602/dry+bean+dataset.zip"
ARFF_NAME = "DryBeanDataset/Dry_Bean_Dataset.arff"


def main() -> None:
    print(f"Downloading {URL}")
    with urllib.request.urlopen(URL) as response:
        payload = response.read()
    print(f"Downloaded {len(payload) / 1024 / 1024:.1f} MB")

    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        with archive.open(ARFF_NAME) as handle:
            data, _ = arff.loadarff(io.StringIO(handle.read().decode("utf-8")))

    frame = pd.DataFrame(data)
    # scipy returns ARFF nominal attributes as bytes.
    frame[TARGET_COLUMN] = frame[TARGET_COLUMN].str.decode("utf-8")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RAW_CSV, index=False)

    print(f"Wrote {RAW_CSV} — {len(frame)} rows, {frame.shape[1]} columns")
    print(frame[TARGET_COLUMN].value_counts().to_string())


if __name__ == "__main__":
    main()
