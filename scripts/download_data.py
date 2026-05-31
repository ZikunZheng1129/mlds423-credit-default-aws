#!/usr/bin/env python3
"""Download the UCI Default of Credit Card Clients dataset from KaggleHub.

This script downloads:
    uciml/default-of-credit-card-clients-dataset

and saves:
    data/raw/default_of_credit_card_clients.csv

It is intended to make the local ML pipeline reproducible without manually
placing the CSV file.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

DATASET_HANDLE = "uciml/default-of-credit-card-clients-dataset"
EXPECTED_FILENAME = "UCI_Credit_Card.csv"
TARGET_PATH = Path("data/raw/default_of_credit_card_clients.csv")


def find_csv(dataset_dir: Path) -> Path:
    """Find the expected CSV file inside the KaggleHub download directory."""
    direct_path = dataset_dir / EXPECTED_FILENAME
    if direct_path.exists():
        return direct_path

    matches = list(dataset_dir.rglob(EXPECTED_FILENAME))
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find {EXPECTED_FILENAME} under KaggleHub download directory: "
        f"{dataset_dir}"
    )


def main() -> int:
    """Download the dataset and save it to the project's raw data folder."""
    try:
        import kagglehub
    except ImportError as exc:
        print(
            "ERROR: kagglehub could not be imported. Install dependencies with:\n"
            "  pip install -r requirements.txt\n"
            "or:\n"
            "  pip install kagglehub\n"
            f"Import error details: {exc}",
            file=sys.stderr,
        )
        return 1

    try:
        dataset_path = kagglehub.dataset_download(DATASET_HANDLE)
        dataset_dir = Path(dataset_path)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(
            f"ERROR: Failed to download KaggleHub dataset {DATASET_HANDLE}: {exc}",
            file=sys.stderr,
        )
        return 1

    try:
        source_csv = find_csv(dataset_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    try:
        TARGET_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_csv, TARGET_PATH)
        df = pd.read_csv(TARGET_PATH)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(
            f"ERROR: Failed to copy/read dataset from {source_csv} to "
            f"{TARGET_PATH}: {exc}",
            file=sys.stderr,
        )
        return 1

    print("Dataset downloaded successfully.")
    print(f"KaggleHub directory: {dataset_dir}")
    print(f"Source CSV: {source_csv}")
    print(f"Target CSV: {TARGET_PATH}")
    print(f"Data shape: {df.shape}")
    print("\nPreview:")
    print(df.head())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
