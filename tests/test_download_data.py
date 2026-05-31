"""Tests for the KaggleHub data download helper without real network calls."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd

from scripts import download_data


def test_find_csv_finds_nested_expected_file(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    csv_path = nested / download_data.EXPECTED_FILENAME
    csv_path.write_text("a,b\n1,2\n", encoding="utf-8")

    assert download_data.find_csv(tmp_path) == csv_path


def test_main_downloads_and_copies_with_mocked_kagglehub(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset_dir = tmp_path / "kagglehub"
    dataset_dir.mkdir()
    source_csv = dataset_dir / download_data.EXPECTED_FILENAME
    pd.DataFrame({"ID": [1], "default.payment.next.month": [0]}).to_csv(
        source_csv,
        index=False,
    )
    target_path = tmp_path / "data" / "raw" / "default_of_credit_card_clients.csv"

    fake_kagglehub = types.SimpleNamespace(
        dataset_download=lambda _handle: str(dataset_dir)
    )
    monkeypatch.setitem(sys.modules, "kagglehub", fake_kagglehub)
    monkeypatch.setattr(download_data, "TARGET_PATH", target_path)

    exit_code = download_data.main()

    assert exit_code == 0
    assert target_path.exists()
    copied = pd.read_csv(target_path)
    assert copied.shape == (1, 2)

