"""Quick sanity checks for Phase 3 Fcgr datasets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd


def summarize_distribution(series: pd.Series, label: str) -> None:
    values = series.dropna().astype(float)
    if values.empty:
        print(f"{label}: no values")
        return
    stats = values.describe(percentiles=[0.1, 0.5, 0.9])
    print(f"\n{label} summary:")
    print(stats.to_string())


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Sanity checks for Phase 3 Fcgr data.")
    parser.add_argument("--data-path", required=True, help="Path to phase3_fcgr_merged.csv")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    data_path = Path(args.data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Missing data file: {data_path}")

    df = pd.read_csv(data_path)
    if "binding_kd_nm" in df.columns:
        summarize_distribution(df["binding_kd_nm"], "binding_kd_nm")
    if "log_kd" in df.columns:
        summarize_distribution(df["log_kd"], "log_kd")

    if {"glycan_name", "binding_kd_nm"}.issubset(df.columns):
        print("\nMedian binding_kd_nm by glycan_name (top 10):")
        summary = (
            df.groupby("glycan_name")["binding_kd_nm"]
            .median()
            .sort_values()
            .head(10)
        )
        print(summary.to_string())


if __name__ == "__main__":
    main()
