#!/usr/bin/env python3
"""Validate cfg_rfu_measurements.csv output."""

from __future__ import annotations

import argparse

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/cfg_rfu_measurements.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    print(f"Rows: {len(df)}")
    print(f"Experiments: {df['experiment_id'].nunique()}")
    print(f"Glycans: {df['glycan_id'].nunique()}")

    rfu = df["rfu_raw"].dropna()
    print(f"RFU min/max: {rfu.min()} / {rfu.max()}")
    print(f"RFU std: {rfu.std()}")
    print(f"Unique RFU values: {rfu.nunique()}")

    if len(df) < 600:
        print("WARNING: <600 rows")
    if rfu.nunique() <= 1:
        print("WARNING: RFU values do not vary")
    if (rfu == 5000.0).all():
        print("WARNING: All RFUs are 5000.0 (placeholder)")


if __name__ == "__main__":
    main()
