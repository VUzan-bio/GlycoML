"""Validate GenePix parser output."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    base_dir = Path(__file__).resolve().parents[1]
    csv_file = base_dir / "data" / "processed" / "cfg_rfu_measurements.csv"

    print("=" * 80)
    print("GENEPIX OUTPUT VALIDATION")
    print("=" * 80)
    print()

    if not csv_file.exists():
        print(f"ERROR: {csv_file} does not exist")
        print("Run genepix_parser_clean.py first")
        return

    df = pd.read_csv(csv_file)

    print(f"✓ File loaded: {csv_file}")
    print(f"  Shape: {df.shape}")
    print()

    required_cols = ["experiment_id", "glycan_id", "rfu_raw", "rfu_normalized", "conclusive"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f"✗ Missing columns: {missing}")
        return

    print("✓ All required columns present")
    print()

    print("RFU Statistics:")
    print(f"  Min: {df['rfu_raw'].min():.1f}")
    print(f"  Max: {df['rfu_raw'].max():.1f}")
    print(f"  Mean: {df['rfu_raw'].mean():.1f}")
    print(f"  Median: {df['rfu_raw'].median():.1f}")
    print(f"  Std: {df['rfu_raw'].std():.1f}")
    print()

    neg_count = int((df["rfu_raw"] < 0).sum())
    print(f"Negative RFU values: {neg_count}", end="")
    print(" ✓" if neg_count == 0 else " ✗")
    print()

    nan_count = int(df["rfu_raw"].isna().sum())
    print(f"NaN RFU values: {nan_count}", end="")
    print(" ✓" if nan_count == 0 else " ✗")
    print()

    print("Experiments:")
    exp_counts = df.groupby("experiment_id").size()
    for exp_id, count in exp_counts.items():
        print(f"  {exp_id}: {count} rows")
    print(f"  Total unique: {df['experiment_id'].nunique()}")
    print()

    print(f"Unique glycans: {df['glycan_id'].nunique()}")
    print("Expected: ~609 (CFG v5.2)")
    print()

    conclusive_count = int(df["conclusive"].sum())
    total_count = len(df)
    print("Data Quality:")
    print(f"  Conclusive (RFU >= 2000): {conclusive_count} / {total_count} ({conclusive_count/total_count*100:.1f}%)")
    print()

    print("First 5 rows:")
    print(df[["experiment_id", "glycan_id", "glycan_name", "rfu_raw", "rfu_normalized", "conclusive"]].head(5).to_string(index=False))
    print()

    print("=" * 80)
    checks = [
        ("Negative RFU values", neg_count == 0),
        ("NaN RFU values", nan_count == 0),
        ("Expected glycans (~609)", df["glycan_id"].nunique() >= 500),
        ("Expected experiments (6)", df["experiment_id"].nunique() >= 6),
        ("RFU range (0-50000)", df["rfu_raw"].min() >= 0 and df["rfu_raw"].max() < 50000),
    ]

    all_pass = all(result for _, result in checks)
    for check_name, result in checks:
        status = "✓" if result else "✗"
        print(f"{status} {check_name}")

    print()
    if all_pass:
        print("✓✓✓ VALIDATION PASSED ✓✓✓")
        print("Output is ready for training")
    else:
        print("✗✗✗ VALIDATION FAILED ✗✗✗")
    print("=" * 80)


if __name__ == "__main__":
    main()
