"""GenePix RFU Parser - FINAL (uses RIGHT PANEL)."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


def _infer_glycan_names(df_raw: pd.DataFrame, right_names: pd.Series) -> pd.Series:
    right_nonnull = right_names.dropna().astype(str).str.strip()
    if right_nonnull.empty:
        return pd.Series(["Unknown"] * len(df_raw), index=df_raw.index)

    has_alpha = right_nonnull.str.contains(r"[A-Za-z]", regex=True)
    unique_count = right_nonnull.nunique()
    if has_alpha.mean() > 0.5 and unique_count > 50:
        return right_names.astype(str)

    left_panel_ids = pd.to_numeric(df_raw.iloc[:, 0], errors="coerce")
    left_names = df_raw.iloc[:, 1].astype(str)
    name_mapping = dict(zip(left_panel_ids, left_names))
    mapped = left_panel_ids.map(name_mapping)
    return mapped.fillna("Unknown")


def parse_genepix_file(filepath: Path, sheet_name: str = "Sheet1") -> Optional[pd.DataFrame]:
    """Parse a GenePix file using RIGHT PANEL columns (28-32)."""
    print(f"  Parsing {filepath.name} ({sheet_name})...", end=" ")

    try:
        df_raw = pd.read_excel(filepath, sheet_name=sheet_name)
    except Exception as exc:
        print(f"✗ ERROR: {exc}")
        return None

    if df_raw.shape[1] < 33:
        print("✗ ERROR: Expected at least 33 columns for right panel")
        return None

    right_panel = df_raw.iloc[:, 28:33].copy()
    right_panel.columns = ["glycan_id", "glycan_name_candidate", "rfu_raw", "stdev", "cv"]

    right_panel = right_panel.dropna(subset=["glycan_id", "rfu_raw"]).copy()
    right_panel["glycan_id"] = pd.to_numeric(right_panel["glycan_id"], errors="coerce")
    right_panel["rfu_raw"] = pd.to_numeric(right_panel["rfu_raw"], errors="coerce")
    right_panel["stdev"] = pd.to_numeric(right_panel["stdev"], errors="coerce")
    right_panel["cv"] = pd.to_numeric(right_panel["cv"], errors="coerce")
    right_panel = right_panel.dropna(subset=["glycan_id", "rfu_raw"]).copy()
    right_panel["glycan_id"] = right_panel["glycan_id"].astype(int)

    neg_count = int((right_panel["rfu_raw"] < 0).sum())
    if neg_count > 0:
        print(f"(removing {neg_count} negative RFU values)", end=" ")
    right_panel = right_panel[right_panel["rfu_raw"] >= 0].copy()

    sample_header = str(df_raw.columns[29]) if df_raw.shape[1] > 29 else "Unknown"
    inferred_names = _infer_glycan_names(df_raw, df_raw.iloc[:, 29])
    right_panel["glycan_name"] = inferred_names.iloc[right_panel.index].astype(str).str.strip()
    right_panel["sample_name"] = sample_header
    right_panel.loc[right_panel["glycan_name"].isin(["nan", "None", ""]), "glycan_name"] = "Unknown"

    # Deduplicate glycan IDs by averaging right-panel replicates
    if right_panel["glycan_id"].duplicated().any():
        right_panel = (
            right_panel.groupby("glycan_id", as_index=False)
            .agg(
                glycan_name=("glycan_name", "first"),
                sample_name=("sample_name", "first"),
                rfu_raw=("rfu_raw", "mean"),
                stdev=("stdev", "mean"),
                cv=("cv", "mean"),
            )
            .reset_index(drop=True)
        )

    right_panel = right_panel.sort_values("glycan_id").reset_index(drop=True)

    print(f"✓ {len(right_panel)} glycans")
    return right_panel


def extract_sample_info(filepath: Path) -> Tuple[str, str, Optional[str]]:
    """Extract sample information from filename."""
    filename = filepath.stem
    mapping = {
        "1-21_16983": ("Sample_1-21", "serum", None),
        "19-166_2_17105": ("Sample_19-166", "serum", None),
        "2367AD12.1_17046": ("Sample_2367AD12.1", "serum", None),
        "A-16_16985": ("Sample_A-16", "serum", None),
        "Balbc-1_16977": ("Balbc-1", "sera", None),
        "J558_16986": ("Sample_J558", "serum", None),
    }
    for key, (name, stype, conc) in mapping.items():
        if key in filename:
            return name, stype, conc
    return filename, "unknown", None


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    input_dir = base_dir / "data" / "raw" / "cfg_arrays_raw"
    output_file = base_dir / "data" / "processed" / "cfg_rfu_measurements.csv"

    print("=" * 80)
    print("GENEPIX RFU PARSER - RIGHT PANEL ONLY")
    print("=" * 80)
    print()

    files_to_parse = [
        ("1-21_16983_GenePix_v5.2_RESULTS.xls", "Sheet1"),
        ("19-166_2_17105_v5.2_GenePix467_RESULTS.xls", "Sheet1"),
        ("2367AD12.1_17046_GenePix_v5.2_RESULTS.xls", "Sheet1"),
        ("A-16_16985_GenePixv5.2_RESULTS.xls", "Sheet1"),
        ("Balbc-1_16977_IgGIgM_v5.2_RESULTS.xls", "IgG"),
        ("Balbc-1_16977_IgGIgM_v5.2_RESULTS.xls", "IgM"),
        ("J558_16986_V5.2_GenePixP575_Results.xls", "Sheet1"),
    ]

    all_data = []
    for filename, sheet_name in files_to_parse:
        filepath = input_dir / filename
        if not filepath.exists():
            print(f"⚠ File not found: {filepath}")
            continue

        df = parse_genepix_file(filepath, sheet_name)
        if df is None or df.empty:
            continue

        sample_name, sample_type, _ = extract_sample_info(filepath)
        df["experiment_id"] = filepath.stem
        df["sample_name"] = df["sample_name"].where(df["sample_name"] != "Unknown", sample_name)
        df["sample_type"] = sample_type
        df["sheet_name"] = sheet_name
        df["file_source"] = filepath.name

        max_rfu = df["rfu_raw"].max()
        df["rfu_normalized"] = (df["rfu_raw"] / max_rfu * 100).round(2) if max_rfu > 0 else 0.0
        df["conclusive"] = df["rfu_raw"] >= 2000
        df["timestamp"] = datetime.now().isoformat()

        all_data.append(df)

    if not all_data:
        print("No data parsed.")
        return

    combined = pd.concat(all_data, ignore_index=True)

    cols_order = [
        "experiment_id",
        "sample_name",
        "sheet_name",
        "glycan_id",
        "glycan_name",
        "rfu_raw",
        "stdev",
        "cv",
        "rfu_normalized",
        "conclusive",
        "file_source",
        "timestamp",
    ]
    combined = combined[cols_order]

    combined.to_csv(output_file, index=False)

    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"Saved to: {output_file}")
    print(f"Total rows: {len(combined)}")
    print(f"Unique experiments: {combined['experiment_id'].nunique()}")
    print(f"Unique glycans: {combined['glycan_id'].nunique()}")
    print(f"RFU min/max: {combined['rfu_raw'].min():.1f} / {combined['rfu_raw'].max():.1f}")
    print(f"RFU mean: {combined['rfu_raw'].mean():.1f}")
    print(f"RFU std: {combined['rfu_raw'].std():.1f}")
    print(f"Conclusive rows (RFU >= 2000): {combined['conclusive'].sum()}")
    if combined["rfu_raw"].max() < 100:
        print("WARNING: RFU max < 100; check if columns are correct.")
    if combined["rfu_raw"].mean() < 500:
        print("ERROR: RFU mean < 500; likely wrong columns.")
    print()
    print("First 5 rows:")
    print(combined.head(5)[["experiment_id", "glycan_id", "glycan_name", "rfu_raw", "rfu_normalized", "conclusive"]].to_string(index=False))
    print()
    print("✓ DONE")


if __name__ == "__main__":
    main()
