"""Build Phase 3 Fcgr-Fc dataset by joining Fc metadata to Fcgr binding rows.

Joins Fcgr-Fc Kd data to antibody Fc metadata on exact fc_sequence when possible.
If no exact matches are found, antibody_* columns remain NaN but the Fcgr-Fc
dataset is still valid for model training.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import pandas as pd


REQUIRED_FCGR_COLUMNS = {
    "fcgr_name",
    "fcgr_sequence",
    "fc_sequence",
    "glycan_name",
    "glycan_structure",
    "binding_kd_nm",
}

REQUIRED_FC_COLUMNS = {
    "antibody_id",
    "antibody_name",
    "fc_sequence",
    "fc_start",
    "fc_end",
    "full_heavy_chain",
    "light_chain",
    "num_glycosites",
}


def _validate_columns(df: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def build_phase3_fcgr_dataset(
    antibody_fc_path: Path,
    fcgr_fc_path: Path,
    out_path: Optional[Path] = None,
    *,
    fail_on_unmatched: bool = False,
    min_coverage: float = 0.9,
) -> pd.DataFrame:
    """Join Fcgr-Fc binding data to antibody Fc metadata on fc_sequence."""
    antibody_fc_path = Path(antibody_fc_path)
    fcgr_fc_path = Path(fcgr_fc_path)

    if not antibody_fc_path.exists():
        raise FileNotFoundError(f"Missing antibody Fc metadata: {antibody_fc_path}")
    if not fcgr_fc_path.exists():
        raise FileNotFoundError(f"Missing Fcgr-Fc binding data: {fcgr_fc_path}")

    fc_meta = pd.read_csv(antibody_fc_path)
    fcgr_fc = pd.read_csv(fcgr_fc_path)

    _validate_columns(fc_meta, REQUIRED_FC_COLUMNS, "antibody_fc_regions.csv")
    _validate_columns(fcgr_fc, REQUIRED_FCGR_COLUMNS, "fcgr_fc_training_data.csv")

    fc_meta["fc_sequence"] = fc_meta["fc_sequence"].fillna("").astype(str)
    fcgr_fc["fc_sequence"] = fcgr_fc["fc_sequence"].fillna("").astype(str)

    empty_fc_meta = (fc_meta["fc_sequence"] == "").sum()
    if empty_fc_meta:
        logging.warning("Dropping %s antibody rows with empty fc_sequence.", empty_fc_meta)
        fc_meta = fc_meta[fc_meta["fc_sequence"] != ""].copy()

    dup_counts = (
        fc_meta.groupby("fc_sequence")["antibody_id"]
        .nunique()
        .sort_values(ascending=False)
    )
    n_multi = int((dup_counts > 1).sum())
    logging.info(
        "antibody_fc_regions: %d unique fc_sequence, %d with >1 antibody_id (max=%d)",
        dup_counts.shape[0],
        n_multi,
        int(dup_counts.max()) if not dup_counts.empty else 0,
    )

    fc_meta = (
        fc_meta.sort_values(["fc_sequence", "num_glycosites"], ascending=[True, False])
        .drop_duplicates(subset=["fc_sequence"], keep="first")
        .copy()
    )

    phase3 = fcgr_fc.merge(
        fc_meta[
            [
                "antibody_id",
                "antibody_name",
                "fc_sequence",
                "fc_start",
                "fc_end",
                "full_heavy_chain",
                "light_chain",
                "num_glycosites",
            ]
        ],
        on="fc_sequence",
        how="left",
        validate="many_to_one",
        suffixes=("", "_ab"),
    )

    total_rows = len(phase3)
    matched_rows = phase3["antibody_id"].notna().sum()
    coverage = matched_rows / total_rows if total_rows else 0.0

    fcgr_sequences = set(fcgr_fc["fc_sequence"].dropna().unique())
    unmatched_sequences = sorted(
        seq for seq in fcgr_sequences if seq and seq not in set(fc_meta["fc_sequence"].unique())
    )

    logging.info("Fcgr rows: %s", total_rows)
    logging.info("Rows matched to antibody Fc metadata: %s (%.1f%%)", matched_rows, coverage * 100.0)
    logging.info("Unmatched fc_sequence values: %s", len(unmatched_sequences))
    if unmatched_sequences:
        logging.info("Unmatched fc_sequence samples: %s", unmatched_sequences[:10])
    if total_rows and matched_rows == 0:
        logging.info("No antibody Fc matches found; leaving antibody_* columns empty.")

    phase3["phase3_id"] = (
        phase3["antibody_id"].fillna("NA").astype(str)
        + "|"
        + phase3["fcgr_name"].astype(str)
        + "|"
        + phase3["glycan_name"].astype(str)
    )

    dupes = phase3["phase3_id"].duplicated().sum()
    if dupes:
        logging.warning("Detected %s duplicate phase3_id values.", dupes)

    if fail_on_unmatched and coverage < min_coverage:
        raise ValueError(
            f"Coverage {coverage:.2%} below threshold {min_coverage:.2%} (matched {matched_rows}/{total_rows})."
        )

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        phase3.to_csv(out_path, index=False)

    return phase3


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Phase 3 Fcgr dataset.")
    parser.add_argument("--antibody-fc", required=True, help="Path to antibody_fc_regions.csv")
    parser.add_argument("--fcgr-fc", required=True, help="Path to fcgr_fc_training_data.csv")
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument(
        "--fail-on-unmatched",
        action="store_true",
        help="Fail if antibody match coverage falls below --min-coverage.",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.9,
        help="Minimum match coverage when --fail-on-unmatched is set.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args(argv)
    phase3 = build_phase3_fcgr_dataset(
        antibody_fc_path=Path(args.antibody_fc),
        fcgr_fc_path=Path(args.fcgr_fc),
        out_path=Path(args.out),
        fail_on_unmatched=args.fail_on_unmatched,
        min_coverage=args.min_coverage,
    )
    matched_rows = phase3["antibody_id"].notna().sum()
    total_rows = len(phase3)
    coverage = matched_rows / total_rows if total_rows else 0.0
    logging.info(
        "Phase3 rows=%d matched=%d coverage=%.1f%% out=%s",
        total_rows,
        matched_rows,
        coverage * 100.0,
        args.out,
    )
    print(
        f"Phase3 rows={total_rows} matched={matched_rows} "
        f"coverage={coverage:.1%} out={args.out}"
    )


if __name__ == "__main__":
    main()
