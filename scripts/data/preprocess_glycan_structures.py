#!/usr/bin/env python
"""Create lightweight glycan features for Phase 2 baselines."""

from __future__ import annotations

import argparse
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd


MONOSAC_TYPES = ["Glc", "GlcNAc", "Gal", "GalNAc", "Man", "Fuc", "Neu5Ac", "Neu5Gc", "Xyl"]


def _count_monosaccharides(iupac: str) -> List[float]:
    counts = []
    for token in MONOSAC_TYPES:
        counts.append(float(iupac.count(token)))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess glycan structures for Phase 2.")
    parser.add_argument(
        "--input",
        default="data/processed/phase2_merged_dataset.csv",
        help="Merged Phase 2 dataset",
    )
    parser.add_argument(
        "--output",
        default="data/processed/glycan_features.pkl",
        help="Output pickle with glycan feature vectors",
    )
    parser.add_argument(
        "--csv_out",
        default="data/processed/glycan_features.csv",
        help="Output CSV summary",
    )
    args = parser.parse_args()

    merged = pd.read_csv(args.input)
    if "glycan_id" not in merged.columns:
        raise ValueError("Merged dataset missing glycan_id column.")

    glycan_records: Dict[str, List[float]] = {}
    missing_iupac = 0

    for _, row in merged.iterrows():
        glycan_id = str(row.get("glycan_id", "")).strip()
        if not glycan_id:
            continue
        iupac = str(row.get("glycan_iupac", "") or "")
        if not iupac:
            missing_iupac += 1
        counts = _count_monosaccharides(iupac)
        total = float(sum(counts))
        length = float(len(iupac))
        features = counts + [total, length]
        glycan_records[glycan_id] = features

    with open(args.output, "wb") as handle:
        pickle.dump(glycan_records, handle)

    feature_cols = [f"count_{token}" for token in MONOSAC_TYPES] + ["count_total", "iupac_len"]
    out_rows = [
        {"glycan_id": gid, **{col: val for col, val in zip(feature_cols, feats)}}
        for gid, feats in glycan_records.items()
    ]
    pd.DataFrame(out_rows).to_csv(args.csv_out, index=False)

    print(f"Processing {len(glycan_records):,} unique glycan structures...")
    print("✓ Computed monosaccharide composition features")
    print(f"✓ Saved to {args.output}")
    print(f"✓ CSV summary written to {args.csv_out}")
    if missing_iupac:
        print(f"⚠️ {missing_iupac:,} rows missing glycan_iupac (features use zeros)")


if __name__ == "__main__":
    main()
