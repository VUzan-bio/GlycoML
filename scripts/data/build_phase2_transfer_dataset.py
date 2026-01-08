#!/usr/bin/env python
"""Build a Phase 2 training CSV with sequences + glycan structures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 2 training CSV for transfer learning.")
    parser.add_argument(
        "--input",
        default="data/processed/phase2_lectin_glycan.jsonl",
        help="Input JSONL with lectin sequences and glycan structures",
    )
    parser.add_argument(
        "--output",
        default="data/processed/phase2_transfer_training.csv",
        help="Output CSV for Phase 2 training",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=50.0,
        help="Percentile for RFU thresholding (default: 50 for median).",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    records = load_jsonl(input_path)
    rows: List[Dict[str, Any]] = []
    for record in records:
        lectin_seq = record.get("lectin_seq") or record.get("lectin_sequence")
        glycan_smiles = record.get("glycan_smiles") or record.get("glycan_iupac")
        rfu_raw = record.get("binding_rfu")

        if not lectin_seq or not glycan_smiles:
            continue

        rows.append(
            {
                "lectin_sequence": lectin_seq,
                "glycan_smiles": glycan_smiles,
                "rfu_raw": rfu_raw,
            }
        )

    if not rows:
        raise SystemExit("No usable rows found. Check input JSONL.")

    df = pd.DataFrame(rows)
    df["rfu_raw"] = pd.to_numeric(df["rfu_raw"], errors="coerce")
    rfu_values = df["rfu_raw"].dropna().to_numpy()
    if rfu_values.size == 0:
        raise SystemExit("No numeric RFU values found to build binding labels.")
    threshold = float(np.percentile(rfu_values, args.threshold_percentile))
    df["binding"] = (df["rfu_raw"] >= threshold).astype(int)
    if df["binding"].nunique() == 1:
        df["binding"] = (df["rfu_raw"] > threshold).astype(int)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✓ Wrote {len(df)} rows to {output_path}")
    print(f"  Threshold percentile: {args.threshold_percentile:.1f}")
    print(f"  RFU threshold: {threshold:.2f}")
    print(f"  Positive samples: {int(df['binding'].sum())}")
    print(f"  Negative samples: {len(df) - int(df['binding'].sum())}")
    if df["binding"].nunique() == 1:
        print("⚠️ Binding labels are degenerate; consider a different percentile or regression.")


if __name__ == "__main__":
    main()
