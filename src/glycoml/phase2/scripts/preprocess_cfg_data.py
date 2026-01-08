"""Preprocess CFG lectin array data into a model-ready CSV."""

from __future__ import annotations

import argparse
import csv
import math
import pickle
import random
from typing import Dict, List, Optional, Tuple


def load_glycan_library(path: str) -> Dict[str, Dict[str, str]]:
    library: Dict[str, Dict[str, str]] = {}
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            glycan_id = parts[0]
            smiles = parts[1]
            iupac = parts[2] if len(parts) > 2 else ""
            library[glycan_id] = {"smiles": smiles, "iupac": iupac}
    return library


def normalize_values(values: List[float], method: str) -> List[float]:
    if method == "none":
        return values
    if method == "log1p":
        return [math.log1p(max(v, 0.0)) for v in values]
    if method == "minmax":
        vmin = min(values)
        vmax = max(values)
        if vmax == vmin:
            return [0.0 for _ in values]
        return [(v - vmin) / (vmax - vmin) for v in values]
    raise ValueError(f"Unknown normalization method: {method}")


def split_indices(n: int, val_ratio: float, test_ratio: float, seed: int) -> Dict[str, List[int]]:
    rng = random.Random(seed)
    indices = list(range(n))
    rng.shuffle(indices)
    n_val = int(n * val_ratio)
    n_test = int(n * test_ratio)
    return {
        "val": indices[:n_val],
        "test": indices[n_val : n_val + n_test],
        "train": indices[n_val + n_test :],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess CFG lectin array data.")
    parser.add_argument("--input_csv", required=True, help="Raw CFG CSV with lectin/glycan data")
    parser.add_argument("--output_csv", required=True, help="Output cleaned CSV")
    parser.add_argument("--glycan_library", help="Optional glycan library mapping IDs to SMILES")
    parser.add_argument("--normalize", default="log1p", choices=["log1p", "minmax", "none"])
    parser.add_argument("--label_threshold", type=float, help="Create label if provided")
    parser.add_argument("--splits_out", default="data/train_val_test_splits.pkl")
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    glycan_library: Dict[str, Dict[str, str]] = {}
    if args.glycan_library:
        glycan_library = load_glycan_library(args.glycan_library)

    rows: List[Dict[str, str]] = []
    rfu_values: List[float] = []

    with open(args.input_csv, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            lectin_id = (row.get("lectin_id") or row.get("lectin") or "").strip()
            lectin_sequence = (row.get("lectin_sequence") or row.get("sequence") or "").strip()
            glycan_id = (row.get("glycan_id") or row.get("glycan") or "").strip()
            glycan_smiles = (row.get("glycan_smiles") or row.get("smiles") or "").strip()
            glycan_iupac = (row.get("glycan_iupac") or row.get("iupac") or "").strip()
            rfu_raw = row.get("rfu") or row.get("binding") or row.get("rfu_raw")

            if not lectin_sequence:
                continue

            if not glycan_smiles and glycan_id and glycan_id in glycan_library:
                glycan_smiles = glycan_library[glycan_id]["smiles"]
                glycan_iupac = glycan_library[glycan_id].get("iupac", glycan_iupac)

            if not glycan_smiles:
                continue

            try:
                rfu = float(rfu_raw) if rfu_raw is not None and rfu_raw != "" else 0.0
            except ValueError:
                rfu = 0.0

            rfu_values.append(rfu)
            rows.append(
                {
                    "lectin_id": lectin_id,
                    "lectin_sequence": lectin_sequence,
                    "glycan_id": glycan_id,
                    "glycan_smiles": glycan_smiles,
                    "glycan_iupac": glycan_iupac,
                    "rfu": str(rfu),
                }
            )

    if not rows:
        raise SystemExit("No valid rows found in input CSV.")

    rfu_norm = normalize_values(rfu_values, args.normalize)

    with open(args.output_csv, "w", newline="") as handle:
        fieldnames = ["lectin_id", "lectin_sequence", "glycan_id", "glycan_smiles", "glycan_iupac", "rfu", "rfu_norm"]
        if args.label_threshold is not None:
            fieldnames.append("label")
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, norm_value in zip(rows, rfu_norm):
            row["rfu_norm"] = f"{norm_value:.6f}"
            if args.label_threshold is not None:
                row["label"] = str(int(float(row["rfu"]) >= args.label_threshold))
            writer.writerow(row)

    splits = split_indices(len(rows), args.val_ratio, args.test_ratio, args.seed)
    with open(args.splits_out, "wb") as handle:
        pickle.dump(splits, handle)

    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    print(f"Saved splits to {args.splits_out}")


if __name__ == "__main__":
    main()
