#!/usr/bin/env python
"""Merge CFG RFU data with UniLectin interactions for Phase 2 training."""

from __future__ import annotations

import argparse
from typing import Dict, List

import pandas as pd


def _build_cfg_frame(cfg: pd.DataFrame) -> pd.DataFrame:
    required = {"experiment_id", "lectin_sample_name", "glycan_id", "cfg_glycan_iupac", "rfu_raw"}
    missing = required - set(cfg.columns)
    if missing:
        raise ValueError(f"CFG data missing columns: {sorted(missing)}")

    out = pd.DataFrame(
        {
            "source": "CFG",
            "lectin_id": "cfg_" + cfg["experiment_id"].astype(str),
            "lectin_name": cfg["lectin_sample_name"].astype(str),
            "lectin_family": "",
            "experiment_id": cfg["experiment_id"].astype(str),
            "glycan_id": cfg["glycan_id"].astype(str),
            "glycan_iupac": cfg["cfg_glycan_iupac"].fillna("").astype(str),
            "rfu_raw": cfg["rfu_raw"].astype(float),
        }
    )
    out["label"] = (out["rfu_raw"] >= 2000).astype(int)
    return out


def _build_unilectin_frame(uni: pd.DataFrame) -> pd.DataFrame:
    required = {"lectin_id", "protein_name"}
    missing = required - set(uni.columns)
    if missing:
        raise ValueError(f"UniLectin data missing columns: {sorted(missing)}")

    glycan_id = ""
    if "glytoucan_id" in uni.columns:
        glycan_id = uni["glytoucan_id"].fillna("")
    elif "ligand" in uni.columns:
        glycan_id = uni["ligand"].fillna("")
    else:
        glycan_id = ""

    glycan_iupac = ""
    if "iupac" in uni.columns:
        glycan_iupac = uni["iupac"].fillna("")

    family = ""
    if "family" in uni.columns:
        family = uni["family"].fillna("")

    out = pd.DataFrame(
        {
            "source": "UniLectin3D",
            "lectin_id": "unilectin_" + uni["lectin_id"].astype(str),
            "lectin_name": uni["protein_name"].astype(str),
            "lectin_family": family.astype(str),
            "experiment_id": "",
            "glycan_id": glycan_id.astype(str),
            "glycan_iupac": glycan_iupac.astype(str),
            "rfu_raw": float("nan"),
            "label": 1,
        }
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge CFG + UniLectin datasets for Phase 2.")
    parser.add_argument(
        "--cfg",
        default="data/processed/cfg_rfu_measurements.csv",
        help="CFG RFU measurements CSV",
    )
    parser.add_argument(
        "--unilectin",
        default="data/interim/unilectin3d_lectin_glycan_interactions.csv",
        help="UniLectin interactions CSV",
    )
    parser.add_argument(
        "--output",
        default="data/processed/phase2_merged_dataset.csv",
        help="Output merged dataset CSV",
    )
    args = parser.parse_args()

    cfg = pd.read_csv(args.cfg)
    uni = pd.read_csv(args.unilectin)

    cfg_frame = _build_cfg_frame(cfg)
    uni_frame = _build_unilectin_frame(uni)

    merged = pd.concat([cfg_frame, uni_frame], ignore_index=True)
    merged.to_csv(args.output, index=False)

    print("Merging Phase 2 datasets...")
    print(f"✓ Loaded CFG: {len(cfg_frame):,} measurements")
    print(f"✓ Loaded UniLectin: {len(uni_frame):,} interactions")
    print(f"✓ Merged dataset: {len(merged):,} total rows")
    print(f"✓ Saved to {args.output}")


if __name__ == "__main__":
    main()
