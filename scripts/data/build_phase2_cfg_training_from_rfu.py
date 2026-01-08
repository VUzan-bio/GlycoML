#!/usr/bin/env python
"""Build Phase 2 training CSV from CFG RFU measurements with lectin sequences."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def tokenize_name(value: str) -> List[str]:
    return [token for token in re.split(r"[^a-z0-9]+", value.lower()) if token]


def load_lectin_sequences(jsonl_path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            name = record.get("lectin_name") or ""
            seq = record.get("lectin_seq") or record.get("lectin_sequence") or ""
            if not name or not seq:
                continue
            mapping[normalize_name(name)] = seq
    return mapping


def match_sequence(sample_name: str, seq_map: Dict[str, str], min_len: int = 4) -> Optional[str]:
    if not sample_name:
        return None
    key = normalize_name(sample_name)
    if key in seq_map:
        return seq_map[key]

    tokens = set(tokenize_name(sample_name))
    best_key = ""
    best_score = 0
    for candidate in seq_map.keys():
        if len(candidate) < min_len:
            continue
        if candidate in key or key in candidate:
            if len(candidate) > len(best_key):
                best_key = candidate
                best_score = 1
            continue
        candidate_tokens = set(tokenize_name(candidate))
        overlap = len(tokens & candidate_tokens)
        if overlap > best_score:
            best_key = candidate
            best_score = overlap
    return seq_map.get(best_key) if best_key else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Phase 2 CFG training CSV from RFU measurements.")
    parser.add_argument(
        "--cfg-rfu",
        default="data/processed/cfg_rfu_measurements.csv",
        help="CFG RFU measurements CSV",
    )
    parser.add_argument(
        "--lectin-jsonl",
        default="data/processed/phase2_lectin_glycan.jsonl",
        help="JSONL with lectin sequences",
    )
    parser.add_argument(
        "--cfg-metadata",
        default="data/metadata/cfg_experiment_metadata.csv",
        help="CFG metadata with experiment_id -> sample_name",
    )
    parser.add_argument(
        "--output",
        default="data/processed/phase2_transfer_training_cfg.csv",
        help="Output CSV for Phase 2 training",
    )
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=60.0,
        help="Percentile for RFU thresholding (default: 60).",
    )
    parser.add_argument(
        "--min-match-length",
        type=int,
        default=4,
        help="Minimum normalized name length for fuzzy matching.",
    )
    parser.add_argument(
        "--allow-unmatched",
        action="store_true",
        help="Use placeholder sequence for unmatched lectins instead of skipping.",
    )
    args = parser.parse_args()

    cfg_path = Path(args.cfg_rfu)
    if not cfg_path.exists():
        raise SystemExit(f"CFG RFU file not found: {cfg_path}")
    lectin_path = Path(args.lectin_jsonl)
    if not lectin_path.exists():
        raise SystemExit(f"Lectin JSONL not found: {lectin_path}")

    cfg = pd.read_csv(cfg_path)
    required = {"lectin_sample_name", "cfg_glycan_iupac", "rfu_raw", "glycan_id"}
    missing = required - set(cfg.columns)
    if missing:
        raise SystemExit(f"CFG RFU missing columns: {sorted(missing)}")

    seq_map = load_lectin_sequences(lectin_path)
    if not seq_map:
        raise SystemExit("No lectin sequences loaded from JSONL.")

    meta_name_map: Dict[str, str] = {}
    meta_path = Path(args.cfg_metadata)
    if meta_path.exists():
        meta = pd.read_csv(meta_path)
        if "experiment_id" in meta.columns and "sample_name" in meta.columns:
            for _, row in meta.iterrows():
                exp_id = str(row.get("experiment_id", "")).strip()
                sample_name = str(row.get("sample_name", "")).strip()
                if exp_id and sample_name:
                    meta_name_map[exp_id] = sample_name

    rows: List[Dict[str, object]] = []
    skipped = 0
    for _, row in cfg.iterrows():
        exp_id = str(row.get("experiment_id", "")).strip()
        metadata_name = meta_name_map.get(exp_id, "")
        lectin_name = metadata_name or str(row.get("lectin_sample_name") or "")
        glycan_iupac = str(row.get("cfg_glycan_iupac") or "")
        if not lectin_name or not glycan_iupac:
            skipped += 1
            continue

        seq = match_sequence(lectin_name, seq_map, min_len=args.min_match_length)
        if not seq:
            if args.allow_unmatched:
                seq = "M" * 50
            else:
                skipped += 1
                continue

        rows.append(
            {
                "lectin_sequence": seq,
                "lectin_name": lectin_name,
                "glycan_smiles": glycan_iupac,
                "glycan_id": row.get("glycan_id"),
                "experiment_id": exp_id,
                "rfu_raw": row.get("rfu_raw"),
            }
        )

    if not rows:
        raise SystemExit("No rows created. Check name matching or allow unmatched.")

    df = pd.DataFrame(rows)
    df["rfu_raw"] = pd.to_numeric(df["rfu_raw"], errors="coerce")
    rfu_values = df["rfu_raw"].dropna().to_numpy()
    if rfu_values.size == 0:
        raise SystemExit("No numeric RFU values found in CFG data.")

    threshold = float(np.percentile(rfu_values, args.threshold_percentile))
    df["binding"] = (df["rfu_raw"] >= threshold).astype(int)
    if df["binding"].nunique() == 1:
        df["binding"] = (df["rfu_raw"] > threshold).astype(int)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✓ Wrote {len(df)} rows to {output_path}")
    print(f"  Matched lectins: {len(df)} | Skipped: {skipped}")
    print(f"  Threshold percentile: {args.threshold_percentile:.1f}")
    print(f"  RFU threshold: {threshold:.2f}")
    print(f"  Positive samples: {int(df['binding'].sum())}")
    print(f"  Negative samples: {len(df) - int(df['binding'].sum())}")
    if df["binding"].nunique() == 1:
        print("⚠️ Binding labels are degenerate; adjust percentile or inspect RFU distribution.")


if __name__ == "__main__":
    main()
