#!/usr/bin/env python
"""Extract Fc regions and glycosite annotations from Phase 1 antibodies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

try:
    from Bio import pairwise2
except ImportError:  # pragma: no cover
    pairwise2 = None


IGG1_FC_REFERENCE = (
    "APELLGGPSVFLFPPKPKDTLMISRTPEVTCVVVDVSHEDPEVKFNWYVDGVEVHNAKTKPREEQYN"
    "STYRVVSVLTVLHQDWLNGKEYKCKVSNKALPAPIEKTISKAKGQPREPQVYTLPPSRDELTKNQV"
    "SLTCLVKGFYPSDIAVEWESNGQPENNYKTTPPVLDSDGSFFLYSKLTVDKSRWQQGNVFSCSVMH"
    "EALHNHYTQKSLSLSPGK"
)


def find_fc_region(heavy_chain_seq: str, min_fc_length: int = 200) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    if not heavy_chain_seq or pairwise2 is None:
        return None, None, None

    alignments = pairwise2.align.localms(
        heavy_chain_seq,
        IGG1_FC_REFERENCE,
        2,
        -1,
        -0.5,
        -0.1,
        one_alignment_only=True,
    )
    if not alignments:
        return None, None, None

    alignment = alignments[0]
    fc_start = alignment.start
    fc_end = alignment.end
    if fc_start is None or fc_end is None:
        return None, None, None
    if fc_end - fc_start < min_fc_length:
        return None, None, None
    return heavy_chain_seq[fc_start:fc_end], fc_start, fc_end


def load_antibodies(path: Path) -> List[Dict[str, Any]]:
    antibodies: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            antibodies.append(json.loads(line))
    return antibodies


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract Fc regions from Phase 1 antibodies.")
    parser.add_argument(
        "--input",
        default="data/processed/phase1_antibodies.jsonl",
        help="Input JSONL with antibody sequences",
    )
    parser.add_argument(
        "--output_csv",
        default="data/processed/antibody_fc_regions.csv",
        help="Output CSV with Fc regions",
    )
    parser.add_argument(
        "--output_glycosites",
        default="data/processed/antibody_glycosites.jsonl",
        help="Output JSONL with glycosite annotations",
    )
    parser.add_argument("--min_fc_length", type=int, default=200)
    args = parser.parse_args()

    if pairwise2 is None:
        raise SystemExit("Biopython is required. Install with: pip install biopython")

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    antibodies = load_antibodies(input_path)
    print(f"Loaded {len(antibodies)} antibodies")

    fc_rows: List[Dict[str, Any]] = []
    glyco_rows: List[Dict[str, Any]] = []

    for ab in antibodies:
        heavy_chain = ab.get("heavy_chain_seq", "")
        light_chain = ab.get("light_chain_seq", "")
        glycosites = ab.get("glycosites", []) or []
        fc_region, fc_start, fc_end = find_fc_region(heavy_chain, min_fc_length=args.min_fc_length)

        if not fc_region:
            continue

        antibody_id = ab.get("pdb_id") or ab.get("antibody_id") or ""
        antibody_name = ab.get("antibody_name") or antibody_id

        fc_rows.append(
            {
                "antibody_id": antibody_id,
                "antibody_name": antibody_name,
                "fc_sequence": fc_region,
                "fc_start": fc_start,
                "fc_end": fc_end,
                "full_heavy_chain": heavy_chain,
                "light_chain": light_chain,
                "num_glycosites": len(glycosites),
            }
        )

        glyco_rows.append({"antibody_id": antibody_id, "glycosites": glycosites})

    if not fc_rows:
        raise SystemExit("No Fc regions extracted. Check sequences or alignment settings.")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(fc_rows)
    df.to_csv(output_csv, index=False)

    output_glycosites = Path(args.output_glycosites)
    output_glycosites.parent.mkdir(parents=True, exist_ok=True)
    with output_glycosites.open("w", encoding="utf-8") as handle:
        for row in glyco_rows:
            handle.write(json.dumps(row) + "\n")

    print(f"✓ Extracted Fc regions from {len(df)} antibodies")
    print(f"✓ Saved to {output_csv}")
    print(f"✓ Glycosite annotations saved to {output_glycosites}")
    print("\nSummary:")
    print(f"  Total Fc regions: {len(df)}")
    print(f"  With glycosites: {(df['num_glycosites'] > 0).sum()}")
    print(f"  Avg glycosites per antibody: {df['num_glycosites'].mean():.1f}")


if __name__ == "__main__":
    main()
