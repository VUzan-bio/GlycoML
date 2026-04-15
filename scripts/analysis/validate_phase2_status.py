#!/usr/bin/env python
"""Report Phase 2 dataset status from CFG + UniLectin inputs."""

from __future__ import annotations

import pandas as pd


def main() -> None:
    cfg = pd.read_csv("data/processed/cfg_rfu_measurements.csv")
    uni = pd.read_csv("data/interim/unilectin3d_lectin_glycan_interactions.csv")

    print("=" * 70)
    print("PHASE 2 DATASET - CURRENT STATUS (27 experiments)")
    print("=" * 70)
    print(f"CFG experiments:        {cfg.experiment_id.nunique():>8}/40 target")
    print(f"CFG measurements:       {len(cfg):>8,}")
    print(
        "CFG strong binders:     "
        f"{(cfg.rfu_raw > 2000).sum():>8,} "
        f"({100 * (cfg.rfu_raw > 2000).sum() / len(cfg):>4.1f}%)"
    )
    print(f"CFG unique glycans:     {cfg.glycan_id.nunique():>8}")
    print(f"CFG mean RFU:           {cfg.rfu_raw.mean():>8.1f}")
    print(f"\nUniLectin interactions: {len(uni):>8,}")
    print(f"UniLectin lectins:      {uni.protein_name.nunique():>8,}")
    print(f"\n{'TOTAL TRAINING DATA:':24} {len(cfg) + len(uni):>8,}")
    print("=" * 70)

    if len(cfg) + len(uni) >= 20000:
        print("\n✅ MEETS MINIMUM THRESHOLD (20,000+ total)")
        print("\n📊 Data Quality Assessment:")
        print(f"   • Coverage: {cfg.experiment_id.nunique()}/40 experiments")
        print(
            f"   • Strong binders: {(cfg.rfu_raw > 2000).sum():,} "
            f"({100 * (cfg.rfu_raw > 2000).sum() / len(cfg):.1f}%)"
        )
        print(f"   • Unique glycans: {cfg.glycan_id.nunique()}")
        print(f"   • Lectin diversity: {uni.protein_name.nunique()} from UniLectin")

        print("\n🎯 RECOMMENDATION:")
        print("   Option A: START TRAINING NOW")
        print("   • Sufficient for Phase 2 proof-of-concept")
        print("   • Can add remaining experiments later as bonus data")
        print("   • Saves 30-45 minutes of downloading")

        print("\n   Option B: Download missing experiments first")
        print("   • Would reach ~27,000 total measurements")
        print("   • More data for better model")
        print("   • Costs 20-30 minutes")

        print("\n   Choose Option A if time-constrained, Option B for optimal results")
    else:
        print(f"\n⚠️ BELOW MINIMUM (have {len(cfg) + len(uni):,}, need 20,000)")
        print("   → Must download missing experiments")


if __name__ == "__main__":
    main()
