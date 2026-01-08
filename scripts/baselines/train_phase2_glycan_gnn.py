#!/usr/bin/env python
"""Train a baseline glycan embedding model (SVD fallback)."""

from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline glycan embeddings for Phase 2.")
    parser.add_argument("--features", default="data/processed/glycan_features.pkl")
    parser.add_argument("--output-dir", default="data/processed")
    parser.add_argument("--model-out", default="models/glycan_gnn_baseline.npz")
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    args = parser.parse_args()

    with open(args.features, "rb") as handle:
        glycan_features: Dict[str, List[float]] = pickle.load(handle)

    glycan_ids = sorted(glycan_features.keys())
    if not glycan_ids:
        raise SystemExit("No glycan features found. Run preprocess_glycan_structures.py first.")

    matrix = np.array([glycan_features[gid] for gid in glycan_ids], dtype=np.float32)
    mean = matrix.mean(axis=0, keepdims=True)
    centered = matrix - mean

    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    dim = min(args.embedding_dim, vt.shape[0])
    components = vt[:dim].T
    embeddings = centered @ components

    os.makedirs(args.output_dir, exist_ok=True)
    embedding_path = os.path.join(args.output_dir, "glycan_embeddings.npy")
    index_path = os.path.join(args.output_dir, "glycan_embeddings_index.csv")
    np.save(embedding_path, embeddings)

    pd.DataFrame({"glycan_id": glycan_ids, "row_index": list(range(len(glycan_ids)))}).to_csv(
        index_path, index=False
    )

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    np.savez(args.model_out, mean=mean.squeeze(), components=components)

    print("Training glycan encoder (baseline SVD)...")
    print(f"✓ Features: {len(glycan_ids):,} glycans, dim {matrix.shape[1]}")
    print(f"✓ Embeddings saved to {embedding_path}")
    print(f"✓ Index saved to {index_path}")
    print(f"✓ Baseline model saved to {args.model_out}")
    print("NOTE: epochs/batch-size/learning-rate are ignored for SVD baseline.")


if __name__ == "__main__":
    main()
