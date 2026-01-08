#!/usr/bin/env python
"""Evaluate Phase 2 binding predictor on a held-out test set."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from phase2_baseline_utils import (
    compute_auc,
    compute_classification_stats,
    hash_text_features,
    load_glycan_embeddings,
    sigmoid,
)


def build_features(
    df: pd.DataFrame, glycan_embeddings: dict[str, np.ndarray], emb_dim: int, lectin_dim: int
) -> np.ndarray:
    features = np.zeros((len(df), emb_dim + lectin_dim), dtype=np.float32)
    for idx, row in df.iterrows():
        glycan_id = str(row.get("glycan_id", "")).strip()
        glycan_vec = glycan_embeddings.get(glycan_id, np.zeros(emb_dim, dtype=np.float32))
        lectin_name = str(row.get("lectin_name") or row.get("lectin_id") or "").strip()
        lectin_vec = hash_text_features(lectin_name, dim=lectin_dim)
        features[idx, :emb_dim] = glycan_vec
        features[idx, emb_dim:] = lectin_vec
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 binding model.")
    parser.add_argument("--test-data", default="data/processed/phase2_test_split.csv")
    parser.add_argument("--model", default="models/phase2_lectin_glycan_predictor.npz")
    parser.add_argument("--glycan-embeddings", default="data/processed/glycan_embeddings.npy")
    parser.add_argument("--glycan-index", default="data/processed/glycan_embeddings_index.csv")
    args = parser.parse_args()

    test_data = pd.read_csv(args.test_data)
    model = np.load(args.model)
    weights = model["weights"]
    bias = float(model["bias"])
    emb_dim = int(model["glycan_dim"])
    lectin_dim = int(model["lectin_dim"])

    glycan_embeddings, _ = load_glycan_embeddings(args.glycan_embeddings, args.glycan_index)
    features = build_features(test_data, glycan_embeddings, emb_dim, lectin_dim)

    y_true = test_data["label"].astype(int).to_numpy()
    logits = features @ weights + bias
    probs = sigmoid(logits)
    preds = (probs >= 0.5).astype(int)

    auc = compute_auc(y_true, probs)
    accuracy, precision, recall, f1 = compute_classification_stats(y_true, preds)
    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        tmp_preds = (probs >= threshold).astype(int)
        _, _, _, tmp_f1 = compute_classification_stats(y_true, tmp_preds)
        if tmp_f1 > best_f1:
            best_f1 = tmp_f1
            best_threshold = threshold

    print("Phase 2 Model Evaluation")
    print("========================")
    print("\nOverall Metrics:")
    print(f"  AUC-ROC: {auc:.3f}")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1: {f1:.3f}")
    print(f"  Optimal threshold (F1): {best_threshold:.2f}")

    opt_preds = (probs >= best_threshold).astype(int)
    opt_acc, opt_precision, opt_recall, opt_f1 = compute_classification_stats(y_true, opt_preds)
    print("\nMetrics @ Optimal Threshold:")
    print(f"  Accuracy: {opt_acc:.3f}")
    print(f"  Precision: {opt_precision:.3f}")
    print(f"  Recall: {opt_recall:.3f}")
    print(f"  F1: {opt_f1:.3f}")

    if "lectin_family" in test_data.columns:
        print("\nBy Lectin Family:")
        for family, group in test_data.groupby("lectin_family"):
            if len(group) < 20:
                continue
            idx = group.index.to_numpy()
            family_auc = compute_auc(y_true[idx], probs[idx])
            if np.isnan(family_auc):
                continue
            print(f"  {family}: AUC {family_auc:.3f} (n={len(group)})")

    if "rfu_raw" in test_data.columns:
        rfu_mask = test_data["rfu_raw"].notna()
        if rfu_mask.any():
            strong_mask = test_data["rfu_raw"] > 2000
            if strong_mask.any():
                strong_true = strong_mask.astype(int).to_numpy()
                precision_s, recall_s, _, _ = compute_classification_stats(strong_true, preds)
                print("\nStrong Binders (RFU > 2000):")
                print(f"  Precision (0.5): {precision_s:.3f}")
                print(f"  Recall (0.5):    {recall_s:.3f}")
                precision_o, recall_o, _, _ = compute_classification_stats(strong_true, opt_preds)
                print(f"  Precision (opt): {precision_o:.3f}")
                print(f"  Recall (opt):    {recall_o:.3f}")


if __name__ == "__main__":
    main()
