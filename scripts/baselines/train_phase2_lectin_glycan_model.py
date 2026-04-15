#!/usr/bin/env python
"""Train a baseline lectin-glycan binding model (logistic regression)."""

from __future__ import annotations

import argparse
import json
import os
from typing import Tuple

import numpy as np
import pandas as pd

from phase2_baseline_utils import (
    compute_auc,
    compute_classification_stats,
    hash_text_features,
    load_glycan_embeddings,
    sigmoid,
)


def build_labels(df: pd.DataFrame) -> np.ndarray:
    labels = []
    for _, row in df.iterrows():
        if "label" in df.columns and pd.notna(row.get("label")):
            labels.append(int(row["label"]))
            continue
        rfu_raw = row.get("rfu_raw")
        if pd.notna(rfu_raw):
            labels.append(int(float(rfu_raw) >= 2000))
        else:
            labels.append(1)
    return np.array(labels, dtype=np.float32)


def build_features(
    df: pd.DataFrame,
    glycan_embeddings: dict[str, np.ndarray],
    emb_dim: int,
    lectin_dim: int,
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


def split_indices(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    train_end = int(0.7 * n)
    val_end = int(0.85 * n)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def batch_indices(indices: np.ndarray, batch_size: int) -> list[np.ndarray]:
    if batch_size <= 0 or batch_size >= len(indices):
        return [indices]
    return [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]


def train_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    lr: float,
    class_weight: str,
) -> Tuple[np.ndarray, float, dict]:
    weights = np.zeros(x_train.shape[1], dtype=np.float32)
    bias = 0.0

    best_auc = -1.0
    best_state = {"weights": weights.copy(), "bias": bias}

    pos = float((y_train == 1).sum())
    neg = float((y_train == 0).sum())
    if class_weight == "balanced" and pos > 0 and neg > 0:
        w_pos = (pos + neg) / (2.0 * pos)
        w_neg = (pos + neg) / (2.0 * neg)
    else:
        w_pos = 1.0
        w_neg = 1.0

    for epoch in range(1, epochs + 1):
        for batch in batch_indices(np.arange(len(x_train)), batch_size):
            xb = x_train[batch]
            yb = y_train[batch]
            logits = xb @ weights + bias
            preds = sigmoid(logits)
            sample_weights = np.where(yb == 1, w_pos, w_neg)
            grad = (preds - yb) * sample_weights
            weights -= lr * (xb.T @ grad) / len(xb)
            bias -= lr * float(grad.mean())

        val_logits = x_val @ weights + bias
        val_probs = sigmoid(val_logits)
        val_weights = np.where(y_val == 1, w_pos, w_neg)
        val_loss = -np.mean(
            val_weights
            * (y_val * np.log(val_probs + 1e-8) + (1.0 - y_val) * np.log(1.0 - val_probs + 1e-8))
        )
        val_auc = compute_auc(y_val.astype(int), val_probs)

        print(f"Epoch {epoch}/{epochs} - Val Loss: {val_loss:.4f} - Val AUC: {val_auc:.3f}")

        if np.isnan(val_auc):
            continue
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {"weights": weights.copy(), "bias": bias}

    return best_state["weights"], float(best_state["bias"]), {"best_val_auc": best_auc}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline lectin-glycan binding model.")
    parser.add_argument("--data", default="data/processed/phase2_merged_dataset.csv")
    parser.add_argument("--glycan-embeddings", default="data/processed/glycan_embeddings.npy")
    parser.add_argument("--glycan-index", default="data/processed/glycan_embeddings_index.csv")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--use-glycan-embeddings", action="store_true")
    parser.add_argument("--lectin-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    parser.add_argument("--model-out", default="models/phase2_lectin_glycan_predictor.npz")
    parser.add_argument("--test-out", default="data/processed/phase2_test_split.csv")
    parser.add_argument("--metrics-out", default="data/processed/phase2_metrics.json")
    args = parser.parse_args()

    data = pd.read_csv(args.data)
    glycan_embeddings, emb_dim = load_glycan_embeddings(args.glycan_embeddings, args.glycan_index)

    labels = build_labels(data)
    features = build_features(data, glycan_embeddings, emb_dim, args.lectin_dim)

    train_idx, val_idx, test_idx = split_indices(len(data), args.seed)
    x_train, y_train = features[train_idx], labels[train_idx]
    x_val, y_val = features[val_idx], labels[val_idx]
    x_test, y_test = features[test_idx], labels[test_idx]

    print("Training lectin-glycan binding model (baseline logistic regression)...")
    print(f"Dataset: {len(data):,} interactions ({len(train_idx):,} train / {len(test_idx):,} test)")

    weights, bias, train_info = train_model(
        x_train, y_train, x_val, y_val, args.epochs, args.batch_size, args.learning_rate, args.class_weight
    )

    test_logits = x_test @ weights + bias
    test_probs = sigmoid(test_logits)
    test_pred = (test_probs >= 0.5).astype(int)

    auc = compute_auc(y_test.astype(int), test_probs)
    accuracy, precision, recall, f1 = compute_classification_stats(y_test.astype(int), test_pred)

    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    np.savez(
        args.model_out,
        weights=weights,
        bias=bias,
        glycan_dim=emb_dim,
        lectin_dim=args.lectin_dim,
    )

    test_frame = data.iloc[test_idx].copy()
    test_frame["prediction"] = test_probs
    test_frame["pred_label"] = test_pred
    test_frame.to_csv(args.test_out, index=False)

    metrics = {
        "val_auc_best": train_info["best_val_auc"],
        "test_auc": auc,
        "test_accuracy": accuracy,
        "test_precision": precision,
        "test_recall": recall,
        "test_f1": f1,
    }
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)
    with open(args.metrics_out, "w") as handle:
        json.dump(metrics, handle, indent=2)

    print(f"\n✓ Best model saved to {args.model_out}")
    print(f"✓ Test split saved to {args.test_out}")
    print("\nTest Set Performance:")
    print(f"  - AUC-ROC:  {auc:.3f}")
    print(f"  - Accuracy: {accuracy:.3f}")
    print(f"  - Precision:{precision:.3f}")
    print(f"  - Recall:   {recall:.3f}")
    print(f"  - F1 Score: {f1:.3f}")


if __name__ == "__main__":
    main()
