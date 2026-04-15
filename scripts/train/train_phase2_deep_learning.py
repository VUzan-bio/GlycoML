#!/usr/bin/env python
"""Train Phase 2 lectin-glycan binding model with deep learning."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

sys.path.append(str(Path(__file__).resolve().parents[2] / "models"))
from esm2_lectin_encoder import ESM2LectinEncoder
from glycan_gnn_encoder import GlycanGNNEncoder
from lectin_glycan_predictor import LectinGlycanPredictor


class LectinGlycanDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        return {
            "lectin_sequence": row["lectin_sequence"],
            "glycan_structure": row["glycan_structure"],
            "label": float(row["label"]),
        }


def collate_fn(batch: List[dict]) -> dict:
    return {
        "lectin_sequences": [item["lectin_sequence"] for item in batch],
        "glycan_structures": [item["glycan_structure"] for item in batch],
        "labels": torch.tensor([item["label"] for item in batch], dtype=torch.float32),
    }


def compute_auc(y_true: List[float], y_score: List[float]) -> float:
    y_true = np.array(y_true, dtype=int)
    y_score = np.array(y_score, dtype=float)
    pos = y_true == 1
    neg = y_true == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(y_score))
    pos_ranks = ranks[pos]
    auc = (pos_ranks.sum() - (pos.sum() * (pos.sum() - 1) / 2.0)) / (pos.sum() * neg.sum())
    return float(auc)


def precision_recall_f1(y_true: List[int], y_pred: List[int]) -> Tuple[float, float, float]:
    y_true = np.array(y_true, dtype=int)
    y_pred = np.array(y_pred, dtype=int)
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return precision, recall, f1


def stratified_split(labels: np.ndarray, val_ratio: float, test_ratio: float, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(labels))
    pos_idx = indices[labels == 1]
    neg_idx = indices[labels == 0]

    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    def split_group(group: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(group)
        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)
        return group[n_val + n_test :], group[:n_val], group[n_val : n_val + n_test]

    pos_train, pos_val, pos_test = split_group(pos_idx)
    neg_train, neg_val, neg_test = split_group(neg_idx)

    train_idx = np.concatenate([pos_train, neg_train])
    val_idx = np.concatenate([pos_val, neg_val])
    test_idx = np.concatenate([pos_test, neg_test])
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def train_epoch(model, dataloader, optimizer, criterion, device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        optimizer.zero_grad()
        logits = model(batch["lectin_sequences"], batch["glycan_structures"]).to(device)
        labels = batch["labels"].to(device)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item())
        all_preds.extend(torch.sigmoid(logits).detach().cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(len(dataloader), 1)
    auc = compute_auc(all_labels, all_preds)
    return avg_loss, auc


def evaluate(model, dataloader, criterion, device) -> Tuple[float, float, List[float], List[float]]:
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            logits = model(batch["lectin_sequences"], batch["glycan_structures"]).to(device)
            labels = batch["labels"].to(device)
            loss = criterion(logits, labels)
            total_loss += float(loss.item())
            all_preds.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(len(dataloader), 1)
    auc = compute_auc(all_labels, all_preds)
    return avg_loss, auc, all_preds, all_labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/phase2_merged_dataset.csv")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-model", default="models/phase2_deep_learning_predictor.pt")
    parser.add_argument("--allow-placeholder", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Training on device: {args.device}")
    df = pd.read_csv(args.data)

    if "label" not in df.columns:
        if "rfu_raw" in df.columns:
            df["label"] = (df["rfu_raw"].fillna(0) >= 2000).astype(int)
        else:
            raise ValueError("Missing label and rfu_raw columns; cannot build targets.")

    if "lectin_sequence" not in df.columns:
        if args.allow_placeholder:
            df["lectin_sequence"] = "M" * 50
            print("WARNING: Using placeholder sequences; ESM2 embeddings are not meaningful.")
        else:
            raise ValueError("lectin_sequence column is required for ESM2. Provide sequences or use --allow-placeholder.")

    if "glycan_structure" not in df.columns:
        if "glycan_iupac" in df.columns:
            df["glycan_structure"] = df["glycan_iupac"].fillna(df["glycan_id"].astype(str))
        else:
            df["glycan_structure"] = df["glycan_id"].astype(str)

    labels = df["label"].astype(int).to_numpy()
    train_idx, val_idx, test_idx = stratified_split(labels, 0.15, 0.15, args.seed)

    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    test_df = df.iloc[test_idx]

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    train_loader = DataLoader(
        LectinGlycanDataset(train_df),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        LectinGlycanDataset(val_df), batch_size=args.batch_size, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        LectinGlycanDataset(test_df), batch_size=args.batch_size, collate_fn=collate_fn
    )

    print("Initializing model...")
    lectin_encoder = ESM2LectinEncoder(embedding_dim=256, freeze_esm=True)
    glycan_encoder = GlycanGNNEncoder(embedding_dim=256, hidden_dim=512, num_layers=3)
    model = LectinGlycanPredictor(lectin_encoder, glycan_encoder, hidden_dims=[512, 256, 128])
    model = model.to(args.device)

    pos = float((train_df["label"] == 1).sum())
    neg = float((train_df["label"] == 0).sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=args.device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val_auc = 0.0
    best_state = None

    print("\nTraining...")
    for epoch in range(args.epochs):
        train_loss, train_auc = train_epoch(model, train_loader, optimizer, criterion, args.device)
        val_loss, val_auc, _, _ = evaluate(model, val_loader, criterion, args.device)
        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f} - Train AUC: {train_auc:.3f} - "
            f"Val Loss: {val_loss:.4f} - Val AUC: {val_auc:.3f}"
        )
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = model.state_dict()

    if best_state is not None:
        torch.save({"model_state_dict": best_state, "val_auc": best_val_auc}, args.output_model)

    print("\nEvaluating on test set...")
    if best_state is not None:
        model.load_state_dict(best_state)
    test_loss, test_auc, test_preds, test_labels = evaluate(
        model, test_loader, criterion, args.device
    )

    test_pred_binary = (np.array(test_preds) > 0.5).astype(int)
    precision, recall, f1 = precision_recall_f1(test_labels, test_pred_binary)

    print(f"\n✓ Test AUC: {test_auc:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1: {f1:.3f}")

    test_df = test_df.copy()
    test_df["predicted_affinity"] = test_preds
    test_df.to_csv("data/processed/phase2_test_predictions_dl.csv", index=False)
    print(f"\n✓ Model saved to {args.output_model}")
    print("✓ Test predictions saved to data/processed/phase2_test_predictions_dl.csv")


if __name__ == "__main__":
    main()
