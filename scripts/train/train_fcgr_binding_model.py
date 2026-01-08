#!/usr/bin/env python
"""Train FcγR-Fc binding model on synthetic literature data."""

from __future__ import annotations

import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "models"))
from esm2_lectin_encoder import ESM2LectinEncoder
from glycan_gnn_encoder import GlycanGNNEncoder
from fcgr_binding_predictor import FcGammaRFcPredictor


class FcgrDataset(Dataset):
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        return {
            "fcgr_sequence": row["fcgr_sequence"],
            "fc_sequence": row["fc_sequence"],
            "glycan_structure": row["glycan_structure"],
            "target": float(row["log_kd"]),
        }


def collate_fn(batch: List[dict]) -> dict:
    return {
        "fcgr_sequences": [item["fcgr_sequence"] for item in batch],
        "fc_sequences": [item["fc_sequence"] for item in batch],
        "glycan_structures": [item["glycan_structure"] for item in batch],
        "targets": torch.tensor([item["target"] for item in batch], dtype=torch.float32),
    }


def split_indices(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_val = max(1, int(0.15 * n))
    n_test = max(1, int(0.15 * n))
    val_idx = indices[:n_val]
    test_idx = indices[n_val : n_val + n_test]
    train_idx = indices[n_val + n_test :]
    return train_idx, val_idx, test_idx


def evaluate(model, loader, criterion, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    preds = []
    targets = []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["fcgr_sequences"],
                batch["fc_sequences"],
                batch["glycan_structures"],
            ).to(device)
            y_true = batch["targets"].to(device)
            loss = criterion(logits, y_true)
            total_loss += float(loss.item())
            preds.extend(logits.cpu().tolist())
            targets.extend(y_true.cpu().tolist())

    mse = float(np.mean((np.array(preds) - np.array(targets)) ** 2))
    avg_loss = total_loss / max(len(loader), 1)
    return avg_loss, mse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed/fcgr_fc_training_data.csv")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-model", default="models/fcgr_fc_binding_predictor.pt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    train_idx, val_idx, test_idx = split_indices(len(df), args.seed)
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    test_df = df.iloc[test_idx]

    train_loader = DataLoader(
        FcgrDataset(train_df),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        FcgrDataset(val_df), batch_size=args.batch_size, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        FcgrDataset(test_df), batch_size=args.batch_size, collate_fn=collate_fn
    )

    print(f"Training on device: {args.device}")
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    fcgr_encoder = ESM2LectinEncoder(embedding_dim=256, freeze_esm=True)
    fc_encoder = ESM2LectinEncoder(embedding_dim=256, freeze_esm=True)
    glycan_encoder = GlycanGNNEncoder(embedding_dim=128, hidden_dim=256, num_layers=3)
    model = FcGammaRFcPredictor(fcgr_encoder, fc_encoder, glycan_encoder, hidden_dims=[512, 256, 128])
    model = model.to(args.device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    best_val = float("inf")
    best_state = None

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            preds = model(
                batch["fcgr_sequences"],
                batch["fc_sequences"],
                batch["glycan_structures"],
            ).to(args.device)
            targets = batch["targets"].to(args.device)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        train_loss = total_loss / max(len(train_loader), 1)
        val_loss, val_mse = evaluate(model, val_loader, criterion, args.device)
        print(
            f"Epoch {epoch + 1}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - Val MSE: {val_mse:.4f}"
        )
        if val_loss < best_val:
            best_val = val_loss
            best_state = model.state_dict()

    if best_state is not None:
        torch.save({"model_state_dict": best_state, "val_loss": best_val}, args.output_model)

    test_loss, test_mse = evaluate(model, test_loader, criterion, args.device)
    print("\nTest Performance:")
    print(f"  MSE: {test_mse:.4f}")
    print(f"✓ Saved model to {args.output_model}")


if __name__ == "__main__":
    main()
