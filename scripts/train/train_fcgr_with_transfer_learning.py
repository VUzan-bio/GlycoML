#!/usr/bin/env python
"""Train FcγR-Fc affinity model with CFG-pretrained glycan encoder."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch_geometric.data import Batch, Data
from torch_geometric.nn import global_mean_pool

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.esm2_lectin_encoder import ESM2LectinEncoder

sys.path.append(str(Path(__file__).parent))
from train_phase2_with_glycan_encoder_export import (
    GlycanGNNEncoder,
    iupac_to_token_graph,
    looks_like_iupac,
    smiles_to_graph,
)
from scripts.utils.glycan_graph_encoder import parse_iupac_condensed


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(requested)
        try:
            dummy_x = torch.zeros((2, 1), device=device)
            dummy_batch = torch.tensor([0, 0], device=device)
            _ = global_mean_pool(dummy_x, dummy_batch)
            return device
        except Exception:
            print("WARNING: torch-scatter CUDA not available; falling back to CPU.")
            return torch.device("cpu")
    return torch.device("cpu")


class FcGammaRFcPredictorTransfer(nn.Module):
    def __init__(self, glycan_encoder_pretrained: GlycanGNNEncoder, freeze_glycan: bool = False) -> None:
        super().__init__()
        self.fcgr_encoder = ESM2LectinEncoder(embedding_dim=256, freeze_esm=True, model_name="esm2_t6_8M_UR50D")
        self.fc_encoder = ESM2LectinEncoder(embedding_dim=256, freeze_esm=True, model_name="esm2_t6_8M_UR50D")
        self.glycan_encoder = glycan_encoder_pretrained

        if freeze_glycan:
            for param in self.glycan_encoder.parameters():
                param.requires_grad = False
            print("✓ Glycan encoder frozen (feature extraction mode)")
        else:
            print("✓ Glycan encoder will be fine-tuned")

        self.fcgr_glycan_interaction = nn.Bilinear(256, 512, 128)
        self.fc_glycan_interaction = nn.Bilinear(256, 512, 128)

        self.mlp = nn.Sequential(
            nn.Linear(256 + 256 + 512 + 128 + 128, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

    def forward(self, fcgr_seqs: List[str], fc_seqs: List[str], glycan_graphs: Data) -> torch.Tensor:
        fcgr_emb = self.fcgr_encoder(fcgr_seqs)
        fc_emb = self.fc_encoder(fc_seqs)
        glycan_emb = self.glycan_encoder(glycan_graphs)
        fcgr_glycan_int = self.fcgr_glycan_interaction(fcgr_emb, glycan_emb)
        fc_glycan_int = self.fc_glycan_interaction(fc_emb, glycan_emb)
        combined = torch.cat([fcgr_emb, fc_emb, glycan_emb, fcgr_glycan_int, fc_glycan_int], dim=1)
        return self.mlp(combined).squeeze(-1)


class FcGammaRDataset(Dataset):
    def __init__(self, csv_path: str, allow_non_smiles: bool = False) -> None:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise SystemExit(f"Input not found: {csv_path}")

        df = pd.read_csv(csv_path)
        required = ["fcgr_sequence", "fc_sequence", "glycan_structure", "binding_kd_nm"]
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if "log_kd" not in df.columns:
            df["log_kd"] = np.log10(df["binding_kd_nm"])
        df = df.dropna(subset=required + ["log_kd"])
        self.df = df.reset_index(drop=True)
        self.allow_non_smiles = allow_non_smiles
        print(f"Loaded {len(self.df)} samples from {csv_path}")
        print(f"  KD range: {self.df['binding_kd_nm'].min():.1f} - {self.df['binding_kd_nm'].max():.1f} nM")

        if not self.allow_non_smiles:
            for idx, row in self.df.iterrows():
                glycan_smiles = row["glycan_structure"]
                try:
                    _ = smiles_to_graph(glycan_smiles)
                except Exception as exc:
                    raise ValueError(f"Invalid glycan_structure at index {idx}: {glycan_smiles}") from exc

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        fcgr_seq = row["fcgr_sequence"]
        fc_seq = row["fc_sequence"]
        glycan_smiles = str(row["glycan_structure"])
        log_kd = torch.tensor(row["log_kd"], dtype=torch.float32)
        if self.allow_non_smiles and looks_like_iupac(glycan_smiles):
            try:
                glycan_graph = parse_iupac_condensed(glycan_smiles)
            except Exception:
                glycan_graph = iupac_to_token_graph(glycan_smiles)
        else:
            try:
                glycan_graph = smiles_to_graph(glycan_smiles, strict=not self.allow_non_smiles)
            except Exception:
                glycan_graph = None
            if glycan_graph is None:
                glycan_graph = iupac_to_token_graph(glycan_smiles)
        return fcgr_seq, fc_seq, glycan_graph, log_kd


def collate_fn(batch):
    fcgr_seqs = [item[0] for item in batch]
    fc_seqs = [item[1] for item in batch]
    glycan_graphs = Batch.from_data_list([item[2] for item in batch])
    log_kds = torch.stack([item[3] for item in batch], dim=0)
    return fcgr_seqs, fc_seqs, glycan_graphs, log_kds


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for fcgr_seqs, fc_seqs, glycan_graphs, log_kds in dataloader:
        glycan_graphs = glycan_graphs.to(device)
        log_kds = log_kds.to(device)
        optimizer.zero_grad()
        preds = model(fcgr_seqs, fc_seqs, glycan_graphs)
        loss = criterion(preds, log_kds)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += float(loss.item())
    return total_loss / max(len(dataloader), 1)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    preds = []
    targets = []
    with torch.no_grad():
        for fcgr_seqs, fc_seqs, glycan_graphs, log_kds in dataloader:
            glycan_graphs = glycan_graphs.to(device)
            log_kds = log_kds.to(device)
            output = model(fcgr_seqs, fc_seqs, glycan_graphs)
            loss = criterion(output, log_kds)
            total_loss += float(loss.item())
            preds.extend(output.cpu().tolist())
            targets.extend(log_kds.cpu().tolist())
    mse = float(np.mean((np.array(preds) - np.array(targets)) ** 2))
    return total_loss / max(len(dataloader), 1), mse


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FcγR model with CFG-pretrained glycan encoder.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--glycan-encoder", required=True)
    parser.add_argument("--output-model", default="models/fcgr_transfer_learning.pt")
    parser.add_argument("--freeze-glycan", action="store_true")
    parser.add_argument("--glycan-lr", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-non-smiles", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FcγR PHASE 3 TRAINING WITH TRANSFER LEARNING")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"FcγR data: {args.data}")
    print(f"Pretrained glycan encoder: {args.glycan_encoder}")
    print(f"Output model: {args.output_model}")
    print(f"Freeze glycan encoder: {args.freeze_glycan}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 80)

    try:
        checkpoint = torch.load(args.glycan_encoder, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(args.glycan_encoder, map_location=device)

    glycan_encoder = GlycanGNNEncoder(
        input_dim=5, hidden_dim=64, embedding_dim=int(checkpoint["embedding_dim"])
    )
    glycan_encoder.load_state_dict(checkpoint["model_state_dict"])

    dataset = FcGammaRDataset(args.data, allow_non_smiles=args.allow_non_smiles)
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    model = FcGammaRFcPredictorTransfer(glycan_encoder, freeze_glycan=args.freeze_glycan)
    model = model.to(device)

    if args.freeze_glycan:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.Adam(
            [
                {"params": model.fcgr_encoder.parameters(), "lr": args.lr},
                {"params": model.fc_encoder.parameters(), "lr": args.lr},
                {"params": model.glycan_encoder.parameters(), "lr": args.glycan_lr},
                {"params": model.fcgr_glycan_interaction.parameters(), "lr": args.lr},
                {"params": model.fc_glycan_interaction.parameters(), "lr": args.lr},
                {"params": model.mlp.parameters(), "lr": args.lr},
            ]
        )

    criterion = nn.MSELoss()
    best_val_mse = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mse = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MSE: {val_mse:.4f}"
        )
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_mse": val_mse,
                    "transfer_learning": True,
                    "glycan_encoder_frozen": args.freeze_glycan,
                    "glycan_encoder_source": args.glycan_encoder,
                },
                args.output_model,
            )
            print(f"  ✓ Saved best model (Val MSE: {val_mse:.4f})")

    _, test_mse = evaluate(model, test_loader, criterion, device)
    print("\nFINAL TEST EVALUATION")
    print(f"Test MSE: {test_mse:.4f}")
    print(f"✓ Model saved to {args.output_model}")


if __name__ == "__main__":
    main()
