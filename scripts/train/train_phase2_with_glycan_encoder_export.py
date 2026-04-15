#!/usr/bin/env python
"""
Train CFG Phase 2 lectin-glycan predictor and export glycan encoder for Phase 3 transfer.

Outputs:
  - models/phase2_cfg_full_model.pt
  - models/phase2_glycan_encoder_pretrained.pt
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool
try:
    from rdkit import Chem
    from rdkit import RDLogger
except ImportError as exc:  # pragma: no cover
    Chem = None
    RDLogger = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.esm2_lectin_encoder import ESM2LectinEncoder
from scripts.utils.glycan_graph_encoder import parse_iupac_condensed


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(requested)
        try:
            dummy_x = torch.zeros((2, 1), device=device)
            dummy_batch = torch.tensor([0, 0], device=device)
            _ = global_max_pool(dummy_x, dummy_batch)
            return device
        except Exception:
            print("WARNING: torch-scatter CUDA not available; falling back to CPU.")
            return torch.device("cpu")
    return torch.device("cpu")


class GlycanGNNEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 5,
        hidden_dim: int = 64,
        embedding_dim: int = 512,
        edge_dim: int = 3,
    ) -> None:
        super().__init__()
        self.edge_dim = edge_dim
        self.node_proj = nn.Linear(input_dim, hidden_dim)
        self.conv1 = GINEConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)),
            edge_dim=edge_dim,
        )
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GINEConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)),
            edge_dim=edge_dim,
        )
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GINEConv(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)),
            edge_dim=edge_dim,
        )
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.embedding_dim = embedding_dim

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.node_proj(x)
        edge_attr = getattr(data, "edge_attr", None)
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), self.edge_dim), device=x.device, dtype=x.dtype)
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.3, training=self.training)

        x = self.conv3(x, edge_index, edge_attr)
        x = self.bn3(x)
        x = F.relu(x)

        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        return self.projection(x)


class CFGLectinGlycanPredictor(nn.Module):
    def __init__(self, lectin_encoder: ESM2LectinEncoder, glycan_encoder: GlycanGNNEncoder, hidden_dims=None) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]
        self.lectin_encoder = lectin_encoder
        self.glycan_encoder = glycan_encoder

        lectin_dim = lectin_encoder.get_embedding_dim()
        glycan_dim = glycan_encoder.embedding_dim
        self.bilinear = nn.Bilinear(lectin_dim, glycan_dim, 128)

        input_dim = lectin_dim + glycan_dim + 128
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.3)])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        self.mlp = nn.Sequential(*layers)

    def forward(self, lectin_seqs: List[str], glycan_graphs: Data) -> torch.Tensor:
        lectin_emb = self.lectin_encoder(lectin_seqs)
        glycan_emb = self.glycan_encoder(glycan_graphs)
        interaction = self.bilinear(lectin_emb, glycan_emb)
        combined = torch.cat([lectin_emb, glycan_emb, interaction], dim=1)
        return self.mlp(combined).squeeze(-1)


def clean_glycan_string(value: str) -> str:
    cleaned = value.strip()
    cleaned = re.sub(r"-Sp\\d+$", "", cleaned)
    cleaned = cleaned.replace("┬á", "")
    return cleaned


def looks_like_iupac(value: str) -> bool:
    if "Sp" in value or "b1-" in value or "a1-" in value:
        return True
    for token in MONOSAC_TOKENS:
        if token in value:
            return True
    return False


def smiles_to_graph(smiles: str, strict: bool = True) -> Optional[Data]:
    if Chem is None:
        raise SystemExit("RDKit is required. Install with: pip install rdkit")
    if RDLogger is not None:
        RDLogger.DisableLog("rdApp.error")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        if strict:
            raise ValueError(f"Invalid SMILES: {smiles}")
        return None

    node_features = []
    for atom in mol.GetAtoms():
        node_features.append(
            [
                atom.GetAtomicNum(),
                atom.GetFormalCharge(),
                int(atom.GetHybridization()),
                float(atom.GetIsAromatic()),
                float(atom.GetTotalNumHs()),
            ]
        )

    x = torch.tensor(node_features, dtype=torch.float32)
    edge_index = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])

    if edge_index:
        edge_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    else:
        edge_tensor = torch.zeros((2, 0), dtype=torch.long)

    return Data(x=x, edge_index=edge_tensor)


MONOSAC_TOKENS = [
    "GlcNAc",
    "GalNAc",
    "Neu5Ac",
    "Neu5Gc",
    "Glc",
    "Gal",
    "Man",
    "Fuc",
    "Xyl",
    "KDN",
    "GlcA",
    "IdoA",
]
MONOSAC_INDEX = {token: idx + 1 for idx, token in enumerate(MONOSAC_TOKENS)}


def iupac_to_token_graph(iupac: str) -> Data:
    tokens: List[int] = []
    for token in MONOSAC_TOKENS:
        count = iupac.count(token)
        tokens.extend([MONOSAC_INDEX[token]] * count)
    if not tokens:
        tokens = [0]

    x = torch.zeros((len(tokens), 5), dtype=torch.float32)
    x[:, 0] = torch.tensor(tokens, dtype=torch.float32)
    if len(tokens) == 1:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
    else:
        edges = [(i, j) for i in range(len(tokens)) for j in range(len(tokens)) if i != j]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = torch.zeros((edge_index.size(1), 3), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


class CFGDataset(Dataset):
    def __init__(self, csv_path: str, allow_non_smiles: bool = False, binding_column: str = "") -> None:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            raise SystemExit(f"Input not found: {csv_path}")

        df = pd.read_csv(csv_path)

        if "binding" not in df.columns:
            if "rfu_raw" in df.columns:
                threshold = df["rfu_raw"].median()
                df["binding"] = (df["rfu_raw"] >= threshold).astype(int)
            elif binding_column:
                if binding_column not in df.columns:
                    raise ValueError(f"Missing binding column override: {binding_column}")
                df["rfu_raw"] = pd.to_numeric(df[binding_column], errors="coerce")
                threshold = df["rfu_raw"].median()
                df["binding"] = (df["rfu_raw"] >= threshold).astype(int)
            else:
                raise ValueError("Missing binding or rfu_raw column.")

        required = ["lectin_sequence", "glycan_smiles", "binding"]
        missing = set(required) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        df = df.dropna(subset=required)
        self.df = df.reset_index(drop=True)
        self.allow_non_smiles = allow_non_smiles

        print(f"Loaded {len(self.df)} samples from {csv_path}")
        print(f"  Positive samples: {int(self.df['binding'].sum())}")
        print(f"  Negative samples: {len(self.df) - int(self.df['binding'].sum())}")

        if not self.allow_non_smiles:
            for idx, row in self.df.iterrows():
                glycan_smiles = row["glycan_smiles"]
                mol = Chem.MolFromSmiles(glycan_smiles) if Chem is not None else None
                if mol is None:
                    raise ValueError(f"Invalid glycan_smiles at index {idx}: {glycan_smiles}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        lectin_seq = row["lectin_sequence"]
        glycan_smiles = clean_glycan_string(str(row["glycan_smiles"]))
        label = torch.tensor(row["binding"], dtype=torch.float32)
        if self.allow_non_smiles and looks_like_iupac(glycan_smiles):
            try:
                glycan_graph = parse_iupac_condensed(glycan_smiles)
            except Exception:
                glycan_graph = iupac_to_token_graph(glycan_smiles)
        else:
            try:
                graph = smiles_to_graph(glycan_smiles, strict=not self.allow_non_smiles)
            except Exception:
                graph = None
            if graph is None:
                glycan_graph = iupac_to_token_graph(glycan_smiles)
            else:
                glycan_graph = graph
        return lectin_seq, glycan_graph, label


def collate_fn(batch: List[Tuple[str, Data, torch.Tensor]]):
    lectin_seqs = [item[0] for item in batch]
    glycan_graphs = Batch.from_data_list([item[1] for item in batch])
    labels = torch.stack([item[2] for item in batch], dim=0)
    return lectin_seqs, glycan_graphs, labels


def summarize_graphs(dataset: Dataset, sample_size: int = 100) -> None:
    size = min(sample_size, len(dataset))
    if size == 0:
        print("No graphs available for debug.")
        return
    node_counts = []
    edge_counts = []
    for idx in range(size):
        _, graph, _ = dataset[idx]
        node_counts.append(graph.num_nodes)
        edge_counts.append(graph.num_edges)
    print("\n=== GRAPH STATISTICS ===")
    print(f"Avg nodes: {np.mean(node_counts):.1f} (std: {np.std(node_counts):.1f})")
    print(f"Avg edges: {np.mean(edge_counts):.1f} (std: {np.std(edge_counts):.1f})")
    isolated = sum(1 for e in edge_counts if e == 0)
    print(f"Isolated nodes (0 edges): {isolated}/{size}")
    print("========================\n")


def debug_parser_examples() -> None:
    examples = [
        "Gala-Sp8",
        "Glc2Man3GlcNAc4",
        "Gal\u03b21-4GlcNAc",
    ]
    print("\n=== PARSER EXAMPLES ===")
    for entry in examples:
        graph = parse_iupac_condensed(entry)
        edge_attr = getattr(graph, "edge_attr", None)
        edge_attr_shape = tuple(edge_attr.shape) if edge_attr is not None else None
        print(f"{entry}: nodes={graph.num_nodes}, edges={graph.num_edges}, edge_attr={edge_attr_shape}")
    print("========================\n")


def train_epoch(model, dataloader, optimizer, criterion, device, debug_grads: bool = False):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    printed = False
    for lectin_seqs, glycan_graphs, labels in dataloader:
        glycan_graphs = glycan_graphs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        preds = model(lectin_seqs, glycan_graphs)
        loss = criterion(preds, labels)
        loss.backward()
        if debug_grads and not printed:
            for name, param in model.glycan_encoder.named_parameters():
                if param.grad is not None:
                    print(f"{name}: grad_norm={param.grad.norm().item():.4f}")
            printed = True
        optimizer.step()

        total_loss += float(loss.item())
        pred_labels = (preds > 0.5).float()
        correct += int((pred_labels == labels).sum().item())
        total += labels.size(0)
    return total_loss / max(len(dataloader), 1), correct / max(total, 1)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for lectin_seqs, glycan_graphs, labels in dataloader:
            glycan_graphs = glycan_graphs.to(device)
            labels = labels.to(device)
            preds = model(lectin_seqs, glycan_graphs)
            loss = criterion(preds, labels)
            total_loss += float(loss.item())
            pred_labels = (preds > 0.5).float()
            correct += int((pred_labels == labels).sum().item())
            total += labels.size(0)
    return total_loss / max(len(dataloader), 1), correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CFG Phase 2 model and export glycan encoder for Phase 3 transfer."
    )
    parser.add_argument("--data", required=True, help="Path to CFG training CSV.")
    parser.add_argument("--output-dir", default="models", help="Directory to save checkpoints.")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--esm-model", default="esm2_t33_650M_UR50D")
    parser.add_argument("--allow-non-smiles", action="store_true")
    parser.add_argument("--debug-graphs", action="store_true")
    parser.add_argument("--debug-grads", action="store_true")
    parser.add_argument("--binding-column", default="")
    args = parser.parse_args()

    device = resolve_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CFG PHASE 2 TRAINING WITH GLYCAN ENCODER EXPORT")
    print("=" * 80)
    print(f"Device: {device}")
    print(f"Data: {args.data}")
    print(f"Output: {output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"ESM model: {args.esm_model}")
    print("=" * 80)

    dataset = CFGDataset(
        args.data, allow_non_smiles=args.allow_non_smiles, binding_column=args.binding_column
    )
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    lectin_encoder = ESM2LectinEncoder(
        embedding_dim=1280, freeze_esm=True, model_name=args.esm_model
    )
    glycan_encoder = GlycanGNNEncoder(input_dim=5, hidden_dim=64, embedding_dim=512)
    model = CFGLectinGlycanPredictor(lectin_encoder, glycan_encoder)
    model = model.to(device)

    if args.debug_graphs:
        summarize_graphs(dataset)
        print("Glycan encoder:")
        print(model.glycan_encoder)
        debug_parser_examples()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, debug_grads=args.debug_grads
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(
            f"Epoch {epoch}/{args.epochs} - "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            full_model_path = output_dir / "phase2_cfg_full_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                full_model_path,
            )

            glycan_encoder_path = output_dir / "phase2_glycan_encoder_pretrained.pt"
            torch.save(
                {
                    "model_state_dict": glycan_encoder.state_dict(),
                    "embedding_dim": glycan_encoder.embedding_dim,
                    "cfg_val_acc": val_acc,
                    "cfg_epoch": epoch,
                },
                glycan_encoder_path,
            )

            print(f"  ✓ Saved best model (Val Acc: {val_acc:.3f})")
            print(f"    - Full model: {full_model_path}")
            print(f"    - Glycan encoder: {glycan_encoder_path}")

    print("\nTraining complete.")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Glycan encoder ready: {output_dir / 'phase2_glycan_encoder_pretrained.pt'}")


if __name__ == "__main__":
    main()
