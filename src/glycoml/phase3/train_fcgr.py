"""Train a Phase 3 Fcgr regression model from the merged Fcgr dataset."""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from glycoml.shared.esm2_embedder import ESM2Embedder


LOGGER = logging.getLogger(__name__)

MONO_ORDER = [
    "Neu5Ac",
    "Neu5Gc",
    "GalNAc",
    "GlcNAc",
    "Gal",
    "Glc",
    "Man",
    "Fuc",
    "Xyl",
    "GlcA",
    "IdoA",
    "KDN",
    "Rha",
    "Ara",
]

MONO_PATTERN = None


def _build_mono_pattern() -> None:
    global MONO_PATTERN
    if MONO_PATTERN is not None:
        return
    tokens = sorted(MONO_ORDER, key=len, reverse=True)
    alternation = "|".join(tokens)
    MONO_PATTERN = re.compile(rf"({alternation})(\d*)")


def glycan_composition_vector(value: str) -> np.ndarray:
    """Return monosaccharide count vector for a glycan composition string."""
    if not value or not isinstance(value, str):
        return np.zeros(len(MONO_ORDER), dtype=np.float32)
    if MONO_PATTERN is None:
        _build_mono_pattern()
    counts = np.zeros(len(MONO_ORDER), dtype=np.float32)
    for mono, count_str in MONO_PATTERN.findall(value):
        count = int(count_str) if count_str else 1
        if mono in MONO_ORDER:
            counts[MONO_ORDER.index(mono)] += count
    return counts


@dataclass
class FcgrExample:
    fc_sequence: str
    fcgr_sequence: str
    glycan_vec: np.ndarray
    target: float


class FcgrDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        target_col: str = "log_kd",
        use_fcgr: bool = True,
        use_glycan: bool = True,
    ) -> None:
        self.use_fcgr = use_fcgr
        self.use_glycan = use_glycan
        self.samples: List[FcgrExample] = []

        for _, row in df.iterrows():
            fc_seq = str(row.get("fc_sequence", "") or "")
            if not fc_seq:
                continue
            fcgr_seq = str(row.get("fcgr_sequence", "") or "")
            glycan_struct = row.get("glycan_structure", "") or ""
            glycan_name = row.get("glycan_name", "") or ""
            glycan_source = glycan_struct if glycan_struct else glycan_name
            glycan_vec = glycan_composition_vector(str(glycan_source))

            target = row.get(target_col)
            if pd.isna(target):
                continue

            self.samples.append(
                FcgrExample(
                    fc_sequence=fc_seq,
                    fcgr_sequence=fcgr_seq,
                    glycan_vec=glycan_vec,
                    target=float(target),
                )
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> FcgrExample:
        return self.samples[idx]


def collate_batch(batch: List[FcgrExample]) -> Tuple[List[str], List[str], torch.Tensor, torch.Tensor]:
    fc_sequences = [item.fc_sequence for item in batch]
    fcgr_sequences = [item.fcgr_sequence for item in batch]
    glycan_vecs = torch.tensor([item.glycan_vec for item in batch], dtype=torch.float32)
    targets = torch.tensor([item.target for item in batch], dtype=torch.float32).view(-1, 1)
    return fc_sequences, fcgr_sequences, glycan_vecs, targets


class FcgrRegressor(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        glycan_dim: int,
        hidden_dim: int = 256,
        use_fcgr: bool = True,
        use_glycan: bool = True,
    ) -> None:
        super().__init__()
        self.use_fcgr = use_fcgr
        self.use_glycan = use_glycan

        self.fc_proj = nn.Linear(embed_dim, hidden_dim)
        self.fcgr_proj = nn.Linear(embed_dim, hidden_dim) if use_fcgr else None
        self.glycan_proj = nn.Linear(glycan_dim, hidden_dim) if use_glycan else None

        input_dim = hidden_dim
        if use_fcgr:
            input_dim += hidden_dim
        if use_glycan:
            input_dim += hidden_dim

        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, fc_emb: torch.Tensor, fcgr_emb: torch.Tensor, glycan_vec: torch.Tensor) -> torch.Tensor:
        features = [self.fc_proj(fc_emb)]
        if self.use_fcgr and self.fcgr_proj is not None:
            features.append(self.fcgr_proj(fcgr_emb))
        if self.use_glycan and self.glycan_proj is not None:
            features.append(self.glycan_proj(glycan_vec))
        combined = torch.cat(features, dim=1)
        return self.head(combined)


def embed_sequences(embedder: ESM2Embedder, sequences: List[str], device: torch.device) -> torch.Tensor:
    embeddings = []
    for seq in sequences:
        emb = embedder.embed_pooled(seq)
        if emb.device != device:
            emb = emb.to(device)
        embeddings.append(emb)
    return torch.stack(embeddings, dim=0)


def train_epoch(
    model: FcgrRegressor,
    embedder: ESM2Embedder,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    losses = []
    for fc_seqs, fcgr_seqs, glycan_vecs, targets in loader:
        fc_emb = embed_sequences(embedder, fc_seqs, device)
        fcgr_emb = embed_sequences(embedder, fcgr_seqs, device)
        glycan_vecs = glycan_vecs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        preds = model(fc_emb, fcgr_emb, glycan_vecs)
        loss = criterion(preds, targets)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


def evaluate(
    model: FcgrRegressor,
    embedder: ESM2Embedder,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    losses = []
    preds_all = []
    targets_all = []
    with torch.no_grad():
        for fc_seqs, fcgr_seqs, glycan_vecs, targets in loader:
            fc_emb = embed_sequences(embedder, fc_seqs, device)
            fcgr_emb = embed_sequences(embedder, fcgr_seqs, device)
            glycan_vecs = glycan_vecs.to(device)
            targets = targets.to(device)
            preds = model(fc_emb, fcgr_emb, glycan_vecs)
            loss = criterion(preds, targets)
            losses.append(loss.item())
            preds_all.append(preds.cpu().numpy())
            targets_all.append(targets.cpu().numpy())
    if preds_all:
        preds_concat = np.vstack(preds_all)
        targets_concat = np.vstack(targets_all)
        mse = float(np.mean((preds_concat - targets_concat) ** 2))
    else:
        mse = 0.0
    return float(np.mean(losses)) if losses else 0.0, mse


def _load_dataframe(path: Path, target: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    df = pd.read_csv(path)
    required = {"fc_sequence", "fcgr_sequence", "glycan_name", "glycan_structure", "binding_kd_nm"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if target == "log_kd" and "log_kd" not in df.columns:
        df = df.copy()
        df["log_kd"] = np.log10(df["binding_kd_nm"].astype(float))
    return df


def _split_dataset(dataset: Dataset, seed: int) -> Tuple[Dataset, Dataset, Dataset]:
    total = len(dataset)
    if total < 3:
        raise ValueError("Dataset too small to split.")
    train_size = int(total * 0.7)
    val_size = int(total * 0.15)
    test_size = total - train_size - val_size
    return random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed),
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Train Phase 3 Fcgr regression model.")
    parser.add_argument("--data-path", required=True, help="Path to phase3_fcgr_merged.csv")
    parser.add_argument("--output-dir", default="outputs/phase3_fcgr", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--esm-model", default="esm2_t6_8M_UR50D")
    parser.add_argument("--target", choices=["log_kd", "binding_kd_nm"], default="log_kd")
    parser.add_argument("--use-fcgr", action="store_true", help="Include Fcgr sequence encoder")
    parser.add_argument("--use-glycan", action="store_true", help="Include glycan composition features")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = _load_dataframe(Path(args.data_path), args.target)
    dataset = FcgrDataset(
        df,
        target_col=args.target,
        use_fcgr=args.use_fcgr,
        use_glycan=args.use_glycan,
    )

    LOGGER.info("Loaded %d samples for Phase 3 training.", len(dataset))

    train_set, val_set, test_set = _split_dataset(dataset, args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    embedder = ESM2Embedder(model_name=args.esm_model, device=device, cache_path=Path("data/cache/esm2_cache.h5"))
    model = FcgrRegressor(
        embed_dim=embedder.embed_dim,
        glycan_dim=len(MONO_ORDER),
        use_fcgr=args.use_fcgr,
        use_glycan=args.use_glycan,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_mse = float("inf")
    history: List[dict] = []

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, embedder, train_loader, optimizer, criterion, device)
        val_loss, val_mse = evaluate(model, embedder, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_mse": val_mse})
        LOGGER.info("Epoch %d/%d - train_loss=%.4f val_loss=%.4f val_mse=%.4f", epoch, args.epochs, train_loss, val_loss, val_mse)

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "val_mse": val_mse,
                    "epoch": epoch,
                    "target": args.target,
                    "use_fcgr": args.use_fcgr,
                    "use_glycan": args.use_glycan,
                },
                output_dir / "model.pt",
            )

    test_loss, test_mse = evaluate(model, embedder, test_loader, criterion, device)
    LOGGER.info("Test loss=%.4f test_mse=%.4f", test_loss, test_mse)

    metrics = {"best_val_mse": best_val_mse, "test_mse": test_mse, "epochs": args.epochs}
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    print(f"Saved model to {output_dir / 'model.pt'}")
    print(f"Metrics: {metrics}")


if __name__ == "__main__":
    main()
