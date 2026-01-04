"""Train the ESM2-based glycosite classifier."""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.esm2_classifier import ESM2Embedder, GlycoMotifClassifier, ModelConfig, extract_motif_embedding, save_classifier
from utils.data_utils import GlycoSample, build_candidate_samples, load_sequence_records, split_records, summarize_samples
from utils.metrics import precision_recall_f1


class SampleDataset(Dataset):
    def __init__(self, samples: List[GlycoSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> GlycoSample:
        return self.samples[idx]


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = bce * ((1 - p_t) ** self.gamma)
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean()


def build_collate_fn(embedder: ESM2Embedder) -> callable:
    def collate(batch: List[GlycoSample]) -> Tuple[torch.Tensor, torch.Tensor]:
        embeddings = []
        labels = []
        for sample in batch:
            residue_embeddings = embedder.embed_sequence(sample.sequence)
            motif_embedding = extract_motif_embedding(residue_embeddings, sample.position)
            embeddings.append(motif_embedding)
            labels.append(sample.label)
        x = torch.stack(embeddings)
        y = torch.tensor(labels, dtype=torch.float32, device=x.device)
        return x, y

    return collate


def evaluate(classifier: GlycoMotifClassifier, loader: DataLoader) -> Tuple[float, float, float]:
    classifier.eval()
    all_labels: List[int] = []
    all_preds: List[int] = []
    with torch.no_grad():
        for x, y in loader:
            logits = classifier(x)
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().tolist()
            labels = y.long().cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels)
    return precision_recall_f1(all_labels, all_preds)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the glycosite classifier.")
    parser.add_argument("--data", required=True, help="Path to thera_sabdab_processed.csv")
    parser.add_argument("--output_dir", default="outputs", help="Directory to save checkpoints")
    parser.add_argument("--model_name", default="esm2_t6_8M_UR50D", help="ESM2 model name")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--use_focal", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    records = load_sequence_records(args.data)
    train_records, val_records, _ = split_records(records, seed=args.seed)
    train_samples = build_candidate_samples(train_records)
    val_samples = build_candidate_samples(val_records)

    train_stats = summarize_samples(train_samples)
    val_stats = summarize_samples(val_samples)
    print(f"Train samples: {train_stats}")
    print(f"Val samples: {val_stats}")

    embedder = ESM2Embedder(args.model_name, device=device)
    classifier = GlycoMotifClassifier(
        embed_dim=embedder.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    train_loader = DataLoader(
        SampleDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=build_collate_fn(embedder),
    )
    val_loader = DataLoader(
        SampleDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=build_collate_fn(embedder),
    )

    if args.use_focal:
        loss_fn: nn.Module = FocalLoss(gamma=args.focal_gamma, alpha=args.focal_alpha)
    else:
        pos_weight = None
        if train_stats["positives"]:
            pos_weight = torch.tensor(
                train_stats["negatives"] / max(train_stats["positives"], 1),
                device=device,
            )
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(classifier.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        classifier.train()
        total_loss = 0.0
        for x, y in train_loader:
            optimizer.zero_grad()
            logits = classifier(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        precision, recall, f1 = evaluate(classifier, val_loader)
        avg_loss = total_loss / max(len(train_loader), 1)
        print(f"Epoch {epoch}: loss={avg_loss:.4f} val_precision={precision:.3f} val_recall={recall:.3f} val_f1={f1:.3f}")

        if f1 > best_f1:
            best_f1 = f1
            config = ModelConfig(
                model_name=args.model_name,
                hidden_dim=args.hidden_dim,
                num_layers=args.num_layers,
                dropout=args.dropout,
            )
            checkpoint_path = os.path.join(args.output_dir, "glyco_classifier.pt")
            save_classifier(checkpoint_path, classifier, config)
            print(f"Saved new best model to {checkpoint_path}")


if __name__ == "__main__":
    main()

