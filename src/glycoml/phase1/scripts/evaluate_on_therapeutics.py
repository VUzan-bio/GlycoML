"""Evaluate the classifier on held-out therapeutics."""

from __future__ import annotations

import argparse
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from ..models.esm2_classifier import ESM2Embedder, extract_motif_embedding, load_classifier
from ..utils.data_utils import GlycoSample, build_candidate_samples, load_sequence_records, split_records, summarize_samples
from ..utils.metrics import precision_recall_f1


class SampleDataset(Dataset):
    def __init__(self, samples: List[GlycoSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> GlycoSample:
        return self.samples[idx]


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


def evaluate(classifier, loader: DataLoader) -> Tuple[float, float, float]:
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
    parser = argparse.ArgumentParser(description="Evaluate glycosite classifier.")
    parser.add_argument("--data", required=True, help="Path to thera_sabdab_processed.csv")
    parser.add_argument("--model", required=True, help="Path to trained classifier checkpoint")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    device = torch.device(args.device)
    classifier, config = load_classifier(args.model, device=device)
    embedder = ESM2Embedder(config.model_name, device=device)

    records = load_sequence_records(args.data)
    _, _, test_records = split_records(records, seed=args.seed)
    test_samples = build_candidate_samples(test_records)

    stats = summarize_samples(test_samples)
    print(f"Test samples: {stats}")

    loader = DataLoader(
        SampleDataset(test_samples),
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=build_collate_fn(embedder),
    )

    precision, recall, f1 = evaluate(classifier, loader)
    print(f"Precision={precision:.3f} Recall={recall:.3f} F1={f1:.3f}")


if __name__ == "__main__":
    main()

