"""Train the lectin-glycan interaction model."""

from __future__ import annotations

import argparse
import os
import pickle
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from models.glycan_encoder import GlycanFingerprintConfig, GlycanFingerprintEncoder, GlycanGCNEncoder
from models.interaction_module import InteractionConfig, InteractionPredictor, save_interaction_model
from models.protein_encoder import ESM2Embedder, LectinEncoder
from utils.data_utils import InteractionSample, build_label_from_threshold, load_interaction_samples, split_samples
from utils.metrics import accuracy, binary_classification_stats, mae, matthews_corrcoef, mse, pearson


class InteractionDataset(Dataset):
    def __init__(self, samples: List[InteractionSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> InteractionSample:
        return self.samples[idx]


def build_target(sample: InteractionSample, target_key: str, label_threshold: Optional[float]) -> float:
    if target_key == "label":
        if sample.label is not None:
            return float(sample.label)
        value = sample.rfu_norm if sample.rfu_norm is not None else sample.rfu
        if label_threshold is None:
            raise ValueError("label_threshold required when label is missing.")
        return float(build_label_from_threshold(value, label_threshold))

    if target_key == "rfu_norm":
        return float(sample.rfu_norm) if sample.rfu_norm is not None else float(sample.rfu)
    if target_key == "rfu":
        return float(sample.rfu)
    raise ValueError(f"Unknown target key: {target_key}")


def build_collate_fn(
    lectin_encoder: LectinEncoder,
    glycan_encoder,
    glycan_encoder_type: str,
    device: torch.device,
    target_key: str,
    label_threshold: Optional[float],
):
    def collate(batch: List[InteractionSample]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        lectin_embeddings = [lectin_encoder.encode(sample.lectin_sequence) for sample in batch]
        lectin_tensor = torch.stack(lectin_embeddings).to(device)

        if glycan_encoder_type == "fingerprint":
            glycan_embeddings = [
                glycan_encoder.encode(sample.glycan_smiles, sample.glycan_iupac) for sample in batch
            ]
            glycan_tensor = torch.stack(glycan_embeddings).to(device)
        else:
            smiles_list = [sample.glycan_smiles for sample in batch]
            glycan_tensor = glycan_encoder(smiles_list).to(device)

        targets = [build_target(sample, target_key, label_threshold) for sample in batch]
        target_tensor = torch.tensor(targets, dtype=torch.float32, device=device)
        return lectin_tensor, glycan_tensor, target_tensor

    return collate


def evaluate_regression(model: InteractionPredictor, loader: DataLoader) -> Tuple[float, float, float]:
    model.eval()
    y_true: List[float] = []
    y_pred: List[float] = []
    with torch.no_grad():
        for lectin, glycan, targets in loader:
            preds = model(lectin, glycan)
            y_true.extend(targets.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    return mse(y_true, y_pred), mae(y_true, y_pred), pearson(y_true, y_pred)


def evaluate_classification(model: InteractionPredictor, loader: DataLoader) -> Tuple[float, float, float, float]:
    model.eval()
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.no_grad():
        for lectin, glycan, targets in loader:
            logits = model(lectin, glycan)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).long()
            y_true.extend(targets.long().cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
    acc = accuracy(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    precision, recall, f1 = binary_classification_stats(y_true, y_pred)
    return acc, mcc, f1, precision


def main() -> None:
    parser = argparse.ArgumentParser(description="Train lectin-glycan interaction model.")
    parser.add_argument("--data", required=True, help="Processed CSV from preprocess_cfg_data.py")
    parser.add_argument("--splits", help="Optional pickle with train/val/test indices")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--task", choices=["regression", "classification"], default="regression")
    parser.add_argument("--target", default=None, help="Target column: rfu_norm, rfu, or label")
    parser.add_argument("--label_threshold", type=float, help="Threshold for generating labels")
    parser.add_argument("--glycan_encoder", choices=["fingerprint", "gcn"], default="fingerprint")
    parser.add_argument("--fingerprint_bits", type=int, default=2048)
    parser.add_argument("--fingerprint_radius", type=int, default=2)
    parser.add_argument("--hidden_dim1", type=int, default=512)
    parser.add_argument("--hidden_dim2", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--use_bilinear", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--model_name", default="esm2_t6_8M_UR50D")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    samples = load_interaction_samples(args.data)
    if not samples:
        raise SystemExit("No samples loaded. Check your CSV columns.")

    if args.splits:
        with open(args.splits, "rb") as handle:
            splits = pickle.load(handle)
        train_samples = [samples[idx] for idx in splits.get("train", [])]
        val_samples = [samples[idx] for idx in splits.get("val", [])]
        test_samples = [samples[idx] for idx in splits.get("test", [])]
    else:
        train_samples, val_samples, test_samples = split_samples(samples, seed=args.seed)

    target_key = args.target
    if target_key is None:
        target_key = "rfu_norm" if args.task == "regression" else "label"

    embedder = ESM2Embedder(args.model_name, device=device)
    lectin_encoder = LectinEncoder(embedder)

    if args.glycan_encoder == "fingerprint":
        glycan_config = GlycanFingerprintConfig(
            radius=args.fingerprint_radius,
            n_bits=args.fingerprint_bits,
            include_physchem=True,
            include_iupac=True,
        )
        glycan_encoder = GlycanFingerprintEncoder(glycan_config)
        glycan_dim = glycan_encoder.feature_size
    else:
        glycan_encoder = GlycanGCNEncoder(out_dim=512)
        glycan_encoder.to(device)
        glycan_dim = glycan_encoder.out_dim

    model_config = InteractionConfig(
        lectin_dim=lectin_encoder.embed_dim,
        glycan_dim=glycan_dim,
        hidden_dim1=args.hidden_dim1,
        hidden_dim2=args.hidden_dim2,
        dropout=args.dropout,
        use_bilinear=args.use_bilinear,
    )
    model = InteractionPredictor(model_config).to(device)

    train_loader = DataLoader(
        InteractionDataset(train_samples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=build_collate_fn(
            lectin_encoder,
            glycan_encoder,
            args.glycan_encoder,
            device,
            target_key,
            args.label_threshold,
        ),
    )
    val_loader = DataLoader(
        InteractionDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=build_collate_fn(
            lectin_encoder,
            glycan_encoder,
            args.glycan_encoder,
            device,
            target_key,
            args.label_threshold,
        ),
    )

    params = list(model.parameters())
    if args.glycan_encoder == "gcn":
        params += list(glycan_encoder.parameters())

    optimizer = torch.optim.Adam(params, lr=args.lr)
    if args.task == "classification":
        loss_fn: nn.Module = nn.BCEWithLogitsLoss()
    else:
        loss_fn = nn.MSELoss()

    os.makedirs(args.output_dir, exist_ok=True)
    best_metric = None
    best_path = os.path.join(args.output_dir, "interaction_model.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.glycan_encoder == "gcn":
            glycan_encoder.train()
        total_loss = 0.0
        for lectin, glycan, targets in train_loader:
            optimizer.zero_grad()
            preds = model(lectin, glycan)
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if args.task == "classification":
            acc, mcc, f1, precision = evaluate_classification(model, val_loader)
            metric = mcc
            print(
                f"Epoch {epoch}: loss={total_loss / max(len(train_loader), 1):.4f} "
                f"val_acc={acc:.3f} val_mcc={mcc:.3f} val_f1={f1:.3f} val_precision={precision:.3f}"
            )
        else:
            val_mse, val_mae, val_pearson = evaluate_regression(model, val_loader)
            metric = -val_mse
            print(
                f"Epoch {epoch}: loss={total_loss / max(len(train_loader), 1):.4f} "
                f"val_mse={val_mse:.4f} val_mae={val_mae:.4f} val_pearson={val_pearson:.3f}"
            )

        if best_metric is None or metric > best_metric:
            best_metric = metric
            meta = {
                "task": args.task,
                "target": target_key,
                "glycan_encoder": args.glycan_encoder,
                "fingerprint_bits": args.fingerprint_bits,
                "fingerprint_radius": args.fingerprint_radius,
                "model_name": args.model_name,
            }
            save_interaction_model(best_path, model, model_config, meta=meta)
            print(f"Saved new best model to {best_path}")

    if test_samples:
        test_loader = DataLoader(
            InteractionDataset(test_samples),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=build_collate_fn(
                lectin_encoder,
                glycan_encoder,
                args.glycan_encoder,
                device,
                target_key,
                args.label_threshold,
            ),
        )
        if args.task == "classification":
            acc, mcc, f1, precision = evaluate_classification(model, test_loader)
            print(f"Test acc={acc:.3f} mcc={mcc:.3f} f1={f1:.3f} precision={precision:.3f}")
        else:
            val_mse, val_mae, val_pearson = evaluate_regression(model, test_loader)
            print(f"Test mse={val_mse:.4f} mae={val_mae:.4f} pearson={val_pearson:.3f}")


if __name__ == "__main__":
    main()
