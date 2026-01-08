"""Train the Phase 2 lectin-glycan binding predictor."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

try:  # pragma: no cover
    import wandb
except Exception:  # pragma: no cover
    wandb = None

from glycoml.shared.esm2_embedder import ESM2Embedder
from glycoml.shared.glycan_tokenizer import GlycanTokenizer
from .data import (
    Phase2DatasetConfig,
    build_labels,
    collate_batch,
    merge_phase2_data,
    stratified_group_split,
    LectinGlycanDataset,
)
from .models.binding_model import BindingModel, BindingModelConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def precision_at_k(y_true: np.ndarray, y_score: np.ndarray, k: int) -> float:
    if len(y_score) == 0:
        return 0.0
    k = min(k, len(y_score))
    idx = np.argsort(y_score)[-k:]
    return float(np.mean(y_true[idx]))


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    def rankdata(a: np.ndarray) -> np.ndarray:
        temp = a.argsort()
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(a))
        return ranks.astype(np.float32)
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: float | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        loss = bce * ((1 - p_t) ** self.gamma)
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        return loss.mean()


def nt_xent(features: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    features = torch.nn.functional.normalize(features, dim=-1)
    noise = torch.randn_like(features) * 0.01
    features_aug = torch.nn.functional.normalize(features + noise, dim=-1)
    batch = features.shape[0]
    logits = features @ features_aug.T / temperature
    labels = torch.arange(batch, device=features.device)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    return loss


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
        metrics["auprc"] = float(average_precision_score(y_true, y_score))
        metrics["precision_at_50"] = precision_at_k(y_true, y_score, 50)
    return metrics


def train_epoch(
    model: BindingModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    loss_bin_fn,
    loss_reg_fn,
    alpha_bin: float,
    alpha_reg: float,
    beta_contrastive: float,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        structure_batch = batch["structure_batch"]
        if structure_batch is not None:
            structure_batch = structure_batch.to(device)
        glycan_graph = batch["glycan_graph"]
        if glycan_graph is not None:
            glycan_graph = glycan_graph.to(device)
        bin_logits, reg_out, features = model(
            batch["lectin_tokens"].to(device),
            batch["lectin_mask"].to(device),
            family_features=batch["family_features"].to(device),
            species_idx=batch["species_idx"].to(device),
            structure_batch=structure_batch,
            glycan_tokens=batch["glycan_tokens"].to(device) if batch["glycan_tokens"] is not None else None,
            glycan_mask=batch["glycan_mask"].to(device) if batch["glycan_mask"] is not None else None,
            glycan_graph=glycan_graph,
            glycan_meta=batch["glycan_meta"].to(device) if batch["glycan_meta"] is not None else None,
            return_features=True,
        )
        labels_bin = batch["labels_bin"].to(device)
        labels_reg = batch["labels_reg"].to(device)

        loss_bin = loss_bin_fn(bin_logits, labels_bin)
        mask_reg = labels_reg > 0
        if mask_reg.any():
            loss_reg = loss_reg_fn(reg_out[mask_reg], labels_reg[mask_reg])
        else:
            loss_reg = torch.tensor(0.0, device=device)

        loss = alpha_bin * loss_bin + alpha_reg * loss_reg
        if beta_contrastive > 0 and features is not None:
            loss = loss + beta_contrastive * nt_xent(features)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / max(len(loader), 1)


def evaluate(model: BindingModel, loader: DataLoader, device: torch.device) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    model.eval()
    preds: List[Dict[str, object]] = []
    y_true: List[float] = []
    y_score: List[float] = []
    y_reg_true: List[float] = []
    y_reg_pred: List[float] = []
    with torch.no_grad():
        for batch in loader:
            structure_batch = batch["structure_batch"]
            if structure_batch is not None:
                structure_batch = structure_batch.to(device)
            glycan_graph = batch["glycan_graph"]
            if glycan_graph is not None:
                glycan_graph = glycan_graph.to(device)
            bin_logits, reg_out, _ = model(
                batch["lectin_tokens"].to(device),
                batch["lectin_mask"].to(device),
                family_features=batch["family_features"].to(device),
                species_idx=batch["species_idx"].to(device),
                structure_batch=structure_batch,
                glycan_tokens=batch["glycan_tokens"].to(device) if batch["glycan_tokens"] is not None else None,
                glycan_mask=batch["glycan_mask"].to(device) if batch["glycan_mask"] is not None else None,
                glycan_graph=glycan_graph,
                glycan_meta=batch["glycan_meta"].to(device) if batch["glycan_meta"] is not None else None,
            )
            probs = torch.sigmoid(bin_logits).cpu().numpy()
            reg_vals = reg_out.cpu().numpy()
            labels_bin = batch["labels_bin"].cpu().numpy()
            labels_reg = batch["labels_reg"].cpu().numpy()

            y_true.extend(labels_bin.tolist())
            y_score.extend(probs.tolist())
            y_reg_true.extend(labels_reg.tolist())
            y_reg_pred.extend(reg_vals.tolist())
            for lectin_id, lectin_name, glycan_id, glycan_iupac, glytoucan_id, prob, reg, label in zip(
                batch["lectin_ids"],
                batch["lectin_names"],
                batch["glycan_ids"],
                batch["glycan_iupac"],
                batch["glytoucan_ids"],
                probs,
                reg_vals,
                labels_bin,
            ):
                preds.append(
                    {
                        "lectin_id": lectin_id,
                        "lectin_name": lectin_name,
                        "glycan_id": glycan_id,
                        "glycan_iupac": glycan_iupac,
                        "glytoucan_id": glytoucan_id,
                        "pred_bin": float(prob),
                        "pred_reg": float(reg),
                        "actual_bin": float(label),
                    }
                )

    metrics = compute_metrics(np.array(y_true), np.array(y_score))
    if len(y_reg_true) > 1:
        y_reg_true_arr = np.array(y_reg_true)
        y_reg_pred_arr = np.array(y_reg_pred)
        metrics["pearson"] = float(np.corrcoef(y_reg_true_arr, y_reg_pred_arr)[0, 1])
        metrics["spearman"] = spearman_corr(y_reg_true_arr, y_reg_pred_arr)
        metrics["rmse"] = float(np.sqrt(np.mean((y_reg_true_arr - y_reg_pred_arr) ** 2)))
    return metrics, preds


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 2 binding model")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--unilectin-path")
    parser.add_argument("--cfg-metadata-path")
    parser.add_argument("--ligands-path")
    parser.add_argument("--output-dir", default="outputs/phase2_model")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--label-mode", choices=["binary", "regression", "multi"], default="multi")
    parser.add_argument("--rfu-threshold", type=float, default=2000.0)
    parser.add_argument("--glycan-encoder", choices=["graph", "token"], default="graph")
    parser.add_argument("--use-structure", action="store_true")
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--nan-handling", choices=["zero", "drop"], default="zero")
    parser.add_argument("--require-labels", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--project", default="glycoml-phase2")
    parser.add_argument("--ablation", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.ablation == "no_struct":
        args.use_structure = False
    if args.ablation == "token_glycan":
        args.glycan_encoder = "token"
    use_cross_attention = args.ablation != "no_crossattn"
    if args.ablation == "single_task":
        args.label_mode = "binary"

    df = merge_phase2_data(
        Path(args.data_path),
        Path(args.unilectin_path) if args.unilectin_path else None,
        Path(args.cfg_metadata_path) if args.cfg_metadata_path else None,
        Path(args.ligands_path) if args.ligands_path else None,
    )
    if args.require_labels:
        df = df[df["binding_value"].notna() | df.get("rfu_normalized").notna()].copy()
    df = build_labels(df, rfu_threshold=args.rfu_threshold, nan_handling=args.nan_handling)
    df = df.reset_index(drop=True)
    if args.max_samples:
        df = df.sample(n=min(args.max_samples, len(df)), random_state=args.seed)
        df = df.reset_index(drop=True)

    tokenizer = GlycanTokenizer()
    tokenizer.build(df["glycan_iupac"].fillna("").astype(str).tolist())
    tokenizer.save(output_dir / "glycan_vocab.json")

    cache_path = Path("data/cache/esm2_cache.h5")
    embedder = ESM2Embedder(model_name="esm2_t33_650M_UR50D", cache_path=cache_path)

    dataset_cfg = Phase2DatasetConfig(use_structure=args.use_structure, glycan_encoder=args.glycan_encoder)
    dataset = LectinGlycanDataset(
        df,
        embedder,
        tokenizer,
        Path("data/cache"),
        dataset_cfg,
        allow_network=args.allow_network,
    )

    train_idx, val_idx, test_idx = stratified_group_split(df, "label_bin", seed=args.seed)
    train_ds = torch.utils.data.Subset(dataset, train_idx)
    val_ds = torch.utils.data.Subset(dataset, val_idx)
    test_ds = torch.utils.data.Subset(dataset, test_idx)

    collate = lambda batch: collate_batch(batch, use_graph=args.glycan_encoder == "graph", use_structure=args.use_structure)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)

    config = BindingModelConfig()
    config.use_graph = args.glycan_encoder == "graph"
    config.use_cross_attention = use_cross_attention
    config.lectin_config.family_dim = getattr(dataset, "family_dim", config.lectin_config.family_dim)
    config.lectin_config.esm_dim = embedder.embed_dim
    config.lectin_config.use_structure = args.use_structure
    config.glycan_token_config.vocab_size = tokenizer.vocab_size()
    config.glycan_token_config.meta_dim = 11
    config.glycan_graph_config.meta_dim = 11

    model = BindingModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pos_weight = torch.tensor([3.0], device=device)
    if args.label_mode == "binary":
        loss_reg = torch.nn.MSELoss()
    else:
        loss_reg = torch.nn.SmoothL1Loss()

    if args.label_mode == "binary":
        loss_bin_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_bin_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    alpha_bin = 0.6
    alpha_reg = 0.4 if args.label_mode != "binary" else 0.0
    beta_contrastive = 0.2 if args.label_mode == "multi" else 0.0

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.wandb and wandb is not None:
        wandb.init(project=args.project, config=vars(args))

    best_metric = -1.0
    history: Dict[str, Dict[str, float]] = {}
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            loss_bin_fn,
            loss_reg,
            alpha_bin,
            alpha_reg,
            beta_contrastive,
        )
        val_metrics, _ = evaluate(model, val_loader, device)
        val_score = val_metrics.get("auprc", 0.0)
        history[str(epoch)] = {"train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}}

        if args.wandb and wandb is not None:
            wandb.log({"epoch": epoch, "train_loss": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})

        if val_score > best_metric:
            best_metric = val_score
            torch.save(model.state_dict(), output_dir / "model.pt")

        metrics, preds = evaluate(model, test_loader, device)
        (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    config_path = output_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump({"args": vars(args), "model": asdict(config)}))

    preds_path = output_dir / "predictions.csv"
    import pandas as pd

    pred_df = pd.DataFrame(preds)
    pred_df.to_csv(preds_path, index=False)

    try:
        import seaborn as sns
        import matplotlib.pyplot as plt

        top_lectins = pred_df["lectin_id"].value_counts().nlargest(20).index
        top_glycans = pred_df["glycan_id"].value_counts().nlargest(20).index
        subset = pred_df[pred_df["lectin_id"].isin(top_lectins) & pred_df["glycan_id"].isin(top_glycans)]
        pivot = subset.pivot_table(index="lectin_id", columns="glycan_id", values="pred_bin", aggfunc="mean")
        plt.figure(figsize=(12, 8))
        sns.heatmap(pivot.fillna(0), cmap="viridis")
        plt.tight_layout()
        plt.savefig(output_dir / "binding_heatmap.png")
        plt.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()
