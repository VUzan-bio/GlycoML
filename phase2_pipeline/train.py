"""Train a lectin-glycan binding model on the GlycoML Phase 2 dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from data import (
    GlycanTokenizer,
    GlycoDataset,
    ESMEmbedder,
    build_samples,
    collate_batch,
    compute_label,
    filter_phase2,
    load_phase2_csv,
    merge_external_data,
    load_sequences,
    normalize_binding_value,
    stratified_group_split,
)
from model import CrossAttentionModel, ModelConfig


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


def evaluate(
    model: CrossAttentionModel,
    loader: DataLoader,
    device: torch.device,
    task: str,
) -> Tuple[float, Dict[str, float], List[Dict[str, object]]]:
    model.eval()
    losses = []
    preds: List[Dict[str, object]] = []
    y_true_all: List[float] = []
    y_score_all: List[float] = []
    if task == "classification":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.MSELoss()

    with torch.no_grad():
        for batch in loader:
            lectin_tokens = batch["lectin_tokens"].to(device)
            lectin_mask = batch["lectin_mask"].to(device)
            glycan_tokens = batch["glycan_tokens"].to(device)
            glycan_mask = batch["glycan_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(lectin_tokens, lectin_mask, glycan_tokens, glycan_mask)
            loss = loss_fn(logits, labels)
            losses.append(loss.item())

            if task == "classification":
                scores = torch.sigmoid(logits).cpu().numpy()
            else:
                scores = logits.cpu().numpy()
            labels_np = labels.cpu().numpy()
            y_true_all.extend(labels_np.tolist())
            y_score_all.extend(scores.tolist())

            for lectin_id, glycan_id, score, label in zip(
                batch["lectin_ids"], batch["glycan_ids"], scores, labels_np
            ):
                preds.append(
                    {
                        "lectin_id": lectin_id,
                        "glycan_id": glycan_id,
                        "score": float(score),
                        "label": float(label),
                    }
                )

    metrics: Dict[str, float] = {"loss": float(np.mean(losses)) if losses else 0.0}
    y_true_arr = np.array(y_true_all)
    y_score_arr = np.array(y_score_all)
    if task == "classification" and len(np.unique(y_true_arr)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true_arr, y_score_arr))
        metrics["auprc"] = float(average_precision_score(y_true_arr, y_score_arr))
        metrics["precision_at_50"] = precision_at_k(y_true_arr, y_score_arr, 50)
    elif task == "regression":
        metrics["rmse"] = float(np.sqrt(np.mean((y_true_arr - y_score_arr) ** 2)))
    return metrics["loss"], metrics, preds


def plot_heatmap(pred_df: pd.DataFrame, path: Path, top_n: int = 20) -> None:
    if pred_df.empty:
        return
    counts = pred_df["lectin_id"].value_counts().nlargest(top_n)
    glycans = pred_df["glycan_id"].value_counts().nlargest(top_n)
    subset = pred_df[
        pred_df["lectin_id"].isin(counts.index) & pred_df["glycan_id"].isin(glycans.index)
    ]
    pivot = subset.pivot_table(
        index="lectin_id",
        columns="glycan_id",
        values="score",
        aggfunc="mean",
    )
    plt.figure(figsize=(12, 8))
    plt.imshow(pivot.fillna(0).values, aspect="auto", cmap="viridis")
    plt.colorbar(label="Predicted score")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=90, fontsize=6)
    plt.yticks(range(len(pivot.index)), pivot.index, fontsize=6)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train GlycoML Phase 2 model")
    parser.add_argument("--data-path", default="data/glycoml_phase2_unified_lectin_glycan_interactions.csv")
    parser.add_argument("--unilectin-path", default="data/unilectin3d_lectin_glycan_interactions.csv")
    parser.add_argument("--cfg-metadata-path", default="data/cfg_experiment_metadata.csv")
    parser.add_argument("--output-dir", default="outputs/phase2_model")
    parser.add_argument("--label-mode", choices=["classification", "regression"], default="classification")
    parser.add_argument("--rfu-threshold", type=float, default=2000.0)
    parser.add_argument("--min-seq-len", type=int, default=50)
    parser.add_argument("--esm-model", default="esm2_t6_8M_UR50D")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_phase2_csv(Path(args.data_path))
    df = merge_external_data(
        df,
        Path(args.unilectin_path) if args.unilectin_path else None,
        Path(args.cfg_metadata_path) if args.cfg_metadata_path else None,
    )
    df = filter_phase2(df, min_seq_len=args.min_seq_len)
    df["binding_value_norm"] = df.apply(normalize_binding_value, axis=1)
    df["label"] = df.apply(lambda row: compute_label(row, args.label_mode, args.rfu_threshold), axis=1)

    sequences = load_sequences(df, output_dir / "sequence_cache", allow_network=args.allow_network, min_len=args.min_seq_len)
    samples = build_samples(df, sequences, args.label_mode, args.rfu_threshold)

    if not samples:
        raise SystemExit("No valid samples after filtering.")

    sample_df = pd.DataFrame(
        [
            {
                "lectin_id": s.lectin_id,
                "glycan_id": s.glycan_id,
                "label": s.label,
            }
            for s in samples
        ]
    )
    train_idx, val_idx, test_idx = stratified_group_split(sample_df, "label", seed=args.seed)

    tokenizer = GlycanTokenizer()
    tokenizer.build([s.glycan_iupac for s in samples])
    tokenizer.save(output_dir / "glycan_vocab.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    esm_embedder = ESMEmbedder(args.esm_model, output_dir / "esm_cache", device)

    train_set = GlycoDataset([samples[i] for i in train_idx], esm_embedder, tokenizer)
    val_set = GlycoDataset([samples[i] for i in val_idx], esm_embedder, tokenizer)
    test_set = GlycoDataset([samples[i] for i in test_idx], esm_embedder, tokenizer)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    cfg = ModelConfig(
        lectin_dim=esm_embedder.model.embed_dim,
        glycan_vocab=tokenizer.vocab_size(),
        task=args.label_mode,
    )
    model = CrossAttentionModel(cfg).to(device)

    if args.label_mode == "classification":
        loss_fn = torch.nn.BCEWithLogitsLoss()
    else:
        loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_loss = float("inf")
    patience = 0
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for batch in train_loader:
            lectin_tokens = batch["lectin_tokens"].to(device)
            lectin_mask = batch["lectin_mask"].to(device)
            glycan_tokens = batch["glycan_tokens"].to(device)
            glycan_mask = batch["glycan_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(lectin_tokens, lectin_mask, glycan_tokens, glycan_mask)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        val_loss, val_metrics, _ = evaluate(model, val_loader, device, args.label_mode)
        history.append({"epoch": epoch, "train_loss": float(np.mean(epoch_losses)), **val_metrics})
        print(f"Epoch {epoch}: train_loss={history[-1]['train_loss']:.4f} val_loss={val_loss:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            torch.save({"model_state": model.state_dict(), "config": cfg.__dict__}, output_dir / "model.pt")
        else:
            patience += 1
            if patience >= args.patience:
                break

    test_loss, test_metrics, preds = evaluate(model, test_loader, device, args.label_mode)
    metrics = {
        "best_val_loss": best_loss,
        "test_loss": test_loss,
        **test_metrics,
        "samples": len(samples),
        "train_size": len(train_set),
        "val_size": len(val_set),
        "test_size": len(test_set),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pred_df = pd.DataFrame(preds)
    pred_df.to_csv(output_dir / "predictions.csv", index=False)
    plot_heatmap(pred_df, output_dir / "binding_heatmap.png")

    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print("Training complete. Outputs saved to", output_dir)


if __name__ == "__main__":
    main()
