"""Evaluate Phase 2 binding model and generate predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch

try:  # pragma: no cover
    import seaborn as sns
except Exception:  # pragma: no cover
    sns = None

import matplotlib.pyplot as plt

from glycoml.shared.esm2_embedder import ESM2Embedder
from glycoml.shared.glycan_tokenizer import GlycanTokenizer
from .data import Phase2DatasetConfig, build_labels, collate_batch, merge_phase2_data, LectinGlycanDataset
from .models.binding_model import BindingModel, BindingModelConfig


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    metrics: Dict[str, float] = {}
    if len(np.unique(y_true)) > 1:
        metrics["auroc"] = float(roc_auc_score(y_true, y_score))
        metrics["auprc"] = float(average_precision_score(y_true, y_score))
    return metrics


def plot_heatmap(pred_df: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    if pred_df.empty:
        return
    top_lectins = pred_df["lectin_id"].value_counts().nlargest(top_n).index
    top_glycans = pred_df["glycan_id"].value_counts().nlargest(top_n).index
    subset = pred_df[pred_df["lectin_id"].isin(top_lectins) & pred_df["glycan_id"].isin(top_glycans)]
    pivot = subset.pivot_table(index="lectin_id", columns="glycan_id", values="pred_bin", aggfunc="mean")
    plt.figure(figsize=(12, 8))
    if sns:
        sns.heatmap(pivot.fillna(0), cmap="viridis")
    else:
        plt.imshow(pivot.fillna(0).values, aspect="auto", cmap="viridis")
        plt.colorbar(label="pred_bin")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Phase 2 binding model")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--unilectin-path")
    parser.add_argument("--cfg-metadata-path")
    parser.add_argument("--ligands-path")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="outputs/phase2_model")
    parser.add_argument("--glycan-encoder", choices=["graph", "token"], default="graph")
    parser.add_argument("--use-structure", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = merge_phase2_data(
        Path(args.data_path),
        Path(args.unilectin_path) if args.unilectin_path else None,
        Path(args.cfg_metadata_path) if args.cfg_metadata_path else None,
        Path(args.ligands_path) if args.ligands_path else None,
    )
    df = build_labels(df)

    tokenizer = GlycanTokenizer()
    tokenizer.build(df["glycan_iupac"].fillna("").astype(str).tolist())
    embedder = ESM2Embedder(model_name="esm2_t33_650M_UR50D", cache_path=Path("data/cache/esm2_cache.h5"))

    dataset_cfg = Phase2DatasetConfig(use_structure=args.use_structure, glycan_encoder=args.glycan_encoder)
    dataset = LectinGlycanDataset(df, embedder, tokenizer, Path("data/cache"), dataset_cfg)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, args.glycan_encoder == "graph", args.use_structure),
    )

    config = BindingModelConfig()
    config.use_graph = args.glycan_encoder == "graph"
    config.lectin_config.family_dim = getattr(dataset, "family_dim", config.lectin_config.family_dim)
    config.lectin_config.esm_dim = embedder.embed_dim
    config.lectin_config.use_structure = args.use_structure
    config.glycan_token_config.vocab_size = tokenizer.vocab_size()
    model = BindingModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    preds = []
    y_true = []
    y_score = []
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
            labels = batch["labels_bin"].cpu().numpy()
            y_true.extend(labels.tolist())
            y_score.extend(probs.tolist())
            for lectin_id, lectin_name, glycan_id, glycan_iupac, glytoucan_id, prob, reg in zip(
                batch["lectin_ids"],
                batch["lectin_names"],
                batch["glycan_ids"],
                batch["glycan_iupac"],
                batch["glytoucan_ids"],
                probs,
                reg_out.cpu().numpy(),
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
                    }
                )

    pred_df = pd.DataFrame(preds)
    pred_df.to_csv(output_dir / "predictions.csv", index=False)

    metrics = compute_metrics(np.array(y_true), np.array(y_score))
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_heatmap(pred_df, output_dir / "binding_heatmap.png")

if __name__ == "__main__":
    main()
