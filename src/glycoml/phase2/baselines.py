"""Baseline models for Phase 2 lectin-glycan prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
import json
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from glycoml.shared.esm2_embedder import ESM2Embedder
from glycoml.shared.glycan_tokenizer import GlycanTokenizer
from .data import build_labels, merge_phase2_data


def build_feature_matrix(df: pd.DataFrame, embedder: ESM2Embedder, tokenizer: GlycanTokenizer) -> Tuple[np.ndarray, np.ndarray]:
    lectin_vecs = []
    glycan_vecs = []
    for _, row in df.iterrows():
        seq = str(row.get("lectin_sequence") or "")
        if not seq:
            continue
        lectin_emb = embedder.embed_pooled(seq).cpu().numpy()
        tokens = tokenizer.encode(str(row.get("glycan_iupac") or ""))
        token_vec = np.zeros(tokenizer.vocab_size(), dtype=np.float32)
        for tok in tokens:
            if tok < token_vec.shape[0]:
                token_vec[tok] += 1.0
        lectin_vecs.append(lectin_emb)
        glycan_vecs.append(token_vec)
    if not lectin_vecs:
        return np.zeros((0, 1)), np.zeros((0,))
    X = np.concatenate([np.stack(lectin_vecs), np.stack(glycan_vecs)], axis=1)
    y = df["label_bin"].values[: X.shape[0]].astype(int)
    return X, y


def run_baselines(data_path: Path, output_dir: Path) -> None:
    df = merge_phase2_data(data_path)
    df = build_labels(df)
    tokenizer = GlycanTokenizer()
    tokenizer.build(df["glycan_iupac"].fillna("").astype(str).tolist())
    embedder = ESM2Embedder(model_name="esm2_t6_8M_UR50D", cache_path=Path("data/cache/esm2_cache.h5"))

    X, y = build_feature_matrix(df, embedder, tokenizer)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "svm": SVC(kernel="rbf", probability=True),
        "rf": RandomForestClassifier(n_estimators=500, random_state=42),
        "mlp": MLPClassifier(hidden_layer_sizes=(512, 128), max_iter=300),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        results[name] = {
            "auroc": float(roc_auc_score(y_test, probs)),
            "auprc": float(average_precision_score(y_test, probs)),
        }

    (output_dir / "baseline_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 2 baseline models")
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--output-dir", default="outputs/phase2_baselines")
    args = parser.parse_args()
    run_baselines(Path(args.data_path), Path(args.output_dir))


if __name__ == "__main__":
    main()
