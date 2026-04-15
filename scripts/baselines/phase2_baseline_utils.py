"""Shared helpers for Phase 2 baseline scripts."""

from __future__ import annotations

import hashlib
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def _stable_hash(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def hash_text_features(text: str, dim: int = 64, ngram: int = 2) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    text = text or ""
    if len(text) < ngram:
        return vec
    for idx in range(len(text) - ngram + 1):
        token = text[idx : idx + ngram]
        bucket = _stable_hash(token) % dim
        vec[bucket] += 1.0
    if vec.sum() > 0:
        vec /= vec.sum()
    return vec


def load_glycan_embeddings(
    embeddings_path: str, index_path: str
) -> Tuple[Dict[str, np.ndarray], int]:
    embeddings = np.load(embeddings_path)
    index = pd.read_csv(index_path)
    if "glycan_id" not in index.columns or "row_index" not in index.columns:
        raise ValueError("Index file must include glycan_id and row_index columns.")
    emb_map: Dict[str, np.ndarray] = {}
    for _, row in index.iterrows():
        gid = str(row["glycan_id"])
        emb_map[gid] = embeddings[int(row["row_index"])]
    emb_dim = embeddings.shape[1]
    return emb_map, emb_dim


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def compute_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    pos = y_true == 1
    neg = y_true == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(y_score))
    pos_ranks = ranks[pos]
    auc = (pos_ranks.sum() - (pos.sum() * (pos.sum() - 1) / 2.0)) / (pos.sum() * neg.sum())
    return float(auc)


def compute_classification_stats(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float, float]:
    tp = float(((y_true == 1) & (y_pred == 1)).sum())
    tn = float(((y_true == 0) & (y_pred == 0)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1.0)
    precision = tp / max(tp + fp, 1.0)
    recall = tp / max(tp + fn, 1.0)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return accuracy, precision, recall, f1
