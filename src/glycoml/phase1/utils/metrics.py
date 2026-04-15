"""Metrics for binary classification."""

from __future__ import annotations

from typing import Iterable, Tuple


def precision_recall_f1(y_true: Iterable[int], y_pred: Iterable[int]) -> Tuple[float, float, float]:
    tp = fp = fn = 0
    for true, pred in zip(y_true, y_pred):
        if pred == 1 and true == 1:
            tp += 1
        elif pred == 1 and true == 0:
            fp += 1
        elif pred == 0 and true == 1:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def accuracy(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    correct = total = 0
    for true, pred in zip(y_true, y_pred):
        correct += int(true == pred)
        total += 1
    return correct / total if total else 0.0

