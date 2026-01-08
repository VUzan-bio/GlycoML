"""Metrics for regression and classification."""

from __future__ import annotations

from typing import Iterable, Tuple
import math


def mse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    total = 0.0
    count = 0
    for true, pred in zip(y_true, y_pred):
        diff = true - pred
        total += diff * diff
        count += 1
    return total / count if count else 0.0


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    total = 0.0
    count = 0
    for true, pred in zip(y_true, y_pred):
        total += abs(true - pred)
        count += 1
    return total / count if count else 0.0


def pearson(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = list(y_true)
    y_pred = list(y_pred)
    n = len(y_true)
    if n == 0:
        return 0.0
    mean_true = sum(y_true) / n
    mean_pred = sum(y_pred) / n
    num = sum((t - mean_true) * (p - mean_pred) for t, p in zip(y_true, y_pred))
    den_true = math.sqrt(sum((t - mean_true) ** 2 for t in y_true))
    den_pred = math.sqrt(sum((p - mean_pred) ** 2 for p in y_pred))
    if den_true == 0.0 or den_pred == 0.0:
        return 0.0
    return num / (den_true * den_pred)


def accuracy(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    correct = 0
    total = 0
    for true, pred in zip(y_true, y_pred):
        correct += int(true == pred)
        total += 1
    return correct / total if total else 0.0


def matthews_corrcoef(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    tp = tn = fp = fn = 0
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            tp += 1
        elif true == 0 and pred == 0:
            tn += 1
        elif true == 0 and pred == 1:
            fp += 1
        elif true == 1 and pred == 0:
            fn += 1
    numerator = (tp * tn) - (fp * fn)
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    return numerator / math.sqrt(denominator) if denominator else 0.0


def binary_classification_stats(y_true: Iterable[int], y_pred: Iterable[int]) -> Tuple[float, float, float]:
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
