"""Data utilities for lectin-glycan interaction modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import csv
import random


@dataclass(frozen=True)
class InteractionSample:
    lectin_id: str
    lectin_sequence: str
    glycan_id: str
    glycan_smiles: str
    glycan_iupac: Optional[str]
    rfu: float
    rfu_norm: Optional[float] = None
    label: Optional[int] = None


def load_interaction_samples(csv_path: str) -> List[InteractionSample]:
    samples: List[InteractionSample] = []
    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            lectin_id = (row.get("lectin_id") or row.get("lectin") or "").strip()
            lectin_sequence = (row.get("lectin_sequence") or row.get("sequence") or "").strip()
            glycan_id = (row.get("glycan_id") or row.get("glycan") or "").strip()
            glycan_smiles = (row.get("glycan_smiles") or row.get("smiles") or "").strip()
            glycan_iupac = (row.get("glycan_iupac") or row.get("iupac") or "").strip()
            rfu_raw = row.get("rfu") or row.get("binding") or row.get("rfu_raw")
            rfu_norm_raw = row.get("rfu_norm") or row.get("binding_norm")
            label_raw = row.get("label")

            if not lectin_sequence or not glycan_smiles:
                continue

            try:
                rfu = float(rfu_raw) if rfu_raw is not None and rfu_raw != "" else 0.0
            except ValueError:
                rfu = 0.0

            rfu_norm = None
            if rfu_norm_raw is not None and rfu_norm_raw != "":
                try:
                    rfu_norm = float(rfu_norm_raw)
                except ValueError:
                    rfu_norm = None

            label = None
            if label_raw is not None and label_raw != "":
                try:
                    label = int(label_raw)
                except ValueError:
                    label = None

            samples.append(
                InteractionSample(
                    lectin_id=lectin_id,
                    lectin_sequence=lectin_sequence,
                    glycan_id=glycan_id,
                    glycan_smiles=glycan_smiles,
                    glycan_iupac=glycan_iupac,
                    rfu=rfu,
                    rfu_norm=rfu_norm,
                    label=label,
                )
            )
    return samples


def split_samples(
    samples: List[InteractionSample],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 13,
) -> Tuple[List[InteractionSample], List[InteractionSample], List[InteractionSample]]:
    rng = random.Random(seed)
    samples = list(samples)
    rng.shuffle(samples)
    n_total = len(samples)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    val_samples = samples[:n_val]
    test_samples = samples[n_val : n_val + n_test]
    train_samples = samples[n_val + n_test :]
    return train_samples, val_samples, test_samples


def summarize_samples(samples: Iterable[InteractionSample]) -> Dict[str, int]:
    total = 0
    with_label = 0
    for sample in samples:
        total += 1
        with_label += int(sample.label is not None)
    return {
        "total": total,
        "with_label": with_label,
    }


def build_label_from_threshold(value: float, threshold: float) -> int:
    return int(value >= threshold)
