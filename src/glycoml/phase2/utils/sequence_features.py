"""Sequence feature helpers for lightweight baselines."""

from __future__ import annotations

import hashlib
from typing import Dict, List

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def hashed_kmer_counts(sequence: str, k: int = 3, dim: int = 1024) -> List[float]:
    """Return hashed k-mer counts with a fixed dimension."""
    sequence = sequence.strip().upper()
    counts = [0.0] * dim
    if k <= 0 or len(sequence) < k:
        return counts
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i : i + k]
        if any(aa not in AMINO_ACIDS for aa in kmer):
            continue
        digest = hashlib.md5(kmer.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % dim
        counts[idx] += 1.0
    return counts
