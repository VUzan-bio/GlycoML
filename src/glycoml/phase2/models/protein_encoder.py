"""Protein encoder based on ESM2 embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

from ...shared.esm2_embedder import ESM2Embedder


@dataclass
class ProteinEncoderConfig:
    model_name: str = "esm2_t6_8M_UR50D"
    cache_size: int = 128
    pooling: str = "mean"


class LectinEncoder:
    """Produce fixed-size lectin embeddings from sequence and optional pLDDT weights."""

    def __init__(self, embedder: ESM2Embedder, pooling: str = "mean"):
        self.embedder = embedder
        self.pooling = pooling
        self.embed_dim = embedder.embed_dim

    def _pool(self, residue_embeddings: torch.Tensor, plddt: Optional[Sequence[float]] = None) -> torch.Tensor:
        if plddt is None or len(plddt) != residue_embeddings.shape[0]:
            return residue_embeddings.mean(dim=0)
        weights = torch.tensor(plddt, dtype=residue_embeddings.dtype, device=residue_embeddings.device) / 100.0
        if torch.all(weights == 0):
            return residue_embeddings.mean(dim=0)
        weights = weights / weights.sum()
        return (residue_embeddings * weights.unsqueeze(-1)).sum(dim=0)

    def encode(self, sequence: str, plddt: Optional[Sequence[float]] = None) -> torch.Tensor:
        residue_embeddings = self.embedder.embed_sequence(sequence)
        return self._pool(residue_embeddings, plddt)

    def encode_batch(self, sequences: List[str]) -> torch.Tensor:
        embeddings = [self.encode(seq) for seq in sequences]
        return torch.stack(embeddings)


def parse_plddt_from_pdb(pdb_path: str, chain_id: Optional[str] = None) -> Dict[int, float]:
    """Parse pLDDT scores from AlphaFold-style PDB (B-factor for CA atoms)."""
    plddt: Dict[int, float] = {}
    with open(pdb_path, "r") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            if chain_id and line[21].strip() != chain_id:
                continue
            try:
                res_id = int(line[22:26].strip())
                b_factor = float(line[60:66].strip())
            except ValueError:
                continue
            plddt[res_id - 1] = b_factor
    return plddt
