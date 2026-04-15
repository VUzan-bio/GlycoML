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
    """Parse B-factor / pLDDT per residue, keyed by 0-based sequence index.

    Enumerates CA atoms in order so insertion codes (e.g. Kabat H100A/H100B)
    produce distinct sequence indices. Only blank or 'A' altLoc is kept to
    avoid double-counting residues with alternate conformers. Caller is
    responsible for interpreting the value: pLDDT (0-100) for predicted
    structures, Debye-Waller temperature factor (A^2) for experimental ones.
    """
    plddt: Dict[int, float] = {}
    seen: set = set()
    seq_index = 0
    with open(pdb_path, "r") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            alt_loc = line[16]
            if alt_loc not in (" ", "A"):
                continue
            if chain_id and line[21].strip() != chain_id:
                continue
            try:
                res_seq = int(line[22:26].strip())
                b_factor = float(line[60:66].strip())
            except ValueError:
                continue
            icode = line[26]
            key = (line[21], res_seq, icode)
            if key in seen:
                continue
            seen.add(key)
            plddt[seq_index] = b_factor
            seq_index += 1
    return plddt
