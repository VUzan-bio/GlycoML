"""Protein encoder based on ESM2 embeddings."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import warnings

import torch
from torch import nn

try:
    import esm
except ImportError:  # pragma: no cover
    esm = None


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class ProteinEncoderConfig:
    model_name: str = "esm2_t6_8M_UR50D"
    cache_size: int = 128
    pooling: str = "mean"


class ESM2Embedder:
    """Wraps ESM2 to provide per-residue embeddings with caching."""

    def __init__(self, model_name: str, device: torch.device, cache_size: int = 128):
        self.model_name = model_name
        self.device = device
        self.cache_size = cache_size
        self.cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()
        self.model = None
        self.alphabet = None
        self.batch_converter = None
        self.embed_dim = 64
        self.fallback_embedding = nn.Embedding(len(AMINO_ACIDS) + 1, self.embed_dim).to(device)

        if esm is None:
            warnings.warn("ESM not installed; using a small trainable embedding instead.")
            return

        try:
            if hasattr(esm.pretrained, model_name):
                self.model, self.alphabet = getattr(esm.pretrained, model_name)()
            else:
                self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
            self.model = self.model.to(device)
            self.model.eval()
            self.embed_dim = getattr(self.model, "embed_dim", self.embed_dim)
            self.batch_converter = self.alphabet.get_batch_converter()
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"Failed to load ESM model '{model_name}': {exc}. Using fallback embedding.")
            self.model = None
            self.alphabet = None
            self.batch_converter = None

    def _cache_set(self, sequence: str, embedding: torch.Tensor) -> None:
        if sequence in self.cache:
            self.cache.move_to_end(sequence)
            return
        self.cache[sequence] = embedding
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)

    def _fallback_embed(self, sequence: str) -> torch.Tensor:
        seq = sequence.strip().upper()
        indices = [AMINO_ACIDS.find(aa) for aa in seq]
        indices = [idx if idx >= 0 else len(AMINO_ACIDS) for idx in indices]
        tokens = torch.tensor(indices, dtype=torch.long, device=self.device)
        return self.fallback_embedding(tokens)

    def embed_sequence(self, sequence: str) -> torch.Tensor:
        """Return per-residue embeddings for a sequence (L, D)."""
        if sequence in self.cache:
            return self.cache[sequence]

        if self.model is None or self.batch_converter is None:
            embedding = self._fallback_embed(sequence)
            self._cache_set(sequence, embedding)
            return embedding

        data = [("seq", sequence)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)
        with torch.no_grad():
            outputs = self.model(tokens, repr_layers=[self.model.num_layers], return_contacts=False)
        representations = outputs["representations"][self.model.num_layers]
        embedding = representations[0, 1 : len(sequence) + 1].detach()
        self._cache_set(sequence, embedding)
        return embedding


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
