"""Unified ESM2 embedder utilities for GlycoML."""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional, Tuple
import warnings

import torch
from torch import nn

try:  # pragma: no cover - optional dependency
    import esm
except ImportError:  # pragma: no cover
    esm = None

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class ESM2Embedder:
    """ESM2 wrapper with optional fallback and caching."""

    MAX_SEQ_LENGTH = {
        "esm2_t6_8M_UR50D": 2048,
        "esm2_t12_35M_UR50D": 2048,
        "esm2_t30_150M_UR50D": 2048,
        "esm2_t33_650M_UR50D": 1024,
        "esm2_t48_15B_UR50D": 512,
    }

    def __init__(
        self,
        model_name: str = "esm2_t6_8M_UR50D",
        device: Optional[torch.device] = None,
        cache_dir: Optional[Path] = None,
        cache_size: int = 128,
        max_len: Optional[int] = None,
        fallback_dim: int = 64,
    ) -> None:
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_size = cache_size
        self._cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()

        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.embed_dim = fallback_dim
        self.max_len = max_len or self.MAX_SEQ_LENGTH.get(model_name, 1024)
        self.model = None
        self.alphabet = None
        self.batch_converter = None
        self.fallback_embedding = nn.Embedding(len(AMINO_ACIDS) + 1, self.embed_dim).to(self.device)

        if esm is None:
            warnings.warn("ESM not installed; using a small trainable embedding instead.")
            return

        try:
            if hasattr(esm.pretrained, model_name):
                self.model, self.alphabet = getattr(esm.pretrained, model_name)()
            else:
                loader = getattr(esm.pretrained, "load_model_and_alphabet", None)
                if loader is None:
                    raise RuntimeError("esm.pretrained.load_model_and_alphabet not available")
                self.model, self.alphabet = loader(model_name)
            self.model = self.model.to(self.device)
            self.model.eval()
            self.embed_dim = getattr(self.model, "embed_dim", self.embed_dim)
            self.batch_converter = self.alphabet.get_batch_converter()
        except Exception as exc:  # pragma: no cover
            warnings.warn(f"Failed to load ESM model '{model_name}': {exc}. Using fallback embedding.")
            self.model = None
            self.alphabet = None
            self.batch_converter = None

    def _hash_text(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _cache_path(self, sequence: str) -> Optional[Path]:
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{self._hash_text(sequence)}.pt"

    def _cache_set(self, sequence: str, embedding: torch.Tensor) -> None:
        if sequence in self._cache:
            self._cache.move_to_end(sequence)
            return
        self._cache[sequence] = embedding
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _fallback_embed(self, sequence: str) -> torch.Tensor:
        seq = sequence.strip().upper()
        indices = [AMINO_ACIDS.find(aa) for aa in seq]
        indices = [idx if idx >= 0 else len(AMINO_ACIDS) for idx in indices]
        tokens = torch.tensor(indices, dtype=torch.long, device=self.device)
        return self.fallback_embedding(tokens)

    def embed_sequence(self, sequence: str) -> torch.Tensor:
        """Return per-residue embeddings for a sequence (L, D)."""
        if not sequence:
            return torch.zeros((1, self.embed_dim), dtype=torch.float32, device=self.device)

        sequence = sequence.strip()
        if self.max_len:
            sequence = sequence[: self.max_len]

        if sequence in self._cache:
            return self._cache[sequence]

        cache_path = self._cache_path(sequence)
        if cache_path and cache_path.exists():
            embedding = torch.load(cache_path, map_location="cpu").to(self.device)
            self._cache_set(sequence, embedding)
            return embedding

        if self.model is None or self.batch_converter is None:
            embedding = self._fallback_embed(sequence)
            self._cache_set(sequence, embedding)
            if cache_path:
                torch.save(embedding.detach().cpu(), cache_path)
            return embedding

        data = [("seq", sequence)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)
        with torch.no_grad():
            outputs = self.model(tokens, repr_layers=[self.model.num_layers], return_contacts=False)
        representations = outputs["representations"][self.model.num_layers]
        embedding = representations[0, 1 : len(sequence) + 1].detach()
        self._cache_set(sequence, embedding)
        if cache_path:
            torch.save(embedding.detach().cpu(), cache_path)
        return embedding

    def embed(self, sequence: str) -> torch.Tensor:
        """Alias for embed_sequence."""
        return self.embed_sequence(sequence)

    def embed_batch(self, sequences: List[str]) -> List[torch.Tensor]:
        embeddings: List[torch.Tensor] = []
        for seq in sequences:
            embeddings.append(self.embed_sequence(seq))
        return embeddings

        if return_lengths:
            return all_embeddings, np.array(all_lengths, dtype=int)
        return all_embeddings

    def embed_with_structure_guidance(
        self,
        seq: str,
        plddt_scores: np.ndarray,
        temperature: float = 0.1,
    ) -> torch.Tensor:
        embeddings = self.embed_sequence(seq)
        if plddt_scores.shape[0] != embeddings.shape[0]:
            raise ValueError(
                f"pLDDT shape {plddt_scores.shape} != sequence length {embeddings.shape[0]}"
            )

        plddt_norm = np.clip(plddt_scores, 0, 100) / 100.0
        weights = torch.tensor(plddt_norm, dtype=embeddings.dtype, device=embeddings.device)
        weights = weights ** (1.0 / temperature)
        weights = weights / (weights.sum() + 1e-10)
        return embeddings * weights.unsqueeze(-1)

    def get_model_info(self) -> Dict[str, Optional[Union[str, int]]]:
        return {
            "model_name": self.model_name,
            "embedding_dim": self.embed_dim,
            "max_sequence_length": self.MAX_SEQ_LENGTH.get(self.model_name),
            "device": self.device,
            "num_parameters": sum(p.numel() for p in self.model.parameters()),
            "use_lora": self.use_lora,
            "lora_rank": self.lora_rank if self.use_lora else None,
        }
