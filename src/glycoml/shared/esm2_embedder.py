"""Unified ESM2 embedder utilities for GlycoML."""

from __future__ import annotations

import hashlib
import logging
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch
from torch import nn

try:  # pragma: no cover - optional dependency
    import esm
except ImportError:  # pragma: no cover
    esm = None

try:  # pragma: no cover - optional dependency
    import h5py
except ImportError:  # pragma: no cover
    h5py = None


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


class HDF5EmbeddingCache:
    """Lightweight HDF5 cache for variable-length embeddings."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _open(self):
        if h5py is None:
            return None
        return h5py.File(self.path, "a")

    def get(self, key: str) -> Optional[Dict[str, np.ndarray]]:
        handle = self._open()
        if handle is None:
            return None
        with handle as h5:
            if key not in h5:
                return None
            group = h5[key]
            result: Dict[str, np.ndarray] = {}
            for name in ("tokens", "mean", "cls"):
                if name in group:
                    result[name] = group[name][()]
            return result

    def set(self, key: str, tokens: np.ndarray, pooled: np.ndarray, cls_token: np.ndarray) -> None:
        handle = self._open()
        if handle is None:
            return
        with handle as h5:
            group = h5.require_group(key)
            for name, value in ("tokens", tokens), ("mean", pooled), ("cls", cls_token):
                if name in group:
                    del group[name]
                group.create_dataset(name, data=value, compression="gzip")


class ESM2Embedder:
    """ESM2 wrapper with optional fallback and HDF5 caching."""

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
        cache_path: Optional[Path] = None,
        cache_size: int = 128,
        max_len: Optional[int] = None,
        fallback_dim: int = 64,
    ) -> None:
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cache_size = cache_size
        self._cache: "OrderedDict[str, torch.Tensor]" = OrderedDict()

        self.cache_path = Path(cache_path) if cache_path else None
        self.h5_cache = HDF5EmbeddingCache(self.cache_path) if self.cache_path else None

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

    def _load_from_h5(self, sequence: str) -> Optional[Dict[str, np.ndarray]]:
        if self.h5_cache is None:
            return None
        key = self._hash_text(sequence)
        return self.h5_cache.get(key)

    def _save_to_h5(self, sequence: str, tokens: torch.Tensor) -> None:
        if self.h5_cache is None:
            return
        key = self._hash_text(sequence)
        tokens_np = tokens.detach().cpu().numpy().astype(np.float32)
        pooled = tokens_np.mean(axis=0)
        cls_token = tokens_np[0] if tokens_np.shape[0] > 0 else pooled
        self.h5_cache.set(key, tokens_np, pooled, cls_token)

    def embed_sequence(self, sequence: str) -> torch.Tensor:
        """Return per-residue embeddings for a sequence (L, D)."""
        if not sequence:
            return torch.zeros((1, self.embed_dim), dtype=torch.float32, device=self.device)

        sequence = sequence.strip()
        if self.max_len:
            sequence = sequence[: self.max_len]

        if sequence in self._cache:
            return self._cache[sequence]

        cached = self._load_from_h5(sequence)
        if cached and "tokens" in cached:
            embedding = torch.tensor(cached["tokens"], dtype=torch.float32, device=self.device)
            self._cache_set(sequence, embedding)
            return embedding

        if self.model is None or self.batch_converter is None:
            embedding = self._fallback_embed(sequence)
            self._cache_set(sequence, embedding)
            self._save_to_h5(sequence, embedding)
            return embedding

        data = [("seq", sequence)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)
        with torch.no_grad():
            outputs = self.model(tokens, repr_layers=[self.model.num_layers], return_contacts=False)
        representations = outputs["representations"][self.model.num_layers]
        embedding = representations[0, 1 : len(sequence) + 1].detach()
        self._cache_set(sequence, embedding)
        self._save_to_h5(sequence, embedding)
        return embedding

    def embed_pooled(self, sequence: str, pool: str = "mean") -> torch.Tensor:
        """Return a pooled embedding (mean or cls)."""
        cached = self._load_from_h5(sequence)
        if cached:
            if pool == "cls" and "cls" in cached:
                return torch.tensor(cached["cls"], dtype=torch.float32, device=self.device)
            if "mean" in cached:
                return torch.tensor(cached["mean"], dtype=torch.float32, device=self.device)
        tokens = self.embed_sequence(sequence)
        if pool == "cls":
            return tokens[0]
        return tokens.mean(dim=0)

    def embed_batch(self, sequences: List[str]) -> List[torch.Tensor]:
        embeddings: List[torch.Tensor] = []
        for seq in sequences:
            embeddings.append(self.embed_sequence(seq))
        return embeddings

    def embed(self, sequence: str) -> torch.Tensor:
        """Alias for embed_sequence."""
        return self.embed_sequence(sequence)
