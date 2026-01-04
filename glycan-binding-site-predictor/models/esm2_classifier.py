"""ESM2-based N-glycosylation site classifier."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import warnings

import torch
from torch import nn

try:
    import esm
except ImportError:  # pragma: no cover - optional dependency
    esm = None


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


@dataclass
class ModelConfig:
    model_name: str = "esm2_t6_8M_UR50D"
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    cache_size: int = 128


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
                # Fallback to generic loader for named checkpoints.
                self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
            self.model = self.model.to(device)
            self.model.eval()
            self.embed_dim = getattr(self.model, "embed_dim", self.embed_dim)
            self.batch_converter = self.alphabet.get_batch_converter()
        except Exception as exc:  # pragma: no cover - defensive
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


def _build_mlp(input_dim: int, hidden_dim: int, num_layers: int, dropout: float) -> nn.Sequential:
    layers = []
    dim = input_dim
    for _ in range(max(num_layers, 1)):
        layers.append(nn.Linear(dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        dim = hidden_dim
    layers.append(nn.Linear(dim, 1))
    return nn.Sequential(*layers)


class GlycoMotifClassifier(nn.Module):
    """Binary classifier for N-X-S/T motif candidates."""

    def __init__(self, embed_dim: int, hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = _build_mlp(embed_dim * 3, hidden_dim, num_layers, dropout)

    def forward(self, motif_embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(motif_embeddings).squeeze(-1)


def extract_motif_embedding(residue_embeddings: torch.Tensor, position: int) -> torch.Tensor:
    """Concatenate embeddings for N, X, and S/T positions (shape: 3 * D)."""
    if position + 2 >= residue_embeddings.shape[0]:
        raise ValueError("Motif position out of range for embedding.")
    return torch.cat(
        [
            residue_embeddings[position],
            residue_embeddings[position + 1],
            residue_embeddings[position + 2],
        ],
        dim=-1,
    )


def save_classifier(path: str, classifier: GlycoMotifClassifier, config: ModelConfig) -> None:
    checkpoint = {
        "model_state": classifier.state_dict(),
        "config": {
            "model_name": config.model_name,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
        },
        "embed_dim": classifier.embed_dim,
    }
    torch.save(checkpoint, path)


def load_classifier(path: str, device: torch.device) -> Tuple[GlycoMotifClassifier, ModelConfig]:
    checkpoint = torch.load(path, map_location=device)
    embed_dim = checkpoint.get("embed_dim", 64)
    cfg_dict = checkpoint.get("config", {})
    config = ModelConfig(
        model_name=cfg_dict.get("model_name", "esm2_t6_8M_UR50D"),
        hidden_dim=cfg_dict.get("hidden_dim", 256),
        num_layers=cfg_dict.get("num_layers", 2),
        dropout=cfg_dict.get("dropout", 0.1),
    )
    classifier = GlycoMotifClassifier(
        embed_dim=embed_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    classifier.load_state_dict(checkpoint["model_state"])
    classifier.to(device)
    classifier.eval()
    return classifier, config

