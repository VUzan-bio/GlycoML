"""ESM2-based N-glycosylation site classifier."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from ...shared.esm2_embedder import ESM2Embedder


@dataclass
class ModelConfig:
    model_name: str = "esm2_t6_8M_UR50D"
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1
    cache_size: int = 128


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

