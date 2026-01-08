"""Interaction predictor for lectin-glycan binding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


@dataclass
class InteractionConfig:
    lectin_dim: int
    glycan_dim: int
    hidden_dim1: int = 512
    hidden_dim2: int = 256
    dropout: float = 0.3
    use_bilinear: bool = False


class BilinearAttention(nn.Module):
    def __init__(self, lectin_dim: int, glycan_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(lectin_dim, glycan_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, lectin: torch.Tensor, glycan: torch.Tensor) -> torch.Tensor:
        score = torch.einsum("bi,ij,bj->b", lectin, self.weight, glycan)
        gate = torch.sigmoid(score).unsqueeze(-1)
        return gate


class InteractionPredictor(nn.Module):
    def __init__(self, config: InteractionConfig):
        super().__init__()
        self.config = config
        self.use_bilinear = config.use_bilinear
        if self.use_bilinear:
            self.attention = BilinearAttention(config.lectin_dim, config.glycan_dim)
        else:
            self.attention = None
        input_dim = config.lectin_dim + config.glycan_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim1),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim1, config.hidden_dim2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim2, 1),
        )

    def forward(self, lectin: torch.Tensor, glycan: torch.Tensor) -> torch.Tensor:
        features = torch.cat([lectin, glycan], dim=-1)
        if self.attention is not None:
            gate = self.attention(lectin, glycan)
            features = features * (1.0 + gate)
        return self.net(features).squeeze(-1)


def save_interaction_model(
    path: str,
    model: InteractionPredictor,
    config: InteractionConfig,
    meta: Optional[dict] = None,
) -> None:
    checkpoint = {
        "model_state": model.state_dict(),
        "config": {
            "lectin_dim": config.lectin_dim,
            "glycan_dim": config.glycan_dim,
            "hidden_dim1": config.hidden_dim1,
            "hidden_dim2": config.hidden_dim2,
            "dropout": config.dropout,
            "use_bilinear": config.use_bilinear,
        },
        "meta": meta or {},
    }
    torch.save(checkpoint, path)


def load_interaction_model(path: str, device: torch.device) -> Tuple[InteractionPredictor, InteractionConfig]:
    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint.get("config", {})
    config = InteractionConfig(
        lectin_dim=cfg.get("lectin_dim", 1280),
        glycan_dim=cfg.get("glycan_dim", 2048),
        hidden_dim1=cfg.get("hidden_dim1", 512),
        hidden_dim2=cfg.get("hidden_dim2", 256),
        dropout=cfg.get("dropout", 0.3),
        use_bilinear=cfg.get("use_bilinear", False),
    )
    model = InteractionPredictor(config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, config
