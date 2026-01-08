"""Neural lectin-glycan binding predictor."""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


class LectinGlycanPredictor(nn.Module):
    def __init__(self, lectin_encoder: nn.Module, glycan_encoder: nn.Module, hidden_dims=None) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        self.lectin_encoder = lectin_encoder
        self.glycan_encoder = glycan_encoder

        input_dim = lectin_encoder.get_embedding_dim() + glycan_encoder.get_embedding_dim()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                ]
            )
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, lectin_sequences: Iterable[str], glycan_structures: Iterable[str]) -> torch.Tensor:
        lectin_emb = self.lectin_encoder(lectin_sequences)
        glycan_emb = self.glycan_encoder(glycan_structures)
        combined = torch.cat([lectin_emb, glycan_emb], dim=1)
        return self.mlp(combined).squeeze(-1)
