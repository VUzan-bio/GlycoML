"""FcγR-Fc binding predictor model."""

from __future__ import annotations

from typing import Iterable, List

import torch
from torch import nn


class FcGammaRFcPredictor(nn.Module):
    def __init__(
        self,
        fcgr_encoder: nn.Module,
        fc_encoder: nn.Module,
        glycan_encoder: nn.Module,
        hidden_dims=None,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        self.fcgr_encoder = fcgr_encoder
        self.fc_encoder = fc_encoder
        self.glycan_encoder = glycan_encoder

        input_dim = (
            fcgr_encoder.get_embedding_dim()
            + fc_encoder.get_embedding_dim()
            + glycan_encoder.get_embedding_dim()
        )

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

    def forward(
        self,
        fcgr_sequences: Iterable[str],
        fc_sequences: Iterable[str],
        glycan_structures: Iterable[str],
    ) -> torch.Tensor:
        fcgr_emb = self.fcgr_encoder(fcgr_sequences)
        fc_emb = self.fc_encoder(fc_sequences)
        glycan_emb = self.glycan_encoder(list(glycan_structures))
        combined = torch.cat([fcgr_emb, fc_emb, glycan_emb], dim=1)
        return self.mlp(combined).squeeze(-1)
