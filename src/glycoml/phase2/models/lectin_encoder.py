"""Lectin encoders for Phase 2 binding prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

try:  # pragma: no cover
    from torch_geometric.data import Batch
    from torch_geometric.nn.models import SchNet
except Exception:  # pragma: no cover
    Batch = None
    SchNet = None


@dataclass
class LectinEncoderConfig:
    esm_dim: int = 1280
    struct_dim: int = 256
    family_dim: int = 73
    species_emb_dim: int = 32
    output_dim: int = 512
    dropout: float = 0.1
    use_structure: bool = False
    schnet_hidden: int = 128
    schnet_interactions: int = 3


class LectinStructureEncoder(nn.Module):
    """Optional structure encoder using SchNet over residue coordinates."""

    def __init__(self, hidden_dim: int = 128, out_dim: int = 256, interactions: int = 3) -> None:
        super().__init__()
        if SchNet is None:
            raise ImportError("torch_geometric is required for structure encoding")
        self.model = SchNet(
            hidden_channels=hidden_dim,
            num_filters=hidden_dim,
            num_interactions=interactions,
            out_channels=out_dim,
            cutoff=10.0,
        )

    def forward(self, batch: Batch) -> torch.Tensor:
        return self.model(batch.z, batch.pos, batch.batch)


def masked_mean(tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(tokens.dtype)
    summed = (tokens * mask.unsqueeze(-1)).sum(dim=1)
    denom = mask.sum(dim=1).clamp_min(1.0)
    return summed / denom.unsqueeze(-1)


class LectinEncoder(nn.Module):
    """Fuse ESM tokens, optional structure embeddings, and metadata."""

    def __init__(self, config: LectinEncoderConfig) -> None:
        super().__init__()
        self.config = config
        self.token_proj = nn.Linear(config.esm_dim, config.output_dim)
        self.family_proj = nn.Linear(config.family_dim, 128) if config.family_dim > 0 else None
        self.species_emb = nn.Embedding(4096, config.species_emb_dim)
        self.use_structure = config.use_structure and SchNet is not None

        if self.use_structure:
            self.structure_encoder = LectinStructureEncoder(
                hidden_dim=config.schnet_hidden,
                out_dim=config.struct_dim,
                interactions=config.schnet_interactions,
            )
        else:
            self.structure_encoder = None

        fusion_dim = config.output_dim + config.struct_dim + 128 + config.species_emb_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, config.output_dim),
            nn.LayerNorm(config.output_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        family_features: Optional[torch.Tensor] = None,
        species_idx: Optional[torch.Tensor] = None,
        structure_batch: Optional[Batch] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_emb = self.token_proj(tokens)
        pooled = masked_mean(token_emb, mask)

        if self.family_proj is not None and family_features is not None:
            family_vec = self.family_proj(family_features)
        else:
            family_vec = torch.zeros((pooled.size(0), 128), device=pooled.device)
        species_vec = (
            self.species_emb(species_idx.clamp_min(0))
            if species_idx is not None
            else torch.zeros((pooled.size(0), self.config.species_emb_dim), device=pooled.device)
        )

        if self.use_structure and structure_batch is not None:
            struct_vec = self.structure_encoder(structure_batch)
        else:
            struct_vec = torch.zeros((pooled.size(0), self.config.struct_dim), device=pooled.device)

        fused = torch.cat([pooled, struct_vec, family_vec, species_vec], dim=-1)
        fused = self.fusion(fused)
        return token_emb, fused
