"""Model definitions for lectin-glycan binding prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class ModelConfig:
    lectin_dim: int
    glycan_vocab: int
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    task: str = "classification"  # classification or regression


class CrossAttentionModel(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.lectin_proj = nn.Linear(cfg.lectin_dim, cfg.hidden_dim)
        self.lectin_norm = nn.LayerNorm(cfg.hidden_dim)

        self.glycan_embed = nn.Embedding(cfg.glycan_vocab, cfg.hidden_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.hidden_dim,
            nhead=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
            dim_feedforward=cfg.hidden_dim * 4,
            activation="gelu",
        )
        self.glycan_encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cfg.hidden_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(cfg.hidden_dim)

        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, 1),
        )

    def forward(
        self,
        lectin_tokens: torch.Tensor,
        lectin_mask: torch.Tensor,
        glycan_tokens: torch.Tensor,
        glycan_mask: torch.Tensor,
    ) -> torch.Tensor:
        lectin_proj = self.lectin_norm(self.lectin_proj(lectin_tokens))
        glycan_embed = self.glycan_embed(glycan_tokens)
        glycan_encoded = self.glycan_encoder(
            glycan_embed, src_key_padding_mask=~glycan_mask
        )

        attn_out, _ = self.cross_attn(
            query=glycan_encoded,
            key=lectin_proj,
            value=lectin_proj,
            key_padding_mask=~lectin_mask,
        )
        attn_out = self.attn_norm(attn_out + glycan_encoded)

        glycan_pool = masked_mean(attn_out, glycan_mask)
        lectin_pool = masked_mean(lectin_proj, lectin_mask)
        features = torch.cat([lectin_pool, glycan_pool], dim=-1)
        logits = self.head(features).squeeze(-1)
        return logits


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).float()
    summed = (values * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1.0)
    return summed / denom


__all__ = ["ModelConfig", "CrossAttentionModel"]
