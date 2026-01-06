"""Cross-attention fusion model for lectin-glycan binding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

from .lectin_encoder import LectinEncoder, LectinEncoderConfig
from .glycan_encoder import GlycanGraphEncoder, GlycanGraphConfig, GlycanTokenEncoder, GlycanTokenConfig


@dataclass
class BindingModelConfig:
    lectin_config: LectinEncoderConfig = LectinEncoderConfig()
    glycan_graph_config: GlycanGraphConfig = GlycanGraphConfig()
    glycan_token_config: GlycanTokenConfig = GlycanTokenConfig()
    use_graph: bool = True
    use_cross_attention: bool = True
    attn_heads: int = 8
    attn_dropout: float = 0.1
    head_dropout: float = 0.2


class CrossAttentionFusion(nn.Module):
    def __init__(self, dim: int = 512, heads: int = 8, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=dim * 2,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(
        self,
        lectin_tokens: torch.Tensor,
        lectin_mask: torch.Tensor,
        glycan_tokens: torch.Tensor,
        glycan_mask: torch.Tensor,
    ) -> torch.Tensor:
        key_padding_mask = ~lectin_mask
        attn_out, _ = self.attn(
            glycan_tokens,
            lectin_tokens,
            lectin_tokens,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        attn_out = self.self_attn(attn_out, src_key_padding_mask=~glycan_mask)
        pooled = (attn_out * glycan_mask.unsqueeze(-1)).sum(dim=1) / glycan_mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        return pooled


class BindingModel(nn.Module):
    """Joint binary + regression predictor for lectin-glycan binding."""

    def __init__(self, config: BindingModelConfig) -> None:
        super().__init__()
        self.config = config
        self.lectin_encoder = LectinEncoder(config.lectin_config)
        if config.use_graph:
            self.glycan_encoder = GlycanGraphEncoder(config.glycan_graph_config)
        else:
            self.glycan_encoder = GlycanTokenEncoder(config.glycan_token_config)
        self.cross_attn = CrossAttentionFusion(dim=config.lectin_config.output_dim, heads=config.attn_heads, dropout=config.attn_dropout)
        self.glycan_proj = nn.Linear(256, config.lectin_config.output_dim)
        self.glycan_token_proj = nn.Linear(config.glycan_token_config.embed_dim, config.lectin_config.output_dim)
        fusion_dim = config.lectin_config.output_dim * 2
        self.use_cross_attention = config.use_cross_attention
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(config.head_dropout),
            nn.LayerNorm(512),
        )
        self.bin_head = nn.Linear(512, 1)
        self.reg_head = nn.Linear(512, 1)

    def forward(
        self,
        lectin_tokens: torch.Tensor,
        lectin_mask: torch.Tensor,
        family_features: Optional[torch.Tensor] = None,
        species_idx: Optional[torch.Tensor] = None,
        structure_batch=None,
        glycan_tokens: Optional[torch.Tensor] = None,
        glycan_mask: Optional[torch.Tensor] = None,
        glycan_graph=None,
        glycan_meta: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        token_emb, lectin_vec = self.lectin_encoder(
            lectin_tokens,
            lectin_mask,
            family_features=family_features,
            species_idx=species_idx,
            structure_batch=structure_batch,
        )
        if glycan_graph is not None:
            glycan_vec = self.glycan_encoder(glycan_graph)
            glycan_vec_proj = self.glycan_proj(glycan_vec)
            glycan_tokens_attn = glycan_vec_proj.unsqueeze(1)
            glycan_mask_attn = torch.ones((glycan_vec.size(0), 1), device=glycan_vec.device, dtype=torch.bool)
        else:
            if glycan_tokens is None or glycan_mask is None:
                raise ValueError("glycan_tokens and glycan_mask required for token encoder")
            glycan_vec, glycan_token_emb = self.glycan_encoder(
                glycan_tokens, glycan_mask, glycan_meta, return_tokens=True
            )
            glycan_vec_proj = self.glycan_proj(glycan_vec)
            glycan_tokens_attn = self.glycan_token_proj(glycan_token_emb)
            glycan_mask_attn = glycan_mask

        if self.use_cross_attention:
            glycan_attn = self.cross_attn(token_emb, lectin_mask, glycan_tokens_attn, glycan_mask_attn)
        else:
            glycan_attn = glycan_vec_proj

        fused = torch.cat([lectin_vec, glycan_attn], dim=-1)
        hidden = self.head(fused)
        bin_logits = self.bin_head(hidden).squeeze(-1)
        reg_out = self.reg_head(hidden).squeeze(-1)
        if return_features:
            return bin_logits, reg_out, hidden
        return bin_logits, reg_out, None
