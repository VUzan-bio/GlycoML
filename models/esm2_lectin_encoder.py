"""ESM2-based lectin protein encoder."""

from __future__ import annotations

from typing import Iterable, List, Tuple

import torch
from torch import nn

import esm


class ESM2LectinEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 320,
        freeze_esm: bool = True,
        model_name: str = "esm2_t6_8M_UR50D",
    ) -> None:
        super().__init__()
        self.esm_model, self.alphabet, self.esm_dim = self._load_model(model_name)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.projection = nn.Linear(self.esm_dim, embedding_dim)
        self._frozen = freeze_esm

        if freeze_esm:
            for param in self.esm_model.parameters():
                param.requires_grad = False

    def forward(self, sequences: Iterable[str]) -> torch.Tensor:
        batch = [(f"seq_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, tokens = self.batch_converter(batch)
        tokens = tokens.to(next(self.parameters()).device)

        with torch.set_grad_enabled(not self._frozen):
            layer_idx = getattr(self.esm_model, "num_layers", 6)
            results = self.esm_model(tokens, repr_layers=[layer_idx], return_contacts=False)
            reps = results["representations"][layer_idx]

        mask = tokens != self.alphabet.padding_idx
        if self.alphabet.cls_idx is not None:
            mask &= tokens != self.alphabet.cls_idx
        if self.alphabet.eos_idx is not None:
            mask &= tokens != self.alphabet.eos_idx

        mask = mask.float()
        pooled = (reps * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1.0)
        return self.projection(pooled)

    def get_embedding_dim(self) -> int:
        return self.projection.out_features

    @staticmethod
    def _load_model(model_name: str) -> Tuple[nn.Module, object, int]:
        registry = {
            "esm2_t6_8M_UR50D": (esm.pretrained.esm2_t6_8M_UR50D, 320),
            "esm2_t12_35M_UR50D": (esm.pretrained.esm2_t12_35M_UR50D, 480),
            "esm2_t30_150M_UR50D": (esm.pretrained.esm2_t30_150M_UR50D, 640),
            "esm2_t33_650M_UR50D": (esm.pretrained.esm2_t33_650M_UR50D, 1280),
            "esm2_t36_3B_UR50D": (esm.pretrained.esm2_t36_3B_UR50D, 2560),
            "esm2_t48_15B_UR50D": (esm.pretrained.esm2_t48_15B_UR50D, 5120),
        }
        if model_name not in registry:
            raise ValueError(f"Unsupported ESM2 model: {model_name}")
        loader, dim = registry[model_name]
        model, alphabet = loader()
        return model, alphabet, dim
