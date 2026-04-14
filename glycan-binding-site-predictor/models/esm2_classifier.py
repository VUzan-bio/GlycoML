"""ESM2-based N-glycosylation site classifier."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
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
    # Size of the sequence window centred on the candidate Asn. NetNGlyc uses
    # a 9-mer (Gupta & Brunak 2002); SPRINT-Gly and DeepNGlyPred use 21.
    # 11 (+/- 5 around N) is the default here because ESM-2 per-residue
    # embeddings already carry long-range context via self-attention, so a
    # moderate local window suffices (Lin et al. 2023).
    window_size: int = 11


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
    """Binary classifier for N-X-S/T motif candidates.

    ``window_size`` sets the number of residues fed to the MLP, centred on the
    candidate Asn. The input is the flat concatenation of ``window_size``
    per-residue embeddings; out-of-range positions are zero-padded by the
    feature extractor so all inputs are a fixed-size ``embed_dim * window``
    vector.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        window_size: int = 11,
    ):
        super().__init__()
        if window_size < 3 or window_size % 2 == 0:
            raise ValueError(
                f"window_size must be odd and >= 3 (got {window_size}); the "
                "Asn is centred at the middle position so the window needs to "
                "be symmetric around it."
            )
        self.embed_dim = embed_dim
        self.window_size = window_size
        self.net = _build_mlp(embed_dim * window_size, hidden_dim, num_layers, dropout)

    def forward(self, motif_embeddings: torch.Tensor) -> torch.Tensor:
        return self.net(motif_embeddings).squeeze(-1)


def extract_motif_embedding(
    residue_embeddings: torch.Tensor,
    position: int,
    window_size: int = 11,
) -> torch.Tensor:
    """Return a fixed-size window of per-residue embeddings centred on ``position``.

    The window extends ``window_size // 2`` residues on each side of the
    candidate Asn. Positions outside ``[0, L)`` are zero-padded so the output
    shape is always ``(embed_dim * window_size,)``. This matches the
    ``GlycoMotifClassifier`` input contract and is the standard trick used by
    NetNGlyc and SPRINT-Gly to handle termini.
    """
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be odd and >= 3.")
    if position < 0 or position >= residue_embeddings.shape[0]:
        raise ValueError(
            f"Motif position {position} out of range for sequence of length "
            f"{residue_embeddings.shape[0]}."
        )

    half = window_size // 2
    L, D = residue_embeddings.shape
    device = residue_embeddings.device
    dtype = residue_embeddings.dtype

    parts: List[torch.Tensor] = []
    for offset in range(-half, half + 1):
        idx = position + offset
        if 0 <= idx < L:
            parts.append(residue_embeddings[idx])
        else:
            parts.append(torch.zeros(D, device=device, dtype=dtype))
    return torch.cat(parts, dim=-1)


def save_classifier(path: str, classifier: GlycoMotifClassifier, config: ModelConfig) -> None:
    checkpoint = {
        "model_state": classifier.state_dict(),
        "config": {
            "model_name": config.model_name,
            "hidden_dim": config.hidden_dim,
            "num_layers": config.num_layers,
            "dropout": config.dropout,
            "window_size": classifier.window_size,
        },
        "embed_dim": classifier.embed_dim,
        "window_size": classifier.window_size,
    }
    torch.save(checkpoint, path)


def load_classifier(path: str, device: torch.device) -> Tuple[GlycoMotifClassifier, ModelConfig]:
    checkpoint = torch.load(path, map_location=device)
    embed_dim = checkpoint.get("embed_dim", 64)
    cfg_dict = checkpoint.get("config", {})
    # Checkpoints written before this change used a 3-residue window (N, X,
    # S/T concatenation). Default to 3 if the field is absent so old
    # checkpoints still load.
    window_size = int(
        checkpoint.get("window_size", cfg_dict.get("window_size", 3))
    )
    config = ModelConfig(
        model_name=cfg_dict.get("model_name", "esm2_t6_8M_UR50D"),
        hidden_dim=cfg_dict.get("hidden_dim", 256),
        num_layers=cfg_dict.get("num_layers", 2),
        dropout=cfg_dict.get("dropout", 0.1),
        window_size=window_size,
    )
    classifier = GlycoMotifClassifier(
        embed_dim=embed_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        dropout=config.dropout,
        window_size=window_size,
    )
    classifier.load_state_dict(checkpoint["model_state"])
    classifier.to(device)
    classifier.eval()
    return classifier, config

