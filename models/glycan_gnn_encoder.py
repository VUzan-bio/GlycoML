"""Graph neural encoder for glycan structures."""

from __future__ import annotations

import re
from typing import List

import torch
from torch import nn

try:
    from torch_geometric.data import Batch, Data
    from torch_geometric.nn import GCNConv, global_mean_pool
except Exception as exc:  # pragma: no cover
    Batch = None
    Data = None
    GCNConv = None
    global_mean_pool = None


class GlycanGNNEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 256, num_layers: int = 3) -> None:
        super().__init__()
        if GCNConv is None or Data is None:
            raise ImportError("torch-geometric is required for GlycanGNNEncoder.")

        self.monosaccharide_vocab = {
            "Glc": 0,
            "Gal": 1,
            "Man": 2,
            "Fuc": 3,
            "Xyl": 4,
            "GlcNAc": 5,
            "GalNAc": 6,
            "ManNAc": 7,
            "Neu5Ac": 8,
            "Neu5Gc": 9,
            "GlcA": 10,
            "IdoA": 11,
            "KDN": 12,
            "<UNK>": 13,
        }

        self.node_embedding = nn.Embedding(len(self.monosaccharide_vocab), hidden_dim)
        self.convs = nn.ModuleList([GCNConv(hidden_dim, hidden_dim) for _ in range(num_layers)])
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.projection = nn.Linear(hidden_dim, embedding_dim)

    def _tokenize(self, glycan_structure: str) -> List[int]:
        tokens: List[int] = []
        for mono in self.monosaccharide_vocab:
            if mono == "<UNK>":
                continue
            count = len(re.findall(mono, glycan_structure))
            if count:
                tokens.extend([self.monosaccharide_vocab[mono]] * count)
        if not tokens:
            tokens = [self.monosaccharide_vocab["<UNK>"]]
        return tokens

    def parse_glycan_to_graph(self, glycan_structure: str) -> "Data":
        nodes = self._tokenize(glycan_structure or "")
        num_nodes = len(nodes)
        if num_nodes == 1:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            edges = [(i, j) for i in range(num_nodes) for j in range(num_nodes) if i != j]
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        return Data(x=torch.tensor(nodes, dtype=torch.long).view(-1, 1), edge_index=edge_index)

    def forward(self, glycan_structures: List[str]) -> torch.Tensor:
        graphs = [self.parse_glycan_to_graph(structure) for structure in glycan_structures]
        batch = Batch.from_data_list(graphs).to(next(self.parameters()).device)

        x = self.node_embedding(batch.x.squeeze(-1))
        for conv, bn in zip(self.convs, self.batch_norms):
            x = conv(x, batch.edge_index)
            x = bn(x)
            x = self.activation(x)
            x = self.dropout(x)

        pooled = global_mean_pool(x, batch.batch)
        return self.projection(pooled)

    def get_embedding_dim(self) -> int:
        return self.projection.out_features
