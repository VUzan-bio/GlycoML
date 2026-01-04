"""Fcgr binding affinity prediction via graph neural networks.

Notes:
- Graph nodes are residues; edges are proximity contacts.
- Output is a binding energy estimate (delta G).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install torch-geometric for Fcgr GNN support") from exc

logger = logging.getLogger(__name__)

AA_FEATURES = {
    "A": [1.88, 0, 1.8],
    "R": [2.65, 1, -4.5],
    "N": [2.3, 0, -3.5],
    "D": [2.2, -1, -3.5],
    "C": [2.04, 0, 2.5],
    "Q": [2.36, 0, -3.5],
    "E": [2.22, -1, -3.5],
    "G": [1.61, 0, -0.4],
    "H": [2.3, 0.1, -3.2],
    "I": [2.36, 0, 4.5],
    "L": [2.36, 0, 3.8],
    "K": [2.71, 1, -3.9],
    "M": [2.3, 0, 1.9],
    "F": [2.36, 0, 2.8],
    "P": [2.14, 0, -1.6],
    "S": [2.16, 0, -0.8],
    "T": [2.25, 0, -0.7],
    "W": [2.36, 0, -0.9],
    "Y": [2.36, 0, -1.3],
    "V": [2.27, 0, 4.2],
    "X": [2.2, 0, 0.0],
}


class FcDomainGCN(nn.Module):
    """Graph convolutional network for Fc domain binding prediction."""

    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 64,
        output_dim: int = 1,
        num_layers: int = 4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.gcn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        dims = [input_dim] + [hidden_dim] * (num_layers - 1)
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            self.gcn_layers.append(GCNConv(in_dim, out_dim))
            self.batch_norms.append(nn.BatchNorm1d(out_dim))

        mlp_input_dim = hidden_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )

    def forward(self, x, edge_index, batch_idx):
        for gcn, bn in zip(self.gcn_layers, self.batch_norms):
            x = gcn(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.3, training=self.training)

        x_mean = global_mean_pool(x, batch_idx)
        x_max = global_max_pool(x, batch_idx)
        x_global = torch.cat([x_mean, x_max], dim=1)
        affinity = self.mlp(x_global)
        return affinity


class FcGraphBuilder:
    """Convert Fc structure to a PyTorch Geometric graph."""

    def __init__(self, distance_threshold: float = 6.5):
        self.distance_threshold = distance_threshold

    def build_from_pdb(self, pdb_path: str, chain: str = "H") -> Data:
        try:
            from Bio.PDB import PDBParser
        except ImportError as exc:
            raise ImportError("Install biopython: `pip install biopython`") from exc

        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("antibody", pdb_path)

        coords = []
        seq = []

        for model in structure:
            for chain_obj in model:
                if chain_obj.id != chain:
                    continue
                for residue in chain_obj:
                    if residue.has_id("CA"):
                        ca = residue["CA"]
                        coords.append(ca.coord)
                        seq.append(self._aa_code(residue.resname))

        coords = np.array(coords)
        node_features = [AA_FEATURES.get(aa, AA_FEATURES["X"]) for aa in seq]
        node_features = torch.tensor(node_features, dtype=torch.float32)

        edges = self._compute_edges(coords)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        return Data(
            x=node_features,
            edge_index=edge_index,
            pos=torch.tensor(coords, dtype=torch.float32),
            seq="".join(seq),
        )

    def build_from_alphafold(
        self,
        seq: str,
        predicted_structure: np.ndarray,
        plddt_scores: Optional[np.ndarray] = None,
    ) -> Data:
        node_features = [AA_FEATURES.get(aa, AA_FEATURES["X"]) for aa in seq]
        node_features = torch.tensor(node_features, dtype=torch.float32)

        if plddt_scores is not None:
            confidence = np.clip(plddt_scores, 0, 100) / 100.0
            node_features = node_features * torch.tensor(confidence, dtype=torch.float32).unsqueeze(-1)

        edges = self._compute_edges(predicted_structure)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

        return Data(
            x=node_features,
            edge_index=edge_index,
            pos=torch.tensor(predicted_structure, dtype=torch.float32),
            seq=seq,
        )

    def _compute_edges(self, coords: np.ndarray) -> list:
        edges = []
        n_residues = coords.shape[0]
        for i in range(n_residues):
            for j in range(i + 1, n_residues):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < self.distance_threshold:
                    edges.append([i, j])
                    edges.append([j, i])
        return edges

    @staticmethod
    def _aa_code(resname: str) -> str:
        code_map = {
            "ALA": "A",
            "ARG": "R",
            "ASN": "N",
            "ASP": "D",
            "CYS": "C",
            "GLN": "Q",
            "GLU": "E",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LEU": "L",
            "LYS": "K",
            "MET": "M",
            "PHE": "F",
            "PRO": "P",
            "SER": "S",
            "THR": "T",
            "TRP": "W",
            "TYR": "Y",
            "VAL": "V",
        }
        return code_map.get(resname, "X")


class FcgrBindingPredictor:
    """High-level API for Fc domain binding prediction."""

    BINDING_RANGES = {
        "FcgrIIIA": (-14, -6),
        "FcgrIIB": (-13, -5),
        "FcgrIII": (-12, -4),
    }

    def __init__(self, model_path: Optional[str] = None, device: str = "cpu"):
        self.device = device
        self.model = FcDomainGCN().to(device)
        self.graph_builder = FcGraphBuilder()

        if model_path and Path(model_path).exists():
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info("Loaded model from %s", model_path)
        else:
            logger.warning("Model not initialized; load weights before inference")

    def predict_from_structure(self, pdb_path: str, chain: str = "H") -> Dict[str, float]:
        data = self.graph_builder.build_from_pdb(pdb_path, chain=chain)
        data = data.to(self.device)

        self.model.eval()
        with torch.no_grad():
            batch_idx = torch.zeros(data.x.shape[0], dtype=torch.long, device=self.device)
            affinity = self.model(data.x, data.edge_index, batch_idx)

        return {
            "binding_energy_kcal_mol": float(affinity.item()),
            "predicted_kd_nM": float(self._energy_to_kd(affinity.item())),
        }

    @staticmethod
    def _energy_to_kd(delta_g: float, temp_kelvin: float = 298.15) -> float:
        r_const = 0.001987
        kd_m = np.exp(delta_g / (r_const * temp_kelvin))
        return kd_m * 1e9
