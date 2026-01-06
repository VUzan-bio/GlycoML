"""Glycan encoders for graph and token representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import hashlib

import torch
from torch import nn

try:  # pragma: no cover
    from rdkit import Chem
    from rdkit.Chem import AllChem
except Exception:  # pragma: no cover
    Chem = None
    AllChem = None

try:  # pragma: no cover
    from torch_geometric.data import Data, Batch
    from torch_geometric.nn import global_max_pool, global_mean_pool
    from torch_geometric.nn.models import SchNet
except Exception:  # pragma: no cover
    Data = None
    Batch = None
    SchNet = None
    global_mean_pool = None
    global_max_pool = None


MONOSAC_TYPES = ["Glc", "GlcNAc", "Gal", "GalNAc", "Man", "Fuc", "Neu5Ac", "Neu5Gc", "Xyl"]


@dataclass
class GlycanGraphConfig:
    hidden_dim: int = 128
    out_dim: int = 256
    interactions: int = 6
    cutoff: float = 5.0
    meta_dim: int = 11


@dataclass
class GlycanTokenConfig:
    vocab_size: int = 128
    embed_dim: int = 128
    num_layers: int = 4
    num_heads: int = 4
    ff_dim: int = 512
    dropout: float = 0.1
    meta_dim: int = 11


def _stable_hash(value: str) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _atom_features(atom) -> List[int]:
    return [
        atom.GetAtomicNum(),
        atom.GetTotalDegree(),
        atom.GetFormalCharge(),
        int(atom.GetIsAromatic()),
    ]


def smiles_to_graph(smiles: str, meta: Optional[torch.Tensor] = None) -> Optional[Data]:
    if Chem is None or Data is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    if AllChem is not None:
        try:
            AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        except Exception:
            pass
    if mol.GetNumConformers() == 0:
        pos = torch.zeros((mol.GetNumAtoms(), 3), dtype=torch.float32)
    else:
        conf = mol.GetConformer()
        coords = [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())]
        pos = torch.tensor([[p.x, p.y, p.z] for p in coords], dtype=torch.float32)
    z = torch.tensor([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=torch.long)
    data = Data(z=z, pos=pos)
    if meta is not None:
        if isinstance(meta, torch.Tensor) and meta.dim() == 1:
            meta = meta.unsqueeze(0)
        data.meta = meta
    return data


class GlycanGraphEncoder(nn.Module):
    """SE(3)-equivariant graph encoder using SchNet."""

    def __init__(self, config: GlycanGraphConfig) -> None:
        super().__init__()
        if SchNet is None:
            raise ImportError("torch_geometric is required for graph encoding")
        self.config = config
        self.model = SchNet(
            hidden_channels=config.hidden_dim,
            num_filters=config.hidden_dim,
            num_interactions=config.interactions,
            out_channels=config.out_dim,
            cutoff=config.cutoff,
        )
        self.proj = nn.Linear(config.out_dim + config.meta_dim, config.out_dim)

    def forward(self, batch) -> torch.Tensor:
        out = self.model(batch.z, batch.pos, batch.batch)
        if hasattr(batch, "meta"):
            meta = batch.meta
            if isinstance(meta, torch.Tensor) and meta.dim() == 1:
                meta = meta.view(out.size(0), -1)
            if isinstance(meta, torch.Tensor):
                meta = meta.to(out.device)
            out = torch.cat([out, meta], dim=-1)
        return self.proj(out)


class GlycanTokenEncoder(nn.Module):
    """Token-based glycan encoder using a Transformer."""

    def __init__(self, config: GlycanTokenConfig) -> None:
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.embed_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.embed_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.proj = nn.Linear(config.embed_dim + config.meta_dim, 256)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        meta: Optional[torch.Tensor] = None,
        return_tokens: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        emb = self.embed(tokens)
        key_padding_mask = ~mask
        emb = self.encoder(emb, src_key_padding_mask=key_padding_mask)
        pooled = (emb * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0).unsqueeze(-1)
        if meta is not None:
            pooled = torch.cat([pooled, meta], dim=-1)
        pooled = self.proj(pooled)
        if return_tokens:
            return pooled, emb
        return pooled


@dataclass
class GlycanFingerprintConfig:
    radius: int = 2
    n_bits: int = 2048
    include_physchem: bool = True
    include_iupac: bool = True


def _fallback_fingerprint(smiles: str, n_bits: int) -> torch.Tensor:
    vec = torch.zeros(n_bits, dtype=torch.float32)
    if not smiles:
        return vec
    for i in range(len(smiles) - 1):
        token = smiles[i : i + 2]
        idx = _stable_hash(token) % n_bits
        vec[idx] = 1.0
    return vec


def _rdkit_fingerprint(smiles: str, radius: int, n_bits: int) -> torch.Tensor:
    if Chem is None or AllChem is None:
        return _fallback_fingerprint(smiles, n_bits)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _fallback_fingerprint(smiles, n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = list(fp)
    return torch.tensor(arr, dtype=torch.float32)


def count_monosaccharides(iupac: Optional[str]) -> List[float]:
    if not iupac:
        return [0.0 for _ in MONOSAC_TYPES]
    counts: List[float] = []
    for token in MONOSAC_TYPES:
        counts.append(float(iupac.count(token)))
    return counts


class GlycanFingerprintEncoder:
    def __init__(self, config: GlycanFingerprintConfig):
        self.config = config
        self._cache: Dict[str, torch.Tensor] = {}
        self.iupac_dim = len(MONOSAC_TYPES) if config.include_iupac else 0
        self.feature_size = config.n_bits + self.iupac_dim

    def encode(self, smiles: str, iupac: Optional[str] = None) -> torch.Tensor:
        key = f"{smiles}|{iupac or ''}"
        if key in self._cache:
            return self._cache[key]
        fp = _rdkit_fingerprint(smiles, self.config.radius, self.config.n_bits)
        features = [fp]
        if self.config.include_iupac:
            features.append(torch.tensor(count_monosaccharides(iupac), dtype=torch.float32))
        vector = torch.cat(features)
        self._cache[key] = vector
        return vector


class GlycanGCNEncoder(nn.Module):
    """Compatibility wrapper for graph encoding."""

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3, out_dim: int = 256):
        super().__init__()
        self.encoder = GlycanGraphEncoder(
            GlycanGraphConfig(hidden_dim=hidden_dim, out_dim=out_dim, interactions=num_layers)
        )

    def forward(self, batch) -> torch.Tensor:
        if isinstance(batch, list) and Batch is not None:
            graphs = [smiles_to_graph(smiles) for smiles in batch]
            graphs = [g for g in graphs if g is not None]
            if not graphs:
                raise ValueError("No valid glycan graphs to encode")
            batch = Batch.from_data_list(graphs)
        return self.encoder(batch)
