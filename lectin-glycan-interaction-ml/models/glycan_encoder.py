"""Glycan encoders for fingerprints or graph convolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence
import hashlib
import re
import warnings

import torch
from torch import nn

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski
except ImportError:  # pragma: no cover
    Chem = None
    AllChem = None
    Descriptors = None
    Crippen = None
    Lipinski = None

try:
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, global_mean_pool
except ImportError:  # pragma: no cover
    Data = None
    GCNConv = None
    global_mean_pool = None


COMMON_MONOSACCHARIDES = [
    "Gal",
    "Glc",
    "GlcNAc",
    "GalNAc",
    "Man",
    "ManNAc",
    "Fuc",
    "Neu5Ac",
    "Neu5Gc",
    "KDN",
    "Kdo",
    "Xyl",
    "GlcA",
    "GalA",
    "IdoA",
    "Rha",
    "Ara",
]

# IUPAC 2-Carb-38 / SNFG (Varki et al., Glycobiology 2015) condensed
# abbreviations. Legacy aliases ("NeuAc" -> "Neu5Ac", "NeuGc" -> "Neu5Gc") are
# accepted on input and normalised to the canonical token.
_MONOSACCHARIDE_ALIASES = {
    "NeuAc": "Neu5Ac",
    "NeuGc": "Neu5Gc",
    "Sia": "Neu5Ac",
}

# Longest-first alternation so that e.g. "GalNAc" is matched as a whole token
# before "Gal" swallows the "Gal" prefix. The regex engine is greedy within an
# alternation only when options are ordered longest-first, so we enforce that
# here explicitly. No right-boundary lookahead is imposed because IUPAC
# condensed notation freely follows a residue with lowercase anomer letters
# (``a``/``b`` for alpha/beta) and longest-first ordering already prevents
# "Gal" from matching inside "GalNAc".
_GLYCAN_TOKEN_ALTERNATION = sorted(
    set(COMMON_MONOSACCHARIDES) | set(_MONOSACCHARIDE_ALIASES.keys()),
    key=lambda name: (-len(name), name),
)
GLYCAN_TOKEN_RE = re.compile(
    "(" + "|".join(re.escape(token) for token in _GLYCAN_TOKEN_ALTERNATION) + ")"
)


def tokenize_iupac(iupac: str) -> List[str]:
    """Tokenise an IUPAC-condensed glycan string into canonical monosaccharides.

    Follows SNFG (Varki et al., Glycobiology 2015) and IUPAC 2-Carb-38.
    - Aliases (NeuAc, NeuGc, Sia) are mapped to canonical Neu5Ac/Neu5Gc.
    - Residues embedded within longer names ("Gal" inside "GalNAc") are NOT
      counted, because the alternation is evaluated longest-first.
    """
    if not iupac:
        return []
    raw_tokens = GLYCAN_TOKEN_RE.findall(iupac)
    return [_MONOSACCHARIDE_ALIASES.get(tok, tok) for tok in raw_tokens]


@dataclass
class GlycanFingerprintConfig:
    radius: int = 2
    n_bits: int = 2048
    include_physchem: bool = True
    include_iupac: bool = True


def _stable_hash(value: str) -> int:
    digest = hashlib.md5(value.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


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
    """Compute a Morgan (ECFP-style) fingerprint that is sensitive to glycan
    stereochemistry.

    ``useChirality=True`` is mandatory for glycobiology: anomeric configuration
    (alpha vs beta), D/L epimerism, and 1->3 vs 1->6 linkages all manifest as
    different stereocentres at atom resolution. The caller is responsible for
    providing isomeric SMILES (RDKit ``MolToSmiles(mol, isomericSmiles=True)``
    or glycowork/glypy-generated SMILES). If chirality is stripped upstream,
    the fingerprint collapses alpha/beta sugars to the same bits.
    """
    if Chem is None or AllChem is None:
        return _fallback_fingerprint(smiles, n_bits)
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return _fallback_fingerprint(smiles, n_bits)
    if not any(atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for atom in mol.GetAtoms()):
        warnings.warn(
            "Morgan fingerprint computed with useChirality=True but input "
            "SMILES lacks stereochemistry descriptors (@, @@). "
            "Alpha/beta anomers will map to identical bits."
        )
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius, nBits=n_bits, useChirality=True, useFeatures=False
    )
    return torch.tensor(list(fp), dtype=torch.float32)


def _rdkit_physchem(smiles: str) -> List[float]:
    if Chem is None or Descriptors is None or Crippen is None or Lipinski is None:
        counts = [
            smiles.count("O"),
            smiles.count("N"),
            smiles.count("S"),
            smiles.count("P"),
            float(len(smiles)),
        ]
        return [float(x) for x in counts]
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    return [
        float(Lipinski.NumHDonors(mol)),
        float(Lipinski.NumHAcceptors(mol)),
        float(Descriptors.MolWt(mol)),
        float(Crippen.MolLogP(mol)),
        float(Lipinski.NumRotatableBonds(mol)),
    ]


def count_monosaccharides(iupac: Optional[str]) -> List[float]:
    """Return per-residue counts in the canonical vocabulary order.

    Uses ``tokenize_iupac`` so that sub-tokens embedded inside longer names do
    not inflate the count (e.g. ``GalNAc`` no longer contributes to ``Gal``,
    ``Neu5Ac`` no longer contributes to ``Ac``).
    """
    if not iupac:
        return [0.0 for _ in COMMON_MONOSACCHARIDES]
    tokens = tokenize_iupac(iupac)
    return [float(tokens.count(name)) for name in COMMON_MONOSACCHARIDES]


class GlycanFingerprintEncoder:
    def __init__(self, config: GlycanFingerprintConfig):
        self.config = config
        self._cache: Dict[str, torch.Tensor] = {}
        self.physchem_dim = 5 if config.include_physchem else 0
        self.iupac_dim = len(COMMON_MONOSACCHARIDES) if config.include_iupac else 0
        self.feature_size = config.n_bits + self.physchem_dim + self.iupac_dim

    def encode(self, smiles: str, iupac: Optional[str] = None) -> torch.Tensor:
        key = f"{smiles}|{iupac or ''}"
        if key in self._cache:
            return self._cache[key]
        fp = _rdkit_fingerprint(smiles, self.config.radius, self.config.n_bits)
        features = [fp]
        if self.config.include_physchem:
            features.append(torch.tensor(_rdkit_physchem(smiles), dtype=torch.float32))
        if self.config.include_iupac:
            features.append(torch.tensor(count_monosaccharides(iupac), dtype=torch.float32))
        vector = torch.cat(features)
        self._cache[key] = vector
        return vector


class GlycanGCNEncoder(nn.Module):
    """Optional graph convolution encoder. Requires rdkit and torch-geometric."""

    def __init__(self, hidden_dim: int = 128, num_layers: int = 3, out_dim: int = 512):
        super().__init__()
        if Chem is None or Data is None or GCNConv is None:
            raise ImportError("GlycanGCNEncoder requires rdkit and torch-geometric.")
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.out_dim = out_dim
        self.atom_dim = 8

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(self.atom_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, out_dim))

    def _atom_features(self, atom) -> List[float]:
        atomic_num = atom.GetAtomicNum()
        return [
            float(atomic_num),
            float(atom.GetIsAromatic()),
            float(atom.GetFormalCharge()),
            float(atom.GetTotalDegree()),
            float(int(atom.GetHybridization())),
            float(atom.GetTotalNumHs()),
            float(atom.IsInRing()),
            1.0,
        ]

    def _smiles_to_graph(self, smiles: str) -> Data:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")
        atom_features = [self._atom_features(atom) for atom in mol.GetAtoms()]
        edges = []
        for bond in mol.GetBonds():
            start = bond.GetBeginAtomIdx()
            end = bond.GetEndAtomIdx()
            edges.append((start, end))
            edges.append((end, start))
        if not edges:
            edges = [(0, 0)]
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.tensor(atom_features, dtype=torch.float32)
        return Data(x=x, edge_index=edge_index)

    def forward(self, smiles_list: Sequence[str]) -> torch.Tensor:
        graphs = [self._smiles_to_graph(smiles) for smiles in smiles_list]
        batch = []
        node_offset = 0
        xs = []
        edge_indices = []
        for idx, graph in enumerate(graphs):
            xs.append(graph.x)
            edge_indices.append(graph.edge_index + node_offset)
            node_offset += graph.x.shape[0]
            batch.extend([idx] * graph.x.shape[0])
        x = torch.cat(xs, dim=0)
        edge_index = torch.cat(edge_indices, dim=1)

        device = next(self.parameters()).device
        x = x.to(device)
        edge_index = edge_index.to(device)
        batch_tensor = torch.tensor(batch, dtype=torch.long, device=device)

        for conv in self.convs[:-1]:
            x = torch.relu(conv(x, edge_index))
        x = self.convs[-1](x, edge_index)
        pooled = global_mean_pool(x, batch_tensor)
        return pooled
