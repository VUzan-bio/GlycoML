"""IUPAC condensed glycan parser to simple graph representation."""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

import torch
from torch_geometric.data import Data


MONOSACCHARIDE_TYPES: Dict[str, int] = {
    "GlcNAc": 0,
    "GalNAc": 1,
    "Neu5Ac": 2,
    "Neu5Gc": 3,
    "Glc": 4,
    "Gal": 5,
    "Man": 6,
    "Fuc": 7,
    "Xyl": 8,
    "KDN": 9,
    "GlcA": 10,
    "IdoA": 11,
    "Rha": 12,
    "Ara": 13,
    "Unknown": 14,
}

ANOMER_MAP = {
    "a": 0,
    "b": 1,
    "\u03b1": 0,
    "\u03b2": 1,
}

TOKENS_SORTED = sorted(MONOSACCHARIDE_TYPES.keys(), key=len, reverse=True)
TOKEN_PATTERN = re.compile("|".join(re.escape(tok) for tok in TOKENS_SORTED if tok != "Unknown"))


def clean_iupac(value: str) -> str:
    cleaned = value.strip()
    cleaned = re.sub(r"-Sp\d+$", "", cleaned)
    cleaned = cleaned.replace("┬á", "")
    return cleaned


def is_composition(value: str) -> bool:
    if any(ch in value for ch in ("-", "(", ")", "[", "]")):
        return False
    return bool(re.match(r"^([A-Za-z0-9]+\d*)+$", value))


def extract_composition_tokens(value: str) -> List[str]:
    tokens: List[str] = []
    idx = 0
    while idx < len(value):
        match = TOKEN_PATTERN.match(value, idx)
        if not match:
            idx += 1
            continue
        token = match.group(0)
        idx = match.end()
        count_str = ""
        while idx < len(value) and value[idx].isdigit():
            count_str += value[idx]
            idx += 1
        count = int(count_str) if count_str else 1
        tokens.extend([token] * count)
    return tokens


def extract_ordered_tokens(value: str) -> List[Tuple[str, int, int]]:
    pattern = re.compile(r"([A-Z][a-zA-Z0-9]*)([\u03b1\u03b2ab]?)(\d)?-?(\d)?")
    matches = pattern.findall(value)
    tokens: List[Tuple[str, int, int]] = []
    for mono_name, anomer, pos_from, pos_to in matches:
        if not mono_name:
            continue
        anomer_idx = ANOMER_MAP.get(anomer, 2)
        bond_pos = int(pos_to) if pos_to and pos_to.isdigit() else 2
        tokens.append((mono_name, anomer_idx, bond_pos))
    return tokens


def parse_iupac_condensed(iupac_string: str) -> Data:
    iupac_string = clean_iupac(iupac_string)
    if not iupac_string:
        return _single_node("Unknown")

    if is_composition(iupac_string):
        tokens = extract_composition_tokens(iupac_string)
        return _build_linear_graph([(token, 0, 2) for token in tokens])

    tokens = extract_ordered_tokens(iupac_string)
    if not tokens:
        return _single_node("Unknown")
    return _build_linear_graph(tokens)


def _single_node(token: str) -> Data:
    mono_type = MONOSACCHARIDE_TYPES.get(token, MONOSACCHARIDE_TYPES["Unknown"])
    x = torch.tensor([[mono_type, 0, 1, 0, 0]], dtype=torch.float32)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, 3), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def _build_linear_graph(tokens: List[Tuple[str, int, int]]) -> Data:
    nodes = []
    edges = []
    edge_attrs = []
    for idx, (token, anomer, bond_pos) in enumerate(tokens):
        mono_type = MONOSACCHARIDE_TYPES.get(token, MONOSACCHARIDE_TYPES["Unknown"])
        nodes.append([mono_type, float(anomer), 1.0 if idx == 0 else 0.0, float(idx), 0.0])
        if idx > 0:
            edges.append([idx - 1, idx])
            edge_attrs.append([float(anomer), float(bond_pos), 0.0])

    x = torch.tensor(nodes, dtype=torch.float32)
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
