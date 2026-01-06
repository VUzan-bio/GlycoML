"""IUPAC and SMILES helpers for glycans."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None

try:  # pragma: no cover - optional dependency
    import glypy
except Exception:  # pragma: no cover
    glypy = None


class GlycanTokenizer:
    """Simple IUPAC tokenizer for glycans."""

    TOKEN_REGEX = re.compile(r"[A-Za-z]+\d*|\d+|\(|\)|\[|\]|\-|\+|,|:|;|\.|/")

    def __init__(self) -> None:
        self.token_to_id: Dict[str, int] = {"<pad>": 0, "<unk>": 1}
        self.id_to_token: Dict[int, str] = {0: "<pad>", 1: "<unk>"}

    def build(self, texts: Iterable[str]) -> None:
        for text in texts:
            for token in self.tokenize(text):
                if token not in self.token_to_id:
                    idx = len(self.token_to_id)
                    self.token_to_id[token] = idx
                    self.id_to_token[idx] = token

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []
        return self.TOKEN_REGEX.findall(text)

    def encode(self, text: str) -> List[int]:
        tokens = self.tokenize(text)
        return [self.token_to_id.get(tok, 1) for tok in tokens]

    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.token_to_id, indent=2), encoding="utf-8")

    def load(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.token_to_id = {str(k): int(v) for k, v in data.items()}
        self.id_to_token = {int(v): str(k) for k, v in self.token_to_id.items()}


def canonicalize_smiles(smiles: str) -> Optional[str]:
    """Return canonical RDKit SMILES when available."""
    if not smiles:
        return None
    if Chem is None:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)


def iupac_to_smiles(iupac: str) -> Optional[str]:
    """Attempt to convert IUPAC to SMILES using available parsers."""
    if not iupac:
        return None

    if Chem is not None:
        mol = Chem.MolFromSmiles(iupac)
        if mol is not None:
            return Chem.MolToSmiles(mol, canonical=True)

    if glypy is None:
        return None

    try:
        glycan = glypy.io.iupac.loads(iupac)
    except Exception:
        return None

    try:
        glycoct = glypy.io.glycoct.dumps(glycan)
    except Exception:
        return None

    if Chem is None:
        return None

    mol = Chem.MolFromSmiles(glycoct)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True)
