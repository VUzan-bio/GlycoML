"""Phase 2 dataset utilities for lectin-glycan binding prediction."""

from __future__ import annotations

import ast
import difflib
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

try:  # pragma: no cover
    from torch_geometric.data import Batch, Data
except Exception:  # pragma: no cover
    Batch = None
    Data = None

try:  # pragma: no cover
    from Bio.PDB import MMCIFParser
except Exception:  # pragma: no cover
    MMCIFParser = None

from glycoml.shared.esm2_embedder import ESM2Embedder
from glycoml.shared.glycan_tokenizer import GlycanTokenizer, canonicalize_smiles, iupac_to_smiles
from .models.glycan_encoder import MONOSAC_TYPES, smiles_to_graph


LOGGER = logging.getLogger(__name__)


UNIFIED_MAP = {
    "lectinid": "lectin_id",
    "lectin_id": "lectin_id",
    "lectinname": "lectin_name",
    "lectin_name": "lectin_name",
    "lectinorigin": "lectin_origin",
    "lectin_origin": "lectin_origin",
    "lectinspecies": "lectin_species",
    "lectin_species": "lectin_species",
    "lectinuniprot": "lectin_uniprot",
    "lectin_uniprot": "lectin_uniprot",
    "lectinpdb": "lectin_pdb",
    "lectin_pdb": "lectin_pdb",
    "glycanid": "glycan_id",
    "glycan_id": "glycan_id",
    "glycaniupac": "glycan_iupac",
    "glycan_iupac": "glycan_iupac",
    "glycanglytoucanid": "glycan_glytoucan_id",
    "glycan_glytoucan_id": "glycan_glytoucan_id",
    "glycanmonosaccomposition": "glycan_monosac_composition",
    "glycan_monosac_composition": "glycan_monosac_composition",
    "bindingvalue": "binding_value",
    "binding_value": "binding_value",
    "bindingunit": "binding_unit",
    "binding_unit": "binding_unit",
    "bindingsource": "binding_source",
    "binding_source": "binding_source",
    "bindingconfidence": "binding_confidence",
    "binding_confidence": "binding_confidence",
    "bindingsourcemetadatajson": "binding_source_metadata_json",
    "binding_source_metadata_json": "binding_source_metadata_json",
}

UNILECTIN_MAP = {
    "lectinid": "lectin_id",
    "lectin_id": "lectin_id",
    "pdb": "pdb",
    "proteinname": "protein_name",
    "protein_name": "protein_name",
    "uniprot": "uniprot",
    "origin": "origin",
    "originspecies": "origin",
    "species": "species",
    "speciesid": "species_id",
    "species_id": "species_id",
    "fold": "fold",
    "class": "class",
    "family": "family",
    "resolution": "resolution",
    "ligand": "ligand",
    "monosac": "monosac",
    "iupac": "iupac",
    "glycoct": "glycoct",
    "glytoucanid": "glytoucan_id",
    "glytoucan_id": "glytoucan_id",
    "timestamp": "timestamp",
}

CFG_MAP = {
    "experimentid": "experiment_id",
    "experiment_id": "experiment_id",
    "arrayversion": "array_version",
    "array_version": "array_version",
    "glycanid": "glycan_id",
    "glycan_id": "glycan_id",
    "glycaniupac": "glycan_iupac",
    "glycan_iupac": "glycan_iupac",
    "glycanglytoucanid": "glytoucan_id",
    "glytoucan_id": "glytoucan_id",
    "lectinsamplename": "lectin_name",
    "samplename": "lectin_name",
    "sample_name": "lectin_name",
    "lectin_name": "lectin_name",
    "rfuraw": "rfu_raw",
    "rfu_raw": "rfu_raw",
    "rfunormalized": "rfu_normalized",
    "rfu_normalized": "rfu_normalized",
    "normalization": "normalization",
    "status": "status",
    "investigator": "investigator",
    "conclusive": "conclusive",
}

LIGAND_MAP = {
    "glycanid": "glycan_id",
    "glycan_id": "glycan_id",
    "iupac": "iupac",
    "smiles": "smiles",
    "glytoucanid": "glytoucan_id",
    "glytoucan_id": "glytoucan_id",
    "monosac_composition": "monosac_composition",
    "monosaccomposition": "monosac_composition",
    "branch_count": "branch_count",
    "charge": "charge",
    "mass": "mass",
}


def _normalize_columns(df: pd.DataFrame, mapping: Dict[str, str]) -> pd.DataFrame:
    rename: Dict[str, str] = {}
    for col in df.columns:
        key = re.sub(r"[^a-z0-9]", "", col.lower())
        if key in mapping:
            rename[col] = mapping[key]
    return df.rename(columns=rename)


def normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _parse_monosac(value: object) -> Dict[str, float]:
    if value is None or value == "":
        return {}
    if isinstance(value, dict):
        return {str(k): float(v) for k, v in value.items()}
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, dict):
            return {str(k): float(v) for k, v in parsed.items()}
    except Exception:
        return {}
    return {}


def _monosac_vector(value: object) -> List[float]:
    parsed = _parse_monosac(value)
    return [float(parsed.get(token, 0.0)) for token in MONOSAC_TYPES]


def _fuzzy_match(value: str, candidates: List[str], cutoff: float) -> Optional[str]:
    if not value:
        return None
    matches = difflib.get_close_matches(value, candidates, n=1, cutoff=cutoff)
    return matches[0] if matches else None


def load_unified(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = _normalize_columns(df, UNIFIED_MAP)
    df["glytoucan_id"] = df.get("glycan_glytoucan_id", "")
    return df


def load_unilectin(path: Optional[Path]) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return _normalize_columns(df, UNILECTIN_MAP)


def load_cfg(path: Optional[Path]) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return _normalize_columns(df, CFG_MAP)


def load_ligands(path: Optional[Path]) -> pd.DataFrame:
    if not path or not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return _normalize_columns(df, LIGAND_MAP)


def merge_phase2_data(
    unified_path: Path,
    unilectin_path: Optional[Path] = None,
    cfg_path: Optional[Path] = None,
    ligands_path: Optional[Path] = None,
    fuzzy_cutoff: float = 0.85,
) -> pd.DataFrame:
    unified = load_unified(unified_path)
    unilectin = load_unilectin(unilectin_path)
    cfg = load_cfg(cfg_path)
    ligands = load_ligands(ligands_path)

    if not unilectin.empty:
        if "glytoucan_id" in unified.columns:
            unified = unified.merge(
                unilectin,
                how="left",
                left_on=["lectin_id", "glytoucan_id"],
                right_on=["lectin_id", "glytoucan_id"],
                suffixes=("", "_unilectin"),
            )
        else:
            unified = unified.merge(
                unilectin,
                how="left",
                left_on=["lectin_id"],
                right_on=["lectin_id"],
                suffixes=("", "_unilectin"),
            )

    if not cfg.empty:
        cfg_names = sorted(cfg["lectin_name"].dropna().unique().tolist())
        cfg["lectin_name_match"] = cfg["lectin_name"].apply(lambda x: _fuzzy_match(str(x), cfg_names, fuzzy_cutoff))
        if "glytoucan_id" in cfg.columns:
            if "glycan_glytoucan_id" in unified.columns:
                unified = unified.merge(
                    cfg,
                    how="left",
                    left_on=["lectin_name", "glycan_glytoucan_id"],
                    right_on=["lectin_name_match", "glytoucan_id"],
                    suffixes=("", "_cfg"),
                )
            elif "glytoucan_id" in unified.columns:
                unified = unified.merge(
                    cfg,
                    how="left",
                    left_on=["lectin_name", "glytoucan_id"],
                    right_on=["lectin_name_match", "glytoucan_id"],
                    suffixes=("", "_cfg"),
                )
            else:
                unified = unified.merge(
                    cfg,
                    how="left",
                    left_on=["lectin_name"],
                    right_on=["lectin_name_match"],
                    suffixes=("", "_cfg"),
                )
        else:
            unified = unified.merge(
                cfg,
                how="left",
                left_on=["lectin_name"],
                right_on=["lectin_name_match"],
                suffixes=("", "_cfg"),
            )

    if not ligands.empty and "glycan_glytoucan_id" in unified.columns:
        ligand_map = {
            "glytoucanid": "glytoucan_id",
            "glytoucan_id": "glytoucan_id",
            "iupac": "iupac",
            "glycoct": "glycoct",
            "monosaccomposition": "monosac_composition",
            "monosac_composition": "monosac_composition",
        }
        ligand_cols = {}
        for col in ligands.columns:
            key = normalize_col(col)
            if key in ligand_map:
                ligand_cols[col] = ligand_map[key]
        ligands_norm = ligands.rename(columns=ligand_cols)
        if "glytoucan_id" in ligands_norm.columns:
            needed = ["glytoucan_id"]
            for opt in ("iupac", "glycoct", "monosac_composition"):
                if opt in ligands_norm.columns:
                    needed.append(opt)
            ligands_subset = ligands_norm[needed].copy()
            unified = unified.merge(
                ligands_subset,
                left_on="glycan_glytoucan_id",
                right_on="glytoucan_id",
                how="left",
                suffixes=("", "_ligand"),
            )

    return unified


def build_labels(
    df: pd.DataFrame,
    rfu_threshold: float = 2000.0,
    nan_handling: str = "zero",
) -> pd.DataFrame:
    df = df.copy()
    rfu = df["rfu_normalized"].fillna(df["binding_value"]).astype(float)
    if nan_handling == "drop":
        df = df[~rfu.isna()].copy()
    rfu = rfu.fillna(0.0)
    df["rfu_value"] = rfu
    df["label_bin"] = ((rfu >= rfu_threshold) | (df.get("conclusive", False).fillna(False))).astype(float)
    df["label_reg"] = np.log10(rfu + 1.0)
    return df


def build_family_vocab(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    folds = sorted(df.get("fold", pd.Series()).dropna().unique())
    classes = sorted(df.get("class", pd.Series()).dropna().unique())
    families = sorted(df.get("family", pd.Series()).dropna().unique())
    return {
        "fold": {name: idx for idx, name in enumerate(folds)},
        "class": {name: idx for idx, name in enumerate(classes)},
        "family": {name: idx for idx, name in enumerate(families)},
    }


def family_onehot(row: pd.Series, vocab: Dict[str, Dict[str, int]]) -> torch.Tensor:
    fold_map = vocab["fold"]
    class_map = vocab["class"]
    family_map = vocab["family"]
    vec = torch.zeros(len(fold_map) + len(class_map) + len(family_map), dtype=torch.float32)
    if row.get("fold") in fold_map:
        vec[fold_map[row["fold"]]] = 1.0
    if row.get("class") in class_map:
        vec[len(fold_map) + class_map[row["class"]]] = 1.0
    if row.get("family") in family_map:
        vec[len(fold_map) + len(class_map) + family_map[row["family"]]] = 1.0
    return vec


class SequenceStore:
    """Cache UniProt sequences on disk."""

    def __init__(self, cache_dir: Path, allow_network: bool = False, rate_limit: float = 0.2) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.allow_network = allow_network
        self.rate_limit = rate_limit
        self._last_request = 0.0

    def _throttle(self) -> None:
        delta = time.time() - self._last_request
        if delta < self.rate_limit:
            time.sleep(self.rate_limit - delta)

    def _cache_path(self, uniprot_id: str) -> Path:
        return self.cache_dir / f"{uniprot_id}.fasta"

    def get_sequence(self, uniprot_id: str) -> Optional[str]:
        uniprot_id = str(uniprot_id).strip()
        if not uniprot_id:
            return None
        cache_path = self._cache_path(uniprot_id)
        if cache_path.exists():
            lines = cache_path.read_text(encoding="utf-8").splitlines()
            return "".join(line.strip() for line in lines if not line.startswith(">")) or None
        if not self.allow_network:
            return None
        import requests

        url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
        self._throttle()
        resp = requests.get(url, timeout=20)
        self._last_request = time.time()
        if resp.status_code != 200:
            return None
        cache_path.write_text(resp.text, encoding="utf-8")
        lines = resp.text.splitlines()
        return "".join(line.strip() for line in lines if not line.startswith(">")) or None


def load_structure_coords(pdb_id: str, uniprot_id: str, cache_dir: Path, allow_network: bool) -> Optional[torch.Tensor]:
    if MMCIFParser is None:
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    candidates: List[Path] = []
    if pdb_id:
        candidates.append(cache_dir / f"{pdb_id}.cif")
    if uniprot_id:
        candidates.append(cache_dir / f"AF-{uniprot_id}-F1-model_v4.cif")

    for path in candidates:
        if path.exists():
            return _parse_cif_coords(path)

    if not allow_network:
        return None

    import requests

    if pdb_id:
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        target = cache_dir / f"{pdb_id}.cif"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            target.write_bytes(resp.content)
            return _parse_cif_coords(target)

    if uniprot_id:
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v4.cif"
        target = cache_dir / f"AF-{uniprot_id}-F1-model_v4.cif"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            target.write_bytes(resp.content)
            return _parse_cif_coords(target)
    return None


def _parse_cif_coords(path: Path) -> Optional[torch.Tensor]:
    parser = MMCIFParser(QUIET=True)
    try:
        structure = parser.get_structure(path.stem, str(path))
    except Exception:
        return None
    coords: List[List[float]] = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    atom = residue["CA"]
                    coords.append(atom.get_coord().tolist())
    if not coords:
        return None
    return torch.tensor(coords, dtype=torch.float32)


@dataclass
class Phase2DatasetConfig:
    use_structure: bool = False
    glycan_encoder: str = "graph"
    require_labels: bool = False


class LectinGlycanDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        esm_embedder: ESM2Embedder,
        glycan_tokenizer: GlycanTokenizer,
        cache_dir: Path,
        config: Phase2DatasetConfig,
        family_vocab: Optional[Dict[str, Dict[str, int]]] = None,
        species_map: Optional[Dict[int, int]] = None,
        allow_network: bool = False,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.esm_embedder = esm_embedder
        self.glycan_tokenizer = glycan_tokenizer
        self.cache_dir = cache_dir
        self.config = config
        self.allow_network = allow_network
        self.sequence_store = SequenceStore(cache_dir / "sequences", allow_network=allow_network)
        self.structure_cache = cache_dir / "pdb"
        self.family_vocab = family_vocab or build_family_vocab(df)
        self.family_dim = (
            len(self.family_vocab["fold"])
            + len(self.family_vocab["class"])
            + len(self.family_vocab["family"])
        )
        if species_map is None:
            species_ids = sorted(df.get("species_id", pd.Series()).dropna().unique().tolist())
            self.species_map = {int(val): idx + 1 for idx, val in enumerate(species_ids)}
        else:
            self.species_map = species_map

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        row = self.df.iloc[idx]
        uniprot_id = row.get("lectin_uniprot", "")
        if pd.isna(uniprot_id):
            uniprot_id = ""
        sequence = self.sequence_store.get_sequence(uniprot_id)
        if not sequence:
            sequence = str(row.get("lectin_sequence", "") or "")
        tokens = self.esm_embedder.embed_sequence(sequence)
        mask = torch.ones(tokens.shape[0], dtype=torch.bool)
        family_feat = family_onehot(row, self.family_vocab)
        species_val = row.get("species_id")
        species_idx = 0
        if species_val is not None and not pd.isna(species_val):
            species_idx = self.species_map.get(int(species_val), 0)

        structure_coords = None
        if self.config.use_structure:
            structure_coords = load_structure_coords(
                str(row.get("lectin_pdb") or ""),
                str(row.get("lectin_uniprot") or ""),
                self.structure_cache,
                self.allow_network,
            )

        meta = torch.tensor(
            _monosac_vector(row.get("monosac_composition") or row.get("glycan_monosac_composition"))
            + [float(row.get("branch_count") or 0.0), float(row.get("charge") or 0.0)],
            dtype=torch.float32,
        )

        glycan_tokens = None
        glycan_mask = None
        glycan_graph = None
        glycan_smiles = row.get("smiles") or ""
        if pd.isna(glycan_smiles):
            glycan_smiles = ""
        glycan_smiles = canonicalize_smiles(str(glycan_smiles) or "")
        if not glycan_smiles:
            glycan_smiles = iupac_to_smiles(str(row.get("glycan_iupac") or "")) or ""

        if self.config.glycan_encoder == "graph":
            glycan_graph = smiles_to_graph(glycan_smiles, meta=meta)
            if glycan_graph is None and Data is not None:
                glycan_graph = Data(z=torch.tensor([6], dtype=torch.long), pos=torch.zeros((1, 3)), meta=meta)
        else:
            tokens_list = self.glycan_tokenizer.encode(str(row.get("glycan_iupac") or ""))
            glycan_tokens = torch.tensor(tokens_list, dtype=torch.long)
            glycan_mask = torch.ones(len(tokens_list), dtype=torch.bool)

        label_bin_val = row.get("label_bin", 0.0)
        label_reg_val = row.get("label_reg", 0.0)
        if label_bin_val is None or pd.isna(label_bin_val):
            label_bin_val = 0.0
        if label_reg_val is None or pd.isna(label_reg_val):
            label_reg_val = 0.0
        label_bin = float(label_bin_val)
        label_reg = float(label_reg_val)

        return {
            "lectin_tokens": tokens,
            "lectin_mask": mask,
            "family_features": family_feat,
            "species_idx": torch.tensor(species_idx, dtype=torch.long),
            "structure_coords": structure_coords,
            "glycan_tokens": glycan_tokens,
            "glycan_mask": glycan_mask,
            "glycan_graph": glycan_graph,
            "glycan_meta": meta,
            "label_bin": torch.tensor(label_bin, dtype=torch.float32),
            "label_reg": torch.tensor(label_reg, dtype=torch.float32),
            "lectin_id": str(row.get("lectin_id")),
            "glycan_id": str(row.get("glycan_id")),
            "lectin_name": str(row.get("lectin_name") or ""),
            "glycan_iupac": str(row.get("glycan_iupac") or ""),
            "glytoucan_id": str(row.get("glytoucan_id") or row.get("glycan_glytoucan_id") or ""),
        }


def collate_batch(batch: List[Dict[str, object]], use_graph: bool, use_structure: bool) -> Dict[str, object]:
    lectin_tokens = torch.nn.utils.rnn.pad_sequence([b["lectin_tokens"] for b in batch], batch_first=True)
    lectin_mask = torch.nn.utils.rnn.pad_sequence([b["lectin_mask"] for b in batch], batch_first=True, padding_value=0).bool()
    family = torch.stack([b["family_features"] for b in batch])
    species = torch.stack([b["species_idx"] for b in batch])
    labels_bin = torch.stack([b["label_bin"] for b in batch])
    labels_reg = torch.stack([b["label_reg"] for b in batch])

    structure_batch = None
    if use_structure and Batch is not None and Data is not None:
        coords_list = [b["structure_coords"] for b in batch]
        data_list = []
        for coords in coords_list:
            if coords is None:
                coords = torch.zeros((1, 3), dtype=torch.float32)
            z = torch.full((coords.shape[0],), 6, dtype=torch.long)
            data_list.append(Data(z=z, pos=coords))
        structure_batch = Batch.from_data_list(data_list)

    if use_graph:
        if Batch is None:
            raise RuntimeError("torch_geometric is required for graph batching")
        graph_list = [b["glycan_graph"] for b in batch]
        if any(graph is None for graph in graph_list):
            raise RuntimeError("Missing glycan graphs in graph mode; check RDKit/torch_geometric installation")
        glycan_batch = Batch.from_data_list(graph_list)
        glycan_tokens = None
        glycan_mask = None
        glycan_meta = None
    else:
        glycan_tokens = torch.nn.utils.rnn.pad_sequence(
            [b["glycan_tokens"] for b in batch], batch_first=True, padding_value=0
        )
        glycan_mask = torch.nn.utils.rnn.pad_sequence(
            [b["glycan_mask"] for b in batch], batch_first=True, padding_value=0
        ).bool()
        glycan_meta = torch.stack([b["glycan_meta"] for b in batch])
        glycan_batch = None

    return {
        "lectin_tokens": lectin_tokens,
        "lectin_mask": lectin_mask,
        "family_features": family,
        "species_idx": species,
        "structure_batch": structure_batch,
        "glycan_tokens": glycan_tokens,
        "glycan_mask": glycan_mask,
        "glycan_graph": glycan_batch,
        "glycan_meta": glycan_meta,
        "labels_bin": labels_bin,
        "labels_reg": labels_reg,
        "lectin_ids": [b["lectin_id"] for b in batch],
        "glycan_ids": [b["glycan_id"] for b in batch],
        "lectin_names": [b["lectin_name"] for b in batch],
        "glycan_iupac": [b["glycan_iupac"] for b in batch],
        "glytoucan_ids": [b["glytoucan_id"] for b in batch],
    }


def stratified_group_split(
    df: pd.DataFrame,
    label_col: str,
    seed: int,
    train_size: float = 0.8,
    val_size: float = 0.1,
) -> Tuple[List[int], List[int], List[int]]:
    from sklearn.model_selection import StratifiedGroupKFold

    labels = df[label_col].values
    groups = (df["lectin_id"].astype(str) + "||" + df["glycan_id"].astype(str)).values
    indices = np.arange(len(df))
    splitter = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    train_idx, test_idx = next(splitter.split(indices, labels, groups))
    remaining = df.iloc[train_idx]
    train_idx2, val_idx = next(
        StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(
            np.arange(len(remaining)),
            remaining[label_col].values,
            (remaining["lectin_id"].astype(str) + "||" + remaining["glycan_id"].astype(str)).values,
        )
    )
    train_idx = remaining.iloc[train_idx2].index.tolist()
    val_idx = remaining.iloc[val_idx].index.tolist()
    return train_idx, val_idx, test_idx.tolist()
