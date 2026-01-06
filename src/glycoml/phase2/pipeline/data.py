"""Data utilities for GlycoML Phase 2 training."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
import torch
from torch.utils.data import Dataset

from ...shared.esm2_embedder import ESM2Embedder
from ...shared.glycan_tokenizer import GlycanTokenizer


PHASE2_COLUMNS = [
    "lectin_id",
    "lectin_name",
    "lectin_origin",
    "lectin_species",
    "lectin_uniprot",
    "lectin_pdb",
    "glycan_id",
    "glycan_iupac",
    "glycan_glytoucan_id",
    "glycan_monosac_composition",
    "binding_value",
    "binding_unit",
    "binding_source",
    "binding_confidence",
    "binding_source_metadata_json",
]


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_phase2_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename_map = {col: col.strip() for col in df.columns}
    df = df.rename(columns=rename_map)
    missing = [col for col in PHASE2_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Phase 2 CSV missing columns: {missing}")
    return df


def merge_external_data(
    phase2_df: pd.DataFrame,
    unilectin_path: Optional[Path] = None,
    cfg_metadata_path: Optional[Path] = None,
) -> pd.DataFrame:
    df = phase2_df.copy()
    if unilectin_path and unilectin_path.exists():
        unilectin_df = pd.read_csv(unilectin_path)
        df = df.merge(
            unilectin_df,
            how="left",
            left_on=["lectin_uniprot", "glycan_iupac"],
            right_on=["uniprot", "iupac"],
            suffixes=("", "_unilectin"),
        )
    if cfg_metadata_path and cfg_metadata_path.exists():
        cfg_df = pd.read_csv(cfg_metadata_path)
        df["experiment_id"] = df["binding_source_metadata_json"].fillna("").apply(_extract_experiment_id)
        df = df.merge(
            cfg_df,
            how="left",
            left_on="experiment_id",
            right_on="experiment_id",
            suffixes=("", "_cfg"),
        )
    return df


def _extract_experiment_id(metadata_json: str) -> str:
    if not metadata_json:
        return ""
    try:
        data = json.loads(metadata_json)
    except json.JSONDecodeError:
        return ""
    return str(data.get("experiment_id") or "")


def normalize_binding_value(row: pd.Series) -> float:
    value = _safe_float(row.get("binding_value"), default=0.0)
    unit = str(row.get("binding_unit") or "").lower()
    if unit == "nm" and value > 0:
        kd_nm = value
        f_bound = 1.0 / (1.0 + kd_nm / 1000.0)
        return f_bound * 10000.0
    return value


def compute_label(row: pd.Series, mode: str, rfu_threshold: float) -> float:
    value = normalize_binding_value(row)
    if mode == "regression":
        return value
    confidence = str(row.get("binding_confidence") or "").lower()
    if confidence in {"high", "medium"}:
        return 1.0
    if value >= rfu_threshold:
        return 1.0
    return 0.0


def filter_phase2(df: pd.DataFrame, min_seq_len: int) -> pd.DataFrame:
    df = df.copy()
    df["lectin_uniprot"] = df["lectin_uniprot"].fillna("")
    df["glycan_iupac"] = df["glycan_iupac"].fillna("")
    df = df[df["glycan_iupac"] != ""]
    df = df[df["lectin_uniprot"] != ""]
    df = df.reset_index(drop=True)
    return df


class SequenceStore:
    """Cache UniProt sequences on disk."""

    def __init__(self, cache_dir: Path, allow_network: bool = True, rate_limit: float = 0.2) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.allow_network = allow_network
        self.rate_limit = rate_limit
        self._last_request = 0.0
        self.session = requests.Session()

    def _throttle(self) -> None:
        delta = time.time() - self._last_request
        if delta < self.rate_limit:
            time.sleep(self.rate_limit - delta)

    def _cache_path(self, uniprot_id: str) -> Path:
        return self.cache_dir / f"{uniprot_id}.fasta"

    def get_sequence(self, uniprot_id: str) -> Optional[str]:
        uniprot_id = uniprot_id.strip()
        if not uniprot_id:
            return None
        cache_path = self._cache_path(uniprot_id)
        if cache_path.exists():
            lines = cache_path.read_text(encoding="utf-8").splitlines()
            return "".join(line.strip() for line in lines if not line.startswith(">")) or None
        if not self.allow_network:
            return None
        url = f"https://www.uniprot.org/uniprot/{uniprot_id}.fasta"
        self._throttle()
        resp = self.session.get(url, timeout=20)
        self._last_request = time.time()
        if resp.status_code != 200:
            return None
        cache_path.write_text(resp.text, encoding="utf-8")
        lines = resp.text.splitlines()
        return "".join(line.strip() for line in lines if not line.startswith(">")) or None


ESMEmbedder = ESM2Embedder


@dataclass
class Sample:
    lectin_id: str
    glycan_id: str
    lectin_sequence: str
    glycan_iupac: str
    label: float
    binding_value: float


class GlycoDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Sample],
        esm_embedder: ESMEmbedder,
        glycan_tokenizer: GlycanTokenizer,
    ) -> None:
        self.samples = samples
        self.esm_embedder = esm_embedder
        self.glycan_tokenizer = glycan_tokenizer

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        sample = self.samples[idx]
        lectin_emb = self.esm_embedder.embed(sample.lectin_sequence)
        glycan_ids = self.glycan_tokenizer.encode(sample.glycan_iupac)
        return {
            "lectin_tokens": lectin_emb,
            "glycan_tokens": torch.tensor(glycan_ids, dtype=torch.long),
            "label": torch.tensor(sample.label, dtype=torch.float32),
            "binding_value": torch.tensor(sample.binding_value, dtype=torch.float32),
            "lectin_id": sample.lectin_id,
            "glycan_id": sample.glycan_id,
        }


def collate_batch(batch: List[Dict[str, object]]) -> Dict[str, object]:
    lectin_seqs = [item["lectin_tokens"] for item in batch]
    glycan_seqs = [item["glycan_tokens"] for item in batch]
    labels = torch.stack([item["label"] for item in batch])
    values = torch.stack([item["binding_value"] for item in batch])

    lectin_lens = torch.tensor([seq.shape[0] for seq in lectin_seqs], dtype=torch.long)
    glycan_lens = torch.tensor([seq.shape[0] for seq in glycan_seqs], dtype=torch.long)

    lectin_padded = torch.nn.utils.rnn.pad_sequence(lectin_seqs, batch_first=True)
    glycan_padded = torch.nn.utils.rnn.pad_sequence(glycan_seqs, batch_first=True, padding_value=0)

    lectin_mask = torch.arange(lectin_padded.shape[1])[None, :] < lectin_lens[:, None]
    glycan_mask = torch.arange(glycan_padded.shape[1])[None, :] < glycan_lens[:, None]

    return {
        "lectin_tokens": lectin_padded,
        "lectin_mask": lectin_mask,
        "glycan_tokens": glycan_padded,
        "glycan_mask": glycan_mask,
        "labels": labels,
        "binding_values": values,
        "lectin_ids": [item["lectin_id"] for item in batch],
        "glycan_ids": [item["glycan_id"] for item in batch],
    }


def build_samples(
    df: pd.DataFrame,
    sequences: Dict[str, str],
    label_mode: str,
    rfu_threshold: float,
) -> List[Sample]:
    samples: List[Sample] = []
    for _, row in df.iterrows():
        uniprot = str(row.get("lectin_uniprot") or "")
        sequence = sequences.get(uniprot, "")
        if not sequence:
            continue
        glycan_iupac = str(row.get("glycan_iupac") or "")
        if not glycan_iupac:
            continue
        label = compute_label(row, label_mode, rfu_threshold)
        binding_value = normalize_binding_value(row)
        samples.append(
            Sample(
                lectin_id=str(row.get("lectin_id") or ""),
                glycan_id=str(row.get("glycan_id") or ""),
                lectin_sequence=sequence,
                glycan_iupac=glycan_iupac,
                label=label,
                binding_value=binding_value,
            )
        )
    return samples


def stratified_group_split(
    df: pd.DataFrame,
    label_col: str,
    seed: int,
    train_size: float = 0.8,
    val_size: float = 0.1,
) -> Tuple[List[int], List[int], List[int]]:
    labels = df[label_col].values
    groups = (df["lectin_id"].astype(str) + "||" + df["glycan_id"].astype(str)).values
    indices = np.arange(len(df))
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
        train_idx, temp_idx = next(sgkf.split(indices, labels, groups))
        temp_labels = labels[temp_idx]
        temp_groups = groups[temp_idx]
        sgkf_val = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        val_idx_rel, test_idx_rel = next(sgkf_val.split(temp_idx, temp_labels, temp_groups))
        val_idx = temp_idx[val_idx_rel]
        test_idx = temp_idx[test_idx_rel]
        return list(train_idx), list(val_idx), list(test_idx)
    except Exception:
        from sklearn.model_selection import StratifiedShuffleSplit

        sss = StratifiedShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
        train_idx, temp_idx = next(sss.split(indices, labels))
        temp_labels = labels[temp_idx]
        val_ratio = val_size / (1.0 - train_size)
        sss_val = StratifiedShuffleSplit(n_splits=1, train_size=val_ratio, random_state=seed)
        val_idx_rel, test_idx_rel = next(sss_val.split(temp_idx, temp_labels))
        val_idx = temp_idx[val_idx_rel]
        test_idx = temp_idx[test_idx_rel]
        return list(train_idx), list(val_idx), list(test_idx)


def load_sequences(
    df: pd.DataFrame,
    cache_dir: Path,
    allow_network: bool,
    min_len: int = 50,
) -> Dict[str, str]:
    store = SequenceStore(cache_dir, allow_network=allow_network)
    sequences: Dict[str, str] = {}
    for uniprot_id in sorted(set(df["lectin_uniprot"].fillna("").tolist())):
        seq = store.get_sequence(uniprot_id)
        if seq and len(seq) >= min_len:
            sequences[uniprot_id] = seq
    return sequences
