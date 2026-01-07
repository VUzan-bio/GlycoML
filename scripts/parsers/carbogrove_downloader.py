"""Download CFG-derived RFU data from CarboGrove."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from io import StringIO
from typing import Dict, Optional

import pandas as pd
import requests

DEFAULT_URL = "https://glycoinfo.org/carbogrove/download/all_data.tsv"

LOGGER = logging.getLogger(__name__)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _metadata_lookup(metadata_df: pd.DataFrame) -> Dict[str, Dict[str, str]]:
    meta = {}
    if metadata_df.empty:
        return meta
    df = metadata_df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    for _, row in df.iterrows():
        name = str(row.get("sample_name") or "").strip().lower()
        if not name:
            continue
        meta[name] = {
            "experiment_id": row.get("experiment_id"),
            "array_version": str(row.get("array_version") or ""),
            "investigator": str(row.get("investigator") or ""),
        }
    return meta


def _conclusive(rfu_raw: float, rfu_norm: float, threshold: float) -> bool:
    if rfu_raw > 0:
        return rfu_raw >= threshold
    norm_threshold = min(100.0, threshold / 100.0)
    return rfu_norm >= norm_threshold


def fetch(
    metadata_df: pd.DataFrame,
    url: str = DEFAULT_URL,
    rfu_threshold: float = 2000.0,
    timeout: int = 60,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    logger = logger or LOGGER
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("CarboGrove fetch failed: %s", exc)
        return pd.DataFrame()

    df = pd.read_csv(StringIO(resp.text), sep="\t")
    df.columns = [str(c).strip().lower() for c in df.columns]

    if "source" in df.columns:
        df = df[df["source"].astype(str).str.lower() == "cfg"]

    rename_map = {
        "lectin_name": "lectin_sample_name",
        "glycan_glytoucan_id": "glytoucan_id",
        "binding_score": "rfu_raw",
        "normalized": "rfu_normalized",
        "glycan_id": "glycan_id",
        "iupac": "cfg_glycan_iupac",
    }
    df = df.rename(columns=rename_map)

    meta_lookup = _metadata_lookup(metadata_df)
    records = []
    for _, row in df.iterrows():
        lectin_name = str(row.get("lectin_sample_name") or "").strip()
        meta = meta_lookup.get(lectin_name.lower(), {})
        exp_id = meta.get("experiment_id")
        glycan_id = row.get("glycan_id")
        try:
            exp_id_val = int(exp_id) if exp_id is not None else 0
        except Exception:
            exp_id_val = 0
        try:
            glycan_id_val = int(float(glycan_id)) if glycan_id is not None else 0
        except Exception:
            glycan_id_val = 0
        rfu_raw = float(row.get("rfu_raw") or 0.0)
        rfu_norm = float(row.get("rfu_normalized") or 0.0)
        records.append(
            {
                "experiment_id": exp_id_val,
                "array_version": meta.get("array_version", ""),
                "glycan_id": glycan_id_val,
                "cfg_glycan_iupac": str(row.get("cfg_glycan_iupac") or ""),
                "glytoucan_id": str(row.get("glytoucan_id") or ""),
                "lectin_sample_name": lectin_name,
                "rfu_raw": rfu_raw,
                "rfu_normalized": rfu_norm,
                "normalization_method": "carbogrove",
                "stdev": 0.0,
                "cv": 0.0,
                "investigator": meta.get("investigator", ""),
                "data_source": "CarboGrove",
                "conclusive": _conclusive(rfu_raw, rfu_norm, rfu_threshold),
                "timestamp": utc_timestamp(),
            }
        )

    return pd.DataFrame(records)
