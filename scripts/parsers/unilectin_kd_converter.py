"""Convert UniLectin KD values to RFU proxies for CFG rescue."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Dict, Optional

import pandas as pd

LOGGER = logging.getLogger(__name__)

KD_PATTERN = re.compile(r"([0-9]*\.?[0-9]+)\s*(pM|nM|uM|mM|M)", re.IGNORECASE)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _parse_kd_value(value: object) -> Optional[float]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value)
    match = KD_PATTERN.search(text)
    if not match:
        return None
    number = float(match.group(1))
    unit = match.group(2).lower()
    if unit == "pm":
        return number / 1000.0
    if unit == "nm":
        return number
    if unit == "um":
        return number * 1000.0
    if unit == "mm":
        return number * 1_000_000.0
    if unit == "m":
        return number * 1_000_000_000.0
    return None


def _extract_kd_from_json(payload: str) -> Optional[float]:
    if not payload:
        return None
    try:
        data = json.loads(payload)
    except Exception:
        return _parse_kd_value(payload)
    candidates = ["kd", "kd_nm", "kd_value", "kd_nM", "KD"]
    for key in candidates:
        if key in data:
            kd_val = _parse_kd_value(data[key])
            if kd_val is not None:
                return kd_val
    for value in data.values():
        kd_val = _parse_kd_value(value)
        if kd_val is not None:
            return kd_val
    return None


def _kd_to_rfu(kd_nm: float) -> float:
    return 10000.0 / (1.0 + kd_nm / 1000.0)


def _conclusive(rfu_raw: float, rfu_norm: float, threshold: float) -> bool:
    if rfu_raw > 0:
        return rfu_raw >= threshold
    norm_threshold = min(100.0, threshold / 100.0)
    return rfu_norm >= norm_threshold


def extract(
    unilectin_path: str,
    metadata_df: pd.DataFrame,
    rfu_threshold: float = 2000.0,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    logger = logger or LOGGER
    df = pd.read_csv(unilectin_path)
    df.columns = [str(c).strip().lower() for c in df.columns]

    meta_lookup = {}
    if not metadata_df.empty:
        meta_df = metadata_df.copy()
        meta_df.columns = [str(c).strip().lower() for c in meta_df.columns]
        for _, row in meta_df.iterrows():
            name = str(row.get("sample_name") or "").strip().lower()
            if name:
                meta_lookup[name] = {
                    "experiment_id": row.get("experiment_id"),
                    "array_version": str(row.get("array_version") or ""),
                    "investigator": str(row.get("investigator") or ""),
                }

    records = []
    for _, row in df.iterrows():
        kd_val = _extract_kd_from_json(row.get("bindingsourcemetadatajson") or row.get("binding_source_metadata_json") or "")
        if kd_val is None:
            kd_val = _parse_kd_value(row.get("binding_value")) if str(row.get("binding_unit")).lower() in {"nm", "nM".lower()} else None
        if kd_val is None:
            continue
        rfu_raw = _kd_to_rfu(kd_val)
        rfu_norm = min(100.0, rfu_raw / 100.0)
        lectin_name = str(row.get("lectin_name") or row.get("protein_name") or "")
        meta = meta_lookup.get(lectin_name.strip().lower(), {})
        exp_id = meta.get("experiment_id")
        try:
            exp_id_val = int(exp_id) if exp_id is not None else 0
        except Exception:
            exp_id_val = 0
        records.append(
            {
                "experiment_id": exp_id_val,
                "array_version": meta.get("array_version", ""),
                "glycan_id": row.get("glycan_id") or 0,
                "cfg_glycan_iupac": str(row.get("glycan_iupac") or row.get("iupac") or ""),
                "glytoucan_id": str(row.get("glytoucan_id") or row.get("glycan_glytoucan_id") or ""),
                "lectin_sample_name": lectin_name,
                "rfu_raw": float(rfu_raw),
                "rfu_normalized": float(rfu_norm),
                "normalization_method": "kd_proxy",
                "stdev": 0.0,
                "cv": 0.0,
                "investigator": meta.get("investigator", ""),
                "data_source": "UniLectin",
                "conclusive": _conclusive(rfu_raw, rfu_norm, rfu_threshold),
                "timestamp": utc_timestamp(),
            }
        )

    return pd.DataFrame(records)
