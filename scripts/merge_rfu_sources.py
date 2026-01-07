"""Merge and deduplicate RFU sources for CFG rescue."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import pandas as pd

RFU_COLUMNS = [
    "experiment_id",
    "array_version",
    "glycan_id",
    "cfg_glycan_iupac",
    "glytoucan_id",
    "lectin_sample_name",
    "rfu_raw",
    "rfu_normalized",
    "normalization_method",
    "stdev",
    "cv",
    "investigator",
    "data_source",
    "conclusive",
    "timestamp",
]

PRIORITY_MAP = {
    "CFG_manual": 1,
    "GlycoPattern": 2,
    "CarboGrove": 3,
    "UniLectin": 4,
    "Merged": 5,
}


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in RFU_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[RFU_COLUMNS]


def _score_row(row: pd.Series) -> float:
    raw = row.get("rfu_raw")
    if raw is not None and not pd.isna(raw):
        try:
            raw_val = float(raw)
            if raw_val > 0:
                return raw_val
        except Exception:
            pass
    norm = row.get("rfu_normalized")
    if norm is not None and not pd.isna(norm):
        try:
            return float(norm)
        except Exception:
            return 0.0
    return 0.0


def flag_outliers(df: pd.DataFrame, max_value: float = 1e6, logger: Optional[logging.Logger] = None) -> None:
    logger = logger or logging.getLogger(__name__)
    if df.empty:
        return
    for idx, row in df.iterrows():
        value = _score_row(row)
        if value < 0 or value > max_value:
            logger.warning("Outlier RFU detected", extra={"row": idx, "value": value})


def deduplicate_rfus(
    df: pd.DataFrame,
    priority_map: Optional[Dict[str, int]] = None,
    variance_tol: float = 0.1,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    logger = logger or logging.getLogger(__name__)
    if df.empty:
        return ensure_columns(df)

    df = ensure_columns(df)
    priority_map = priority_map or PRIORITY_MAP
    df["priority"] = df["data_source"].map(priority_map).fillna(99).astype(int)
    df["score"] = df.apply(_score_row, axis=1)

    merged_rows = []
    group_cols = ["experiment_id", "glycan_id", "lectin_sample_name"]
    for _, group in df.groupby(group_cols, dropna=False):
        if len(group) == 1:
            merged_rows.append(group.iloc[0])
            continue
        values = group["score"].replace(0, np.nan).dropna()
        if len(values) >= 2 and values.min() > 0:
            if values.max() <= values.min() * (1 + variance_tol):
                row = group.iloc[0].copy()
                row["rfu_raw"] = float(values.mean())
                if group["rfu_normalized"].notna().any():
                    row["rfu_normalized"] = float(group["rfu_normalized"].dropna().mean())
                row["data_source"] = "Merged"
                row["normalization_method"] = "merged"
                row["timestamp"] = utc_timestamp()
                merged_rows.append(row)
                continue
        selected = group.sort_values("priority").iloc[0]
        merged_rows.append(selected)

    result = pd.DataFrame(merged_rows)
    result = result.drop(columns=["priority", "score"], errors="ignore")
    return ensure_columns(result)


def finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = ensure_columns(df)
    df = df.copy()
    df["timestamp"] = df["timestamp"].fillna(utc_timestamp())
    return df
