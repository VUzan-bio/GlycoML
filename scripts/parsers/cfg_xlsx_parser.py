"""Parse CFG manual XLSX downloads into RFU measurements."""
from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _normalize_column_name(name: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "", str(name).lower())
    return value


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    col_map = {}
    for col in df.columns:
        key = _normalize_column_name(col)
        if key in {"glycanid", "glycan_id", "glycan"}:
            col_map[col] = "glycan_id"
        elif key in {"rfu", "rawrfu", "rfu_raw"}:
            col_map[col] = "rfu_raw"
        elif key in {"normalized", "normalizedrfu", "rfu_normalized"}:
            col_map[col] = "rfu_normalized"
        elif key in {"stdev", "stddev", "sd"}:
            col_map[col] = "stdev"
        elif key in {"cv", "percentcv", "coeffvar"}:
            col_map[col] = "cv"
        elif key in {"name", "glycanname", "iupac"}:
            col_map[col] = "cfg_glycan_iupac"
        elif key in {"glytoucan", "glytoucanid"}:
            col_map[col] = "glytoucan_id"
        elif "rfu" in key and "rep" in key:
            col_map[col] = f"rfu_rep_{len(col_map)}"
        elif "norm" in key and "rep" in key:
            col_map[col] = f"norm_rep_{len(col_map)}"
    return df.rename(columns=col_map)


def _metadata_lookup(metadata_df: pd.DataFrame) -> Dict[int, Dict[str, str]]:
    meta = {}
    if metadata_df.empty:
        return meta
    df = metadata_df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "experiment_id" not in df.columns and "experimentid" in df.columns:
        df = df.rename(columns={"experimentid": "experiment_id"})
    for _, row in df.iterrows():
        try:
            exp_id = int(row.get("experiment_id"))
        except Exception:
            continue
        meta[exp_id] = {
            "array_version": str(row.get("array_version") or ""),
            "investigator": str(row.get("investigator") or ""),
            "lectin_sample_name": str(row.get("sample_name") or ""),
        }
    return meta


def _extract_experiment_id(xlsx_path: Path, metadata_sheet: pd.DataFrame, metadata_ids: List[int]) -> Optional[int]:
    match = re.search(r"(\d{4,6})", xlsx_path.stem)
    if match:
        candidate = int(match.group(1))
        if candidate in metadata_ids or not metadata_ids:
            return candidate
    for value in metadata_sheet.astype(str).values.flatten().tolist():
        match = re.search(r"(\d{4,6})", value)
        if match:
            candidate = int(match.group(1))
            if candidate in metadata_ids or not metadata_ids:
                return candidate
    return None


def _select_data_sheet(xl: pd.ExcelFile) -> str:
    for name in xl.sheet_names:
        preview = xl.parse(name, nrows=5)
        preview = normalize_columns(preview)
        cols = set(preview.columns)
        if "glycan_id" in cols and ("rfu_raw" in cols or "rfu_normalized" in cols):
            return name
    if len(xl.sheet_names) > 1:
        return xl.sheet_names[1]
    return xl.sheet_names[0]


def _compute_stats(values: List[float]) -> Tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values, dtype=float)
    mean = float(np.mean(arr))
    stdev = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    cv = float(stdev / mean * 100.0) if mean > 0 else 0.0
    return mean, stdev, cv


def _conclusive(rfu_raw: float, rfu_norm: float, threshold: float) -> bool:
    if rfu_raw > 0:
        return rfu_raw >= threshold
    norm_threshold = min(100.0, threshold / 100.0)
    return rfu_norm >= norm_threshold


def parse_single_xlsx(
    xlsx_path: Path,
    metadata_df: pd.DataFrame,
    rfu_threshold: float = 2000.0,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    logger = logger or LOGGER
    xl = pd.ExcelFile(xlsx_path)
    metadata_sheet = xl.parse(xl.sheet_names[0]) if xl.sheet_names else pd.DataFrame()
    metadata_ids = sorted(_metadata_lookup(metadata_df).keys())
    exp_id = _extract_experiment_id(xlsx_path, metadata_sheet, metadata_ids)
    if exp_id is None:
        logger.warning("Could not determine experiment_id for %s", xlsx_path)
        return pd.DataFrame()

    data_sheet = _select_data_sheet(xl)
    data_df = xl.parse(data_sheet)
    data_df = normalize_columns(data_df)

    meta_lookup = _metadata_lookup(metadata_df).get(exp_id, {})
    array_version = meta_lookup.get("array_version", "")
    investigator = meta_lookup.get("investigator", "")
    lectin_sample = meta_lookup.get("lectin_sample_name", "")

    records = []
    for _, row in data_df.iterrows():
        glycan_id = row.get("glycan_id")
        if pd.isna(glycan_id):
            continue
        try:
            glycan_id_val = int(float(glycan_id))
        except Exception:
            continue

        raw_cols = [c for c in data_df.columns if c.startswith("rfu_rep_")]
        norm_cols = [c for c in data_df.columns if c.startswith("norm_rep_")]

        raw_vals = [row[c] for c in raw_cols if pd.notna(row.get(c))]
        norm_vals = [row[c] for c in norm_cols if pd.notna(row.get(c))]

        rfu_raw = row.get("rfu_raw")
        rfu_norm = row.get("rfu_normalized")
        stdev = row.get("stdev")
        cv = row.get("cv")

        if raw_vals:
            rfu_raw, stdev_calc, cv_calc = _compute_stats([float(v) for v in raw_vals])
            if not stdev or pd.isna(stdev):
                stdev = stdev_calc
            if not cv or pd.isna(cv):
                cv = cv_calc
        if norm_vals:
            rfu_norm, _, _ = _compute_stats([float(v) for v in norm_vals])

        rfu_raw = float(rfu_raw) if pd.notna(rfu_raw) else 0.0
        rfu_norm = float(rfu_norm) if pd.notna(rfu_norm) else 0.0
        stdev = float(stdev) if pd.notna(stdev) else 0.0
        cv = float(cv) if pd.notna(cv) else 0.0

        records.append(
            {
                "experiment_id": exp_id,
                "array_version": array_version,
                "glycan_id": glycan_id_val,
                "cfg_glycan_iupac": str(row.get("cfg_glycan_iupac") or ""),
                "glytoucan_id": str(row.get("glytoucan_id") or ""),
                "lectin_sample_name": lectin_sample,
                "rfu_raw": rfu_raw,
                "rfu_normalized": rfu_norm,
                "normalization_method": "cfg_xlsx",
                "stdev": stdev,
                "cv": cv,
                "investigator": investigator,
                "data_source": "CFG_manual",
                "conclusive": _conclusive(rfu_raw, rfu_norm, rfu_threshold),
                "timestamp": utc_timestamp(),
            }
        )

    return pd.DataFrame(records)


def parse_all(
    downloads_dir: str | Path,
    metadata_df: pd.DataFrame,
    rfu_threshold: float = 2000.0,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    logger = logger or LOGGER
    downloads_path = Path(downloads_dir)
    if not downloads_path.exists():
        logger.warning("CFG downloads directory not found: %s", downloads_path)
        return pd.DataFrame()

    all_frames = []
    for xlsx_path in sorted(downloads_path.glob("*.xlsx")):
        try:
            frame = parse_single_xlsx(xlsx_path, metadata_df, rfu_threshold, logger)
            if not frame.empty:
                all_frames.append(frame)
        except Exception as exc:
            logger.warning("Failed to parse %s: %s", xlsx_path, exc)
    return pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
