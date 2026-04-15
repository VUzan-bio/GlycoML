"""Fetch CFG RFU measurements from GlycoPattern API."""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional

import pandas as pd
import requests

BASE_URL = "https://glycopattern.emory.edu/api"

LOGGER = logging.getLogger(__name__)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


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


def _get_json(session: requests.Session, url: str, retries: int = 5, timeout: int = 30) -> List[Dict[str, object]]:
    backoff = 1.0
    for attempt in range(retries):
        resp = session.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code in {429, 500, 502, 503, 504}:
            time.sleep(backoff)
            backoff *= 2
            continue
        resp.raise_for_status()
    raise RuntimeError(f"Failed to fetch {url} after {retries} retries")


def _conclusive(rfu_raw: float, rfu_norm: float, threshold: float) -> bool:
    if rfu_raw > 0:
        return rfu_raw >= threshold
    norm_threshold = min(100.0, threshold / 100.0)
    return rfu_norm >= norm_threshold


def fetch_all(
    metadata_df: pd.DataFrame,
    rate_limit: float = 1.0,
    max_experiments: int = 0,
    rfu_threshold: float = 2000.0,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    logger = logger or LOGGER
    session = requests.Session()
    try:
        experiments = _get_json(session, f"{BASE_URL}/experiments")
    except requests.RequestException as exc:
        logger.warning("GlycoPattern API unreachable: %s", exc)
        return pd.DataFrame()
    except Exception as exc:
        logger.warning("GlycoPattern API error: %s", exc)
        return pd.DataFrame()
    if max_experiments and max_experiments > 0:
        experiments = experiments[:max_experiments]

    meta_lookup = _metadata_lookup(metadata_df)
    records = []

    for exp in experiments:
        exp_id = exp.get("id") or exp.get("experiment_id")
        if exp_id is None:
            continue
        try:
            exp_id_val = int(exp_id)
        except Exception:
            continue
        data_url = f"{BASE_URL}/experiments/{exp_id_val}/data"
        try:
            data_rows = _get_json(session, data_url)
        except requests.RequestException as exc:
            logger.warning("GlycoPattern fetch failed for %s: %s", exp_id_val, exc)
            continue
        except Exception as exc:
            logger.warning("GlycoPattern data error for %s: %s", exp_id_val, exc)
            continue

        meta = meta_lookup.get(exp_id_val, {})
        array_version = str(exp.get("array_version") or meta.get("array_version") or "")
        investigator = str(meta.get("investigator") or "")
        lectin_sample = str(exp.get("lectin") or meta.get("lectin_sample_name") or "")

        for row in data_rows:
            glycan_id = row.get("glycan_id") or row.get("glycan")
            if glycan_id is None:
                continue
            try:
                glycan_id_val = int(float(glycan_id))
            except Exception:
                continue
            rfu_raw = float(row.get("rfu") or row.get("rfu_raw") or 0.0)
            rfu_norm = float(row.get("normalized") or row.get("rfu_normalized") or 0.0)
            records.append(
                {
                    "experiment_id": exp_id_val,
                    "array_version": array_version,
                    "glycan_id": glycan_id_val,
                    "cfg_glycan_iupac": str(row.get("iupac") or row.get("name") or ""),
                    "glytoucan_id": str(row.get("glytoucan_id") or ""),
                    "lectin_sample_name": lectin_sample,
                    "rfu_raw": rfu_raw,
                    "rfu_normalized": rfu_norm,
                    "normalization_method": "api_normalized" if rfu_norm else "api_raw",
                    "stdev": float(row.get("stdev") or 0.0),
                    "cv": float(row.get("cv") or 0.0),
                    "investigator": investigator,
                    "data_source": "GlycoPattern",
                    "conclusive": _conclusive(rfu_raw, rfu_norm, rfu_threshold),
                    "timestamp": utc_timestamp(),
                }
            )
        time.sleep(rate_limit)

    return pd.DataFrame(records)
