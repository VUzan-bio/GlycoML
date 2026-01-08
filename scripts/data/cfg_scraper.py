#!/usr/bin/env python3
"""CFG scraper (BeautifulSoup) to download and parse RFU tables."""

from __future__ import annotations

import argparse
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.functionalglycomics.org"
LIST_URLS = [
    f"{BASE_URL}/glycan-array",
    f"{BASE_URL}/glycomics/publicdata/selectedScreens.jsp",
]

TARGET_IDS = ["primscreen_6599", "primscreen_6592", "primscreen_6598"]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_str(val: object) -> str:
    return "" if val is None else str(val)


def normalize_col(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def load_metadata(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def resolve_experiment_ids(metadata: pd.DataFrame, primary_ids: List[str]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    if metadata.empty:
        return mapping

    if "primary_screen_id" in metadata.columns:
        meta = metadata.copy()
        meta["primary_screen_id"] = meta["primary_screen_id"].astype(str).str.lower()
        for pid in primary_ids:
            match = meta[meta["primary_screen_id"] == pid.lower()]
            if not match.empty:
                exp_id = int(match.iloc[0]["experiment_id"])
                mapping[pid] = exp_id
    return mapping


def fetch_html(url: str) -> str:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def find_detail_links(html: str) -> List[str]:
    soup = BeautifulSoup(html, "lxml")
    links: List[str] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        text = (anchor.get_text() or "").strip().lower()
        if any(target in href.lower() for target in TARGET_IDS) or any(target in text for target in TARGET_IDS):
            links.append(href)
    return links


def find_download_link(html: str, base_url: str) -> Optional[str]:
    soup = BeautifulSoup(html, "lxml")
    candidates: List[str] = []
    for anchor in soup.find_all("a", href=True):
        href = anchor["href"]
        text = (anchor.get_text() or "").lower()
        if any(ext in href.lower() for ext in [".xlsx", ".xlsm", ".xls", ".csv", ".tsv"]):
            candidates.append(href)
        elif "download" in href.lower() or "download" in text:
            candidates.append(href)

    for button in soup.find_all(attrs={"onclick": True}):
        onclick = button["onclick"]
        match = re.search(r"(https?://[\w\./\-\?=_%]+)", onclick)
        if match:
            candidates.append(match.group(1))

    if not candidates:
        return None

    preferred_exts = [".xlsm", ".xlsx", ".xls", ".csv", ".tsv"]

    def score(href: str) -> int:
        for idx, ext in enumerate(preferred_exts):
            if href.lower().endswith(ext):
                return idx
        return len(preferred_exts)

    candidates = sorted(set(candidates), key=score)
    return urljoin(base_url, candidates[0])


def download_file(url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = os.path.basename(urlparse(url).path) or "cfg_download"
    path = output_dir / filename
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    return path


def find_local_file(downloads_dir: Path, pid: str, exp_id: Optional[int]) -> Optional[Path]:
    candidates = []
    if downloads_dir.exists():
        for path in downloads_dir.iterdir():
            if not path.is_file():
                continue
            name = path.name.lower()
            if any(name.endswith(ext) for ext in (".xlsm", ".xlsx", ".xls", ".csv", ".tsv")):
                candidates.append(path)

    if exp_id:
        exp_token = str(exp_id)
        for path in candidates:
            if exp_token in path.name:
                return path

    pid_token = pid.split("_")[-1]
    for path in candidates:
        if pid.lower() in path.name.lower() or pid_token in path.name:
            return path
    return candidates[0] if candidates else None


def _extract_sample_header(columns: List[str]) -> str:
    markers = ("slide#", "rrequest", "genepix", "cfg#", "alex", "fitc", "pmt")
    candidates: List[str] = []
    for col in columns:
        lower = str(col).lower()
        if any(marker in lower for marker in markers):
            candidates.append(col)
    if not candidates:
        return ""
    return max(candidates, key=lambda val: len(str(val)))


def _group_average(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return df

    rows: List[Dict[str, object]] = []
    for keys, group in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        entry = {col: val for col, val in zip(group_cols, keys)}
        rfu_vals = group["rfu_raw"].astype(float)
        mean_val = float(rfu_vals.mean())
        if mean_val < 0:
            mean_val = 0.0
        if len(rfu_vals) > 1:
            stdev = float(rfu_vals.std(ddof=1))
            cv = float((stdev / mean_val) * 100.0) if mean_val > 0 else 0.0
        else:
            stdev = float(group["stdev"].iloc[0]) if "stdev" in group else 0.0
            cv = float(group["cv"].iloc[0]) if "cv" in group else 0.0
        entry.update(
            {
                "rfu_raw": mean_val,
                "stdev": stdev,
                "cv": cv,
                "cfg_glycan_iupac": group["cfg_glycan_iupac"].dropna().iloc[0] if "cfg_glycan_iupac" in group else "",
                "sample_name": group["sample_name"].dropna().iloc[0] if "sample_name" in group else "",
                "sample_tag": group["sample_tag"].dropna().iloc[0] if "sample_tag" in group else "",
            }
        )
        rows.append(entry)
    return pd.DataFrame(rows)


def extract_experiment_id_from_name(name: str) -> int:
    matches = re.findall(r"(\d{4,6})", name)
    if not matches:
        return 0
    return int(matches[-1])


def _split_columns(df: pd.DataFrame) -> Tuple[Dict[str, str], Dict[str, str]]:
    left: Dict[str, str] = {}
    right: Dict[str, str] = {}
    for col in df.columns:
        col_str = str(col).strip()
        norm = normalize_col(col_str)
        if col_str.endswith(".1"):
            right[norm] = col
        else:
            left[norm] = col
    return left, right


def _find_col(mapping: Dict[str, str], key: str) -> Optional[str]:
    for norm, original in mapping.items():
        if norm.startswith(key):
            return original
    return None


def _find_col_idx(columns: List[object], target: str) -> Optional[int]:
    for idx, col in enumerate(columns):
        if normalize_col(col) == target:
            return idx
    return None


def parse_data_sheet(df: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
    df = df.copy()
    left_map, right_map = _split_columns(df)

    def parse_side(mapping: Dict[str, str]) -> List[Dict[str, object]]:
        chart = _find_col(mapping, "chartid")
        avg = _find_col(mapping, "averagerfu")
        stdev = _find_col(mapping, "stdev")
        cv = _find_col(mapping, "cv")
        name = _find_col(mapping, "sampleconc")

        if not chart or not avg:
            return []

        records: List[Dict[str, object]] = []
        for _, row in df.iterrows():
            gid = row.get(chart)
            if pd.isna(gid):
                continue
            try:
                gid = int(float(gid))
            except Exception:
                continue
            rfu = row.get(avg)
            if pd.isna(rfu):
                continue
            records.append(
                {
                    "glycan_id": gid,
                    "rfu_raw": float(rfu),
                    "stdev": float(row.get(stdev) or 0.0),
                    "cv": float(row.get(cv) or 0.0),
                    "cfg_glycan_iupac": safe_str(row.get(name) or ""),
                }
            )
        return records

    records = parse_side(left_map) + parse_side(right_map)
    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    frame["sample_tag"] = sheet_name
    frame["sample_name"] = ""
    return _group_average(frame, ["sample_tag", "glycan_id"])


def _parse_cfg_data_sheet(df: pd.DataFrame, sheet_name: str, file_stem: str) -> pd.DataFrame:
    df = df.copy()
    cols = list(df.columns)
    chart_idx = _find_col_idx(cols, "chartnumber1")
    rfu_idx = _find_col_idx(cols, "averagerfu1")
    stdev_idx = _find_col_idx(cols, "stdev1")
    cv_idx = _find_col_idx(cols, "cv1")
    name_idx = _find_col_idx(cols, "structureonmasterlist1")

    if chart_idx is None or rfu_idx is None:
        chart_idx = _find_col_idx(cols, "chartnumber")
        rfu_idx = _find_col_idx(cols, "averagerfu")
        stdev_idx = _find_col_idx(cols, "stdev")
        cv_idx = _find_col_idx(cols, "cv")
        name_idx = _find_col_idx(cols, "structureonmasterlist")

    if chart_idx is None or rfu_idx is None:
        return pd.DataFrame()

    if name_idx is None and chart_idx < rfu_idx:
        for idx in range(chart_idx + 1, rfu_idx):
            if df.iloc[:, idx].notna().any():
                name_idx = idx
                break

    sample_name = str(file_stem or sheet_name)
    records: List[Dict[str, object]] = []

    for _, row in df.iterrows():
        gid = row.iloc[chart_idx]
        if pd.isna(gid):
            continue
        try:
            gid_val = int(float(gid))
        except Exception:
            continue

        rfu = row.iloc[rfu_idx]
        if pd.isna(rfu):
            continue
        try:
            rfu_val = float(rfu)
        except Exception:
            continue
        if rfu_val < 0:
            continue

        glycan_name = ""
        if name_idx is not None:
            name_val = row.iloc[name_idx]
            if pd.notna(name_val):
                glycan_name = safe_str(name_val)

        stdev_val = 0.0
        if stdev_idx is not None:
            stdev_cell = row.iloc[stdev_idx]
            stdev_val = float(stdev_cell) if pd.notna(stdev_cell) else 0.0

        cv_val = 0.0
        if cv_idx is not None:
            cv_cell = row.iloc[cv_idx]
            cv_val = float(cv_cell) if pd.notna(cv_cell) else 0.0

        records.append(
            {
                "glycan_id": gid_val,
                "rfu_raw": rfu_val,
                "stdev": stdev_val,
                "cv": cv_val,
                "cfg_glycan_iupac": glycan_name,
                "sample_name": sample_name,
                "sample_tag": sheet_name,
            }
        )

    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    return _group_average(frame, ["sample_name", "sample_tag", "glycan_id"])


def _parse_genepix_sheet(df: pd.DataFrame, sheet_name: str, file_stem: str) -> pd.DataFrame:
    df = df.copy()
    cols = list(df.columns)
    sample_header = _extract_sample_header(cols)
    sample_name = str(sample_header or file_stem or sheet_name)
    if len(cols) < 33:
        print(f"WARNING: Missing right panel in {file_stem}:{sheet_name} ({len(cols)} columns)")
        return pd.DataFrame()

    records: List[Dict[str, object]] = []

    for _, row in df.iterrows():
        gid = row.iloc[28]
        if pd.isna(gid):
            continue
        try:
            gid_val = int(float(gid))
        except Exception:
            continue
        rfu = row.iloc[30]
        if pd.isna(rfu):
            continue
        try:
            rfu_val = float(rfu)
        except Exception:
            continue
        if rfu_val < 0:
            continue
        records.append(
            {
                "glycan_id": gid_val,
                "rfu_raw": rfu_val,
                "stdev": float(row.iloc[31] or 0.0),
                "cv": float(row.iloc[32] or 0.0),
                "cfg_glycan_iupac": safe_str(row.iloc[29] or ""),
                "sample_name": sample_name,
                "sample_tag": sheet_name,
            }
        )

    if not records:
        return pd.DataFrame()
    frame = pd.DataFrame(records)
    mean_rfu = frame["rfu_raw"].mean()
    if mean_rfu < 500:
        print(f"WARNING: Low RFU mean ({mean_rfu:.1f}) in {file_stem}:{sheet_name}")
    return frame


def parse_excel(path: Path) -> pd.DataFrame:
    xls = pd.ExcelFile(path)
    frames: List[pd.DataFrame] = []
    is_genepix_file = any(marker in path.name.lower() for marker in ["genepix", "v5.2", "results"])
    for sheet in xls.sheet_names:
        df = xls.parse(sheet)
        cols_norm = [normalize_col(c) for c in df.columns]
        if is_genepix_file and len(df.columns) >= 33:
            frame = _parse_genepix_sheet(df, sheet, path.stem)
            if not frame.empty:
                frames.append(frame)
        elif not is_genepix_file and any(c.startswith("chartnumber") for c in cols_norm) and any(
            c.startswith("averagerfu") for c in cols_norm
        ):
            frame = _parse_cfg_data_sheet(df, sheet, path.stem)
            if not frame.empty:
                frames.append(frame)
        elif not is_genepix_file:
            if any("avgmeansbwominmax" in c for c in cols_norm) or "data" in sheet.lower():
                frame = parse_data_sheet(df, sheet)
                if not frame.empty:
                    frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def normalize_rfu(df: pd.DataFrame, group_col: str = "sample_key") -> pd.DataFrame:
    if df.empty:
        return df
    if group_col not in df.columns:
        group_col = ""
    if group_col:
        df["rfu_normalized"] = 0.0
        for _, group in df.groupby(group_col, dropna=False):
            max_rfu = group["rfu_raw"].max()
            if max_rfu and max_rfu > 0:
                df.loc[group.index, "rfu_normalized"] = (group["rfu_raw"] / max_rfu) * 100.0
    else:
        max_rfu = df["rfu_raw"].max()
        if max_rfu and max_rfu > 0:
            df["rfu_normalized"] = (df["rfu_raw"] / max_rfu) * 100.0
        else:
            df["rfu_normalized"] = 0.0
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-metadata", default="data/metadata/cfg_experiment_metadata.csv")
    parser.add_argument("--output", default="data/processed/cfg_rfu_measurements.csv")
    parser.add_argument("--downloads-dir", default="data/raw/cfg_arrays_raw")
    parser.add_argument("--local-xlsx", default="")
    parser.add_argument("--use-local-dir", action="store_true", help="Parse all local files in downloads-dir")
    args = parser.parse_args()

    metadata = load_metadata(Path(args.cfg_metadata))
    exp_map = resolve_experiment_ids(metadata, TARGET_IDS)

    downloads_dir = Path(args.downloads_dir)
    downloads_dir.mkdir(parents=True, exist_ok=True)

    downloaded_files: List[Tuple[str, Path]] = []

    if args.local_xlsx:
        downloaded_files.append(("local", Path(args.local_xlsx)))
    elif args.use_local_dir:
        for path in sorted(downloads_dir.iterdir()):
            if not path.is_file():
                continue
            if not any(path.name.lower().endswith(ext) for ext in (".xlsm", ".xlsx", ".xls", ".csv", ".tsv")):
                continue
            downloaded_files.append((path.stem, path))
    else:
        detail_links: List[str] = []
        for url in LIST_URLS:
            try:
                html = fetch_html(url)
                detail_links.extend(find_detail_links(html))
            except Exception:
                continue

        for pid in TARGET_IDS:
            exp_id = exp_map.get(pid)
            if exp_id:
                detail_url = f"{BASE_URL}/glycan-array/{exp_id}"
            else:
                detail_url = None
                for link in detail_links:
                    if pid in link.lower():
                        detail_url = urljoin(BASE_URL, link)
                        break

            if not detail_url:
                print(f"Could not resolve detail page for {pid}")
                continue

            try:
                html = fetch_html(detail_url)
            except Exception as exc:
                print(f"Failed to fetch detail page {detail_url}: {exc}")
                continue

            download_url = find_download_link(html, BASE_URL)
            if not download_url:
                local_path = find_local_file(downloads_dir, pid, exp_id)
                if local_path is None:
                    print(f"No download link found for {pid} at {detail_url}")
                    continue
                downloaded_files.append((pid, local_path))
                print(f"Using local file for {pid}: {local_path}")
                continue

            try:
                path = download_file(download_url, downloads_dir)
                downloaded_files.append((pid, path))
                print(f"Downloaded {pid}: {path}")
            except Exception as exc:
                local_path = find_local_file(downloads_dir, pid, exp_id)
                if local_path is None:
                    print(f"Failed to download {pid}: {exc}")
                    continue
                downloaded_files.append((pid, local_path))
                print(f"Download failed, using local file for {pid}: {local_path}")

    all_records: List[pd.DataFrame] = []
    for pid, path in downloaded_files:
        try:
            df = parse_excel(path)
        except Exception as exc:
            print(f"Failed to parse {path}: {exc}")
            continue

        if df.empty:
            continue

        if "sample_name" not in df.columns:
            df["sample_name"] = ""
        if "sample_tag" not in df.columns:
            df["sample_tag"] = ""
        df["sample_key"] = df["sample_name"].fillna("") + "|" + df["sample_tag"].fillna("")
        df = normalize_rfu(df, group_col="sample_key")
        exp_id = exp_map.get(pid, None)
        if exp_id is None:
            exp_id = extract_experiment_id_from_name(str(path.name))

        array_version = ""
        investigator = ""
        sample_name = ""
        if not metadata.empty and exp_id:
            row = metadata[metadata["experiment_id"] == exp_id]
            if not row.empty:
                array_version = safe_str(row.iloc[0].get("array_version", ""))
                investigator = safe_str(row.iloc[0].get("investigator", ""))
                sample_name = safe_str(row.iloc[0].get("sample_name", ""))

        df["sample_name"] = df["sample_name"].replace("", sample_name).fillna(sample_name)
        df["sample_tag"] = df["sample_tag"].fillna("")
        df["lectin_sample_name"] = df["sample_name"]
        needs_suffix = df["sample_tag"] != ""
        if needs_suffix.any():
            df.loc[needs_suffix, "lectin_sample_name"] = df.loc[needs_suffix].apply(
                lambda row: (
                    f"{row['sample_name']} | {row['sample_tag']}"
                    if row['sample_tag'] and row['sample_tag'] not in str(row['sample_name'])
                    else row['sample_name']
                ),
                axis=1,
            )

        df["experiment_id"] = exp_id
        df["array_version"] = array_version
        df["glytoucan_id"] = ""
        df["normalization_method"] = "percent_max"
        df["investigator"] = investigator
        df["data_source"] = "CFG_scrape"
        df["conclusive"] = df["rfu_raw"] >= 2000
        df["timestamp"] = utc_timestamp()

        all_records.append(df)

    if not all_records:
        print("No RFU data extracted.")
        return

    merged = pd.concat(all_records, ignore_index=True)
    merged = merged.drop_duplicates(subset=["experiment_id", "glycan_id", "lectin_sample_name"])
    cols = [
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
    merged = merged[cols]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.output, index=False)
    print(f"Saved {len(merged)} rows to {args.output}")


if __name__ == "__main__":
    main()
