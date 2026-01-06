#!/usr/bin/env python3
"""Phase 2 data acquisition for UniLectin3D and CFG lectin arrays."""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd
import requests
from bs4 import BeautifulSoup
try:  # pragma: no cover - optional dependency
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
except Exception:  # pragma: no cover
    webdriver = None
    Options = None

DEFAULT_UNILECTIN_BASE = "https://unilectin.unige.ch/api"
DEFAULT_CFG_BASE = "https://www.functionalglycomics.org"

UNILECTIN_INTERACTIONS_ENDPOINT = "/query_unilectin3d"
UNILECTIN_HUMANLECTOME_ENDPOINT = "/query_humanlectome"
UNILECTIN_LIGAND_ENDPOINT = "/query_ligand"

UNILECTIN_COLUMNS = [
    "lectin.lectin_id",
    "pdb",
    "protein_name",
    "uniprot",
    "origin",
    "species",
    "species_id",
    "fold",
    "class",
    "family",
    "resolution",
    "ligand",
    "monosac",
    "iupac",
    "glycoct",
    "glytoucan_id",
]

UNILECTIN_OUTPUT_COLUMNS = [
    "lectin_id",
    "pdb",
    "protein_name",
    "uniprot",
    "origin",
    "species",
    "species_id",
    "fold",
    "class",
    "family",
    "resolution",
    "ligand",
    "monosac",
    "iupac",
    "glycoct",
    "glytoucan_id",
    "timestamp_download",
]

PREDICTED_OUTPUT_COLUMNS = [
    "protein_id",
    "uniprot",
    "protein_name",
    "fold",
    "class",
    "score",
    "species",
    "genus",
    "superkingdom",
    "phylum",
    "length",
    "sequence_hash",
]

LIGAND_OUTPUT_COLUMNS = [
    "ligand_id",
    "iupac",
    "glycoct",
    "glytoucan_id",
    "monosac_composition",
]

CFG_METADATA_COLUMNS = [
    "experiment_id",
    "primary_screen_id",
    "request",
    "sample_name",
    "category",
    "array_version",
    "investigator",
    "conclusive_status",
    "data_url",
    "download_date",
    "file_path",
    "file_size_bytes",
    "file_hash_md5",
]

CFG_RFU_COLUMNS = [
    "experiment_id",
    "array_version",
    "glycan_id_cfg",
    "glycan_iupac",
    "glycan_glytoucan_id",
    "lectin_sample_name",
    "rfu_raw",
    "rfu_normalized",
    "normalization_status",
    "investigator",
    "conclusive",
]

CFG_MAPPING_COLUMNS = [
    "cfg_lectin_name",
    "unilectin3d_lectin_id",
    "uniprot_id",
    "pdb",
    "protein_name",
    "confidence_score",
]

UNIFIED_COLUMNS = [
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


def utc_timestamp() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def safe_str(value: Optional[object]) -> str:
    if value is None:
        return ""
    return str(value)


def compute_md5_bytes(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


def compute_md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    hasher = hashlib.md5()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            hasher.update(chunk)
    return hasher.hexdigest()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def extract_monosaccharide_composition(iupac: str) -> str:
    if not iupac:
        return ""
    tokens = [
        "GlcNAc",
        "GalNAc",
        "Glc",
        "Gal",
        "Man",
        "Fuc",
        "Neu5Ac",
        "NeuAc",
        "Neu",
        "Xyl",
        "Rha",
        "Ara",
        "Kdo",
        "HexNAc",
        "Hex",
    ]
    counts: Dict[str, int] = {}
    for token in tokens:
        matches = re.findall(re.escape(token), iupac)
        if matches:
            counts[token] = len(matches)
    return ",".join(f"{key}:{value}" for key, value in sorted(counts.items()))


def normalize_iupac(iupac: str) -> str:
    if not iupac:
        return ""
    replacements = {
        "Glucose": "Glc",
        "Galactose": "Gal",
        "Mannose": "Man",
        "N-acetylglucosamine": "GlcNAc",
        "N-acetylgalactosamine": "GalNAc",
    }
    normalized = iupac
    for key, value in replacements.items():
        normalized = normalized.replace(key, value)
    return normalized


def write_csv(path: Path, columns: List[str], rows: Iterable[Dict[str, object]]) -> int:
    ensure_dir(path.parent)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({key: safe_str(row.get(key)) for key in columns})
            count += 1
    return count


def append_csv(path: Path, columns: List[str], rows: Iterable[Dict[str, object]]) -> int:
    ensure_dir(path.parent)
    exists = path.exists()
    count = 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        if not exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({key: safe_str(row.get(key)) for key in columns})
            count += 1
    return count


def record_metadata(metadata_path: Path, output_path: Path, row_count: int) -> None:
    file_size = output_path.stat().st_size if output_path.exists() else 0
    file_hash = compute_md5_file(output_path) if output_path.exists() else ""
    append_csv(
        metadata_path,
        ["filename", "row_count", "file_size_bytes", "hash_md5", "timestamp"],
        [
            {
                "filename": output_path.as_posix(),
                "row_count": row_count,
                "file_size_bytes": file_size,
                "hash_md5": file_hash,
                "timestamp": utc_timestamp(),
            }
        ],
    )


def setup_logging(log_dir: Path, level: str) -> Tuple[logging.Logger, logging.Logger]:
    ensure_dir(log_dir)
    logger = logging.getLogger("phase2")
    error_logger = logging.getLogger("phase2_errors")
    if logger.handlers:
        return logger, error_logger
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    error_logger.setLevel(logging.ERROR)

    log_path = log_dir / "phase2_download.log"
    error_path = log_dir / "error.log"

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    error_handler = logging.FileHandler(error_path, encoding="utf-8")
    error_handler.setFormatter(formatter)
    error_logger.addHandler(error_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger, error_logger


@dataclass
class RequestConfig:
    rate_limit: float = 0.1
    retry_max: int = 5
    timeout: int = 15


class ResponseCache:
    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir
        ensure_dir(self.cache_dir)
        self.manifest_path = self.cache_dir / "cache_manifest.json"
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Dict[str, str]]:
        if self.manifest_path.exists():
            try:
                with self.manifest_path.open("r", encoding="utf-8") as handle:
                    return json.load(handle)
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_manifest(self) -> None:
        with self.manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(self.manifest, handle, indent=2)

    def _key_to_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> Optional[object]:
        entry = self.manifest.get(key)
        if not entry:
            return None
        path = self._key_to_path(key)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except json.JSONDecodeError:
            return None

    def set(self, key: str, payload: object) -> None:
        path = self._key_to_path(key)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        md5 = compute_md5_file(path)
        self.manifest[key] = {"path": path.name, "md5": md5, "timestamp": utc_timestamp()}
        self._save_manifest()


class UniLectinClient:
    def __init__(
        self,
        base_url: str,
        request_cfg: RequestConfig,
        cache: ResponseCache,
        logger: logging.Logger,
        error_logger: logging.Logger,
        force_refresh: bool = False,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_cfg = request_cfg
        self.cache = cache
        self.logger = logger
        self.error_logger = error_logger
        self.force_refresh = force_refresh
        self.session = requests.Session()
        self._last_request = 0.0

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < self.request_cfg.rate_limit:
            time.sleep(self.request_cfg.rate_limit - elapsed)

    def _cache_key(self, endpoint: str, params: Dict[str, object], method: str) -> str:
        raw = json.dumps({"endpoint": endpoint, "params": params, "method": method}, sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    def request_json(self, endpoint: str, params: Dict[str, object]) -> List[Dict[str, object]]:
        return self._request_json(endpoint, params, method="get")

    def _request_json(self, endpoint: str, params: Dict[str, object], method: str) -> List[Dict[str, object]]:
        key = self._cache_key(endpoint, params, method)
        if not self.force_refresh:
            cached = self.cache.get(key)
            if isinstance(cached, list):
                return cached

        url = f"{self.base_url}{endpoint}"
        for attempt in range(self.request_cfg.retry_max):
            self._throttle()
            try:
                start = time.time()
                if method == "post_json":
                    response = self.session.post(url, json=params, timeout=self.request_cfg.timeout)
                elif method == "post_form":
                    response = self.session.post(url, data=params, timeout=self.request_cfg.timeout)
                else:
                    response = self.session.get(url, params=params, timeout=self.request_cfg.timeout)
                elapsed = time.time() - start
                self._last_request = time.time()
                status = response.status_code
                if status == 200:
                    data = self._coerce_rows(response)
                    if not data and "ERROR" in response.text:
                        self.error_logger.error("UniLectin response error: %s", response.text[:200])
                    self.logger.info(
                        "UniLectin %s %s status=%s rows=%s time=%.2fs",
                        method.upper(),
                        endpoint,
                        status,
                        len(data),
                        elapsed,
                    )
                    self.cache.set(key, data)
                    return data
                if status in {429, 500, 502, 503, 504}:
                    backoff = 2 ** attempt
                    self.logger.warning("UniLectin status %s, retrying in %ss", status, backoff)
                    time.sleep(backoff)
                    continue
                self.error_logger.error("UniLectin request failed (%s): %s", status, response.text[:200])
                return []
            except requests.RequestException as exc:
                backoff = 2 ** attempt
                self.error_logger.error("UniLectin request error: %s", exc)
                time.sleep(backoff)
        return []

    def request_json_with_fallback(self, endpoint: str, params: Dict[str, object]) -> List[Dict[str, object]]:
        rows = self._request_json(endpoint, params, method="get")
        if rows:
            return rows
        fallback = {
            UNILECTIN_INTERACTIONS_ENDPOINT: "/getlectins",
            UNILECTIN_HUMANLECTOME_ENDPOINT: "/gethumanlectome",
            UNILECTIN_LIGAND_ENDPOINT: "/getligands",
        }.get(endpoint)
        if not fallback:
            return rows
        self.logger.warning("UniLectin endpoint %s returned 0 rows; trying fallback %s", endpoint, fallback)
        rows = self._request_json(fallback, params, method="post_json")
        if rows:
            return rows
        return self._request_json(fallback, params, method="post_form")

    def _coerce_rows(self, response: requests.Response) -> List[Dict[str, object]]:
        try:
            data = response.json()
        except json.JSONDecodeError:
            return []
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        if isinstance(data, dict) and "columns" in data and "rows" in data:
            columns = data.get("columns") or []
            rows = data.get("rows") or []
            return [dict(zip(columns, row)) for row in rows]
        if isinstance(data, list) and data and isinstance(data[0], list):
            columns = data[0]
            return [dict(zip(columns, row)) for row in data[1:]]
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        return []


def iter_id_ranges(start: int, max_id: int, step: int) -> Iterator[Tuple[int, int]]:
    current = start
    while current <= max_id:
        end = min(current + step - 1, max_id)
        yield current, end
        current = end + 1


def build_range_filter(range_template: str, start: int, end: int) -> str:
    return range_template.format(start=start, end=end)


def build_unilectin_params(
    getcolumns: str,
    limit: int,
    wherecolumn: Optional[str] = None,
    isvalue: Optional[str] = None,
) -> Dict[str, object]:
    params: Dict[str, object] = {"getcolumns": getcolumns, "limit": str(limit)}
    if wherecolumn:
        params["wherecolumn"] = wherecolumn
    if isvalue is not None:
        params["isvalue"] = isvalue
    return params


def stream_unilectin3d_lectins(
    client: UniLectinClient,
    limit: int,
    range_size: int,
    max_id: int,
    range_template: str,
) -> Iterator[Dict[str, object]]:
    full_params = build_unilectin_params(
        ",".join(UNILECTIN_COLUMNS),
        limit,
        wherecolumn="lectin.lectin_id",
        isvalue="%%",
    )
    rows = client.request_json_with_fallback(UNILECTIN_INTERACTIONS_ENDPOINT, full_params)
    if rows:
        for row in rows:
            yield row
        return

    empty_count = 0
    for start, end in iter_id_ranges(1, max_id, range_size):
        params = build_unilectin_params(
            ",".join(UNILECTIN_COLUMNS),
            -1,
            wherecolumn="lectin.lectin_id",
            isvalue=build_range_filter(range_template, start, end),
        )
        rows = client.request_json_with_fallback(UNILECTIN_INTERACTIONS_ENDPOINT, params)
        if not rows:
            empty_count += 1
            if empty_count >= 5:
                break
            continue
        empty_count = 0
        for row in rows:
            yield row


def stream_unilectin3d_predicted(
    client: UniLectinClient,
    superkingdoms: List[str],
    limit: int,
) -> Iterator[Dict[str, object]]:
    base_params = build_unilectin_params(
        "uniprot,protein_name,fold,class,lectomeXplore_score,species,genus,superkingdom,phylum,sequence",
        limit,
    )
    if not superkingdoms:
        for row in client.request_json_with_fallback(UNILECTIN_HUMANLECTOME_ENDPOINT, base_params):
            yield row
        return
    for kingdom in superkingdoms:
        params = build_unilectin_params(
            "uniprot,protein_name,fold,class,lectomeXplore_score,species,genus,superkingdom,phylum,sequence",
            limit,
            wherecolumn="superkingdom",
            isvalue=kingdom,
        )
        for row in client.request_json_with_fallback(UNILECTIN_HUMANLECTOME_ENDPOINT, params):
            yield row


def fetch_unilectin3d_ligands(client: UniLectinClient, limit: int) -> Iterator[Dict[str, object]]:
    params = build_unilectin_params("ligand_id,iupac,glycoct,glytoucan_id", limit)
    for row in client.request_json_with_fallback(UNILECTIN_LIGAND_ENDPOINT, params):
        yield row


def normalize_unilectin_row(row: Dict[str, object]) -> Dict[str, object]:
    lectin_id = row.get("lectin.lectin_id") or row.get("lectin_id") or ""
    normalized = {
        "lectin_id": lectin_id,
        "pdb": row.get("pdb") or "",
        "protein_name": row.get("protein_name") or "",
        "uniprot": row.get("uniprot") or "",
        "origin": row.get("origin") or "",
        "species": row.get("species") or "",
        "species_id": row.get("species_id") or "",
        "fold": row.get("fold") or "",
        "class": row.get("class") or "",
        "family": row.get("family") or "",
        "resolution": row.get("resolution") or "",
        "ligand": row.get("ligand") or "",
        "monosac": row.get("monosac") or "",
        "iupac": normalize_iupac(row.get("iupac") or ""),
        "glycoct": row.get("glycoct") or "",
        "glytoucan_id": row.get("glytoucan_id") or "",
        "timestamp_download": utc_timestamp(),
    }
    return normalized


def normalize_predicted_row(row: Dict[str, object]) -> Dict[str, object]:
    sequence = row.get("sequence") or ""
    seq_hash = hashlib.sha256(sequence.encode("utf-8")).hexdigest() if sequence else ""
    return {
        "protein_id": row.get("protein_id") or row.get("lectin_id") or "",
        "uniprot": row.get("uniprot") or row.get("UniProt_ID") or "",
        "protein_name": row.get("protein_name") or row.get("name") or "",
        "fold": row.get("infer_fold") or row.get("fold") or "",
        "class": row.get("infer_class") or row.get("class") or "",
        "score": row.get("lectomeXplore_score") or row.get("score") or "",
        "species": row.get("species") or "",
        "genus": row.get("genus") or "",
        "superkingdom": row.get("superkingdom") or "",
        "phylum": row.get("phylum") or "",
        "length": len(sequence) if sequence else row.get("length") or "",
        "sequence_hash": seq_hash,
    }


def normalize_ligand_row(row: Dict[str, object]) -> Dict[str, object]:
    iupac = normalize_iupac(row.get("iupac") or "")
    return {
        "ligand_id": row.get("ligand_id") or "",
        "iupac": iupac,
        "glycoct": row.get("glycoct") or "",
        "glytoucan_id": row.get("glytoucan_id") or "",
        "monosac_composition": extract_monosaccharide_composition(iupac),
    }


class CFGClient:
    def __init__(
        self,
        base_url: str,
        request_cfg: RequestConfig,
        logger: logging.Logger,
        error_logger: logging.Logger,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.request_cfg = request_cfg
        self.logger = logger
        self.error_logger = error_logger
        self.session = requests.Session()
        self._last_request = 0.0

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < self.request_cfg.rate_limit:
            time.sleep(self.request_cfg.rate_limit - elapsed)

    def fetch_listing(self, listing_path: str) -> str:
        url = listing_path if listing_path.startswith("http") else f"{self.base_url}{listing_path}"
        self._throttle()
        response = self.session.get(url, timeout=self.request_cfg.timeout)
        self._last_request = time.time()
        if response.status_code != 200:
            raise RuntimeError(f"CFG listing fetch failed ({response.status_code}).")
        return response.text

    def fetch_experiment_page(self, experiment_url: str) -> str:
        self._throttle()
        response = self.session.get(experiment_url, timeout=self.request_cfg.timeout)
        self._last_request = time.time()
        if response.status_code != 200:
            raise RuntimeError(f"CFG experiment fetch failed ({response.status_code}).")
        return response.text


def parse_cfg_listing(html: str) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table is not None:
        rows = table.find_all("tr")
        if not rows:
            return []
        header_cells = [cell.get_text(strip=True).lower() for cell in rows[0].find_all(["th", "td"])]
        if header_cells:
            return _parse_cfg_rows(rows[1:], header_cells)
        return _parse_cfg_rows(rows, [])

    rows = soup.find_all("tr")
    if rows:
        return _parse_cfg_rows(rows, [])

    lines = [line.strip() for line in soup.get_text("\n").splitlines() if "|" in line]
    if not lines:
        return []
    header: List[str] = []
    records = []
    for line in lines:
        parts = [part.strip() for part in line.strip("|").split("|")]
        if not header and any("id" == part.lower() for part in parts):
            header = [part.lower() for part in parts]
            continue
        if not header or len(parts) < len(header):
            continue
        record = {header[idx]: parts[idx] if idx < len(parts) else "" for idx in range(len(header))}
        experiment_id = record.get("id") or record.get("experiment_id") or parts[0]
        record["experiment_id"] = experiment_id
        records.append(record)
    return records


def _parse_cfg_rows(rows: Iterable[object], header_cells: List[str]) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for row in rows:
        cells = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
        if not cells:
            continue
        if any("ContainsDoes not contain" in cell for cell in cells):
            continue
        if header_cells and len(set(header_cells)) == len(header_cells) and len(cells) >= len(header_cells):
            record = {header_cells[idx]: cells[idx] if idx < len(cells) else "" for idx in range(len(header_cells))}
        else:
            primary_screen = ""
            if len(cells) > 1:
                primary_screen = cells[1]
            if not primary_screen and len(cells) > 2:
                primary_screen = cells[2]
            record = {
                "id": cells[0] if len(cells) > 0 else "",
                "primary screen id": primary_screen,
                "request": cells[3] if len(cells) > 3 else "",
                "sample": cells[4] if len(cells) > 4 else "",
                "description": cells[5] if len(cells) > 5 else "",
                "category": cells[6] if len(cells) > 6 else "",
                "glycan array": cells[7] if len(cells) > 7 else "",
                "investigator": cells[8] if len(cells) > 8 else "",
                "conclusive": cells[9] if len(cells) > 9 else "",
            }
        experiment_id = record.get("id") or record.get("experiment_id") or cells[0]
        record["experiment_id"] = experiment_id
        records.append(record)
    return records


def extract_cfg_max_page(html: str) -> int:
    pages = re.findall(r"gotoPage\((\d+),\s*'page'\)", html)
    if not pages:
        return 1
    return max(int(page) for page in pages)


def fetch_cfg_listing_livewire(
    client: CFGClient,
    listing_path: str,
    include_categories: Optional[List[str]],
    max_pages: Optional[int],
) -> List[Dict[str, str]]:
    html = client.fetch_listing(listing_path)
    soup = BeautifulSoup(html, "html.parser")
    snapshot_el = soup.find(attrs={"wire:snapshot": True})
    csrf_el = soup.find("meta", {"name": "csrf-token"})
    if snapshot_el is None or csrf_el is None:
        return parse_cfg_listing(html)
    snapshot = snapshot_el.get("wire:snapshot")
    csrf = csrf_el.get("content", "")
    if not snapshot or not csrf:
        return parse_cfg_listing(html)
    max_page = extract_cfg_max_page(html)
    if max_pages is not None:
        max_page = min(max_page, max_pages)
    update_url = f"{client.base_url}/livewire/update"
    headers = {
        "Content-Type": "application/json",
        "X-CSRF-TOKEN": csrf,
        "X-Requested-With": "XMLHttpRequest",
        "X-Livewire": "true",
    }
    records: List[Dict[str, str]] = []
    current_snapshot = snapshot
    for page in range(1, max_page + 1):
        client._throttle()
        payload = {
            "_token": csrf,
            "components": [
                {
                    "snapshot": current_snapshot,
                    "updates": {},
                    "calls": [
                        {
                            "path": "",
                            "method": "gotoPage",
                            "params": [page, "page"],
                        }
                    ],
                }
            ],
        }
        try:
            response = client.session.post(
                update_url, json=payload, headers=headers, timeout=client.request_cfg.timeout
            )
            client._last_request = time.time()
        except requests.RequestException as exc:
            client.logger.warning("CFG Livewire request failed for page %s: %s", page, exc)
            continue
        if response.status_code != 200:
            client.logger.warning("CFG Livewire status %s for page %s", response.status_code, page)
            continue
        try:
            data = response.json()
        except json.JSONDecodeError:
            client.logger.warning("CFG Livewire JSON parse failed for page %s", page)
            continue
        component = data.get("components", [{}])[0]
        current_snapshot = component.get("snapshot", current_snapshot)
        effects = component.get("effects", {})
        html_fragment = effects.get("html", "")
        if not html_fragment:
            client.logger.warning("CFG Livewire response missing HTML for page %s", page)
            continue
        fragment_records = parse_cfg_listing(html_fragment)
        if include_categories:
            fragment_records = [
                record
                for record in fragment_records
                if any(
                    record.get("category", "").lower() == target.lower() for target in include_categories
                )
            ]
        records.extend(fragment_records)
        if page == 1 or page % 50 == 0 or page == max_page:
            client.logger.info("CFG Livewire page %d/%d -> %d records", page, max_page, len(records))
    return records


def fetch_cfg_listing_html(
    client: CFGClient,
    listing_path: str,
    use_selenium: bool,
    headless: bool,
    js_wait: float,
) -> str:
    url = listing_path if listing_path.startswith("http") else f"{client.base_url}{listing_path}"
    if not use_selenium:
        return client.fetch_listing(url)
    if webdriver is None or Options is None:
        raise RuntimeError("selenium is not installed; install it or disable --cfg-use-selenium.")
    options = Options()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    time.sleep(js_wait)
    html = driver.page_source
    driver.quit()
    return html


def fetch_cfg_detail_html(
    client: CFGClient,
    detail_url: str,
    use_selenium: bool,
    headless: bool,
    js_wait: float,
) -> str:
    if not use_selenium:
        return client.fetch_experiment_page(detail_url)
    if webdriver is None or Options is None:
        raise RuntimeError("selenium is not installed; install it or disable --cfg-use-selenium.")
    options = Options()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(options=options)
    driver.get(detail_url)
    time.sleep(js_wait)
    html = driver.page_source
    driver.quit()
    return html


def extract_cfg_download_link(html: str, base_url: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for link in soup.find_all("a"):
        href = link.get("href") or ""
        if any(href.lower().endswith(ext) for ext in [".xls", ".xlsx", ".csv", ".tsv"]):
            return href if href.startswith("http") else f"{base_url}{href}"
        if "download" in href.lower():
            return href if href.startswith("http") else f"{base_url}{href}"
    return ""


def extract_cfg_download_form(html: str, base_url: str) -> Tuple[str, Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    form = soup.find("form", {"action": "/glycan-array/download"})
    if not form:
        return "", {}
    action = form.get("action") or "/glycan-array/download"
    action_url = action if action.startswith("http") else f"{base_url}{action}"
    payload: Dict[str, str] = {}
    for input_tag in form.find_all("input"):
        name = input_tag.get("name")
        if not name:
            continue
        payload[name] = input_tag.get("value") or ""
    return action_url, payload


def scrape_cfg_glycan_array_metadata(
    client: CFGClient,
    listing_path: str,
    experiment_limit: int,
    include_categories: Optional[List[str]],
    use_selenium: bool,
    headless: bool,
    js_wait: float,
) -> List[Dict[str, str]]:
    if use_selenium:
        html = fetch_cfg_listing_html(client, listing_path, use_selenium, headless, js_wait)
        records = parse_cfg_listing(html)
    else:
        max_pages = None
        if experiment_limit > 0:
            max_pages = max(1, (experiment_limit + 9) // 10)
        records = fetch_cfg_listing_livewire(client, listing_path, include_categories, max_pages)
    if include_categories and use_selenium:
        filtered: List[Dict[str, str]] = []
        for record in records:
            category = record.get("category", "")
            if any(category.lower() == target.lower() for target in include_categories):
                filtered.append(record)
        records = filtered
    if experiment_limit > 0:
        records = records[:experiment_limit]
    metadata: List[Dict[str, str]] = []
    for record in records:
        experiment_id = record.get("experiment_id", "")
        experiment_url = f"{client.base_url}/glycan-array/{experiment_id}"
        metadata.append(
            {
                "experiment_id": experiment_id,
                "primary_screen_id": record.get("primary screen id", record.get("primary_screen_id", "")),
                "request": record.get("request", ""),
                "sample_name": record.get("sample", record.get("sample_name", "")),
                "category": record.get("category", ""),
                "array_version": record.get("glycan array", record.get("array_version", "")),
                "investigator": record.get("investigator", ""),
                "conclusive_status": record.get("conclusive", record.get("conclusive_status", "")),
                "data_url": "",
                "download_payload": {},
                "detail_url": experiment_url,
            }
        )
    return metadata


def download_cfg_file(
    session: requests.Session,
    url: str,
    dest_path: Path,
    timeout: int,
    retry_max: int,
    method: str = "get",
    payload: Optional[Dict[str, str]] = None,
) -> Tuple[bool, str, int]:
    for attempt in range(retry_max):
        try:
            if method == "post":
                response = session.post(url, data=payload or {}, timeout=timeout)
            else:
                response = session.get(url, timeout=timeout)
            if response.status_code == 200:
                content_type = response.headers.get("content-type", "").lower()
                if "text/html" in content_type:
                    return False, "", 0
                dest_path.write_bytes(response.content)
                md5 = compute_md5_bytes(response.content)
                return True, md5, len(response.content)
            if response.status_code in {429, 500, 502, 503, 504}:
                time.sleep(2 ** attempt)
                continue
            return False, "", 0
        except requests.RequestException:
            time.sleep(2 ** attempt)
    return False, "", 0


def download_cfg_arrays(
    client: CFGClient,
    metadata: List[Dict[str, str]],
    output_dir: Path,
    workers: int,
    retry_max: int,
    recaptcha_token: Optional[str],
) -> List[Dict[str, str]]:
    if recaptcha_token is None:
        client.logger.warning("CFG downloads skipped: recaptcha token not provided.")
        return metadata
    ensure_dir(output_dir)
    updated: List[Dict[str, str]] = []
    futures = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for record in metadata:
            url = record.get("data_url", "")
            if not url:
                detail_url = record.get("detail_url", "")
                if detail_url:
                    try:
                        exp_html = client.fetch_experiment_page(detail_url)
                        url, payload = extract_cfg_download_form(exp_html, client.base_url)
                        if not url:
                            url = extract_cfg_download_link(exp_html, client.base_url)
                    except Exception as exc:
                        client.logger.warning("CFG detail fetch failed for %s: %s", detail_url, exc)
                        url = ""
                        payload = {}
                    record["data_url"] = url
                    record["download_payload"] = payload
            else:
                payload = record.get("download_payload") if isinstance(record.get("download_payload"), dict) else {}
            if not url:
                updated.append(record)
                continue
            experiment_id = record.get("experiment_id", "")
            if payload and recaptcha_token is not None:
                payload["g-recaptcha-response"] = recaptcha_token
            file_name = payload.get("file_name") if payload else ""
            ext = Path(file_name).suffix if file_name else Path(url).suffix
            if not ext:
                ext = ".dat"
            dest_path = output_dir / f"{experiment_id}{ext}"
            method = "post" if payload else "get"
            future = executor.submit(
                download_cfg_file,
                client.session,
                url,
                dest_path,
                client.request_cfg.timeout,
                retry_max,
                method,
                payload if payload else None,
            )
            futures[future] = (experiment_id, url)
            record["file_path"] = dest_path.as_posix()
            updated.append(record)
        for future in as_completed(futures):
            success, _, _ = future.result()
            experiment_id, url = futures[future]
            if not success:
                client.logger.warning("CFG download failed for %s (%s)", experiment_id, url)
    for record in updated:
        file_path = record.get("file_path", "")
        if file_path and Path(file_path).exists():
            record["file_size_bytes"] = str(Path(file_path).stat().st_size)
            record["file_hash_md5"] = compute_md5_file(Path(file_path))
            record["download_date"] = utc_timestamp()
    return updated


def read_cfg_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    if path.suffix.lower() in {".csv"}:
        return pd.read_csv(path)
    return pd.read_csv(path, sep=None, engine="python")


def parse_cfg_rfu_data(
    metadata: List[Dict[str, str]],
    output_path: Path,
    glycan_lookup: Dict[str, Dict[str, str]],
    logger: logging.Logger,
) -> int:
    row_count = 0
    ensure_dir(output_path.parent)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=CFG_RFU_COLUMNS,
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        writer.writeheader()
        for record in metadata:
            file_path = record.get("file_path", "")
            if not file_path or not Path(file_path).exists():
                continue
            try:
                df = read_cfg_file(Path(file_path))
            except Exception as exc:
                logger.warning("CFG parse failed for %s: %s", file_path, exc)
                continue
            if df.empty:
                continue
            glycan_col = df.columns[0]
            lectin_cols = df.columns[1:]
            max_val = pd.to_numeric(df[lectin_cols].stack(), errors="coerce").max()
            normalization_status = "normalized" if max_val <= 1 else "raw->normalized"
            for _, row in df.iterrows():
                glycan_id = safe_str(row.get(glycan_col))
                lookup = glycan_lookup.get(glycan_id, {})
                glycan_iupac = lookup.get("iupac", "")
                glycan_gt = lookup.get("glytoucan_id", "")
                for lectin_col in lectin_cols:
                    value = row.get(lectin_col)
                    if pd.isna(value):
                        continue
                    try:
                        rfu_raw = float(value)
                    except (TypeError, ValueError):
                        continue
                    rfu_norm = rfu_raw / max_val if max_val and max_val > 1 else rfu_raw
                    writer.writerow(
                        {
                            "experiment_id": record.get("experiment_id", ""),
                            "array_version": record.get("array_version", ""),
                            "glycan_id_cfg": glycan_id,
                            "glycan_iupac": glycan_iupac,
                            "glycan_glytoucan_id": glycan_gt,
                            "lectin_sample_name": lectin_col,
                            "rfu_raw": rfu_raw,
                            "rfu_normalized": rfu_norm,
                            "normalization_status": normalization_status,
                            "investigator": record.get("investigator", ""),
                            "conclusive": record.get("conclusive_status", ""),
                        }
                    )
                    row_count += 1
    return row_count


def map_cfg_lectins_to_unilectin(
    cfg_measurements_path: Path,
    unilectin_path: Path,
    output_path: Path,
    logger: logging.Logger,
) -> int:
    if not cfg_measurements_path.exists() or not unilectin_path.exists():
        return 0
    cfg_names: set[str] = set()
    for chunk in pd.read_csv(cfg_measurements_path, usecols=["lectin_sample_name"], chunksize=20000):
        cfg_names.update(name for name in chunk["lectin_sample_name"].dropna().unique())
    unilectin_df = pd.read_csv(unilectin_path)
    mapping_rows: List[Dict[str, object]] = []
    for name in sorted(cfg_names):
        candidates = unilectin_df[unilectin_df["protein_name"].str.contains(name, case=False, na=False)]
        confidence = 0.0
        row = None
        if not candidates.empty:
            row = candidates.iloc[0]
            confidence = 1.0
        else:
            best_score = 0.0
            for _, candidate in unilectin_df.iterrows():
                cand_name = safe_str(candidate.get("protein_name"))
                score = difflib_ratio(name, cand_name)
                if score > best_score:
                    best_score = score
                    row = candidate
            confidence = best_score
        mapping_rows.append(
            {
                "cfg_lectin_name": name,
                "unilectin3d_lectin_id": row.get("lectin_id") if row is not None else "",
                "uniprot_id": row.get("uniprot") if row is not None else "",
                "pdb": row.get("pdb") if row is not None else "",
                "protein_name": row.get("protein_name") if row is not None else "",
                "confidence_score": round(confidence, 3),
            }
        )
    count = write_csv(output_path, CFG_MAPPING_COLUMNS, mapping_rows)
    logger.info("CFG->UniLectin mapping rows: %d", count)
    return count


def difflib_ratio(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()


def merge_unilectin_cfg_data(
    unilectin_path: Path,
    cfg_measurements_path: Path,
    cfg_mapping_path: Path,
    output_path: Path,
    logger: logging.Logger,
) -> int:
    ensure_dir(output_path.parent)
    if not unilectin_path.exists():
        return 0
    unilectin_df = pd.read_csv(unilectin_path)
    mapping_df = pd.read_csv(cfg_mapping_path) if cfg_mapping_path.exists() else pd.DataFrame()
    mapping = {}
    if not mapping_df.empty:
        mapping = {
            safe_str(row["cfg_lectin_name"]): row
            for _, row in mapping_df.iterrows()
            if safe_str(row.get("cfg_lectin_name"))
        }

    row_count = 0
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=UNIFIED_COLUMNS,
            quoting=csv.QUOTE_MINIMAL,
            lineterminator="\n",
        )
        writer.writeheader()
        for _, row in unilectin_df.iterrows():
            metadata = {
                "source": "UniLectin3D",
                "pdb": safe_str(row.get("pdb")),
                "resolution": safe_str(row.get("resolution")),
            }
            writer.writerow(
                {
                    "lectin_id": safe_str(row.get("lectin_id")),
                    "lectin_name": safe_str(row.get("protein_name")),
                    "lectin_origin": safe_str(row.get("origin")),
                    "lectin_species": safe_str(row.get("species")),
                    "lectin_uniprot": safe_str(row.get("uniprot")),
                    "lectin_pdb": safe_str(row.get("pdb")),
                    "glycan_id": "",
                    "glycan_iupac": safe_str(row.get("iupac")),
                    "glycan_glytoucan_id": safe_str(row.get("glytoucan_id")),
                    "glycan_monosac_composition": extract_monosaccharide_composition(
                        safe_str(row.get("iupac"))
                    ),
                    "binding_value": "",
                    "binding_unit": "nM",
                    "binding_source": "UniLectin3D",
                    "binding_confidence": "high",
                    "binding_source_metadata_json": json.dumps(metadata),
                }
            )
            row_count += 1
        if cfg_measurements_path.exists():
            for chunk in pd.read_csv(cfg_measurements_path, chunksize=50000):
                for _, row in chunk.iterrows():
                    cfg_name = safe_str(row.get("lectin_sample_name"))
                    map_row = mapping.get(cfg_name, {})
                    metadata = {
                        "source": "CFG",
                        "experiment_id": safe_str(row.get("experiment_id")),
                        "array_version": safe_str(row.get("array_version")),
                        "investigator": safe_str(row.get("investigator")),
                        "conclusive": safe_str(row.get("conclusive")),
                    }
                    writer.writerow(
                        {
                            "lectin_id": safe_str(map_row.get("unilectin3d_lectin_id")),
                            "lectin_name": cfg_name,
                            "lectin_origin": "",
                            "lectin_species": "",
                            "lectin_uniprot": safe_str(map_row.get("uniprot_id")),
                            "lectin_pdb": safe_str(map_row.get("pdb")),
                            "glycan_id": safe_str(row.get("glycan_id_cfg")),
                            "glycan_iupac": safe_str(row.get("glycan_iupac")),
                            "glycan_glytoucan_id": safe_str(row.get("glycan_glytoucan_id")),
                            "glycan_monosac_composition": extract_monosaccharide_composition(
                                safe_str(row.get("glycan_iupac"))
                            ),
                            "binding_value": safe_str(row.get("rfu_raw")),
                            "binding_unit": "RFU",
                            "binding_source": "CFG",
                            "binding_confidence": "high"
                            if safe_str(row.get("conclusive")) == "Data"
                            else "medium",
                            "binding_source_metadata_json": json.dumps(metadata),
                        }
                    )
                    row_count += 1
    logger.info("Unified dataset rows: %d", row_count)
    return row_count


def orchestrate_phase2_data_download(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    log_dir = Path(args.log_dir)
    logger, error_logger = setup_logging(log_dir, args.log_level)

    request_cfg = RequestConfig(rate_limit=args.rate_limit, retry_max=args.retry_max, timeout=args.timeout)
    cache_dir = output_dir / "cache"
    cache = ResponseCache(cache_dir)
    unilectin_output = output_dir / "unilectin3d_lectin_glycan_interactions.csv"
    predicted_output = output_dir / "unilectin3d_predicted_lectins.csv"
    ligand_output = output_dir / "unilectin3d_ligands.csv"
    if args.skip_unilectin:
        unilectin_count = write_csv(unilectin_output, UNILECTIN_OUTPUT_COLUMNS, [])
        predicted_count = write_csv(predicted_output, PREDICTED_OUTPUT_COLUMNS, [])
        ligand_count = write_csv(ligand_output, LIGAND_OUTPUT_COLUMNS, [])
        logger.info("Skipped UniLectin downloads.")
    else:
        unilectin_client = UniLectinClient(
            args.unilectin_base_url,
            request_cfg,
            cache,
            logger,
            error_logger,
            force_refresh=args.force_refresh,
        )
        unilectin_rows = (
            normalize_unilectin_row(row)
            for row in stream_unilectin3d_lectins(
                unilectin_client,
                args.unilectin_limit,
                args.unilectin_range_size,
                args.unilectin_max_id,
                args.unilectin_range_template,
            )
        )
        unilectin_count = write_csv(unilectin_output, UNILECTIN_OUTPUT_COLUMNS, unilectin_rows)
        logger.info("Wrote UniLectin interactions: %d rows", unilectin_count)

        predicted_rows = (
            normalize_predicted_row(row)
            for row in stream_unilectin3d_predicted(
                unilectin_client,
                args.superkingdoms,
                args.unilectin_limit,
            )
        )
        predicted_count = write_csv(predicted_output, PREDICTED_OUTPUT_COLUMNS, predicted_rows)
        logger.info("Wrote predicted lectins: %d rows", predicted_count)

        ligand_rows = (
            normalize_ligand_row(row) for row in fetch_unilectin3d_ligands(unilectin_client, args.unilectin_limit)
        )
        ligand_count = write_csv(ligand_output, LIGAND_OUTPUT_COLUMNS, ligand_rows)
        logger.info("Wrote ligands: %d rows", ligand_count)
        if ligand_count == 0 and unilectin_count > 0:
            unilectin_df = pd.read_csv(unilectin_output)
            ligand_rows_fallback: List[Dict[str, object]] = []
            seen: set[str] = set()
            for _, row in unilectin_df.iterrows():
                iupac = safe_str(row.get("iupac"))
                glycoct = safe_str(row.get("glycoct"))
                glytoucan_id = safe_str(row.get("glytoucan_id"))
                if not (iupac or glycoct or glytoucan_id):
                    continue
                ligand_id = glytoucan_id or f"IUPAC_{hashlib.md5(iupac.encode('utf-8')).hexdigest()}"
                if ligand_id in seen:
                    continue
                seen.add(ligand_id)
                ligand_rows_fallback.append(
                    {
                        "ligand_id": ligand_id,
                        "iupac": iupac,
                        "glycoct": glycoct,
                        "glytoucan_id": glytoucan_id,
                        "monosac_composition": extract_monosaccharide_composition(iupac),
                    }
                )
            ligand_count = write_csv(ligand_output, LIGAND_OUTPUT_COLUMNS, ligand_rows_fallback)
            logger.info("Derived ligands from UniLectin interactions: %d rows", ligand_count)

    cfg_client = CFGClient(
        args.cfg_base_url,
        request_cfg,
        logger,
        error_logger,
    )
    include_categories = []
    if args.cfg_include_categories:
        include_categories = [item.strip() for item in args.cfg_include_categories.split(",") if item.strip()]
    cfg_metadata = scrape_cfg_glycan_array_metadata(
        cfg_client,
        args.cfg_listing_path,
        args.cfg_experiment_limit,
        include_categories or None,
        args.cfg_use_selenium,
        args.headless_browser,
        args.cfg_js_wait,
    )
    cfg_metadata_path = output_dir / "cfg_experiment_metadata.csv"
    cfg_metadata_count = write_csv(cfg_metadata_path, CFG_METADATA_COLUMNS, cfg_metadata)
    logger.info("Wrote CFG metadata: %d rows", cfg_metadata_count)
    cfg_raw_dir = output_dir / "cfg_arrays_raw"
    cfg_metadata = download_cfg_arrays(
        cfg_client,
        cfg_metadata,
        cfg_raw_dir,
        args.parallel_workers,
        args.retry_max,
        args.cfg_recaptcha_token,
    )
    cfg_metadata_count = write_csv(cfg_metadata_path, CFG_METADATA_COLUMNS, cfg_metadata)
    logger.info("Updated CFG metadata: %d rows", cfg_metadata_count)

    glycan_lookup: Dict[str, Dict[str, str]] = {}
    if ligand_output.exists():
        for _, row in pd.read_csv(ligand_output).iterrows():
            ligand_id = safe_str(row.get("ligand_id"))
            if ligand_id:
                glycan_lookup[ligand_id] = {
                    "iupac": safe_str(row.get("iupac")),
                    "glytoucan_id": safe_str(row.get("glytoucan_id")),
                }

    cfg_rfu_output = output_dir / "cfg_rfu_measurements.csv"
    cfg_rfu_count = parse_cfg_rfu_data(cfg_metadata, cfg_rfu_output, glycan_lookup, logger)
    logger.info("Wrote CFG RFU measurements: %d rows", cfg_rfu_count)

    cfg_mapping_output = output_dir / "cfg_to_unilectin_lectin_mapping.csv"
    map_count = map_cfg_lectins_to_unilectin(cfg_rfu_output, unilectin_output, cfg_mapping_output, logger)

    unified_output = output_dir / "glycoml_phase2_unified_lectin_glycan_interactions.csv"
    unified_count = merge_unilectin_cfg_data(unilectin_output, cfg_rfu_output, cfg_mapping_output, unified_output, logger)

    metadata_path = output_dir / "download_metadata.csv"
    record_metadata(metadata_path, unilectin_output, unilectin_count)
    record_metadata(metadata_path, predicted_output, predicted_count)
    record_metadata(metadata_path, ligand_output, ligand_count)
    record_metadata(metadata_path, cfg_metadata_path, cfg_metadata_count)
    record_metadata(metadata_path, cfg_rfu_output, cfg_rfu_count)
    record_metadata(metadata_path, cfg_mapping_output, map_count)
    record_metadata(metadata_path, unified_output, unified_count)

    logger.info("Phase 2 acquisition complete.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 2 UniLectin3D + CFG downloader")
    parser.add_argument("--unilectin-limit", type=int, default=-1)
    parser.add_argument("--cfg-experiment-limit", type=int, default=0)
    parser.add_argument("--output-dir", default="./data")
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--retry-max", type=int, default=5)
    parser.add_argument("--parallel-workers", type=int, default=5)
    parser.add_argument("--rate-limit", type=float, default=0.1)
    parser.add_argument("--timeout", type=int, default=15)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--skip-unilectin", action="store_true", help="Skip UniLectin3D downloads")

    parser.add_argument("--unilectin-base-url", default=DEFAULT_UNILECTIN_BASE)
    parser.add_argument("--unilectin-range-size", type=int, default=1000)
    parser.add_argument("--unilectin-max-id", type=int, default=50000)
    parser.add_argument("--unilectin-range-template", default="{start}-{end}")
    parser.add_argument("--superkingdoms", nargs="*", default=[])

    parser.add_argument("--cfg-base-url", default=DEFAULT_CFG_BASE)
    parser.add_argument("--cfg-listing-path", default=f"{DEFAULT_CFG_BASE}/glycan-array")
    parser.add_argument("--cfg-include-categories", help="Comma-separated CFG categories to include")
    parser.add_argument("--cfg-use-selenium", action="store_true", help="Use selenium for JS-rendered CFG pages")
    parser.add_argument("--headless-browser", action="store_true", help="Run selenium in headless mode")
    parser.add_argument("--cfg-js-wait", type=float, default=5.0, help="Seconds to wait for CFG JS tables to load")
    parser.add_argument("--cfg-recaptcha-token", help="Optional reCAPTCHA token for CFG downloads")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    orchestrate_phase2_data_download(args)


if __name__ == "__main__":
    main()
