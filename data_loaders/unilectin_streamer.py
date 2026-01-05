"""Streaming utilities for lectin-glycan binding data from UniLectin3D."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd
import requests
from tqdm import tqdm

# Optional RDKit dependency
try:  # pragma: no cover - optional heavy dependency
    from rdkit import Chem
except Exception:  # pragma: no cover
    Chem = None

UNILECTIN_ENDPOINTS = [
    "https://unilectin.unige.ch/api/getlectins",
]
GLYTOUCAN_ENDPOINT = "https://glytoucan.org/api/glycans/{glytoucan_id}"
UNIPROT_FASTA = "https://www.uniprot.org/uniprot/{uniprot_id}.fasta"

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"


def _setup_logger(log_path: Path, verbose: bool) -> logging.Logger:
    logger = logging.getLogger("unilectin_streamer")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if verbose:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
    return logger


def _now() -> float:
    return time.time()


def _looks_like_iupac(value: str) -> bool:
    tokens = ["GAL", "GLC", "MAN", "NEU", "FUC", "GLCNAC", "GALNAC"]
    upper = value.upper()
    return any(token in upper for token in tokens)


class UniLectinStreamer:
    """Stream lectin-glycan binding pairs from UniLectin3D.

    Parameters
    ----------
    filter_family : str, optional
        Restrict lectins to a family (e.g., "Siglec").
    filter_organism : str, optional
        Restrict lectins to an organism class (e.g., "Human").
    cache_dir : str, default "./data/unilectin_cache"
        Directory for cached metadata, sequences, and SMILES strings.
    rate_limit : float, default 0.2
        Minimum seconds between HTTP calls.
    verbose : bool, default True
        Emit logs to stdout in addition to file logging.
    max_pairs : int, optional
        Maximum number of pairs to yield.
    allow_iupac_fallback : bool, default True
        Use IUPAC strings when SMILES cannot be resolved.
    """

    def __init__(
        self,
        filter_family: Optional[str] = None,
        filter_organism: Optional[str] = None,
        cache_dir: str = "./data/unilectin_cache",
        rate_limit: float = 0.2,
        verbose: bool = True,
        max_pairs: Optional[int] = None,
        allow_iupac_fallback: bool = True,
    ) -> None:
        self.filter_family = filter_family
        self.filter_organism = filter_organism
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.seq_dir = self.cache_dir / "sequences"
        self.seq_dir.mkdir(parents=True, exist_ok=True)
        self.smiles_dir = self.cache_dir / "smiles"
        self.smiles_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = max(rate_limit, 0.0)
        self.max_pairs = max_pairs
        self.allow_iupac_fallback = allow_iupac_fallback
        self.iupac_fallback: set[str] = set()
        self._last_request: float = 0.0
        self.metadata_path = self.cache_dir / "lectin_metadata.json"
        self.manifest_path = self.cache_dir / "cache_manifest.json"
        self.log_path = Path("logs") / "unilectin_streamer.log"
        self.logger = _setup_logger(self.log_path, verbose)
        self.session = requests.Session()

    def _throttle(self) -> None:
        delta = _now() - self._last_request
        if delta < self.rate_limit:
            time.sleep(self.rate_limit - delta)

    def _save_json(self, path: Path, payload: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def _load_manifest(self) -> Dict[str, object]:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as handle:
                    return json.load(handle)
            except json.JSONDecodeError:
                return {}
        return {}

    def fetch_all_lectins_metadata(self) -> List[Dict]:
        """Retrieve full metadata from UniLectin, using a 7-day cache.

        Returns
        -------
        list of dict
            Raw metadata records as returned by the UniLectin API.
        """
        manifest = self._load_manifest()
        meta_entry = manifest.get("lectin_metadata", {})
        if (
            meta_entry
            and isinstance(meta_entry, dict)
            and meta_entry.get("path")
            and (Path(meta_entry["path"]).exists())
            and (_now() - float(meta_entry.get("timestamp", 0)) < 7 * 24 * 3600)
        ):
            try:
                with open(meta_entry["path"], "r", encoding="utf-8") as handle:
                    cached = json.load(handle)
                self.logger.info("Loaded UniLectin metadata from cache")
                return cached
            except Exception:
                pass

        query = {
            "getcolumns": "lectin.lectin_id,protein_name,ligand,iupac,glytoucan_id,uniprot,family,origin,species",
            "wherecolumn": "lectin.lectin_id",
            "isvalue": "%%",
            "limit": "-1",
        }
        payload: Optional[List[Dict]] = None
        for endpoint in UNILECTIN_ENDPOINTS:
            self._throttle()
            try:
                resp = self.session.post(endpoint, json=query, timeout=30)
                self._last_request = _now()
                resp.raise_for_status()
                parsed = resp.json()
                if isinstance(parsed, list) and parsed:
                    header = parsed[0]
                    rows = parsed[1:]
                    if isinstance(header, list):
                        payload = [dict(zip(header, row)) for row in rows if isinstance(row, list)]
                        break
            except Exception as exc:  # pragma: no cover - network dependent
                self.logger.error("UniLectin metadata fetch failed (%s): %s", endpoint, exc)
                continue

        if payload is None:
            return []

        self._save_json(self.metadata_path, payload)
        manifest["lectin_metadata"] = {"path": str(self.metadata_path), "timestamp": _now()}
        self._save_json(self.manifest_path, manifest)
        self.logger.info("Fetched %d UniLectin records", len(payload))
        return payload

    def validate_lectin_sequence(self, seq: str) -> bool:
        seq_clean = seq.strip().upper()
        if len(seq_clean) <= 50:
            return False
        valid = set("ACDEFGHIKLMNPQRSTVWY")
        return all(ch in valid for ch in seq_clean)

    def fetch_lectin_sequence(self, uniprot_id: str) -> Optional[str]:
        """Download a lectin sequence from UniProt.

        Parameters
        ----------
        uniprot_id : str
            UniProt accession.

        Returns
        -------
        str or None
            Amino acid sequence if valid, otherwise None.
        """
        uniprot_id = uniprot_id.strip()
        cache_path = self.seq_dir / f"{uniprot_id}.fasta"
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as handle:
                lines = handle.read().splitlines()
            seq = "".join(line.strip() for line in lines if not line.startswith(">"))
            if seq:
                return seq

        url = UNIPROT_FASTA.format(uniprot_id=uniprot_id)
        self._throttle()
        try:
            resp = self.session.get(url, timeout=20)
            self._last_request = _now()
            resp.raise_for_status()
            lines = resp.text.splitlines()
            seq = "".join(line.strip() for line in lines if not line.startswith(">"))
        except Exception as exc:  # pragma: no cover - network dependent
            self.logger.warning("Failed to fetch UniProt %s: %s", uniprot_id, exc)
            return None

        if not self.validate_lectin_sequence(seq):
            self.logger.warning("Sequence failed validation for %s", uniprot_id)
            return None
        cache_path.write_text(resp.text, encoding="utf-8")
        return seq

    def validate_glycan_smiles(self, smiles: str) -> bool:
        if Chem is None:  # pragma: no cover - optional dependency
            return bool(smiles)
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False
        if mol.GetNumAtoms() < 5:
            return False
        mw = sum(atom.GetMass() for atom in mol.GetAtoms())
        return 100.0 <= mw <= 2000.0

    def fetch_glycan_smiles(self, glytoucan_id: str, iupac_name: str) -> Optional[str]:
        """Resolve a glycan SMILES string from GlyTouCan or IUPAC.

        Parameters
        ----------
        glytoucan_id : str
            GlyTouCan accession.
        iupac_name : str
            IUPAC glycan description used as fallback.

        Returns
        -------
        str or None
            Canonical SMILES string if resolved and valid.
        """
        glytoucan_id = (glytoucan_id or "").strip()
        cache_key = glytoucan_id or iupac_name.replace(" ", "_")[:32]
        cache_path = self.smiles_dir / f"{cache_key}.smi"
        if cache_path.exists():
            smiles_cached = cache_path.read_text(encoding="utf-8").strip()
            if smiles_cached:
                if self.allow_iupac_fallback and _looks_like_iupac(smiles_cached):
                    self.iupac_fallback.add(smiles_cached)
                return smiles_cached

        smiles_val: Optional[str] = None
        if glytoucan_id:
            url = GLYTOUCAN_ENDPOINT.format(glytoucan_id=glytoucan_id)
            self._throttle()
            try:
                resp = self.session.get(url, timeout=20)
                self._last_request = _now()
                if resp.status_code == 200:
                    payload = resp.json()
                    smiles_val = payload.get("smiles") or payload.get("structure", {}).get("smiles")
            except Exception as exc:  # pragma: no cover
                self.logger.warning("Failed GlyTouCan fetch for %s: %s", glytoucan_id, exc)

        if smiles_val is None and iupac_name and self.allow_iupac_fallback:
            smiles_val = iupac_name
            self.iupac_fallback.add(smiles_val)
            cache_path.write_text(smiles_val, encoding="utf-8")
            self.logger.warning("Using IUPAC fallback for %s", glytoucan_id or iupac_name)
            return smiles_val

        if smiles_val and self.validate_glycan_smiles(smiles_val):
            cache_path.write_text(smiles_val, encoding="utf-8")
            return smiles_val

        self.logger.warning("Invalid or missing SMILES for %s", glytoucan_id or iupac_name)
        return None

    def normalize_binding_affinity(self, kd_nm: float) -> Tuple[float, str]:
        """Convert Kd (nM) to RFU scale and binding class."""
        kd_nm = float(kd_nm)
        f_bound = 1.0 / (1.0 + kd_nm / 1000.0)
        rfu = f_bound * 10000.0
        if rfu > 5000:
            cls = "strong"
        elif rfu > 2000:
            cls = "medium"
        else:
            cls = "weak"
        return rfu, cls

    def _stream_from_records(
        self,
        records: Iterable[Dict],
    ) -> Iterator[Tuple[str, str, str, str, str, str, str, float, float, str, str]]:
        filtered = []
        for record in records:
            family = str(record.get("family") or "")
            organism = str(record.get("origin") or record.get("species") or "")
            if self.filter_family:
                if self.filter_family.lower() not in family.lower():
                    continue
            if self.filter_organism:
                needle = self.filter_organism.lower()
                haystack = organism.lower()
                if needle not in haystack:
                    if needle in {"mammalian", "mammal", "mammals"} and "animal" in haystack:
                        pass
                    else:
                        continue
            filtered.append(record)

        def _kd_key(rec: Dict) -> float:
            try:
                return float(rec.get("kd_value"))
            except Exception:
                return float("inf")

        filtered.sort(key=_kd_key)
        count = 0
        for record in tqdm(filtered, desc="Lectin-glycan pairs", unit="pair"):
            lectin_id = str(record.get("lectin_id") or record.get("lectin.lectin_id") or "").strip()
            lectin_name = str(record.get("protein_name") or lectin_id or "unknown")
            glytoucan_id = str(record.get("glytoucan_id") or "").strip()
            iupac = str(record.get("iupac") or "").strip()
            uniprot_id = str(record.get("uniprot") or "").strip()
            family = str(record.get("family") or "")
            organism = str(record.get("origin") or record.get("species") or "")
            kd_value = record.get("kd_value")
            method = str(record.get("method") or "")
            try:
                kd_nm = float(kd_value) if kd_value is not None else float("nan")
            except Exception:
                kd_nm = float("nan")

            seq = self.fetch_lectin_sequence(uniprot_id) if uniprot_id else ""
            if not seq or not self.validate_lectin_sequence(seq):
                self.logger.warning("Skipping %s due to invalid sequence", lectin_id)
                continue

            smiles = self.fetch_glycan_smiles(glytoucan_id, iupac)
            if not smiles:
                self.logger.warning("Skipping %s due to invalid SMILES", lectin_id)
                continue
            if smiles not in self.iupac_fallback and not self.validate_glycan_smiles(smiles):
                self.logger.warning("Skipping %s due to invalid SMILES", lectin_id)
                continue

            if not pd.isna(kd_nm):
                kd_molar = kd_nm * 1e-9
                if not (1e-12 <= kd_molar <= 1e-3):
                    self.logger.warning("K_d out of range for %s", lectin_id)

            rfu, cls = self.normalize_binding_affinity(kd_nm if not pd.isna(kd_nm) else 1000.0)

            self.logger.info("Pair %s (%s) → %s RFU %.0f", lectin_id, family, cls, rfu)
            yield (
                lectin_id,
                lectin_name,
                seq,
                family or organism,
                iupac,
                smiles,
                glytoucan_id,
                kd_nm,
                rfu,
                cls,
                method,
            )
            count += 1
            if self.max_pairs and count >= self.max_pairs:
                self.logger.info("Reached max_pairs limit (%d).", self.max_pairs)
                break
            self._throttle()

    def stream_pairs(self) -> Iterator[Tuple[str, str, str, str, str, str, str, float, float, str, str]]:
        """Stream lectin-glycan pairs with validation.

        Yields
        ------
        tuple
            (lectin_id, lectin_name, lectin_seq, family, iupac, smiles, glytoucan_id, kd_nm, rfu, class, method)
        """
        metadata = self.fetch_all_lectins_metadata()
        yield from self._stream_from_records(metadata)

    def stream_pairs_from_records(
        self,
        records: Iterable[Dict],
    ) -> Iterator[Tuple[str, str, str, str, str, str, str, float, float, str, str]]:
        """Stream lectin-glycan pairs from pre-fetched metadata records."""
        yield from self._stream_from_records(records)

    def stream_pairs_filtered(
        self,
        binding_strength: Optional[str] = None,
        min_sequence_length: int = 50,
    ) -> Iterator[Tuple[str, str, str, str, str, str, str, float, float, str, str]]:
        """Wrapper over :meth:`stream_pairs` with simple filtering."""
        for item in self.stream_pairs():
            lectin_id, lectin_name, seq, family, iupac, smiles, glytoucan_id, kd_nm, rfu, cls, method = item
            if binding_strength and cls != binding_strength:
                continue
            if len(seq) < min_sequence_length:
                continue
            yield item


__all__ = ["UniLectinStreamer"]
