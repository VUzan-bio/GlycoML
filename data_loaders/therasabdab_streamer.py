"""Streaming utilities for therapeutic antibody structures from RCSB/thera-SAbDab.

This module provides the :class:`TheraSAbDabStreamer` which queries RCSB for
antibody PDB identifiers, downloads and caches structures, extracts sequences,
identifies glycosylation motifs, and computes per-site SASA and pLDDT values.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import pandas as pd
import requests
from Bio.PDB import PDBParser, Polypeptide
from Bio.SeqUtils import seq1
from Bio.PDB.Structure import Structure
from tqdm import tqdm

# Optional heavy-weight dependency; handled gracefully if absent.
try:
    import freesasa  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    freesasa = None

RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_PDB_DOWNLOAD = "https://files.rcsb.org/download/{pdb_id}.pdb"

LOG_FORMAT = "[%(asctime)s] [%(levelname)s] %(message)s"


def _setup_logger(log_path: Path, verbose: bool) -> logging.Logger:
    """Configure module logger.

    Parameters
    ----------
    log_path : Path
        Destination path for the log file.
    verbose : bool
        Whether to also emit logs to stdout.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger("therasabdab_streamer")
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


def _motif_type(seq: str, idx: int) -> str:
    return f"N{seq[idx + 1]}{seq[idx + 2]}"


def _find_nglyco(sequence: str) -> List[Tuple[int, str]]:
    """Find N-X-[S/T] motifs where X is not proline.

    Parameters
    ----------
    sequence : str
        Amino acid sequence (uppercase expected).

    Returns
    -------
    list of tuple
        List of (position, motif_type) pairs using 1-based indexing.

    Examples
    --------
    >>> _find_nglyco("ANST")
    [(2, 'NXT')]
    """
    hits: List[Tuple[int, str]] = []
    seq = sequence.upper()
    for i in range(len(seq) - 2):
        if seq[i] != "N" or seq[i + 1] == "P":
            continue
        if seq[i + 2] in {"S", "T"}:
            hits.append((i + 1, _motif_type(seq, i)))
    return hits


@dataclass
class ChainData:
    chain_id: str
    sequence: str
    residues: List[object]


@dataclass
class GlycoSite:
    position: int
    residue: str
    motif_type: str
    chain: str
    plddt: float
    sasa: float
    accessibility_rank: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "position": self.position,
            "residue": self.residue,
            "motif_type": self.motif_type,
            "chain": self.chain,
            "plddt": self.plddt,
            "sasa": self.sasa,
            "accessibility_rank": self.accessibility_rank,
        }


class TheraSAbDabStreamer:
    """Stream therapeutic antibody structures and glycosylation sites.

    Parameters
    ----------
    pdb_ids : list of str, optional
        PDB identifiers to process. If None, the RCSB search API is queried.
    cache_dir : str, default "./data/pdb_cache"
        Directory for cached PDB downloads and manifest.
    rate_limit : float, default 1.0
        Minimum seconds between HTTP requests to respect rate limits.
    verbose : bool, default True
        Whether to log to stdout in addition to file logging.
    query_limit : int, default 500
        Maximum number of PDB IDs to return from RCSB query.
    resume_from_checkpoint : bool, default False
        Whether to skip PDB IDs already present in a checkpoint JSONL file.
    checkpoint_file : str, optional
        Path to a JSONL file produced by the orchestrator for resuming streams.

    Examples
    --------
    >>> streamer = TheraSAbDabStreamer(pdb_ids=["1HZH"], rate_limit=0.0, verbose=False)
    >>> isinstance(streamer.pdb_ids, list)
    True
    """

    def __init__(
        self,
        pdb_ids: Optional[List[str]] = None,
        cache_dir: str = "./data/pdb_cache",
        rate_limit: float = 1.0,
        verbose: bool = True,
        query_limit: int = 500,
        resume_from_checkpoint: bool = False,
        checkpoint_file: Optional[str] = None,
    ) -> None:
        self.pdb_ids = pdb_ids or []
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit = max(rate_limit, 0.0)
        self.query_limit = query_limit
        self._last_request: float = 0.0
        self.resume_from_checkpoint = resume_from_checkpoint
        self.checkpoint_file = Path(checkpoint_file) if checkpoint_file else None
        self.streamed_pdb_ids: set[str] = set()
        self.manifest_path = self.cache_dir / "cache_manifest.json"
        self.manifest: Dict[str, Dict[str, object]] = self._load_manifest()
        self.log_path = Path("logs") / "therasabdab_streamer.log"
        self.logger = _setup_logger(self.log_path, verbose)
        self.session = requests.Session()
        if self.resume_from_checkpoint and self.checkpoint_file and self.checkpoint_file.exists():
            self._load_checkpoint()

    def _load_checkpoint(self) -> None:
        if not self.checkpoint_file:
            return
        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    pdb_id = record.get("pdb_id")
                    if pdb_id:
                        self.streamed_pdb_ids.add(str(pdb_id))
            self.logger.info("Loaded checkpoint with %d PDB IDs", len(self.streamed_pdb_ids))
        except Exception as exc:
            self.logger.warning("Failed to load checkpoint: %s", exc)

    def _load_manifest(self) -> Dict[str, Dict[str, object]]:
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as handle:
                    return json.load(handle)
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_manifest(self) -> None:
        with open(self.manifest_path, "w", encoding="utf-8") as handle:
            json.dump(self.manifest, handle, indent=2)

    def _throttle(self) -> None:
        delta = _now() - self._last_request
        if delta < self.rate_limit:
            time.sleep(self.rate_limit - delta)

    def query_rcsb_antibodies(self) -> List[str]:
        """Query RCSB for antibody-like entries.

        Returns
        -------
        list of str
            PDB identifiers limited to ``query_limit`` entries.
        """
        query = {
            "query": {
                "type": "terminal",
                "service": "full_text",
                "parameters": {"value": "antibody OR immunoglobulin"},
            },
            "request_options": {
                "return_all_hits": True,
                "results_content_type": ["experimental"],
                "sort": [{"sort_by": "score", "direction": "desc"}],
            },
            "return_type": "entry",
        }
        self._throttle()
        try:
            response = self.session.post(RCSB_SEARCH_URL, json=query, timeout=20)
            self._last_request = _now()
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:  # pragma: no cover - network dependent
            self.logger.error("RCSB query failed: %s", exc)
            return []

        ids: List[str] = [item.get("identifier", "").upper() for item in payload.get("result_set", [])]
        ids = [pid for pid in ids if pid]
        if self.query_limit:
            ids = ids[: self.query_limit]
        self.logger.info("RCSB query returned %d entries", len(ids))
        self.pdb_ids = ids
        return ids

    def download_pdb(self, pdb_id: str) -> Optional[Path]:
        """Download a PDB file with caching and validation.

        Parameters
        ----------
        pdb_id : str
            Four-character PDB identifier.

        Returns
        -------
        Path or None
            Local filesystem path to the cached PDB, or None on failure.
        """
        pdb_id = pdb_id.lower().strip()
        target = self.cache_dir / f"{pdb_id}.pdb"
        if target.exists() and target.stat().st_size > 1024:
            return target

        url = RCSB_PDB_DOWNLOAD.format(pdb_id=pdb_id)
        backoff = self.rate_limit or 1.0
        attempts = 0
        while attempts < 5:
            self._throttle()
            try:
                resp = self.session.get(url, timeout=15)
                self._last_request = _now()
                status = resp.status_code
                if status == 200:
                    target.write_bytes(resp.content)
                    if target.stat().st_size < 1024:
                        raise ValueError("Downloaded file too small")
                    self.manifest[pdb_id] = {
                        "path": str(target),
                        "size": target.stat().st_size,
                        "timestamp": _now(),
                    }
                    self._save_manifest()
                    self.logger.info("Downloaded %s", pdb_id)
                    return target
                if status in {404}:
                    self.logger.warning("PDB %s not found (404)", pdb_id)
                    return None
                if status in {429, 500, 502, 503}:
                    attempts += 1
                    sleep_for = backoff * (2 ** (attempts - 1))
                    self.logger.warning("Rate/server issue for %s (status %s). Retrying in %.1fs", pdb_id, status, sleep_for)
                    time.sleep(sleep_for)
                    continue
                resp.raise_for_status()
            except Exception as exc:
                attempts += 1
                sleep_for = backoff * (2 ** (attempts - 1))
                self.logger.error("Download failed for %s (attempt %d): %s", pdb_id, attempts, exc)
                time.sleep(sleep_for)
        return None

    def _parse_structure(self, pdb_path: Path) -> Optional[Structure]:
        try:
            parser = PDBParser(QUIET=True)
            return parser.get_structure(pdb_path.stem, str(pdb_path))
        except Exception as exc:  # pragma: no cover - parsing may fail on rare files
            self.logger.error("Failed to parse %s: %s", pdb_path, exc)
            return None

    def validate_pdb(self, pdb_file: Path) -> bool:
        """Basic validation for downloaded structures."""
        if not pdb_file.exists() or pdb_file.stat().st_size < 1024:
            return False
        structure = self._parse_structure(pdb_file)
        if structure is None:
            return False
        for model in structure:
            for chain in model:
                residues = [res for res in chain if Polypeptide.is_aa(res, standard=False)]
                if residues:
                    return True
        return False

    def _chain_data(self, structure: Structure) -> List[ChainData]:
        chains: List[ChainData] = []
        for model in structure:
            for chain in model:
                sequence_chars: List[str] = []
                residues: List[object] = []
                for residue in chain:
                    if not Polypeptide.is_aa(residue, standard=False):
                        continue
                    residues.append(residue)
                    try:
                        sequence_chars.append(seq1(residue.get_resname()))
                    except Exception:
                        sequence_chars.append("X")
                if sequence_chars:
                    chains.append(ChainData(chain.id, "".join(sequence_chars), residues))
        return chains

    def extract_sequences(self, pdb_file: Path) -> Tuple[Optional[str], Optional[str]]:
        """Extract heavy and light chain sequences from a PDB file.

        Parameters
        ----------
        pdb_file : Path
            Local PDB path.

        Returns
        -------
        tuple
            (heavy_sequence, light_sequence). Each item may be None if missing.
        """
        structure = self._parse_structure(pdb_file)
        if structure is None:
            return None, None
        chains = self._chain_data(structure)
        heavy_ids = {"H", "A"}
        light_ids = {"L", "B"}
        heavy_seq = next((c.sequence for c in chains if c.chain_id in heavy_ids), None)
        light_seq = next((c.sequence for c in chains if c.chain_id in light_ids), None)
        if heavy_seq is None and chains:
            heavy_seq = chains[0].sequence
        if light_seq is None and len(chains) > 1:
            light_seq = chains[1].sequence
        return heavy_seq, light_seq

    def _extract_plddt(self, chain_data: ChainData) -> List[float]:
        values: List[float] = []
        for residue in chain_data.residues:
            if residue.has_id("CA"):
                values.append(float(residue["CA"].get_bfactor()))
            else:
                values.append(0.0)
        return [max(0.0, min(v, 100.0)) for v in values]

    def _compute_sasa(self, pdb_path: Path, chain_data: ChainData) -> List[float]:
        if freesasa is None or not pdb_path.exists():  # pragma: no cover - optional dependency
            return [0.0] * len(chain_data.sequence)
        try:
            structure = freesasa.Structure(str(pdb_path))
            result = freesasa.calc(structure)
            residue_areas = result.residueAreas()
            sasa_values: List[float] = []
            for residue in chain_data.residues:
                res_id = residue.get_id()[1]
                chain_map = residue_areas.get(chain_data.chain_id, {})
                area_obj = chain_map.get(str(res_id))
                sasa = float(area_obj.total) if area_obj is not None else 0.0
                sasa_values.append(sasa)
            return sasa_values
        except Exception:  # pragma: no cover - depends on binary
            return [0.0] * len(chain_data.sequence)

    def find_glycosites(self, sequence: str, chain_id: str, pdb_structure: Structure, pdb_path: Optional[Path] = None) -> pd.DataFrame:
        """Identify N-X-[S/T] motifs and annotate with SASA and pLDDT.

        Parameters
        ----------
        sequence : str
            Amino acid sequence for the chain.
        chain_id : str
            Chain identifier within the structure.
        pdb_structure : Structure
            Parsed Bio.PDB structure.
        pdb_path : Path, optional
            Path to the PDB file for SASA computation with FreeSASA.

        Returns
        -------
        pandas.DataFrame
            DataFrame with columns [position, residue, motif_type, chain, plddt, sasa, accessibility_rank].
        """
        motifs = _find_nglyco(sequence)
        if not motifs:
            return pd.DataFrame(columns=["position", "residue", "motif_type", "chain", "plddt", "sasa", "accessibility_rank"])

        chain_obj = None
        for model in pdb_structure:
            if chain_id in model:
                chain_obj = model[chain_id]
                break
        if chain_obj is None:
            self.logger.warning("Chain %s not found in structure", chain_id)
            return pd.DataFrame(columns=["position", "residue", "motif_type", "chain", "plddt", "sasa", "accessibility_rank"])

        residues = [res for res in chain_obj if Polypeptide.is_aa(res, standard=False)]
        chain_data = ChainData(chain_id=chain_id, sequence=sequence, residues=residues)
        plddt_values = self._extract_plddt(chain_data)
        sasa_values = self._compute_sasa(pdb_path or Path(""), chain_data) if pdb_path else [0.0] * len(chain_data.sequence)

        ranked = sorted(
            motifs,
            key=lambda item: sasa_values[item[0] - 1] if item[0] - 1 < len(sasa_values) else 0.0,
            reverse=True,
        )
        accessibility = {pos: idx + 1 for idx, (pos, _) in enumerate(ranked)}

        sites: List[GlycoSite] = []
        for pos, motif in motifs:
            sasa_val = sasa_values[pos - 1] if pos - 1 < len(sasa_values) else 0.0
            if sasa_val < 10.0:
                self.logger.warning("Residue appears buried (SASA %.2f) at %s%s", sasa_val, chain_id, pos)
            sites.append(
                GlycoSite(
                    position=pos,
                    residue="N",
                    motif_type=motif,
                    chain=chain_id,
                    plddt=plddt_values[pos - 1] if pos - 1 < len(plddt_values) else 0.0,
                    sasa=sasa_val,
                    accessibility_rank=accessibility.get(pos, 0),
                )
            )
        df = pd.DataFrame([site.as_dict() for site in sites])
        return df

    def stream_antibodies(self) -> Iterator[Tuple[str, str, Optional[str], Optional[str], pd.DataFrame]]:
        """Stream antibodies one by one.

        Yields
        ------
        tuple
            (pdb_id, antibody_name, heavy_sequence, light_sequence, glycosites_df)
        """
        if not self.pdb_ids:
            self.query_rcsb_antibodies()
        iterable: Iterable[str] = self.pdb_ids
        if self.resume_from_checkpoint and self.streamed_pdb_ids:
            remaining = [pid for pid in self.pdb_ids if pid not in self.streamed_pdb_ids]
            iterable = remaining
            self.logger.info(
                "Resuming stream: %d processed, %d remaining",
                len(self.streamed_pdb_ids),
                len(remaining),
            )
        for pdb_id in tqdm(iterable, desc="Antibodies", unit="pdb"):
            if self.resume_from_checkpoint and pdb_id in self.streamed_pdb_ids:
                continue
            pdb_path = self.download_pdb(pdb_id)
            if pdb_path is None:
                self.logger.warning("Skipping %s due to download failure", pdb_id)
                continue
            if not self.validate_pdb(pdb_path):
                self.logger.warning("Skipping %s due to validation failure", pdb_id)
                continue
            structure = self._parse_structure(pdb_path)
            if structure is None:
                continue
            chain_data_list = self._chain_data(structure)
            heavy_candidates = [c for c in chain_data_list if c.chain_id in {"H", "A"}]
            light_candidates = [c for c in chain_data_list if c.chain_id in {"L", "B"}]
            heavy_data = heavy_candidates[0] if heavy_candidates else (chain_data_list[0] if chain_data_list else None)
            light_data = None
            if light_candidates:
                light_data = light_candidates[0]
            elif len(chain_data_list) > 1:
                light_data = chain_data_list[1]

            heavy_seq = heavy_data.sequence if heavy_data else None
            light_seq = light_data.sequence if light_data else None

            glyco_frames: List[pd.DataFrame] = []
            if heavy_data:
                glyco_frames.append(self.find_glycosites(heavy_data.sequence, heavy_data.chain_id, structure, pdb_path))
            if light_data:
                glyco_frames.append(self.find_glycosites(light_data.sequence, light_data.chain_id, structure, pdb_path))
            glyco_frames = [frame for frame in glyco_frames if not frame.empty]
            glycosites_df = (
                pd.concat(glyco_frames, ignore_index=True)
                if glyco_frames
                else pd.DataFrame(columns=["position", "residue", "motif_type", "chain", "plddt", "sasa", "accessibility_rank"])
            )
            antibody_name = pdb_id
            self.logger.info("Processed %s with %d glycosites", pdb_id, len(glycosites_df))
            yield pdb_id, antibody_name, heavy_seq, light_seq, glycosites_df


__all__ = ["TheraSAbDabStreamer"]
