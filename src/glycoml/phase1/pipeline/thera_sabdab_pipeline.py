"""Thera-SAbDab antibody glycosylation pipeline.

This module downloads PDB structures, extracts antibody sequences, detects
N-glycosylation motifs, computes pLDDT (from B-factors), and SASA values, and
writes per-site CSV output plus metadata logs.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import urllib.error
import urllib.request

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.PDB import PDBList, PDBParser, Polypeptide
from Bio.PDB.Structure import Structure

DEFAULT_PLDDT = 70.0

LOGGER_NAME = "thera_sabdab_pipeline"


@dataclass(frozen=True)
class InputRecord:
    pdb_id: str
    antibody_name: str
    chain_ids: Optional[List[str]] = None


@dataclass
class ChainData:
    chain_id: str
    sequence: str
    residues: List[object]


@dataclass
class SiteRecord:
    pdb_id: str
    antibody_name: str
    chain_id: str
    position: int
    residue: str
    motif_type: str
    plddt: float
    sasa: float
    accessibility_rank: int


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def normalize_pdb_id(value: str) -> str:
    return value.strip().lower()


def download_pdb(pdb_id: str, cache_dir: Path, logger: logging.Logger) -> Optional[Path]:
    """Download a PDB file from RCSB using Bio.PDB.PDBList.

    Returns the local path if successful, otherwise None.
    """
    pdb_id = normalize_pdb_id(pdb_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    target_path = cache_dir / f"{pdb_id}.pdb"
    if target_path.exists():
        return target_path

    pdb_list = PDBList()
    try:
        downloaded = pdb_list.retrieve_pdb_file(pdb_id, pdir=str(cache_dir), file_format="pdb")
    except Exception as exc:
        logger.warning("Download failed for %s: %s", pdb_id, exc)
        return None

    downloaded_path = Path(downloaded)
    if downloaded_path.exists() and downloaded_path != target_path:
        try:
            downloaded_path.replace(target_path)
        except OSError:
            target_path = downloaded_path
    return target_path if target_path.exists() else None


def load_structure(pdb_path: Path) -> Structure:
    parser = PDBParser(QUIET=True)
    return parser.get_structure(pdb_path.stem, str(pdb_path))


def extract_chain_data(structure: Structure, chain_ids: Optional[Sequence[str]] = None) -> List[ChainData]:
    chains: List[ChainData] = []
    for model in structure:
        for chain in model:
            if chain_ids and chain.id not in chain_ids:
                continue
            residues = []
            seq_chars: List[str] = []
            for residue in chain:
                if not Polypeptide.is_aa(residue, standard=False):
                    continue
                residues.append(residue)
                try:
                    seq_chars.append(Polypeptide.three_to_one(residue.get_resname()))
                except KeyError:
                    seq_chars.append("X")
            if seq_chars:
                chains.append(ChainData(chain_id=chain.id, sequence="".join(seq_chars), residues=residues))
    return chains




def extract_sequence(structure: Structure, chain_id: str) -> Optional[ChainData]:
    """Extract a single chain sequence by chain ID.

    Example:
        chain = extract_sequence(structure, "H")
    """
    chains = extract_chain_data(structure, [chain_id])
    return chains[0] if chains else None


def find_nglyco_sites(sequence: str) -> List[Tuple[int, str]]:
    """Find N-X-S/T motifs (X != P). Returns 1-based positions and motif type.

    Example:
        find_nglyco_sites("ANST") -> [(2, "NXT")]
    """
    sequence = sequence.upper()
    hits: List[Tuple[int, str]] = []
    for i in range(len(sequence) - 2):
        if sequence[i] != "N":
            continue
        if sequence[i + 1] == "P":
            continue
        if sequence[i + 2] in {"S", "T"}:
            motif = f"NX{sequence[i + 2]}"
            hits.append((i + 1, motif))
    return hits




def predict_glycosites(sequence: str) -> List[Tuple[int, str]]:
    """Alias for find_nglyco_sites.

    Example:
        predict_glycosites("ANST") -> [(2, "NXT")]
    """
    return find_nglyco_sites(sequence)


def extract_plddt(chain_data: ChainData, default_value: float = DEFAULT_PLDDT) -> List[float]:
    plddt: List[float] = []
    for residue in chain_data.residues:
        if residue.has_id("CA"):
            plddt.append(float(residue["CA"].get_bfactor()))
        else:
            plddt.append(default_value)
    if not plddt:
        return [default_value] * len(chain_data.sequence)
    return [val if val > 0 else default_value for val in plddt]


def _compute_sasa_dssp(structure: Structure, pdb_path: Path, chain_data: ChainData) -> Optional[List[float]]:
    try:
        from Bio.PDB.DSSP import DSSP
    except Exception:
        return None

    try:
        model = structure[0]
        dssp = DSSP(model, str(pdb_path))
    except Exception:
        return None

    sasa_values: List[float] = []
    for residue in chain_data.residues:
        res_id = residue.get_id()
        key = (chain_data.chain_id, res_id)
        if key in dssp:
            sasa_values.append(float(dssp[key][3]))
        else:
            sasa_values.append(0.0)
    return sasa_values


def _compute_sasa_freesasa(pdb_path: Path, chain_data: ChainData) -> Optional[List[float]]:
    try:
        import freesasa
    except Exception:
        return None

    try:
        structure = freesasa.Structure(str(pdb_path))
        result = freesasa.calc(structure)
    except Exception:
        return None

    residue_areas = result.residueAreas()
    sasa_values: List[float] = []
    for residue in chain_data.residues:
        res_id = residue.get_id()[1]
        chain = chain_data.chain_id
        try:
            sasa_values.append(float(residue_areas[chain][str(res_id)].total))
        except Exception:
            sasa_values.append(0.0)
    return sasa_values


def compute_sasa(structure: Structure, pdb_path: Path, chain_data: ChainData) -> List[float]:
    sasa = _compute_sasa_dssp(structure, pdb_path, chain_data)
    if sasa is not None:
        return sasa
    sasa = _compute_sasa_freesasa(pdb_path, chain_data)
    if sasa is not None:
        return sasa
    return [0.0] * len(chain_data.sequence)


def rank_accessibility(sasa_values: List[float], positions: List[int]) -> Dict[int, int]:
    ranked = sorted(positions, key=lambda pos: sasa_values[pos - 1], reverse=True)
    return {pos: rank + 1 for rank, pos in enumerate(ranked)}


def build_site_records(
    pdb_id: str,
    antibody_name: str,
    chain_data: ChainData,
    plddt_values: List[float],
    sasa_values: List[float],
) -> List[SiteRecord]:
    motifs = find_nglyco_sites(chain_data.sequence)
    if not motifs:
        return []
    ranks = rank_accessibility(sasa_values, [pos for pos, _ in motifs])
    records: List[SiteRecord] = []
    for pos, motif in motifs:
        residue = chain_data.sequence[pos - 1]
        record = SiteRecord(
            pdb_id=pdb_id,
            antibody_name=antibody_name,
            chain_id=chain_data.chain_id,
            position=pos,
            residue=residue,
            motif_type=motif,
            plddt=plddt_values[pos - 1],
            sasa=sasa_values[pos - 1],
            accessibility_rank=ranks.get(pos, 0),
        )
        records.append(record)
    return records


def write_fasta(chain_data: ChainData, output_dir: Path, pdb_id: str, antibody_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = output_dir / f"{pdb_id}_{chain_data.chain_id}.fasta"
    record_id = f"{antibody_name}|{pdb_id}|{chain_data.chain_id}"
    SeqIO.write([SeqIO.SeqRecord(seq=chain_data.sequence, id=record_id, description="")], fasta_path, "fasta")
    return fasta_path


def load_checkpoint(checkpoint_path: Path) -> List[str]:
    if checkpoint_path.exists():
        with open(checkpoint_path, "r") as handle:
            try:
                return json.load(handle)
            except json.JSONDecodeError:
                return []
    return []


def save_checkpoint(checkpoint_path: Path, processed_ids: List[str]) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with open(checkpoint_path, "w") as handle:
        json.dump(sorted(set(processed_ids)), handle, indent=2)


def load_input_csv(path: Path) -> List[InputRecord]:
    records: List[InputRecord] = []
    with open(path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            pdb_id = (row.get("pdb_id") or row.get("pdb") or "").strip()
            name = (row.get("antibody_name") or row.get("name") or pdb_id).strip()
            chains_raw = (row.get("chain_ids") or row.get("chains") or "").strip()
            chain_ids = [c.strip() for c in chains_raw.split(";") if c.strip()] if chains_raw else None
            if pdb_id:
                records.append(InputRecord(pdb_id=pdb_id, antibody_name=name, chain_ids=chain_ids))
    return records


def resolve_uniprot_ids(uniprot_ids: Sequence[str], logger: logging.Logger) -> List[InputRecord]:
    records: List[InputRecord] = []
    for uniprot_id in uniprot_ids:
        uniprot_id = uniprot_id.strip()
        if not uniprot_id:
            continue
        query = {
            "query": {
                "type": "terminal",
                "service": "text",
                "parameters": {
                    "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                    "operator": "exact_match",
                    "value": uniprot_id,
                },
            },
            "request_options": {"return_all_hits": True},
            "return_type": "entry",
        }
        data = json.dumps(query).encode("utf-8")
        req = urllib.request.Request(
            "https://search.rcsb.org/rcsbsearch/v2/query",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            logger.warning("UniProt lookup failed for %s: %s", uniprot_id, exc)
            continue
        hits = payload.get("result_set", [])
        for hit in hits:
            pdb_id = hit.get("identifier", "").lower()
            if pdb_id:
                records.append(InputRecord(pdb_id=pdb_id, antibody_name=uniprot_id))
    return records


def process_record(
    record: InputRecord,
    cache_dir: Path,
    fasta_dir: Path,
    logger: logging.Logger,
) -> Tuple[str, List[SiteRecord], Dict[str, object]]:
    pdb_path = download_pdb(record.pdb_id, cache_dir, logger)
    if pdb_path is None:
        return record.pdb_id, [], {"status": "download_failed"}

    try:
        structure = load_structure(pdb_path)
    except Exception as exc:
        logger.warning("Failed to parse %s: %s", record.pdb_id, exc)
        return record.pdb_id, [], {"status": "parse_failed"}

    chain_data_list = extract_chain_data(structure, record.chain_ids)
    if not chain_data_list:
        return record.pdb_id, [], {"status": "no_chains"}

    all_records: List[SiteRecord] = []
    chain_info: Dict[str, Dict[str, object]] = {}
    for chain_data in chain_data_list:
        plddt = extract_plddt(chain_data)
        sasa = compute_sasa(structure, pdb_path, chain_data)
        chain_records = build_site_records(record.pdb_id, record.antibody_name, chain_data, plddt, sasa)
        all_records.extend(chain_records)
        write_fasta(chain_data, fasta_dir, record.pdb_id, record.antibody_name)
        chain_info[chain_data.chain_id] = {
            "length": len(chain_data.sequence),
            "mean_plddt": float(np.mean(plddt)) if plddt else DEFAULT_PLDDT,
        }

    metadata = {
        "status": "ok",
        "pdb_id": record.pdb_id,
        "antibody_name": record.antibody_name,
        "chains": chain_info,
        "resolution": structure.header.get("resolution"),
        "structure_method": structure.header.get("structure_method"),
    }
    return record.pdb_id, all_records, metadata




def records_to_dataframe(records: Sequence[SiteRecord]) -> pd.DataFrame:
    """Convert site records to a pandas DataFrame.

    Example:
        df = records_to_dataframe(records)
    """
    return pd.DataFrame([record.__dict__ for record in records])

def append_site_records(csv_path: Path, records: List[SiteRecord]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as handle:
        fieldnames = [
            "pdb_id",
            "antibody_name",
            "chain_id",
            "position",
            "residue",
            "motif_type",
            "plddt",
            "sasa",
            "accessibility_rank",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def update_metadata(metadata_path: Path, record_metadata: Dict[str, object]) -> None:
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    if metadata_path.exists():
        with open(metadata_path, "r") as handle:
            try:
                metadata = json.load(handle)
            except json.JSONDecodeError:
                metadata = {}
    else:
        metadata = {}
    if record_metadata.get("pdb_id"):
        metadata[record_metadata["pdb_id"]] = record_metadata
    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle, indent=2)


def run_pipeline(
    input_csv: Optional[Path],
    uniprot_list: Optional[Path],
    output_dir: Path,
    cache_dir: Path,
    max_workers: int = 4,
    checkpoint_path: Optional[Path] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fasta_dir = output_dir / "fastas"
    fasta_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "download_log.txt"
    logger = setup_logger(log_path)

    records: List[InputRecord] = []
    if input_csv:
        records.extend(load_input_csv(input_csv))
    if uniprot_list:
        with open(uniprot_list, "r") as handle:
            ids = [line.strip() for line in handle if line.strip()]
        records.extend(resolve_uniprot_ids(ids, logger))

    if not records:
        logger.error("No input records found.")
        return

    checkpoint_path = checkpoint_path or (output_dir / "checkpoints" / "processed.json")
    processed = set(load_checkpoint(checkpoint_path))

    csv_path = output_dir / "glycosylation_sites.csv"
    metadata_path = output_dir / "structure_metadata.json"

    lock = threading.Lock()

    def should_skip(pdb_id: str) -> bool:
        return pdb_id in processed

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for record in records:
            pdb_id = normalize_pdb_id(record.pdb_id)
            if should_skip(pdb_id):
                logger.info("Skipping %s (checkpoint)", pdb_id)
                continue
            futures[executor.submit(process_record, record, cache_dir, fasta_dir, logger)] = pdb_id

        for future in as_completed(futures):
            pdb_id = futures[future]
            try:
                pdb_id, site_records, metadata = future.result()
            except Exception as exc:
                logger.error("Processing failed for %s: %s", pdb_id, exc)
                continue

            if metadata.get("status") != "ok":
                logger.warning("%s status: %s", pdb_id, metadata.get("status"))
            if site_records:
                append_site_records(csv_path, site_records)
            update_metadata(metadata_path, metadata)
            with lock:
                processed.add(pdb_id)
                save_checkpoint(checkpoint_path, list(processed))

    logger.info("Done. Output CSV: %s", csv_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Thera-SAbDab glycosylation pipeline")
    parser.add_argument("--input_csv", type=Path, help="CSV with pdb_id and antibody_name")
    parser.add_argument("--input_uniprot", type=Path, help="Text file with UniProt IDs")
    parser.add_argument("--output_dir", type=Path, default=Path("pipeline/data"))
    parser.add_argument("--cache_dir", type=Path, default=Path("data/cache/pdb_cache"))
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    run_pipeline(
        input_csv=args.input_csv,
        uniprot_list=args.input_uniprot,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        max_workers=args.max_workers,
        checkpoint_path=args.checkpoint,
    )
