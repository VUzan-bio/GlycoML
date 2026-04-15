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

import math

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.PDB import PDBList, PDBParser, Polypeptide
from Bio.PDB.Structure import Structure

DEFAULT_PLDDT = 70.0

LOGGER_NAME = "thera_sabdab_pipeline"

# wwPDB experimental method strings (lowercased). See
# https://mmcif.wwpdb.org/dictionaries/mmcif_pdbx_v50.dic/Items/_exptl.method.html
EXPERIMENTAL_METHODS = frozenset(
    {
        "x-ray diffraction",
        "solution nmr",
        "solid-state nmr",
        "electron microscopy",
        "electron crystallography",
        "neutron diffraction",
        "fiber diffraction",
        "powder diffraction",
        "solution scattering",
    }
)

# Tien et al., PLoS ONE 2013 "Maximum allowed solvent accessibilities of
# residues in proteins" — empirical max SASA (A^2) from a Gly-X-Gly tripeptide
# reference observed in the PDB. Used to convert absolute SASA to relative SASA
# (RSA). Values from Table 1, "empirical".
MAX_SASA_TIEN_2013_EMPIRICAL = {
    "A": 121.0, "R": 265.0, "N": 187.0, "D": 187.0, "C": 148.0,
    "E": 214.0, "Q": 214.0, "G": 97.0,  "H": 216.0, "I": 195.0,
    "L": 191.0, "K": 230.0, "M": 203.0, "F": 228.0, "P": 154.0,
    "S": 143.0, "T": 163.0, "W": 264.0, "Y": 255.0, "V": 165.0,
}


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
    plddt: Optional[float]
    sasa: float
    rsa: float
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
            mapping = getattr(
                Polypeptide,
                "protein_letters_3to1_extended",
                Polypeptide.protein_letters_3to1,
            )
            for residue in chain:
                if not Polypeptide.is_aa(residue, standard=False):
                    continue
                residues.append(residue)
                resname = residue.get_resname()
                seq_chars.append(mapping.get(resname, "X"))
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


def is_predicted_structure(structure: Structure) -> bool:
    """Return True if the PDB header indicates a computed/predicted model.

    AlphaFold2 (Jumper et al., Nature 2021) and other predictors store their
    per-residue confidence (pLDDT, 0-100) in the B-factor column. Experimental
    methods (X-ray, NMR, EM, etc.) store the Debye-Waller thermal displacement
    factor in A^2 in the same column, which must NOT be interpreted as pLDDT.
    """
    method = (structure.header.get("structure_method") or "").strip().lower()
    if not method:
        return False
    if method in EXPERIMENTAL_METHODS:
        return False
    predicted_keywords = ("predict", "theoretical", "computational", "alphafold", "model")
    return any(keyword in method for keyword in predicted_keywords)


def extract_plddt(
    chain_data: ChainData,
    is_confidence: bool = False,
    default_value: float = DEFAULT_PLDDT,
) -> List[Optional[float]]:
    """Extract per-residue pLDDT from B-factor when the structure is predicted.

    For experimental structures, B-factor is the Debye-Waller temperature
    factor (A^2), NOT a confidence score; returns ``None`` for every residue
    so downstream consumers can skip the pLDDT term in ranking.
    """
    if not is_confidence:
        return [None] * len(chain_data.sequence)

    plddt: List[Optional[float]] = []
    for residue in chain_data.residues:
        if residue.has_id("CA"):
            b = float(residue["CA"].get_bfactor())
            # AlphaFold pLDDT is bounded to [0, 100]. Anything outside that
            # range is treated as missing and replaced by the default.
            plddt.append(b if 0.0 < b <= 100.0 else default_value)
        else:
            plddt.append(default_value)
    if not plddt:
        return [default_value] * len(chain_data.sequence)
    return plddt


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


def absolute_to_rsa(sequence: str, sasa_values: Sequence[float]) -> List[float]:
    """Convert absolute SASA (A^2) to relative SASA (RSA) per Tien et al. 2013.

    RSA[i] = SASA[i] / MAX_SASA[residue_i], clipped to [0, 1.5] to tolerate
    slight over-exposure from terminal residues and unusual packing. Residues
    not in the 20 standard amino acids map to Ala's max (conservative).
    """
    rsa: List[float] = []
    fallback = MAX_SASA_TIEN_2013_EMPIRICAL["A"]
    for residue, sasa in zip(sequence, sasa_values):
        ref = MAX_SASA_TIEN_2013_EMPIRICAL.get(residue.upper(), fallback)
        if ref <= 0:
            rsa.append(0.0)
            continue
        ratio = float(sasa) / ref
        rsa.append(max(0.0, min(ratio, 1.5)))
    return rsa


def rank_accessibility(sasa_values: List[float], positions: List[int]) -> Dict[int, int]:
    ranked = sorted(positions, key=lambda pos: sasa_values[pos - 1], reverse=True)
    return {pos: rank + 1 for rank, pos in enumerate(ranked)}


def build_site_records(
    pdb_id: str,
    antibody_name: str,
    chain_data: ChainData,
    plddt_values: List[Optional[float]],
    sasa_values: List[float],
    rsa_values: Optional[List[float]] = None,
) -> List[SiteRecord]:
    motifs = find_nglyco_sites(chain_data.sequence)
    if not motifs:
        return []
    if rsa_values is None:
        rsa_values = absolute_to_rsa(chain_data.sequence, sasa_values)
    # Rank by RSA (dimensionless, residue-normalised) rather than raw SASA (A^2)
    # so that a 180 A^2 Asn (RSA~0.96) outranks a 180 A^2 Trp (RSA~0.68).
    ranks = rank_accessibility(rsa_values, [pos for pos, _ in motifs])
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
            rsa=rsa_values[pos - 1],
            accessibility_rank=ranks.get(pos, 0),
        )
        records.append(record)
    return records


def write_fasta(chain_data: ChainData, output_dir: Path, pdb_id: str, antibody_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = output_dir / f"{pdb_id}_{chain_data.chain_id}.fasta"
    record_id = f"{antibody_name}|{pdb_id}|{chain_data.chain_id}"
    SeqIO.write([SeqIO.SeqRecord(seq=Seq(chain_data.sequence), id=record_id, description="")], fasta_path, "fasta")
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
        # RCSB Search API v2: restrict accession match to UniProt to avoid
        # spurious GenBank / Ensembl / NORINE ID collisions. See
        # https://search.rcsb.org/#search-attributes (attribute sibling
        # database_name lives on the same reference_sequence_identifiers node
        # and must be combined via an AND group, not concatenated keys).
        query = {
            "query": {
                "type": "group",
                "logical_operator": "and",
                "nodes": [
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                            "operator": "exact_match",
                            "value": uniprot_id,
                        },
                    },
                    {
                        "type": "terminal",
                        "service": "text",
                        "parameters": {
                            "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_name",
                            "operator": "exact_match",
                            "value": "UniProt",
                        },
                    },
                ],
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

    predicted = is_predicted_structure(structure)

    all_records: List[SiteRecord] = []
    chain_info: Dict[str, Dict[str, object]] = {}
    for chain_data in chain_data_list:
        plddt = extract_plddt(chain_data, is_confidence=predicted)
        sasa = compute_sasa(structure, pdb_path, chain_data)
        rsa = absolute_to_rsa(chain_data.sequence, sasa)
        chain_records = build_site_records(
            record.pdb_id, record.antibody_name, chain_data, plddt, sasa, rsa
        )
        all_records.extend(chain_records)
        write_fasta(chain_data, fasta_dir, record.pdb_id, record.antibody_name)
        numeric_plddt = [v for v in plddt if v is not None]
        chain_info[chain_data.chain_id] = {
            "length": len(chain_data.sequence),
            "mean_plddt": float(np.mean(numeric_plddt)) if numeric_plddt else None,
            "mean_rsa": float(np.mean(rsa)) if rsa else None,
        }

    metadata = {
        "status": "ok",
        "pdb_id": record.pdb_id,
        "antibody_name": record.antibody_name,
        "chains": chain_info,
        "resolution": structure.header.get("resolution"),
        "structure_method": structure.header.get("structure_method"),
        "is_predicted_structure": predicted,
    }
    return record.pdb_id, all_records, metadata




def records_to_dataframe(records: Sequence[SiteRecord]) -> pd.DataFrame:
    """Convert site records to a pandas DataFrame.

    Example:
        df = records_to_dataframe(records)
    """
    return pd.DataFrame([record.__dict__ for record in records])

def _record_to_row(record: SiteRecord) -> Dict[str, object]:
    row = dict(record.__dict__)
    # pLDDT is None for experimental structures (where B-factor != confidence);
    # serialise as empty string so downstream consumers don't mis-read "nan".
    plddt = row.get("plddt")
    if plddt is None or (isinstance(plddt, float) and math.isnan(plddt)):
        row["plddt"] = ""
    return row


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
            "rsa",
            "accessibility_rank",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for record in records:
            writer.writerow(_record_to_row(record))


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
