"""Structure-informed ranking for candidate glycosylation sites."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


@dataclass
class SiteScore:
    position: int  # 0-based position along the extracted chain sequence
    plddt: float
    sasa: float
    conservation: float
    score: float


@dataclass
class ChainResidueMap:
    """Maps 0-based sequence index <-> (resseq, icode) tuple from a PDB chain.

    Antibodies use Kabat/Chothia numbering with insertion codes (e.g. H100A,
    H100B, H100C in CDR-H3). Keying structural features by residue number
    alone collapses those inserts; keying by sequence enumeration order
    matches the sequence string extracted elsewhere in the pipeline.
    """

    plddt: Dict[int, float] = field(default_factory=dict)
    residue_ids: List[Tuple[int, str]] = field(default_factory=list)


def parse_plddt_from_pdb(
    pdb_path: str,
    chain_id: Optional[str] = None,
) -> Dict[int, float]:
    """Parse per-residue B-factor from CA atoms, keyed by 0-based sequence index.

    Notes:
    - wwPDB ATOM record format: resSeq cols 23-26, iCode col 27, altLoc col 17.
    - altLoc: only the blank/primary (' ' or 'A') conformer is consumed, so
      structures with alternate rotamers do not double-count a residue.
    - Insertion codes are preserved *implicitly* by enumeration: a residue
      100A following 100 becomes the next sequence index.
    - Interpretation of the B-factor column is caller's responsibility: for
      AlphaFold PDBs it is pLDDT (0-100); for X-ray it is Debye-Waller (A^2).
    """
    chain_map = parse_residue_map_from_pdb(pdb_path, chain_id=chain_id)
    return dict(chain_map.plddt)


def parse_residue_map_from_pdb(
    pdb_path: str,
    chain_id: Optional[str] = None,
) -> ChainResidueMap:
    """Return a ChainResidueMap with per-position B-factors and residue ids."""
    result = ChainResidueMap()
    seen: set = set()
    seq_index = 0
    with open(pdb_path, "r") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            alt_loc = line[16]
            if alt_loc not in (" ", "A"):
                continue
            if chain_id and line[21].strip() != chain_id:
                continue
            try:
                res_seq = int(line[22:26].strip())
                b_factor = float(line[60:66].strip())
            except ValueError:
                continue
            icode = line[26]
            key = (line[21], res_seq, icode)
            if key in seen:
                continue
            seen.add(key)
            result.plddt[seq_index] = b_factor
            result.residue_ids.append((res_seq, icode.strip()))
            seq_index += 1
    return result


def load_sasa_from_csv(csv_path: str, chain_id: Optional[str] = None) -> Dict[int, float]:
    """Load per-residue relative SASA (RSA) from the pipeline CSV.

    Accepts either of two column layouts:
    - ``chain, position, rsa``  (preferred; dimensionless, Tien 2013 normalised)
    - ``chain, position, sasa`` (absolute A^2; used only when ``rsa`` is absent)

    Positions are 1-based in the CSV and converted to 0-based sequence index
    here to match the pipeline's ``build_site_records`` convention.
    """
    values: Dict[int, float] = {}
    with open(csv_path, "r") as handle:
        header = handle.readline().strip().split(",")
        idx_chain = header.index("chain_id") if "chain_id" in header else (
            header.index("chain") if "chain" in header else None
        )
        idx_pos = header.index("position") if "position" in header else None
        idx_rsa = header.index("rsa") if "rsa" in header else None
        idx_sasa = header.index("sasa") if "sasa" in header else None
        idx_value = idx_rsa if idx_rsa is not None else idx_sasa
        if idx_pos is None or idx_value is None:
            raise ValueError(
                "SASA CSV must include 'position' plus either 'rsa' or 'sasa' columns."
            )
        for line in handle:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            if idx_chain is not None and chain_id and parts[idx_chain] != chain_id:
                continue
            pos = int(parts[idx_pos]) - 1
            try:
                values[pos] = float(parts[idx_value])
            except (ValueError, IndexError):
                continue
    return values


def normalize_scores(values: Dict[int, float], max_value: float) -> Dict[int, float]:
    if not values:
        return {}
    return {key: min(max(val / max_value, 0.0), 1.0) for key, val in values.items()}


def rank_sites(
    positions: Iterable[int],
    plddt_scores: Dict[int, float],
    sasa_scores: Optional[Dict[int, float]] = None,
    conservation_scores: Optional[Dict[int, float]] = None,
    plddt_threshold: float = 70.0,
    sasa_threshold: float = 0.2,
) -> List[SiteScore]:
    """Rank candidate sites using confidence, accessibility, and conservation."""
    sasa_scores = sasa_scores or {}
    conservation_scores = conservation_scores or {}

    plddt_norm = normalize_scores(plddt_scores, 100.0)
    sasa_norm = normalize_scores(sasa_scores, 1.0)
    cons_norm = normalize_scores(conservation_scores, 1.0)

    import math

    results: List[SiteScore] = []
    for pos in positions:
        plddt = plddt_scores.get(pos)
        sasa = sasa_scores.get(pos, 0.0)
        conservation = conservation_scores.get(pos, 1.0)
        # For experimental structures the caller may pass None or NaN because
        # the PDB B-factor is a Debye-Waller thermal factor (A^2), not a
        # confidence score. In that case the pLDDT gate is bypassed and the
        # multiplicative factor collapses to 1.0 so the ranker still produces
        # usable output from SASA and conservation alone.
        plddt_missing = plddt is None or (isinstance(plddt, float) and math.isnan(plddt))
        if not plddt_missing and plddt < plddt_threshold:
            continue
        if sasa_scores and sasa < sasa_threshold:
            continue
        plddt_factor = 1.0 if plddt_missing else plddt_norm.get(pos, 0.0)
        sasa_factor = sasa_norm.get(pos, 1.0) if sasa_scores else 1.0
        score = plddt_factor * sasa_factor * cons_norm.get(pos, 1.0)
        results.append(
            SiteScore(
                position=pos,
                plddt=float("nan") if plddt_missing else float(plddt),
                sasa=sasa,
                conservation=conservation,
                score=score,
            )
        )
    results.sort(key=lambda item: item.score, reverse=True)
    return results

