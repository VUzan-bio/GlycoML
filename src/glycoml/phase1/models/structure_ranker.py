"""Structure-informed ranking for candidate glycosylation sites."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


@dataclass
class SiteScore:
    position: int  # 0-based position
    plddt: float
    sasa: float
    conservation: float
    score: float


def parse_plddt_from_pdb(pdb_path: str, chain_id: Optional[str] = None) -> Dict[int, float]:
    """Parse per-residue pLDDT scores from AlphaFold PDB (B-factor for CA atoms)."""
    plddt: Dict[int, float] = {}
    with open(pdb_path, "r") as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            if chain_id and line[21].strip() != chain_id:
                continue
            try:
                res_id = int(line[22:26].strip())
                b_factor = float(line[60:66].strip())
            except ValueError:
                continue
            plddt[res_id - 1] = b_factor
    return plddt


def load_sasa_from_csv(csv_path: str, chain_id: Optional[str] = None) -> Dict[int, float]:
    """Load per-residue SASA from a CSV with columns: chain, position, sasa."""
    sasa: Dict[int, float] = {}
    with open(csv_path, "r") as handle:
        header = handle.readline().strip().split(",")
        idx_chain = header.index("chain") if "chain" in header else None
        idx_pos = header.index("position") if "position" in header else None
        idx_sasa = header.index("sasa") if "sasa" in header else None
        if idx_pos is None or idx_sasa is None:
            raise ValueError("SASA CSV must include 'position' and 'sasa' columns.")
        for line in handle:
            if not line.strip():
                continue
            parts = line.strip().split(",")
            if idx_chain is not None and chain_id and parts[idx_chain] != chain_id:
                continue
            pos = int(parts[idx_pos]) - 1
            sasa[pos] = float(parts[idx_sasa])
    return sasa


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

    results: List[SiteScore] = []
    for pos in positions:
        plddt = plddt_scores.get(pos, 0.0)
        sasa = sasa_scores.get(pos, 0.0)
        conservation = conservation_scores.get(pos, 1.0)
        if plddt < plddt_threshold:
            continue
        if sasa_scores and sasa < sasa_threshold:
            continue
        score = plddt_norm.get(pos, 0.0) * (sasa_norm.get(pos, 1.0) if sasa_scores else 1.0) * cons_norm.get(pos, 1.0)
        results.append(
            SiteScore(
                position=pos,
                plddt=plddt,
                sasa=sasa,
                conservation=conservation,
                score=score,
            )
        )
    results.sort(key=lambda item: item.score, reverse=True)
    return results

