"""Dataset utilities for glycosite prediction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple
import csv
import random

from .sequence import find_nglyco_motifs, parse_site_list


@dataclass(frozen=True)
class SequenceRecord:
    record_id: str
    chain: str
    sequence: str
    glyco_sites: List[int]


@dataclass(frozen=True)
class GlycoSample:
    record_id: str
    chain: str
    sequence: str
    position: int  # 0-based N position
    label: int


def _clean_sequence(value: Optional[str]) -> str:
    return (value or "").strip().replace(" ", "").upper()


def load_sequence_records(csv_path: str) -> List[SequenceRecord]:
    """Load sequence records from a CSV file.

    Supported schemas:
    - id, chain, sequence, glyco_sites
    - id, heavy_seq, light_seq, heavy_glyco_sites, light_glyco_sites
    - id, heavy_seq, light_seq, glyco_sites (with H:/L: prefixes)
    """
    records: List[SequenceRecord] = []
    with open(csv_path, "r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_idx, row in enumerate(reader, start=1):
            record_id = (row.get("id") or row.get("record_id") or row.get("name") or f"row_{row_idx}").strip()

            sequence = _clean_sequence(row.get("sequence"))
            if sequence:
                chain = (row.get("chain") or row.get("chain_id") or "?").strip() or "?"
                sites = parse_site_list(row.get("glyco_sites"), chain)
                records.append(SequenceRecord(record_id=record_id, chain=chain, sequence=sequence, glyco_sites=sites))
                continue

            heavy_seq = _clean_sequence(row.get("heavy_seq"))
            light_seq = _clean_sequence(row.get("light_seq"))
            heavy_sites = row.get("heavy_glyco_sites") or row.get("glyco_sites") or ""
            light_sites = row.get("light_glyco_sites") or row.get("glyco_sites") or ""

            if heavy_seq:
                sites = parse_site_list(heavy_sites, chain="H")
                records.append(SequenceRecord(record_id=record_id, chain="H", sequence=heavy_seq, glyco_sites=sites))
            if light_seq:
                sites = parse_site_list(light_sites, chain="L")
                records.append(SequenceRecord(record_id=record_id, chain="L", sequence=light_seq, glyco_sites=sites))

    return records


def build_candidate_samples(records: Iterable[SequenceRecord]) -> List[GlycoSample]:
    """Expand sequence records into motif candidates with labels."""
    samples: List[GlycoSample] = []
    for record in records:
        motif_positions = find_nglyco_motifs(record.sequence)
        glyco_set = set(record.glyco_sites)
        for pos in motif_positions:
            label = 1 if pos in glyco_set else 0
            samples.append(
                GlycoSample(
                    record_id=record.record_id,
                    chain=record.chain,
                    sequence=record.sequence,
                    position=pos,
                    label=label,
                )
            )
    return samples


def split_records(
    records: List[SequenceRecord],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 13,
) -> Tuple[List[SequenceRecord], List[SequenceRecord], List[SequenceRecord]]:
    """Split records into train/val/test lists."""
    rng = random.Random(seed)
    records = list(records)
    rng.shuffle(records)
    n_total = len(records)
    n_val = int(n_total * val_ratio)
    n_test = int(n_total * test_ratio)
    val_records = records[:n_val]
    test_records = records[n_val : n_val + n_test]
    train_records = records[n_val + n_test :]
    return train_records, val_records, test_records


def summarize_samples(samples: Iterable[GlycoSample]) -> Dict[str, int]:
    """Return basic counts for logging."""
    total = 0
    positives = 0
    for sample in samples:
        total += 1
        positives += int(sample.label == 1)
    return {
        "total": total,
        "positives": positives,
        "negatives": total - positives,
    }

