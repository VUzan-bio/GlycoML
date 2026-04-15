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


def _stratum_label(record: SequenceRecord) -> int:
    """Stratification label: 1 if the record carries at least one annotated
    glycosite, else 0.

    This is the classification label aggregated to the record (not the motif)
    level, which is the correct grouping key for stratified sampling. With
    record-level stratification the train/val/test splits preserve the
    observed fraction of glycosylated antibodies, preventing the pathological
    case where the validation set contains no positives (common with random
    splits at small N; Jefferis 2009, Nat. Rev. Drug Discov.).
    """
    return 1 if record.glyco_sites else 0


def split_records(
    records: List[SequenceRecord],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 13,
    stratify: bool = True,
) -> Tuple[List[SequenceRecord], List[SequenceRecord], List[SequenceRecord]]:
    """Stratified split of records into train / val / test lists.

    Stratification key is :func:`_stratum_label` (glycosylated vs not). Each
    stratum is shuffled independently and split in the requested ratio so the
    per-class proportion is preserved across folds. Pass ``stratify=False`` to
    recover the prior random-shuffle behaviour.
    """
    if val_ratio < 0 or test_ratio < 0 or val_ratio + test_ratio >= 1.0:
        raise ValueError(
            f"Invalid split ratios val={val_ratio}, test={test_ratio}: must be "
            "non-negative and sum to < 1.0."
        )

    rng = random.Random(seed)
    records = list(records)

    if not stratify:
        rng.shuffle(records)
        strata = {0: records}
    else:
        strata = {}
        for record in records:
            strata.setdefault(_stratum_label(record), []).append(record)
        for bucket in strata.values():
            rng.shuffle(bucket)

    train_records: List[SequenceRecord] = []
    val_records: List[SequenceRecord] = []
    test_records: List[SequenceRecord] = []

    for bucket in strata.values():
        n = len(bucket)
        n_val = int(round(n * val_ratio))
        n_test = int(round(n * test_ratio))
        # If a stratum is too small to supply at least one record per split,
        # prefer train over val over test so the rare class never disappears.
        n_val = min(n_val, max(0, n - 1))
        n_test = min(n_test, max(0, n - n_val - 1))
        val_records.extend(bucket[:n_val])
        test_records.extend(bucket[n_val : n_val + n_test])
        train_records.extend(bucket[n_val + n_test :])

    # Reshuffle within each split so strata are interleaved.
    for split in (train_records, val_records, test_records):
        rng.shuffle(split)

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

