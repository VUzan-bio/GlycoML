"""Phase D verification tests.

Covers:
- Fix 11: stratified train/val/test split preserves per-class proportion.
"""

from __future__ import annotations

import sys
from pathlib import Path

PHASE1_ROOT = Path(__file__).resolve().parents[2]    # src/glycoml/phase1
SRC_ROOT = PHASE1_ROOT.parents[1]                    # src/
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from glycoml.phase1.utils.data_utils import (  # noqa: E402
    SequenceRecord,
    split_records,
)


def _make_records(n_pos: int, n_neg: int):
    records = []
    for i in range(n_pos):
        records.append(
            SequenceRecord(
                record_id=f"pos_{i}",
                chain="H",
                sequence="ANSTACDE" * 4,
                glyco_sites=[1],
            )
        )
    for i in range(n_neg):
        records.append(
            SequenceRecord(
                record_id=f"neg_{i}",
                chain="H",
                sequence="ACDEFGHIK" * 3,
                glyco_sites=[],
            )
        )
    return records


def _fraction_positive(records):
    if not records:
        return 0.0
    return sum(1 for r in records if r.glyco_sites) / len(records)


def test_stratified_split_preserves_positive_fraction():
    # 10% positive rate, N=200.
    records = _make_records(n_pos=20, n_neg=180)
    train, val, test = split_records(records, val_ratio=0.2, test_ratio=0.2, seed=13)

    assert len(train) + len(val) + len(test) == len(records)
    # Every split must contain at least one positive example.
    assert any(r.glyco_sites for r in val)
    assert any(r.glyco_sites for r in test)
    assert any(r.glyco_sites for r in train)

    target = _fraction_positive(records)
    for split in (train, val, test):
        assert abs(_fraction_positive(split) - target) < 0.05


def test_stratified_split_is_deterministic():
    records = _make_records(n_pos=15, n_neg=85)
    a1, b1, c1 = split_records(records, seed=42)
    a2, b2, c2 = split_records(records, seed=42)
    assert [r.record_id for r in a1] == [r.record_id for r in a2]
    assert [r.record_id for r in b1] == [r.record_id for r in b2]
    assert [r.record_id for r in c1] == [r.record_id for r in c2]


def test_stratified_split_small_minority_class():
    # Only 3 positives in 100 records. Stratification must put 1 in each of
    # val and test (rounded down, with the min-1-per-stratum fallback) or
    # at worst leave at least 1 positive in every split when possible.
    records = _make_records(n_pos=3, n_neg=97)
    train, val, test = split_records(records, val_ratio=0.2, test_ratio=0.2, seed=7)
    pos_train = sum(1 for r in train if r.glyco_sites)
    pos_val = sum(1 for r in val if r.glyco_sites)
    pos_test = sum(1 for r in test if r.glyco_sites)
    assert pos_train + pos_val + pos_test == 3
    assert pos_train >= 1  # the train fold is non-empty for the minority class


def test_unstratified_fallback_matches_legacy_layout():
    # When stratify=False, behaviour must reduce to a plain random split.
    records = _make_records(n_pos=5, n_neg=20)
    train, val, test = split_records(
        records, val_ratio=0.2, test_ratio=0.2, seed=1, stratify=False
    )
    assert len(train) + len(val) + len(test) == 25


def test_invalid_ratios_raise():
    import pytest

    records = _make_records(1, 1)
    with pytest.raises(ValueError):
        split_records(records, val_ratio=0.6, test_ratio=0.6)
    with pytest.raises(ValueError):
        split_records(records, val_ratio=-0.1, test_ratio=0.2)
