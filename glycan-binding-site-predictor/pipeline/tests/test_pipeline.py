import math

from pipeline.thera_sabdab_pipeline import (
    ChainData,
    MAX_SASA_TIEN_2013_EMPIRICAL,
    absolute_to_rsa,
    build_site_records,
    find_nglyco_sites,
    rank_accessibility,
)


def test_find_nglyco_sites_skips_proline():
    sequence = "NPTNATNQS"
    sites = find_nglyco_sites(sequence)
    assert (4, "NXT") in sites
    assert all(pos != 1 for pos, _ in sites)


def test_rank_accessibility_orders_by_sasa():
    sasa_values = [10.0, 5.0, 15.0]
    positions = [1, 2, 3]
    ranks = rank_accessibility(sasa_values, positions)
    assert ranks[3] == 1
    assert ranks[1] == 2
    assert ranks[2] == 3


def test_build_site_records_assigns_rank():
    chain = ChainData(chain_id="H", sequence="ANST", residues=[object()] * 4)
    plddt = [70.0, 75.0, 80.0, 85.0]
    sasa = [1.0, 2.0, 3.0, 4.0]
    records = build_site_records("1abc", "test", chain, plddt, sasa)
    assert len(records) == 1
    record = records[0]
    assert record.position == 2
    assert record.accessibility_rank == 1
    # RSA must be auto-computed when omitted, using Tien 2013 empirical max.
    expected_rsa = 2.0 / MAX_SASA_TIEN_2013_EMPIRICAL["N"]
    assert math.isclose(record.rsa, expected_rsa, rel_tol=1e-6)


def test_absolute_to_rsa_matches_tien_table():
    # Asn near its empirical max should saturate toward 1.0.
    rsa = absolute_to_rsa("N", [MAX_SASA_TIEN_2013_EMPIRICAL["N"]])
    assert math.isclose(rsa[0], 1.0, rel_tol=1e-6)
    # Trp with small SASA stays close to 0.
    rsa = absolute_to_rsa("W", [5.0])
    assert rsa[0] < 0.05
    # Over-exposure is clipped at 1.5.
    rsa = absolute_to_rsa("G", [1000.0])
    assert rsa[0] == 1.5
    # Non-standard residue falls back to Ala reference (does not raise).
    rsa = absolute_to_rsa("X", [50.0])
    assert 0.0 < rsa[0] < 1.5
