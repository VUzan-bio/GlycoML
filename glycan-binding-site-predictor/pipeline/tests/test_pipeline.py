import math

from pipeline.thera_sabdab_pipeline import (
    ChainData,
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
