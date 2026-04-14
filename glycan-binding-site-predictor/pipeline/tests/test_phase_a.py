"""Phase A verification tests.

Covers:
- Fix 1: B-factor vs pLDDT gating.
- Fix 2: absolute SASA -> RSA via Tien et al. 2013 empirical max table.
- Fix 3: PDB insertion-code preservation (Kabat CDR-H3 100A/100B/100C).
- Fix 4: RCSB Search API v2 UniProt scoping.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.thera_sabdab_pipeline import (  # noqa: E402
    ChainData,
    EXPERIMENTAL_METHODS,
    absolute_to_rsa,
    extract_plddt,
    is_predicted_structure,
    resolve_uniprot_ids,
)
import importlib.util

# Load models.structure_ranker directly from its file path to bypass the
# package __init__ (which imports torch-dependent modules not required here).
_sr_path = ROOT / "models" / "structure_ranker.py"
_spec = importlib.util.spec_from_file_location("structure_ranker_direct", _sr_path)
structure_ranker = importlib.util.module_from_spec(_spec)
sys.modules["structure_ranker_direct"] = structure_ranker
_spec.loader.exec_module(structure_ranker)
parse_plddt_from_pdb = structure_ranker.parse_plddt_from_pdb
parse_residue_map_from_pdb = structure_ranker.parse_residue_map_from_pdb
rank_sites = structure_ranker.rank_sites


# ---------------------------------------------------------------------------
# Fix 1: B-factor vs pLDDT gating
# ---------------------------------------------------------------------------


def _fake_residue(b_factor: float):
    ca = SimpleNamespace(get_bfactor=lambda: b_factor)
    return SimpleNamespace(has_id=lambda name: name == "CA", __getitem__=lambda self, key: ca if key == "CA" else None)


class _FakeResidue:
    def __init__(self, b_factor):
        self._ca = SimpleNamespace(get_bfactor=lambda: b_factor)

    def has_id(self, name):
        return name == "CA"

    def __getitem__(self, key):
        if key == "CA":
            return self._ca
        raise KeyError(key)


def _fake_structure(method: str):
    return SimpleNamespace(header={"structure_method": method})


def test_is_predicted_structure_xray_is_experimental():
    assert not is_predicted_structure(_fake_structure("X-RAY DIFFRACTION"))
    assert not is_predicted_structure(_fake_structure("Solution NMR"))
    assert not is_predicted_structure(_fake_structure("ELECTRON MICROSCOPY"))


def test_is_predicted_structure_theoretical_is_predicted():
    assert is_predicted_structure(_fake_structure("theoretical model"))
    assert is_predicted_structure(_fake_structure("Predicted"))
    assert is_predicted_structure(_fake_structure("alphafold monomer v4"))


def test_is_predicted_structure_empty_is_conservative():
    # Missing header => treat as experimental. Callers must opt-in to pLDDT.
    assert not is_predicted_structure(_fake_structure(""))
    assert not is_predicted_structure(_fake_structure(None))


def test_experimental_method_set_is_canonical_wwpdb():
    for m in ["x-ray diffraction", "solution nmr", "electron microscopy",
              "solid-state nmr", "neutron diffraction", "fiber diffraction"]:
        assert m in EXPERIMENTAL_METHODS


def test_extract_plddt_returns_none_for_experimental():
    chain = ChainData(chain_id="H", sequence="ACDE", residues=[_FakeResidue(45.0)] * 4)
    values = extract_plddt(chain, is_confidence=False)
    assert values == [None, None, None, None]


def test_extract_plddt_reads_bfactor_for_predicted():
    residues = [_FakeResidue(b) for b in (85.0, 92.0, 60.0, 45.0)]
    chain = ChainData(chain_id="H", sequence="ACDE", residues=residues)
    values = extract_plddt(chain, is_confidence=True)
    assert values == [85.0, 92.0, 60.0, 45.0]


def test_extract_plddt_clamps_out_of_range_confidence():
    # pLDDT must be in [0, 100]; anything outside is treated as missing and
    # replaced by DEFAULT_PLDDT (70.0).
    residues = [_FakeResidue(b) for b in (0.0, 150.0, -5.0, 50.0)]
    chain = ChainData(chain_id="H", sequence="ACDE", residues=residues)
    values = extract_plddt(chain, is_confidence=True)
    assert values[0] == 70.0  # 0.0 -> default (pLDDT of exactly 0 is unusable)
    assert values[1] == 70.0  # 150 -> out of range
    assert values[2] == 70.0  # negative -> out of range
    assert values[3] == 50.0


# ---------------------------------------------------------------------------
# Fix 2: absolute SASA -> RSA (Tien 2013 empirical)
# ---------------------------------------------------------------------------


def test_absolute_to_rsa_preserves_length():
    assert len(absolute_to_rsa("ACDE", [10.0, 20.0, 30.0, 40.0])) == 4


def test_absolute_to_rsa_residue_specific_scaling():
    # Same raw SASA should produce different RSA for residues with different
    # max exposure; Asn (187) vs Trp (264).
    rsa = absolute_to_rsa("NW", [100.0, 100.0])
    assert rsa[0] > rsa[1]
    assert math.isclose(rsa[0], 100.0 / 187.0, rel_tol=1e-6)
    assert math.isclose(rsa[1], 100.0 / 264.0, rel_tol=1e-6)


# ---------------------------------------------------------------------------
# Fix 3: Insertion codes (Kabat CDR-H3 H100/100A/100B)
# ---------------------------------------------------------------------------


def _write_kabat_pdb(tmp_path: Path) -> Path:
    # Minimal PDB excerpt with CDR-H3 insertions (H100, H100A, H100B) and an
    # altLoc pair on the first residue. Column layout follows the wwPDB spec:
    # cols 13-16 atom name, col 17 altLoc, col 22 chainID, cols 23-26 resSeq,
    # col 27 iCode, cols 61-66 B-factor.
    lines = [
        "ATOM      1  CA ATYR H  99     10.000  10.000  10.000  1.00 20.00           C",
        "ATOM      2  CA BTYR H  99     10.100  10.100  10.100  0.50 20.00           C",
        "ATOM      3  CA  GLY H 100     11.000  11.000  11.000  1.00 30.00           C",
        "ATOM      4  CA  ARG H 100A    12.000  12.000  12.000  1.00 40.00           C",
        "ATOM      5  CA  SER H 100B    13.000  13.000  13.000  1.00 50.00           C",
        "ATOM      6  CA  ASP H 101     14.000  14.000  14.000  1.00 60.00           C",
    ]
    path = tmp_path / "kabat.pdb"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_insertion_codes_produce_distinct_indices(tmp_path):
    pdb = _write_kabat_pdb(tmp_path)
    plddt = parse_plddt_from_pdb(str(pdb), chain_id="H")
    # 5 distinct residues (99, 100, 100A, 100B, 101); altLoc B of 99 dropped.
    assert list(plddt.keys()) == [0, 1, 2, 3, 4]
    assert plddt[0] == 20.0  # Tyr 99 (altLoc A)
    assert plddt[1] == 30.0  # Gly 100
    assert plddt[2] == 40.0  # Arg 100A
    assert plddt[3] == 50.0  # Ser 100B
    assert plddt[4] == 60.0  # Asp 101


def test_residue_map_preserves_icode(tmp_path):
    pdb = _write_kabat_pdb(tmp_path)
    chain_map = parse_residue_map_from_pdb(str(pdb), chain_id="H")
    assert chain_map.residue_ids == [
        (99, ""), (100, ""), (100, "A"), (100, "B"), (101, ""),
    ]


# ---------------------------------------------------------------------------
# rank_sites interoperability: nan pLDDT must not zero the score
# ---------------------------------------------------------------------------


def test_rank_sites_handles_missing_plddt_as_nan():
    positions = [0, 1, 2]
    plddt = {0: float("nan"), 1: float("nan"), 2: float("nan")}
    sasa = {0: 0.9, 1: 0.3, 2: 0.6}
    results = rank_sites(positions, plddt, sasa_scores=sasa)
    assert [r.position for r in results] == [0, 2, 1]  # ordered by SASA desc


def test_rank_sites_still_gates_on_sasa():
    plddt = {0: float("nan"), 1: float("nan")}
    sasa = {0: 0.9, 1: 0.1}  # second below sasa_threshold=0.2
    results = rank_sites([0, 1], plddt, sasa_scores=sasa)
    assert [r.position for r in results] == [0]


# ---------------------------------------------------------------------------
# Fix 4: RCSB UniProt scoping
# ---------------------------------------------------------------------------


def test_resolve_uniprot_ids_query_has_database_name_filter(monkeypatch):
    captured = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return b'{"result_set": []}'

    def fake_urlopen(req, timeout=20):  # noqa: ARG001
        import json

        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse()

    import pipeline.thera_sabdab_pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod.urllib.request, "urlopen", fake_urlopen)

    logger = SimpleNamespace(warning=lambda *a, **k: None)
    resolve_uniprot_ids(["P01857"], logger)

    payload = captured["payload"]
    assert payload["query"]["type"] == "group"
    assert payload["query"]["logical_operator"] == "and"
    attrs = [node["parameters"]["attribute"] for node in payload["query"]["nodes"]]
    values = [node["parameters"]["value"] for node in payload["query"]["nodes"]]
    assert any(a.endswith(".database_accession") for a in attrs)
    assert any(a.endswith(".database_name") for a in attrs)
    assert "UniProt" in values
    assert "P01857" in values
