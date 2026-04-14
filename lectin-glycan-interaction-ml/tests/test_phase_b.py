"""Phase B verification tests (glycan representations).

Covers:
- Fix 5: monosaccharide tokenisation (IUPAC 2-Carb-38 / SNFG, Varki 2015).
- Fix 6: Morgan fingerprint chirality (Rogers & Hahn 2010; Bojar 2021).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# Load models/glycan_encoder.py directly without running models/__init__.py,
# which pulls in torch_geometric-dependent modules not required here.
def _load_module(mod_name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


glycan_encoder = _load_module("glycan_encoder_direct", "models/glycan_encoder.py")
tokenize_iupac = glycan_encoder.tokenize_iupac
count_monosaccharides = glycan_encoder.count_monosaccharides
COMMON_MONOSACCHARIDES = glycan_encoder.COMMON_MONOSACCHARIDES
GlycanFingerprintEncoder = glycan_encoder.GlycanFingerprintEncoder
GlycanFingerprintConfig = glycan_encoder.GlycanFingerprintConfig

try:
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore

    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False


# ---------------------------------------------------------------------------
# Fix 5: monosaccharide tokeniser
# ---------------------------------------------------------------------------


def _count(iupac: str, token: str) -> int:
    idx = COMMON_MONOSACCHARIDES.index(token)
    return int(count_monosaccharides(iupac)[idx])


def test_galnac_does_not_inflate_gal_count():
    # The prior bug: "GalNAc" substring matched "Gal" via str.count.
    assert _count("GalNAcb1-3Galb1-4Glc", "Gal") == 1
    assert _count("GalNAcb1-3Galb1-4Glc", "GalNAc") == 1
    assert _count("GalNAcb1-3Galb1-4Glc", "Glc") == 1


def test_sialylated_n_glycan_trisaccharide_motif():
    # Neu5Acalpha2-3Galbeta1-4GlcNAc: one each of Neu5Ac, Gal, GlcNAc.
    iupac = "Neu5Aca2-3Galb1-4GlcNAc"
    assert _count(iupac, "Neu5Ac") == 1
    assert _count(iupac, "Gal") == 1
    assert _count(iupac, "GlcNAc") == 1
    # Glc and GalNAc must not appear as phantom substring matches.
    assert _count(iupac, "Glc") == 0
    assert _count(iupac, "GalNAc") == 0
    assert _count(iupac, "Neu5Gc") == 0


def test_legacy_aliases_normalise_to_canonical():
    # NeuAc (older nomenclature) must map to Neu5Ac.
    tokens = tokenize_iupac("NeuAca2-6Galb1-4GlcNAc")
    assert "Neu5Ac" in tokens
    assert "NeuAc" not in tokens


def test_branched_n_glycan_core():
    # Core pentasaccharide of all N-glycans (Varki 2022):
    # Manalpha1-3(Manalpha1-6)Manbeta1-4GlcNAcbeta1-4GlcNAc
    iupac = "Mana1-3(Mana1-6)Manb1-4GlcNAcb1-4GlcNAc"
    assert _count(iupac, "Man") == 3
    assert _count(iupac, "GlcNAc") == 2


def test_empty_or_none_input_returns_zero_vector():
    zeros = [0.0] * len(COMMON_MONOSACCHARIDES)
    assert count_monosaccharides("") == zeros
    assert count_monosaccharides(None) == zeros


def test_vocabulary_covers_sialic_acids_and_uronic_acids():
    # Requirement from Phase B plan: expand vocabulary to include sialic acid
    # family, uronic acids, and bacterial/plant sugars (Varki 2015).
    for required in ("Neu5Ac", "Neu5Gc", "KDN", "Kdo", "GlcA", "IdoA"):
        assert required in COMMON_MONOSACCHARIDES


# ---------------------------------------------------------------------------
# Fix 6: Morgan fingerprint chirality
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not HAS_RDKIT, reason="rdkit not installed")
def test_morgan_fingerprint_distinguishes_anomers():
    # alpha-D-Glucose vs beta-D-Glucose differ only in the anomeric C1
    # stereochemistry. useChirality=True must make their fingerprints differ.
    alpha_glc = "OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O"
    beta_glc = "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"

    cfg = GlycanFingerprintConfig(radius=2, n_bits=2048,
                                  include_physchem=False, include_iupac=False)
    enc = GlycanFingerprintEncoder(cfg)
    fp_alpha = enc.encode(alpha_glc)
    fp_beta = enc.encode(beta_glc)
    assert fp_alpha.shape == fp_beta.shape
    # The two fingerprints must differ in at least one bit. Without
    # useChirality=True this assertion fails (the historical bug).
    assert (fp_alpha - fp_beta).abs().sum().item() > 0


@pytest.mark.skipif(not HAS_RDKIT, reason="rdkit not installed")
def test_morgan_fingerprint_achiral_input_warns():
    import warnings

    cfg = GlycanFingerprintConfig(radius=2, n_bits=1024,
                                  include_physchem=False, include_iupac=False)
    enc = GlycanFingerprintEncoder(cfg)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        enc.encode("OCC(O)C(O)C(O)C(O)CO")  # sugar alcohol, no stereo
    messages = [str(w.message) for w in caught]
    assert any("stereochemistry" in m for m in messages)
