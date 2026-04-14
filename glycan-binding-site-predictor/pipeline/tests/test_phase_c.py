"""Phase C verification tests.

Covers:
- Fix 8: FcgrBindingPredictor refuses to silently return delta_g=0.
- Fix 9: wider classifier window with zero-padding at termini.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")


def _load(name: str, rel_path: str):
    path = ROOT / rel_path
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


classifier_mod = _load("esm2_classifier_direct", "models/esm2_classifier.py")
fcgr_mod = _load("fcgr_binding_module_direct", "models/fcgr_binding_module.py")

GlycoMotifClassifier = classifier_mod.GlycoMotifClassifier
extract_motif_embedding = classifier_mod.extract_motif_embedding
FcgrBindingPredictor = fcgr_mod.FcgrBindingPredictor


# ---------------------------------------------------------------------------
# Fix 9: classifier window
# ---------------------------------------------------------------------------


def test_extract_motif_embedding_default_window_is_11():
    L, D = 20, 8
    emb = torch.randn(L, D)
    out = extract_motif_embedding(emb, position=10)
    assert out.shape == (11 * D,)


def test_extract_motif_embedding_zero_pads_left_terminus():
    L, D = 20, 4
    emb = torch.arange(L * D, dtype=torch.float32).reshape(L, D)
    # Position 0 means the left 5 slots of an 11-wide window must be zero.
    out = extract_motif_embedding(emb, position=0, window_size=11)
    assert out.shape == (11 * D,)
    left_pad = out[: 5 * D]
    centre = out[5 * D : 6 * D]
    right_real = out[6 * D : 11 * D]
    assert torch.all(left_pad == 0)
    assert torch.allclose(centre, emb[0])
    assert torch.allclose(right_real, emb[1:6].reshape(-1))


def test_extract_motif_embedding_zero_pads_right_terminus():
    L, D = 15, 3
    emb = torch.arange(L * D, dtype=torch.float32).reshape(L, D)
    out = extract_motif_embedding(emb, position=L - 1, window_size=11)
    # Right half (positions L..L+4) are out of range -> zero padded.
    right_pad = out[6 * D : 11 * D]
    assert torch.all(right_pad == 0)


def test_extract_motif_embedding_rejects_even_window():
    emb = torch.zeros(10, 4)
    with pytest.raises(ValueError, match="odd"):
        extract_motif_embedding(emb, position=4, window_size=10)


def test_classifier_input_dim_scales_with_window():
    clf3 = GlycoMotifClassifier(embed_dim=16, window_size=3)
    clf11 = GlycoMotifClassifier(embed_dim=16, window_size=11)
    # First Linear layer's input dim should be embed_dim * window_size.
    first3 = next(m for m in clf3.net if isinstance(m, torch.nn.Linear))
    first11 = next(m for m in clf11.net if isinstance(m, torch.nn.Linear))
    assert first3.in_features == 16 * 3
    assert first11.in_features == 16 * 11


def test_classifier_forward_accepts_windowed_input():
    clf = GlycoMotifClassifier(embed_dim=16, window_size=11)
    batch = torch.randn(4, 16 * 11)
    logits = clf(batch)
    assert logits.shape == (4,)


def test_classifier_rejects_even_window():
    with pytest.raises(ValueError, match="odd"):
        GlycoMotifClassifier(embed_dim=16, window_size=4)


# ---------------------------------------------------------------------------
# Fix 8: FcgrBindingPredictor refuses to produce a silent zero
# ---------------------------------------------------------------------------


def test_fcgr_predictor_raises_without_model():
    predictor = FcgrBindingPredictor(model_path=None)
    with pytest.raises(NotImplementedError, match="FcDomainGCN|model"):
        predictor.predict_delta_g("ACDEFGHIKLMNPQRSTVWY", glyco_sites=[])


def test_fcgr_predictor_raises_with_nonmodule_checkpoint(tmp_path):
    ckpt = tmp_path / "not_a_module.pt"
    torch.save({"weights": torch.zeros(3)}, ckpt)
    predictor = FcgrBindingPredictor(model_path=str(ckpt))
    # The raw state_dict is rejected; model stays None; predict raises.
    assert predictor.model is None
    with pytest.raises(NotImplementedError):
        predictor.predict_delta_g("NSTG", glyco_sites=[0])
