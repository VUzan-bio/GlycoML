import pytest
import torch

from glycoml.shared.glycan_tokenizer import GlycanTokenizer
from glycoml.phase2.models.binding_model import BindingModel, BindingModelConfig


def test_tokenizer_roundtrip():
    tok = GlycanTokenizer()
    tok.build(["GlcNAcb1-4GlcNAc"])
    ids = tok.encode("GlcNAcb1-4GlcNAc")
    assert len(ids) > 0


def test_binding_model_forward_token():
    config = BindingModelConfig()
    config.use_graph = False
    model = BindingModel(config)

    lectin_tokens = torch.randn(2, 10, config.lectin_config.esm_dim)
    lectin_mask = torch.ones(2, 10, dtype=torch.bool)
    glycan_tokens = torch.randint(0, 10, (2, 5))
    glycan_mask = torch.ones(2, 5, dtype=torch.bool)
    family = torch.zeros(2, config.lectin_config.family_dim)
    species = torch.zeros(2, dtype=torch.long)
    glycan_meta = torch.zeros(2, config.glycan_token_config.meta_dim)

    bin_logits, reg_out, _ = model(
        lectin_tokens,
        lectin_mask,
        family_features=family,
        species_idx=species,
        glycan_tokens=glycan_tokens,
        glycan_mask=glycan_mask,
        glycan_meta=glycan_meta,
    )

    assert bin_logits.shape == (2,)
    assert reg_out.shape == (2,)


def test_binding_model_forward_graph():
    try:
        from torch_geometric.data import Data, Batch
    except Exception:
        pytest.skip("torch_geometric not installed")

    config = BindingModelConfig()
    config.use_graph = True
    model = BindingModel(config)

    lectin_tokens = torch.randn(2, 10, config.lectin_config.esm_dim)
    lectin_mask = torch.ones(2, 10, dtype=torch.bool)
    family = torch.zeros(2, config.lectin_config.family_dim)
    species = torch.zeros(2, dtype=torch.long)

    g1 = Data(z=torch.tensor([6, 6]), pos=torch.randn(2, 3), meta=torch.zeros(config.glycan_graph_config.meta_dim))
    g2 = Data(z=torch.tensor([6, 6, 6]), pos=torch.randn(3, 3), meta=torch.zeros(config.glycan_graph_config.meta_dim))
    batch = Batch.from_data_list([g1, g2])

    bin_logits, reg_out, _ = model(
        lectin_tokens,
        lectin_mask,
        family_features=family,
        species_idx=species,
        glycan_graph=batch,
    )

    assert bin_logits.shape == (2,)
    assert reg_out.shape == (2,)
