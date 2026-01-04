# Lectin-Glycan Interaction Predictor (Phase 2)

Predict lectin-glycan binding signals using a protein encoder (ESM2) and a glycan encoder (fingerprint or GCN).

## Architecture

- Protein encoder: ESM2 embeddings pooled to a fixed-size lectin representation.
- Glycan encoder:
  - Fingerprints (Morgan) + physicochemical features + optional IUPAC counts.
  - Optional GCN over RDKit graphs (requires rdkit + torch-geometric).
- Interaction module: MLP over concatenated lectin and glycan embeddings with optional bilinear gating.

## Repository layout

```
.
|-- data/
|   |-- dataset_schema.md
|   |-- glycan_smiles_library.txt
|   `-- sample_cfg_data.csv
|-- models/
|   |-- protein_encoder.py
|   |-- glycan_encoder.py
|   |-- interaction_module.py
|   `-- benchmark_models.py
|-- scripts/
|   |-- preprocess_cfg_data.py
|   |-- train_interaction_model.py
|   |-- evaluate_benchmarks.py
|   `-- predict_new_lectins.py
|-- utils/
`-- notebooks/
```

## Data preparation

Start from a CFG lectin array export (or a curated CSV).
The preprocessing script normalizes RFU values and creates train/val/test splits.

```
python scripts/preprocess_cfg_data.py \
  --input_csv data/sample_cfg_data.csv \
  --output_csv data/cfg_processed.csv \
  --glycan_library data/glycan_smiles_library.txt \
  --normalize log1p \
  --label_threshold 500
```

## Training

Regression on normalized RFU:
```
python scripts/train_interaction_model.py \
  --data data/cfg_processed.csv \
  --splits data/train_val_test_splits.pkl \
  --task regression \
  --target rfu_norm \
  --output_dir outputs
```

Classification (binder/non-binder):
```
python scripts/train_interaction_model.py \
  --data data/cfg_processed.csv \
  --splits data/train_val_test_splits.pkl \
  --task classification \
  --target label \
  --label_threshold 500 \
  --output_dir outputs
```

Optional GCN glycan encoder:
```
python scripts/train_interaction_model.py \
  --data data/cfg_processed.csv \
  --splits data/train_val_test_splits.pkl \
  --glycan_encoder gcn \
  --output_dir outputs
```

## Baseline benchmarks

```
python scripts/evaluate_benchmarks.py \
  --data data/cfg_processed.csv \
  --splits data/train_val_test_splits.pkl \
  --task classification \
  --label_threshold 500
```

## Predict new lectins

Provide a CSV with columns: lectin_sequence, glycan_smiles (and optional glycan_iupac).

```
python scripts/predict_new_lectins.py \
  --model outputs/interaction_model.pt \
  --input_csv data/sample_cfg_data.csv \
  --output_csv outputs/predictions.csv
```

## Ablation plan (expected)

| Variant | Notes | Target metric |
| --- | --- | --- |
| ESM2 only | No glycan features | MCC ~0.5 |
| Fingerprints | Morgan + physchem | MCC ~0.6 |
| GCN | Graph encoder | MCC ~0.65+ |

## Notes

- ESM2 is optional at runtime. If the `esm` package is missing, a lightweight embedding is used.
- The GCN encoder requires `rdkit` and `torch-geometric`.
- Normalize RFU values before training (log1p or minmax).
