# Phase 2 → Phase 3 Transfer Learning Runbook

This runbook trains a CFG glycan encoder (Phase 2) and transfers it into a low‑data
FcγR–Fc affinity model (Phase 3).

## Requirements

- Python environment with: `torch`, `torch-geometric`, `fair-esm`, `rdkit`.
- CFG training CSV must include:
  - `lectin_sequence`
  - `glycan_smiles`
  - `binding` **or** `rfu_raw` (median threshold is used if `binding` is missing)
- FcγR training CSV must include:
  - `fcgr_sequence`
  - `fc_sequence`
  - `glycan_structure` (SMILES)
  - `binding_kd_nm`

All scripts fail fast if inputs or required columns are missing.

If your CFG dataset lacks sequences or SMILES, build a transfer dataset from
`data/processed/phase2_lectin_glycan.jsonl`:

```bash
python scripts/data/build_phase2_transfer_dataset.py \
  --input data/processed/phase2_lectin_glycan.jsonl \
  --output data/processed/phase2_transfer_training.csv \
  --threshold-percentile 60
```

When using that dataset, pass `--allow-non-smiles` to Phase 2 training because
the glycan strings are IUPAC‑like rather than true SMILES.

If you have parsed CFG RFU measurements in `data/processed/cfg_rfu_measurements.csv`,
you can build a more realistic training set by matching CFG lectin names to
known lectin sequences from the JSONL:

```bash
python scripts/data/build_phase2_cfg_training_from_rfu.py \
  --cfg-rfu data/processed/cfg_rfu_measurements.csv \
  --lectin-jsonl data/processed/phase2_lectin_glycan.jsonl \
  --output data/processed/phase2_transfer_training_cfg.csv \
  --threshold-percentile 60
```

This produces a CSV with real RFU distributions and IUPAC‑like glycan strings
(use `--allow-non-smiles` during Phase 2 training).

---

## 1) Train Phase 2 + Export Glycan Encoder

```bash
python scripts/train/train_phase2_with_glycan_encoder_export.py \
  --data data/interim/glycoml_phase2_unified_lectin_glycan_interactions.csv \
  --output-dir models \
  --epochs 50 \
  --batch-size 16 \
  --lr 1e-4 \
  --device cuda \
  --esm-model esm2_t33_650M_UR50D
```

If using the transfer dataset built above:

```bash
python scripts/train/train_phase2_with_glycan_encoder_export.py \
  --data data/processed/phase2_transfer_training.csv \
  --output-dir models \
  --epochs 50 \
  --batch-size 16 \
  --lr 1e-4 \
  --device cuda \
  --esm-model esm2_t33_650M_UR50D \
  --allow-non-smiles
```

Outputs:
- `models/phase2_cfg_full_model.pt`
- `models/phase2_glycan_encoder_pretrained.pt`

---

## 2) Phase 3 Transfer Learning (Frozen Glycan Encoder)

```bash
python scripts/train/train_fcgr_with_transfer_learning.py \
  --data data/processed/fcgr_fc_training_data.csv \
  --glycan-encoder models/phase2_glycan_encoder_pretrained.pt \
  --output-model models/fcgr_transfer_frozen.pt \
  --freeze-glycan \
  --lr 5e-4 \
  --epochs 100 \
  --batch-size 16 \
  --device cuda
```

---

## 3) Phase 3 Transfer Learning (Fine‑tuned Glycan Encoder)

```bash
python scripts/train/train_fcgr_with_transfer_learning.py \
  --data data/processed/fcgr_fc_training_data.csv \
  --glycan-encoder models/phase2_glycan_encoder_pretrained.pt \
  --output-model models/fcgr_transfer_finetuned.pt \
  --glycan-lr 1e-5 \
  --lr 5e-4 \
  --epochs 100 \
  --batch-size 16 \
  --device cuda
```

---

## 4) Compare Transfer Learning vs Baseline

```bash
python scripts/analysis/compare_transfer_learning_results.py
```

Output:
- `results/transfer_learning_comparison.csv`

---

## Notes

- ESM2 backbone is configurable via `--esm-model` in Phase 2 training.
- Glycan encoder weights are loaded with `weights_only=True` when supported.
- RDKit parsing errors halt training with a clear error message.
