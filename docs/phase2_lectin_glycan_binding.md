# Phase 2 Lectin-Glycan Binding

This document describes the Phase 2 lectin-glycan binding predictor in the refactored `glycoml` package.

## Architecture

- **Lectin encoder**: ESM2 embeddings + optional structure encoder (SchNet) + fold/class/family metadata.
- **Glycan encoder**: Graph encoder (SchNet) or token-based Transformer over IUPAC tokens.
- **Fusion**: Cross-attention from glycan queries to lectin tokens.
- **Heads**: Multi-task binary + regression outputs.

## Training

```bash
python -m glycoml.phase2.train       --data-path data/interim/glycoml_phase2_unified_lectin_glycan_interactions.csv       --unilectin-path data/interim/unilectin3d_lectin_glycan_interactions.csv       --cfg-metadata-path data/metadata/cfg_experiment_metadata.csv       --ligands-path data/metadata/unilectin3d_ligands.csv       --output-dir outputs/phase2_model       --label-mode multi       --glycan-encoder graph       --use-structure       --seed 42
```

## Outputs

- `model.pt`: trained model weights
- `config.yaml`: run configuration
- `predictions.csv`: per-pair predictions
- `metrics.json`: final evaluation metrics
- `history.json`: training curves
- `binding_heatmap.png`: heatmap visualization

## Ablations

Use `--ablation` to run targeted studies:

- `no_struct`: remove structure encoder
- `token_glycan`: swap to token glycan encoder
- `no_crossattn`: skip cross-attention fusion
- `single_task`: binary-only

## Baselines

```bash
python -m glycoml.phase2.baselines       --data-path data/interim/glycoml_phase2_unified_lectin_glycan_interactions.csv       --output-dir outputs/phase2_baselines
```
