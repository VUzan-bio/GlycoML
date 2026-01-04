# Glycan-Binding Site Predictor (Phase 1)

Predict candidate N-glycosylation sites in therapeutic antibodies using ESM-2 embeddings, optional structure ranking, and a placeholder Fcgr impact module.

## Phase 1 modules

1) Sequence-level N-glycosite classifier (ESM-2 embeddings + MLP head)
2) Structure-informed ranking (pLDDT + SASA filtering)
3) Fcgr binding impact stub (hook for your GNN)

## Repository layout

```
.
|-- data/
|   |-- dataset_schema.md
|   |-- sample_sequences.csv
|   `-- sample_sequences.fasta
|-- models/
|   |-- esm2_classifier.py
|   |-- structure_ranker.py
|   `-- fcgr_binding_module.py
|-- scripts/
|   |-- train.py
|   |-- predict.py
|   `-- evaluate_on_therapeutics.py
|-- utils/
`-- notebooks/
```

## Data preparation

Place your curated Thera-SAbDab export at `data/thera_sabdab_processed.csv`.
The loader supports both single-chain and heavy/light schemas. See `data/dataset_schema.md`.
For a quick smoke test, you can point the scripts at `data/sample_sequences.csv`.

## Training

```
python scripts/train.py --data data/thera_sabdab_processed.csv --output_dir outputs
```

### Optional: focal loss

```
python scripts/train.py --data data/thera_sabdab_processed.csv --output_dir outputs --use_focal
```

## Prediction

Single sequence:
```
python scripts/predict.py --model outputs/glyco_classifier.pt --sequence EVQLVNNSTGATV
```

FASTA input:
```
python scripts/predict.py --model outputs/glyco_classifier.pt --fasta data/sample_sequences.fasta
```

Structure ranking (AlphaFold PDB + SASA):
```
python scripts/predict.py --model outputs/glyco_classifier.pt --sequence EVQLVNNSTGATV --pdb path/to/af_model.pdb --sasa_csv path/to/sasa.csv --chain_id H
```

## Evaluation

```
python scripts/evaluate_on_therapeutics.py --data data/thera_sabdab_processed.csv --model outputs/glyco_classifier.pt
```

## Notes

- ESM-2 is optional at runtime. If the `esm` package is unavailable, the code falls back to a lightweight trainable embedding.
- The Fcgr module is a stub; integrate your GNN by replacing `FcgrBindingPredictor`.
- Metrics are computed at the motif level (N in N-X-S/T).
