# Phase 1 Antibody Classification

This document describes the Phase 1 workflow for N-glycosite prediction and Fc engineering.

## Goals

- Predict N-glycosylation sites on antibody sequences.
- Rank sites with structural accessibility and confidence.
- Estimate Fc gamma receptor binding impact.

## Inputs

- Antibody sequences (FASTA or CSV).
- Optional structure predictions (pLDDT, SASA).

## Pipeline

1. ESM2 embeddings for per-residue features.
2. LoRA fine-tuned classifier for glycosite probabilities.
3. Structure-guided ranking with SASA and pLDDT filters.
4. Optional Fc domain GNN for binding impact.

## Outputs

- Per-residue glycosite probabilities.
- Ranked site list with confidence scores.
- Optional Fc gamma receptor delta-G estimates.

## References

See `src/glycoml/phase1/README.md` and `src/glycoml/phase1/data/dataset_schema.md` for details.
