# Phase 1 Antibody Classification

This document describes the Phase 1 workflow for N-glycosite prediction and Fc engineering.

## Goals

- Predict N-glycosylation sequons (N-X-S/T, X != P) on antibody sequences.
- Rank sites with structural accessibility and (where available) model
  confidence.
- Estimate Fc gamma receptor binding impact.

## Inputs

- Antibody sequences (FASTA or CSV).
- Optional PDB structures (experimental or AlphaFold-predicted). For
  AlphaFold/ColabFold outputs the CA B-factor column is interpreted as pLDDT
  in [0, 100]; for experimental structures (X-ray, NMR, EM) the B-factor is
  a Debye-Waller thermal factor (A^2) and the pipeline records pLDDT as
  missing (see `structure_method` header field).

## Pipeline

1. ESM-2 embeddings for per-residue features (Lin et al. 2023). LoRA
   fine-tuning available via `src/glycoml/shared/esm2_embedder.setup_lora()`
   (Hu et al. 2021, `TaskType.FEATURE_EXTRACTION`).
2. Classifier with a configurable context window (default 11 residues,
   centred on the candidate Asn and zero-padded at termini; NetNGlyc uses 9,
   SPRINT-Gly uses 21).
3. Structure-guided ranking using relative SASA (RSA) normalised by the Tien
   et al. 2013 empirical max table and, for predicted structures, pLDDT.
   Insertion codes (e.g. Kabat H100A/H100B in CDR-H3) are preserved by
   sequence-order enumeration.
4. Optional Fc domain GNN for FcgammaR binding impact. The stub predictor
   raises `NotImplementedError` when no weights are loaded rather than
   silently returning `delta_g = 0.0`.

## Outputs

- `glycosylation_sites.csv` columns: `pdb_id, antibody_name, chain_id,
  position, residue, motif_type, plddt, sasa, rsa, accessibility_rank`.
  `plddt` is empty for experimental PDBs; `rsa` is in [0, 1.5] (upper-clip
  tolerates over-exposed termini).
- Ranked site list with composite score plddt_norm * rsa_norm * conservation.
- Optional Fc gamma receptor delta-G estimates (requires trained weights).

## References

- Gavel & von Heijne (1990) sequon consensus
- Jumper et al. (2021) AlphaFold2 / pLDDT definition
- Tien et al. (2013) empirical max SASA table
- Lin et al. (2023) ESM-2
- Hu et al. (2021) LoRA
- Gupta & Brunak (2002) NetNGlyc; Taherzadeh et al. (2019) SPRINT-Gly;
  Pakhrin et al. (2021) DeepNGlyPred
- Bruhns et al. (2009); Dekkers et al. (2017); Shields et al. (2001) for
  FcgammaR affinity references

See `src/glycoml/phase1/README.md` and `src/glycoml/phase1/data/dataset_schema.md` for details.
