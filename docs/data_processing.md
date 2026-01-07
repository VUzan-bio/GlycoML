# Data Processing

This document describes the data acquisition and processing pipeline for CFG and UniLectin sources.

## CFG Pipeline

### GenePix parsing (right panel)

- Input: `data/raw/cfg_arrays_raw/*.xls`
- Parser: `scripts/genepix_parser_clean.py`
- Columns used: 28-32 (Chart Number.1, Structure, AvgMeanS-B w/o MIN/MAX.1, StDev.1, %CV.1)
- Output: `data/processed/cfg_rfu_measurements.csv`

### CFG scraping

- Script: `scripts/cfg_scraper.py`
- Supports local parsing or downloading from CFG.
- Outputs the same processed RFU CSV.

## UniLectin Pipeline

- Script: `scripts/phase2_data_downloader.py`
- Outputs:
  - `data/interim/unilectin3d_lectin_glycan_interactions.csv`
  - `data/metadata/unilectin3d_predicted_lectins.csv`
  - `data/metadata/unilectin3d_ligands.csv`
  - `data/metadata/cfg_experiment_metadata.csv`
  - `data/metadata/cfg_to_unilectin_lectin_mapping.csv`
  - `data/interim/glycoml_phase2_unified_lectin_glycan_interactions.csv`

## Validation

- Script: `scripts/validate_cfg_output.py`
- Script: `scripts/validate_genepix_clean.py`
- Pipeline validation: `data_pipeline/orchestrator.py`

## Directory Layout

- `data/raw`: immutable source files.
- `data/interim`: intermediate merged datasets.
- `data/processed`: clean training-ready datasets.
- `data/metadata`: experiment and mapping metadata.
- `data/cache`: PDB and UniLectin caches.
