# Scripts

This directory contains data acquisition and parsing utilities for CFG and UniLectin sources.

## CFG GenePix (right panel)

- `scripts/genepix_parser_clean.py`: parse GenePix XLS files using right panel columns (28-32).
  - Input: `data/raw/cfg_arrays_raw/*.xls`
  - Output: `data/processed/cfg_rfu_measurements.csv`
- `scripts/validate_genepix_clean.py`: validate the processed CFG RFU output.

```bash
python scripts/genepix_parser_clean.py
python scripts/validate_genepix_clean.py
```

## CFG Scraper (public site or local files)

- `scripts/cfg_scraper.py`: download CFG array files or parse a local directory of GenePix files.

```bash
python scripts/cfg_scraper.py \
  --use-local-dir \
  --downloads-dir data/raw/cfg_arrays_raw \
  --output data/processed/cfg_rfu_measurements.csv
```

## Phase 2 Downloader (UniLectin + CFG)

- `scripts/phase2_data_downloader.py`: orchestrates UniLectin and CFG downloads and writes outputs into
  `data/raw`, `data/metadata`, `data/interim`, and `data/processed`.

```bash
python scripts/phase2_data_downloader.py --output-dir data
```

## Merge and Validate

- `scripts/merge_rfu_sources.py`: merge multiple CFG RFU sources and deduplicate.
- `scripts/validate_cfg_output.py`: sanity checks for `data/processed/cfg_rfu_measurements.csv`.
