# External Data Sources

GlycoML requires public datasets not included in this repository.

## Required Downloads (Total: ~500MB)

### 1. Thera-SAbDab
- URL: http://opig.stats.ox.ac.uk/webapps/newsabdab/therasabdab/
- Download: All therapeutic antibodies (CSV format)
- Place in: `data/external/thera_sabdab.csv`
- Size: ~5MB

### 2. CFG Glycan Arrays
- URL: https://www.functionalglycomics.org/
- Download: Mammalian Array v5.0 (requires free registration)
- Place in: `data/external/cfg_arrays/`
- Size: ~200MB

### 3. UniLectin3D
- URL: https://unilectin.eu/
- Download: Full dataset (JSON)
- Place in: `data/external/unilectin3d.json`
- Size: ~50MB

## Automated Download (Optional)
```bash
python scripts/data/phase2_data_downloader.py --output data/external/
```

## Verification
After processing, use the existing validation scripts as needed:
```bash
python scripts/data/validate_cfg_output.py
python scripts/data/validate_genepix_clean.py
```
