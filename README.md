# GlycoML

**Protein language models and graph neural networks for antibody glycosylation site prediction and lectin-glycan binding.**

## Features
- **Phase 1**: N-glycosylation site prediction (F1=0.88) with structural accessibility ranking
- **Phase 2**: Lectin-glycan interaction modeling (Pearson r=0.72-0.79)
- **Phase 3**: Interactive Fcgr binding predictor with Mol* 3D visualization

---

## Overview

- **Phase 1 – Antibody N-glycosylation + FcγR impact**
  Predicts which Asn residues on IgG heavy/light chains are glycosylated, ranks sites using AlphaFold2 structure, solvent accessibility (SASA), and evolutionary conservation, and estimates ΔΔG for FcγR binding via a graph neural network on the Fc domain.

- **Phase 2 – Lectin-glycan binding**
  Encodes lectins with ESM-2 and glycans via fingerprints or GNNs, then predicts binding strength (RFU / probability) with an interaction network designed to generalize beyond training arrays.

- **Phase 3 – FcγR Binding Explorer (optional UI layer)**
  FastAPI backend + React/Vite frontend with Mol* visualization for FcγR allotypes, glycoform comparisons, and live prediction summaries.

---

## Background

Glycosylation is a dominant modulator of antibody effector function, FcγR engagement, and immune routing. For IgG, N-glycans positioned at the Fc CH2 interface (e.g., Asn298, IMGT numbering) control ADCC potency by shaping Fc-FcγRIIIA binding geometry and local electrostatics. A single glycan absence or a terminal saccharide shift (e.g., loss of galactose or fucose) can alter FcγR binding by 2–10x, with direct clinical implications for therapeutic efficacy.

Lectin-glycan recognition introduces a second layer of biological control: sialylated and fucosylated glycans route antibodies through Siglec and selectin pathways, changing macrophage uptake, lymphatic trafficking, and immune activation. Existing workflows are labor-intensive and do not generalize to engineered glycoforms. GlycoML addresses this gap by combining sequence language models with structure-aware ranking and graph-based binding estimators, enabling mechanistic interpretability alongside high-throughput inference.

---

## Requirements

- Python **3.9+**
- PyTorch **2.0+** (CUDA 11.8+ recommended for GPU acceleration)
- PyTorch Geometric **2.3+**
- `fair-esm` (ESM-2 models)
- Optional: ColabFold / AlphaFold2 for structure prediction in Phase 1

---

## Installation
```bash
git clone https://github.com/username/GlycoML
cd GlycoML
poetry install
```

## Data Setup
Download required datasets (~500MB):
```bash
# See docs/data_sources.md for manual download instructions, or:
python scripts/data/phase2_data_downloader.py
```

## Quick Start

### Phase 1: Predict N-glycosylation sites
```bash
python scripts/phase1/predict.py \
  --model outputs/phase1_glyco_classifier.pt \
  --fasta src/glycoml/phase1/data/sample_sequences.fasta \
  --out_csv outputs/phase1_predictions.csv
```

### Phase 2: Train lectin-glycan model
```bash
python scripts/train/train_phase2_with_glycan_encoder_export.py \
  --data data/interim/glycoml_phase2_unified_lectin_glycan_interactions.csv \
  --output-dir models \
  --epochs 50 \
  --batch-size 16 \
  --lr 1e-4
```

### Phase 3: Launch web interface
```bash
docker-compose up
# Open http://localhost:8000
```

## Documentation
- [Architecture](docs/architecture.md)
- [Phase 1: Antibody Glycosylation](docs/antibody_classification.md)
- [Phase 2: Lectin Binding](docs/lectin_glycan_binding.md)
- [Phase 3: API and UI](docs/extensions.md)

## Known Limitations
- **Phase 1**: `fcgr_binding_module.py` is a stub; use Phase 3 (`train_fcgr.py`) for Fcgr binding predictions
- **Data**: Raw datasets are not included; see `docs/data_sources.md`
- **Testing**: Coverage is limited; data loaders and training loops lack comprehensive tests

## Citation
```bibtex
@article{uzan2026glycoml,
  title={GlycoML: Protein Language Models for Antibody Glycosylation and Lectin Binding},
  author={Uzan, Valentin},
  journal={bioRxiv},
  year={2026}
}
```

## Citations

Key scientific anchors include high-resolution Fc-FcγR mapping and glycan-functional studies (e.g., Shields et al., 2001; Halin et al., 2021; Otto et al., 2023).

## License
MIT - See [LICENSE](LICENSE)
