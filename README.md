# GlycoML

End-to-end machine learning framework for (i) predicting antibody N-glycosylation sites and FcγR binding impact, and (ii) modeling lectin-glycan interactions from sequence and structure. The codebase is organized as a Python monorepo with shared encoders across phases and a lightweight web UI for structural exploration.

---

## Overview

- **Phase 1 – Antibody N-glycosylation + FcγR impact**
  Predicts which Asn residues on IgG heavy/light chains are glycosylated, ranks sites using AlphaFold2 structure, solvent accessibility (SASA), and evolutionary conservation, and estimates ΔΔG for FcγR binding via a graph neural network on the Fc domain.

- **Phase 2 – Lectin-glycan binding**
  Encodes lectins with ESM-2 and glycans via fingerprints or GNNs, then predicts binding strength (RFU / probability) with an interaction network designed to generalize beyond training arrays.

- **Phase 3 – FcγR Binding Explorer (optional UI layer)**
  FastAPI backend + React/Vite frontend with Mol* visualization for FcγR allotypes, glycoform comparisons, and live prediction summaries.

---

## Scientific Rationale (PhD-level summary)

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

From the repository root:

```bash
# Clone
git clone https://github.com/VUzan-bio/GlycoML.git
cd GlycoML

# Option A: pip (editable install)
python -m venv .venv
source .venv/bin/activate             # On Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .

# Option B: Poetry
poetry install
poetry shell
```

This installs the `glycoml` package and all Python dependencies in editable mode, so local code changes are picked up immediately.

---

## CLI Quickstart

### Phase 2 example (lectin-glycan interaction model)

```bash
python -m glycoml.phase2.train \
  --data-path data/interim/glycoml_phase2_unified_lectin_glycan_interactions.csv \
  --output-dir outputs/phase2_model
```

This trains the Phase 2 interaction model on the unified lectin-glycan dataset and writes model weights, metrics, and predictions into `outputs/phase2_model/`.

Analogous scripts exist under `glycoml.phase1` and `scripts/` for Phase 1 training, evaluation, and data preparation (see docstrings and scripts subfolders for task-specific entrypoints).

---

## Web Application (optional)

This repository ships a lightweight FastAPI backend plus a React/Vite UI for Phase 3 visualization. To run locally:

```bash
# API
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000

# Frontend (separate terminal)
cd frontend
npm install
npm run dev
```

The UI expects an API layer that exposes FcγR/glycan predictions and structure endpoints. You can also integrate GlycoML as a library in your own service layer by calling the Phase 1/Phase 2 inference functions and returning JSON.

---

## Repository Layout

- `glycoml/` - core Python package (encoders, models, training loops)
- `backend/` - FastAPI service for Phase 3 visualization
- `frontend/` - React/Vite UI with Mol* viewer
- `data/` - processed datasets and input CSVs
- `scripts/` - utility scripts (rendering, preprocessing, export)
- `outputs/` - model checkpoints, rendered structures, metrics

---

## Citations

Key scientific anchors include high-resolution Fc-FcγR mapping and glycan-functional studies (e.g., Shields et al., 2001; Halin et al., 2021; Otto et al., 2023). If you use GlycoML in a publication, please cite these foundational sources and this repository.

Documentation and releases are hosted at: https://github.com/VUzan-bio/GlycoML
