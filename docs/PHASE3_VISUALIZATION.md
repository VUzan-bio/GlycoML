# Phase3 Fcgr Web Visualization

This document describes the Phase3 visualization stack: rendering structures, serving predictions via FastAPI, and a lightweight React UI with a Molstar viewer.
For live inference and the comparison view, see `docs/PHASE3_EXTENSIONS.md`.

## 1) Render structures and build the manifest

Render high-resolution images and PDBs for each Fcgr-glycan pair and write a manifest JSON:

```bash
python scripts/render_fcgr_structures.py \
  --data data/processed/phase3_fcgr_merged.csv \
  --pdb-template data/structures/fc_fcgr_complex.pdb \
  --glycan-dir data/structures/glycans \
  --output-dir outputs/phase3_pymol
```

Outputs:
- `outputs/phase3_pymol/<fcgr>_<glycan>.png`
- `outputs/phase3_pymol/<fcgr>_<glycan>.pdb`
- `outputs/phase3_pymol/<fcgr>_<glycan>.pse`
- `outputs/phase3_pymol/manifest.json`

If glycan PDBs are unavailable, omit `--glycan-dir`; the script will still render the protein complex and produce a manifest.

## 2) Start the API

```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /api/glycoforms`
- `GET /api/predict?fcgr=<name>&glycan=<name>`
- `GET /api/structure/<fcgr>/<glycan>?format=png|pdb`
- `GET /api/export`
- `POST /api/batch-predict-live`
- `GET /api/snapshot?fcgr=<name>&glycan=<name>`

The API reads `data/processed/phase3_fcgr_merged.csv` and enriches rows with predicted values if present.
To enable live inference, set `USE_MODEL_PREDICTIONS=true` and ensure the model checkpoint exists.

## 3) Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:3000`.

## 4) Docker (optional)

```bash
docker-compose up --build
```

This builds the frontend and runs the FastAPI server in a single container.

## Data expectations

The API expects at least these columns in `data/processed/phase3_fcgr_merged.csv`:
- `fcgr_name`
- `glycan_name`
- `binding_kd_nm`
- `log_kd`
- `glycan_structure`

If `predicted_log_kd` is present, it will be used to compute `predicted_kd_nm` and `delta_g_kcal_mol`.
