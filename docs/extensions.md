# Phase3 Extensions: Live Inference + Comparison View

This document covers the Phase3 extensions on top of the baseline visualization stack:
1) live model inference in the API, and
2) side-by-side comparison view in the frontend.

The API and UI are already wired; this doc clarifies behavior, configuration, and expected outputs.

## 1) Live model inference

### What it does
- Loads the Phase3 PyTorch checkpoint at API startup.
- Uses cached features + the model to produce `predicted_log_kd`, `predicted_kd_nm`,
  and `delta_g_kcal_mol`.
- Preserves backward compatibility: CSV values are still used unless `USE_MODEL_PREDICTIONS=true`.

### Files
- `backend/model_inference.py`: `Phase3Predictor` (ESM2 pooled embeddings + glycan composition features).
- `backend/app.py`: integrates the predictor and exposes live inference endpoints.

### Configuration
Environment variables:
- `PHASE3_MODEL_PATH` (default: `outputs/phase3_fcgr/model.pt`)
- `PHASE3_MODEL_DEVICE` (default: `cpu`)
- `PHASE3_MODEL_VERSION` (default: `phase3_v1`)
- `USE_MODEL_PREDICTIONS` (`true` or `false`, default: `false`)

### API behavior
- `GET /api/predict` returns prediction fields from the model if:
  - `USE_MODEL_PREDICTIONS=true`, or
  - the CSV row is missing `predicted_log_kd`.
- `POST /api/batch-predict-live` returns batch predictions and metadata.
  If no model is available, it returns HTTP 503.

## 2) Comparison view

### What it does
- Lets users compare 2-3 glycoforms at once for a single Fcgr allotype.
- Shows a metrics table and multiple structure viewers in a responsive grid.
- Uses the live batch endpoint by default.

### Files
- `frontend/src/pages/CompareView.tsx`
- `frontend/src/components/GlycoformMultiSelector.tsx`
- `frontend/src/components/ComparisonTable.tsx`
- `frontend/src/components/ComparisonGrid.tsx`
- `frontend/src/api.ts`

### UI rules
- Max 3 glycoforms per compare request.
- Strong binding (<100 nM) = green, weak binding (>1000 nM) = red, else yellow/gray.

## 3) Snapshot endpoint

The API serves high-res images for figure exports:
- `GET /api/snapshot?fcgr=<name>&glycan=<name>`

## 4) Runbook

### Start API
```bash
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

### Start frontend (dev)
```bash
cd frontend
npm install
npm run dev
```

### Enable live inference
```bash
export USE_MODEL_PREDICTIONS=true
export PHASE3_MODEL_PATH=outputs/phase3_fcgr/model.pt
```

## 5) Testing

```bash
pytest tests/test_model_inference.py tests/test_comparison.py
```

These tests:
- verify `/api/predict` returns expected fields from CSV,
- verify `/api/batch-predict-live` returns HTTP 503 when no model is configured,
- validate that the API remains backward compatible.
