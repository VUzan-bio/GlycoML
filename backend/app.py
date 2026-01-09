from __future__ import annotations

import json
import os
import logging
import math
import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from backend.model_inference import Phase3Predictor

LOG = logging.getLogger("glycoml.phase3.api")

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "processed" / "phase3_fcgr_merged.csv"
MANIFEST_PATH = ROOT_DIR / "outputs" / "phase3_pymol" / "manifest.json"
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"
MODEL_PATH = Path(os.getenv("PHASE3_MODEL_PATH", ROOT_DIR / "outputs" / "phase3_fcgr" / "model.pt"))
MODEL_DEVICE = os.getenv("PHASE3_MODEL_DEVICE", "cpu")
MODEL_VERSION = os.getenv("PHASE3_MODEL_VERSION", "phase3_v1")
USE_MODEL_PREDICTIONS = os.getenv("USE_MODEL_PREDICTIONS", "false").lower() == "true"

app = FastAPI(title="GlycoML Phase3 API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_DATA: Optional[pd.DataFrame] = None
_MANIFEST: Dict[str, Dict[str, str]] = {}
_PREDICTOR: Optional[Phase3Predictor] = None
_DATA_VERSION: str = "unknown"


def _safe_key(key: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", key)


def _load_manifest(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        LOG.warning("Manifest not found: %s", path)
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception as exc:
        LOG.warning("Failed to load manifest %s: %s", path, exc)
        return {}


def _load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing data file: {path}")
    df = pd.read_csv(path)
    if "predicted_log_kd" not in df.columns and "log_kd" in df.columns:
        df["predicted_log_kd"] = df["log_kd"]
    if "predicted_kd_nm" not in df.columns and "predicted_log_kd" in df.columns:
        df["predicted_kd_nm"] = (10 ** df["predicted_log_kd"]).astype(float)
    df["delta_g_kcal_mol"] = df.get("predicted_log_kd", 0).astype(float) * -0.593
    df = df.reset_index(drop=True)
    return df


def _hash_data(path: Path) -> str:
    if not path.exists():
        return "missing"
    digest = hashlib.md5(path.read_bytes()).hexdigest()
    return digest[:8]


@app.on_event("startup")
def _startup() -> None:
    global _DATA, _MANIFEST, _PREDICTOR, _DATA_VERSION
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    _DATA = _load_data(DATA_PATH)
    _MANIFEST = _load_manifest(MANIFEST_PATH)
    _DATA_VERSION = _hash_data(DATA_PATH)
    if MODEL_PATH.exists():
        try:
            _PREDICTOR = Phase3Predictor(MODEL_PATH, device=MODEL_DEVICE)
            LOG.info("Loaded Phase3 model from %s", MODEL_PATH)
        except Exception as exc:
            LOG.warning("Failed to load model %s: %s", MODEL_PATH, exc)
    else:
        LOG.warning("Model not found: %s", MODEL_PATH)
    LOG.info("Loaded Phase3 data rows=%d", len(_DATA))


@app.get("/api/glycoforms")
def list_glycoforms() -> Dict[str, Any]:
    if _DATA is None:
        raise HTTPException(500, "Data not loaded")
    cols = [
        "fcgr_name",
        "glycan_name",
        "binding_kd_nm",
        "log_kd",
        "predicted_log_kd",
        "predicted_kd_nm",
        "glycan_structure",
    ]
    records = _DATA[cols].to_dict(orient="records")
    return {"count": len(records), "glycoforms": records}


@app.get("/api/predict")
def predict_binding(
    fcgr: str = Query(..., description="Fcgr name"),
    glycan: str = Query(..., description="Glycan name"),
) -> Dict[str, Any]:
    if _DATA is None:
        raise HTTPException(500, "Data not loaded")
    rows = _DATA[(_DATA["fcgr_name"] == fcgr) & (_DATA["glycan_name"] == glycan)]
    if rows.empty:
        raise HTTPException(404, "Glycoform not found")
    row = rows.iloc[0].to_dict()
    rank = int((_DATA["binding_kd_nm"] < row["binding_kd_nm"]).sum()) + 1
    row["affinity_rank"] = rank
    row["affinity_class"] = _affinity_class(row.get("binding_kd_nm"))
    key = f"{row['fcgr_name']}_{row['glycan_name']}"
    row["structure"] = _MANIFEST.get(key) or _MANIFEST.get(_safe_key(key), {})
    row["model_version"] = MODEL_VERSION
    row["data_version"] = _DATA_VERSION
    row["prediction_timestamp"] = datetime.now(timezone.utc).isoformat()

    if _PREDICTOR and (USE_MODEL_PREDICTIONS or pd.isna(row.get("predicted_log_kd"))):
        preds = _PREDICTOR.predict(
            fcgr_name=row.get("fcgr_name", ""),
            glycan_name=row.get("glycan_name", ""),
            glycan_structure=row.get("glycan_structure", ""),
            fc_sequence=row.get("fc_sequence", ""),
            fcgr_sequence=row.get("fcgr_sequence", ""),
        )
        row.update(preds)
    elif row.get("predicted_log_kd") is not None:
        row["predicted_kd_nm"] = float(10 ** row["predicted_log_kd"])
        row["delta_g_kcal_mol"] = float(-0.593 * row["predicted_log_kd"])
    return row


@app.get("/api/structure/{fcgr}/{glycan}")
def serve_structure(fcgr: str, glycan: str, format: str = "png") -> FileResponse:
    key = f"{fcgr}_{glycan}"
    entry = _MANIFEST.get(key) or _MANIFEST.get(_safe_key(key))
    if not entry:
        raise HTTPException(404, "Structure not found")
    if format == "png":
        path = entry.get("png_path")
    elif format == "pdb":
        path = entry.get("pdb_path")
    else:
        raise HTTPException(400, "Invalid format")
    if not path:
        raise HTTPException(404, "Structure path missing in manifest")
    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(404, f"File not found: {file_path}")
    return FileResponse(file_path)


@app.post("/api/batch-predict")
def batch_predict(pairs: List[Dict[str, str]]) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for pair in pairs:
        fcgr = pair.get("fcgr", "")
        glycan = pair.get("glycan", "")
        try:
            results.append(predict_binding(fcgr=fcgr, glycan=glycan))
        except HTTPException:
            results.append({"fcgr": fcgr, "glycan": glycan, "error": "Not found"})
    return {"results": results}


@app.post("/api/batch-predict-live")
def batch_predict_live(pairs: List[Dict[str, str]]) -> Dict[str, Any]:
    if _DATA is None:
        raise HTTPException(500, "Data not loaded")
    if _PREDICTOR is None:
        raise HTTPException(503, "Model not available")

    rows = []
    for pair in pairs:
        fcgr = pair.get("fcgr", "")
        glycan = pair.get("glycan", "")
        match = _DATA[(_DATA["fcgr_name"] == fcgr) & (_DATA["glycan_name"] == glycan)]
        if match.empty:
            continue
        rows.append(match.iloc[0].to_dict())

    if not rows:
        return {"results": [], "count": 0}

    predictions = _PREDICTOR.predict_batch(rows)
    results = []
    for row, pred in zip(rows, predictions):
        row.update(pred)
        row["model_version"] = MODEL_VERSION
        row["data_version"] = _DATA_VERSION
        row["prediction_timestamp"] = datetime.now(timezone.utc).isoformat()
        results.append(row)

    return {"results": results, "count": len(results)}


@app.get("/api/export")
def export_csv() -> StreamingResponse:
    if _DATA is None:
        raise HTTPException(500, "Data not loaded")
    csv_bytes = _DATA.to_csv(index=False).encode("utf-8")
    return StreamingResponse(
        iter([csv_bytes]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=phase3_fcgr_export.csv"},
    )


@app.get("/api/snapshot")
def snapshot(fcgr: str = Query(...), glycan: str = Query(...)) -> FileResponse:
    key = f"{fcgr}_{glycan}"
    entry = _MANIFEST.get(key) or _MANIFEST.get(_safe_key(key))
    if not entry:
        raise HTTPException(404, "Structure not found")
    path = entry.get("png_path")
    if not path:
        raise HTTPException(404, "PNG path missing in manifest")
    file_path = Path(path)
    if not file_path.exists():
        raise HTTPException(404, f"File not found: {file_path}")
    filename = _safe_key(f"{fcgr}_{glycan}_snapshot.png")
    return FileResponse(
        file_path,
        media_type="image/png",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.get("/")
def root() -> PlainTextResponse:
    return PlainTextResponse("GlycoML Phase3 API is running. See /docs.")


def _affinity_class(kd_nm: Optional[float]) -> str:
    if kd_nm is None or math.isnan(kd_nm):
        return "unknown"
    if kd_nm < 100:
        return "strong"
    if kd_nm > 1000:
        return "weak"
    return "moderate"


if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
