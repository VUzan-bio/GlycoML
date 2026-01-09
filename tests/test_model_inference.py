"""API sanity checks for Phase3 inference behavior."""

from __future__ import annotations

import importlib
import os
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

DATA_PATH = Path("data/processed/phase3_fcgr_merged.csv")

if not DATA_PATH.exists():
    pytest.skip("Missing Phase3 dataset for API tests.", allow_module_level=True)


def _client() -> TestClient:
    os.environ["PHASE3_MODEL_PATH"] = "missing-model.pt"
    os.environ["PHASE3_MODEL_DEVICE"] = "cpu"
    os.environ["USE_MODEL_PREDICTIONS"] = "false"
    import backend.app as app_module

    importlib.reload(app_module)
    return TestClient(app_module.app)


def test_predict_endpoint_returns_metrics() -> None:
    client = _client()
    row = pd.read_csv(DATA_PATH).iloc[0]
    resp = client.get(
        "/api/predict",
        params={"fcgr": row["fcgr_name"], "glycan": row["glycan_name"]},
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["fcgr_name"] == row["fcgr_name"]
    assert payload["glycan_name"] == row["glycan_name"]
    assert "binding_kd_nm" in payload
    assert "predicted_kd_nm" in payload
    assert "delta_g_kcal_mol" in payload


def test_glycoforms_endpoint_lists_rows() -> None:
    client = _client()
    resp = client.get("/api/glycoforms")
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["count"] > 0
    assert len(payload["glycoforms"]) == payload["count"]
