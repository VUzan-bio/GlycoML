"""Comparison view API checks."""

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


def test_batch_predict_works_without_model() -> None:
    client = _client()
    row = pd.read_csv(DATA_PATH).iloc[0]
    resp = client.post(
        "/api/batch-predict",
        json=[{"fcgr": row["fcgr_name"], "glycan": row["glycan_name"]}],
    )
    assert resp.status_code == 200
    payload = resp.json()
    assert "results" in payload
    assert payload["results"]
    assert payload["results"][0]["glycan_name"] == row["glycan_name"]


def test_batch_predict_live_requires_model() -> None:
    client = _client()
    row = pd.read_csv(DATA_PATH).iloc[0]
    resp = client.post(
        "/api/batch-predict-live",
        json=[{"fcgr": row["fcgr_name"], "glycan": row["glycan_name"]}],
    )
    assert resp.status_code == 503
