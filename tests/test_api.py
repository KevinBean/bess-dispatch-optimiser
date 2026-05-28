"""Smoke tests for the FastAPI service layer (TestClient, no network)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
HAS_MODEL = (ROOT / "models" / "forecaster_SA1.pt").exists()

from fastapi.testclient import TestClient  # noqa: E402

from src.bess.api import app  # noqa: E402

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_unknown_region_is_400():
    r = client.get("/forecast?region=ZZ9")
    assert r.status_code == 400


@pytest.mark.skipif(not HAS_MODEL, reason="needs a trained SA1 forecaster")
def test_forecast_shape():
    r = client.get("/forecast?region=SA1")
    assert r.status_code == 200
    j = r.json()
    assert j["region"] == "SA1"
    assert len(j["prices"]) == j["horizon"] > 0


@pytest.mark.skipif(not HAS_MODEL, reason="needs a trained SA1 forecaster")
def test_optimise_returns_revenue_and_schedule():
    r = client.post("/optimise", json={"region": "SA1", "power_mw": 50, "energy_mwh": 100})
    assert r.status_code == 200
    j = r.json()
    assert j["status"] == "Optimal"
    assert "estimated_revenue" in j
    assert len(j["schedule"]["charge_mw"]) == len(j["schedule"]["soc_mwh"])


def test_ask_without_key_is_503(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    r = client.post("/ask", json={"question": "hi"})
    assert r.status_code == 503
