"""FastAPI service layer over the optimiser, forecaster, and advisor.

A small REST surface so the engine can be driven programmatically — by a frontend,
another service (e.g. an EmptyOS connector app), or a curl/CI smoke test — without
going through the Streamlit UI.

    GET  /health                  liveness + which region models are loaded
    GET  /forecast?region=SA1     next-24h price forecast ($/MWh)
    POST /optimise                {region, power_mw?, energy_mwh?, ...} -> dispatch + revenue
    GET  /backtest?region=SA1     day-ahead perfect/forecast/naive revenue
    POST /ask                     {question} -> LangGraph agent answer (needs OPENAI_API_KEY)

Run:  uvicorn src.bess.api:app --reload
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import data
from .config import BatteryConfig, MarketConfig
from .optimiser import optimise_dispatch

ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(
    title="BESS Dispatch Optimiser",
    version="0.1.0",
    description="MILP battery arbitrage + PyTorch price forecast + LangGraph advisor over NEM data.",
)


@lru_cache(maxsize=8)
def _region(region: str):
    """Cache (prices df, forecaster|None) per region."""
    region = region.upper()
    if region not in data.REGIONS:
        raise HTTPException(400, f"unknown region '{region}'; expected one of {data.REGIONS}")
    try:
        df = data.load_processed(region)
    except FileNotFoundError:
        raise HTTPException(404, f"no cached data for {region}; run scripts/fetch_data.py")
    fc = None
    mp = ROOT / "models" / f"forecaster_{region}.pt"
    if mp.exists():
        from .model import Forecaster

        fc = Forecaster().load(str(mp))
    return df, fc


# --- schemas ----------------------------------------------------------------

class OptimiseRequest(BaseModel):
    region: str = "SA1"
    power_mw: float = Field(100.0, gt=0)
    energy_mwh: float = Field(200.0, gt=0)
    round_trip_efficiency: float = Field(0.86, gt=0, le=1)
    degradation_cost_per_mwh: float = Field(2.0, ge=0)


class AskRequest(BaseModel):
    question: str


# --- endpoints --------------------------------------------------------------

@app.get("/health")
def health():
    loaded = [r for r in data.REGIONS if (ROOT / "models" / f"forecaster_{r}.pt").exists()]
    return {"status": "ok", "models": loaded}


@app.get("/forecast")
def forecast(region: str = "SA1"):
    df, fc = _region(region)
    if fc is None:
        raise HTTPException(404, f"no trained forecaster for {region}")
    preds = fc.predict_next(df.iloc[-fc.cfg.lookback :])
    dt_h = MarketConfig(region=region).interval_hours
    return {
        "region": region.upper(),
        "interval_hours": dt_h,
        "horizon": len(preds),
        "mean": round(float(preds.mean()), 2),
        "min": round(float(preds.min()), 2),
        "max": round(float(preds.max()), 2),
        "peak_hours_ahead": round(float(np.argmax(preds)) * dt_h, 1),
        "trough_hours_ahead": round(float(np.argmin(preds)) * dt_h, 1),
        "prices": [round(float(p), 2) for p in preds],
    }


@app.post("/optimise")
def optimise(req: OptimiseRequest):
    df, fc = _region(req.region)
    market = MarketConfig(region=req.region)
    eta = float(np.sqrt(req.round_trip_efficiency))
    battery = BatteryConfig(
        power_mw=req.power_mw, energy_mwh=req.energy_mwh,
        eta_charge=eta, eta_discharge=eta,
        degradation_cost_per_mwh=req.degradation_cost_per_mwh,
    )
    if fc is None:
        raise HTTPException(404, f"no trained forecaster for {req.region}")
    preds = fc.predict_next(df.iloc[-fc.cfg.lookback :])
    res = optimise_dispatch(preds, battery, market)
    s = res.schedule
    dt_h = market.interval_hours
    return {
        "region": req.region.upper(),
        "status": res.status,
        "solver": res.solver,
        "estimated_revenue": round(res.revenue, 2),
        "charge_hours": round(float((s.charge > 1).sum()) * dt_h, 1),
        "discharge_hours": round(float((s.discharge > 1).sum()) * dt_h, 1),
        "schedule": {
            "charge_mw": [round(float(x), 2) for x in s.charge],
            "discharge_mw": [round(float(x), 2) for x in s.discharge],
            "soc_mwh": [round(float(x), 2) for x in s.soc],
            "price": [round(float(x), 2) for x in s.price],
        },
    }


@app.get("/backtest")
def backtest_endpoint(region: str = "SA1", test_start: str = "2024-11-15"):
    df, fc = _region(region)
    if fc is None:
        raise HTTPException(404, f"no trained forecaster for {region}")
    from .backtest import backtest

    res = backtest(df, fc, test_start=test_start,
                   battery=BatteryConfig(), market=MarketConfig(region=region))
    return {
        "region": region.upper(),
        "test_start": test_start,
        "policies": {
            name: {
                "realised_revenue": round(r.realised_revenue, 2),
                "pct_of_perfect": round(r.captured_frac_of_perfect, 4),
            }
            for name, r in res.items()
        },
    }


@app.post("/ask")
def ask_endpoint(req: AskRequest):
    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(503, "advisor disabled: OPENAI_API_KEY not set")
    from .agent import ask

    return {"question": req.question, "answer": ask(req.question)}
