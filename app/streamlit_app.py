"""BESS Dispatch Optimiser & Advisor — public demo (Streamlit).

Three surfaces over the same engine:
  • Forecast & Dispatch — LSTM next-24h price forecast → MILP schedule + revenue
  • Ask the Advisor     — LangGraph agent (forecast / optimise / RAG tools)
  • Backtest            — the day-ahead money chart (perfect vs forecast vs naive)

Reads baked artifacts (trained model, cached prices, Chroma store) from the image.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import json  # noqa: E402

from src.bess import data  # noqa: E402
from src.bess.config import BatteryConfig, MarketConfig  # noqa: E402
from src.bess.optimiser import optimise_dispatch  # noqa: E402

st.set_page_config(page_title="BESS Dispatch Optimiser", page_icon="🔋", layout="wide")

REPO_URL = "https://github.com/KevinBean/bess-dispatch-optimiser"
# Forecast skill vs seasonal-naive on the held-out test split (MAE-based, from training).
FORECAST_SKILL = {"SA1": 0.075, "NSW1": 0.321, "QLD1": 0.175, "TAS1": 0.102, "VIC1": 0.043}


def _load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def available_regions() -> list[str]:
    """Regions that actually have a baked model + cached prices (so the demo can
    serve them). Avoids offering regions that would crash on load."""
    out = []
    for r in data.REGIONS:
        has_data = (ROOT / "data" / "processed" / f"{r}.parquet").exists()
        has_model = (ROOT / "models" / f"forecaster_{r}.pt").exists()
        if has_data and has_model:
            out.append(r)
    return out or [data.REGIONS[0]]


@st.cache_resource
def load_region(region: str):
    df = data.load_processed(region)
    fc = None
    mp = ROOT / "models" / f"forecaster_{region}.pt"
    if mp.exists():
        from src.bess.model import Forecaster

        fc = Forecaster().load(str(mp))
    return df, fc


def battery_from_sidebar() -> BatteryConfig:
    st.sidebar.header("Battery")
    power = st.sidebar.slider("Power (MW)", 10, 300, 100, 10)
    duration = st.sidebar.slider("Duration (h)", 1.0, 4.0, 2.0, 0.5)
    rte = st.sidebar.slider("Round-trip efficiency", 0.80, 0.98, 0.86, 0.01)
    deg = st.sidebar.slider("Degradation cost ($/MWh)", 0.0, 10.0, 2.0, 0.5)
    eta = float(np.sqrt(rte))
    return BatteryConfig(
        power_mw=power, energy_mwh=power * duration,
        eta_charge=eta, eta_discharge=eta, degradation_cost_per_mwh=deg,
    )


st.title("🔋 BESS Dispatch Optimiser & Advisor")
st.caption(
    "MILP arbitrage (PuLP + HiGHS) · PyTorch price forecast · LangGraph advisor · "
    "Chroma RAG over NEM market docs · real AEMO data"
)
with st.expander("What this demonstrates  ·  view the code", expanded=False):
    st.markdown(
        f"""
A grid-scale battery dispatch optimiser & advisor, built end-to-end on **real AEMO
NEM data** to answer: *when should a battery charge/discharge to maximise arbitrage
revenue?*

| Capability | How |
|---|---|
| **Mathematical optimisation** | MILP dispatch — round-trip losses, SoC limits, charge/discharge mutex (PuLP + HiGHS) |
| **Deep learning (PyTorch)** | Multi-horizon LSTM price forecaster, evaluated leakage-free vs a seasonal-naive baseline |
| **Agents (LangGraph)** | Tool-calling `StateGraph` — the optimiser, forecaster, and doc search are tools |
| **RAG / vector DB (Chroma)** | Hugging-Face embeddings over NEM market docs so the agent cites sources |
| **Cloud deploy (AWS / GCP)** | Containerised; this demo runs on Cloud Run |

🔗 **[View the source on GitHub]({REPO_URL})**  ·  five NEM regions · honest perfect-vs-forecast-vs-naive backtest
"""
    )

_regions = available_regions()
region = st.sidebar.selectbox(
    "NEM region", _regions,
    index=_regions.index("SA1") if "SA1" in _regions else 0,
)
battery = battery_from_sidebar()
market = MarketConfig(region=region)

tab1, tab2, tab3 = st.tabs(["⚡ Forecast & Dispatch", "💬 Ask the Advisor", "📈 Backtest"])

with tab1:
    try:
        df, fc = load_region(region)
    except Exception as e:  # noqa: BLE001
        df, fc = None, None
        st.error(f"Data for {region} isn't bundled in this demo ({type(e).__name__}).")
    if fc is None:
        st.warning(f"No trained forecaster for {region} — pick another region.")
    else:
        preds = fc.predict_next(df.iloc[-fc.cfg.lookback :])
        res = optimise_dispatch(preds, battery, market)
        dt_h = market.interval_hours
        hours = np.arange(len(preds)) * dt_h

        skill = FORECAST_SKILL.get(region)
        c1, c2, c3 = st.columns(3)
        c1.metric("Est. 24h arbitrage revenue", f"${res.revenue:,.0f}")
        c2.metric("Forecast skill vs naive (MAE)",
                  f"{skill:+.0%}" if skill is not None else "—")
        c3.metric("Round-trip efficiency", f"{battery.round_trip_efficiency:.0%}")
        st.caption(f"Battery: {battery.power_mw:.0f} MW / {battery.energy_mwh:.0f} MWh "
                   f"({battery.energy_mwh / battery.power_mw:.1f} h duration)")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        ax1.plot(hours, preds, color="#d62728")
        ax1.set_ylabel("Forecast price\n($/MWh)")
        ax1.grid(alpha=0.3)
        s = res.schedule
        ax2.bar(hours, s.discharge, width=dt_h * 0.9, color="#2ca02c", label="discharge")
        ax2.bar(hours, -s.charge, width=dt_h * 0.9, color="#1f77b4", label="charge")
        ax2b = ax2.twinx()
        ax2b.plot(hours, s.soc, color="#7f7f7f", lw=1.5, label="SoC")
        ax2b.set_ylabel("SoC (MWh)")
        ax2.set_xlabel("hours ahead")
        ax2.set_ylabel("power (MW)")
        ax2.legend(loc="upper left")
        ax2.grid(alpha=0.3)
        st.pyplot(fig)
        st.caption("Forecast-based estimate; actual revenue depends on realised prices.")

with tab2:
    st.write("Ask about dispatch, forecasts, revenue, or NEM market mechanics.")
    keyless = not os.environ.get("OPENAI_API_KEY")
    if keyless:
        st.info(
            "🔑 The live LLM advisor is disabled on this public demo (no API key, to "
            "avoid billing exposure) — but here are **real captured answers** below, so "
            "you can see what the LangGraph agent produces. It calls the optimiser, "
            "forecaster, and Chroma RAG store as tools. Run locally with `OPENAI_API_KEY` "
            "to ask your own questions."
        )
        examples = _load_json(ROOT / "docs" / "agent_examples.json")
        if examples and examples.get("examples"):
            st.markdown("##### Example agent answers (captured live)")
            for ex in examples["examples"]:
                with st.expander("💬 " + ex["question"], expanded=False):
                    st.markdown(ex["answer"])
    else:
        q = st.text_input("Question", "What's the optimal dispatch in SA1 tomorrow and how much could I make?")
        if st.button("Ask"):
            with st.spinner("Agent thinking (calling tools)…"):
                try:
                    from src.bess.agent import ask

                    st.markdown(ask(q))
                except Exception as e:  # noqa: BLE001
                    st.error(f"Agent unavailable: {e}\n\nSet OPENAI_API_KEY to enable the advisor.")

with tab3:
    chart = ROOT / "docs" / f"money_chart_{region}.png"
    if not chart.exists():
        chart = ROOT / "docs" / "money_chart.png"  # fallback to SA1 hero image
    if chart.exists():
        st.image(str(chart), caption=f"{region} day-ahead backtest, settled at actual prices.")
    else:
        st.info("Run scripts/run_backtest.py to generate the money chart.")

    bt = _load_json(ROOT / "docs" / f"backtest_{region}.json")
    reading = (
        "**Reading it:** the gap between *perfect* and *forecast* is the cost of "
        "forecast error; *forecast* vs *naive* is whether the LSTM adds value. Most of "
        "perfect-foresight's lead comes from a few unpredictable price-spike days."
    )
    if bt and "policies" in bt:
        pol = bt["policies"]
        fpct = pol.get("forecast", {}).get("pct_of_perfect")
        npct = pol.get("naive", {}).get("pct_of_perfect")
        if fpct is not None and npct is not None:
            if fpct >= npct:
                verdict = (f"Here the LSTM captured **{fpct:.0%}** of perfect-foresight "
                           f"revenue vs **{npct:.0%}** for naive — it adds value in {region}.")
            else:
                verdict = (f"Honest result: in {region} the LSTM captured **{fpct:.0%}** "
                           f"vs naive's **{npct:.0%}** — naive did *better* on revenue here "
                           f"despite the LSTM's stronger MAE. Better point-forecast accuracy "
                           f"doesn't guarantee better dispatch economics.")
            reading += "\n\n" + verdict
    st.markdown(reading)
