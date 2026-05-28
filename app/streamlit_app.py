"""BESS Dispatch Optimiser & Advisor — public demo (Streamlit).

Three surfaces over the same engine:
  • Forecast & Dispatch — LSTM next-24h price forecast → MILP schedule + revenue
  • Ask the Advisor     — LangGraph agent (forecast / optimise / RAG tools)
  • Backtest            — the day-ahead money chart (perfect vs forecast vs naive)

Reads baked artifacts (trained model, cached prices, Chroma store) from the image.
"""

from __future__ import annotations

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

from src.bess import data  # noqa: E402
from src.bess.config import BatteryConfig, MarketConfig  # noqa: E402
from src.bess.optimiser import optimise_dispatch  # noqa: E402

st.set_page_config(page_title="BESS Dispatch Optimiser", page_icon="🔋", layout="wide")


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

region = st.sidebar.selectbox("NEM region", list(data.REGIONS), index=2)
battery = battery_from_sidebar()
market = MarketConfig(region=region)

tab1, tab2, tab3 = st.tabs(["⚡ Forecast & Dispatch", "💬 Ask the Advisor", "📈 Backtest"])

with tab1:
    df, fc = load_region(region)
    if fc is None:
        st.warning(f"No trained forecaster for {region}.")
    else:
        preds = fc.predict_next(df.iloc[-fc.cfg.lookback :])
        res = optimise_dispatch(preds, battery, market)
        dt_h = market.interval_hours
        hours = np.arange(len(preds)) * dt_h

        c1, c2, c3 = st.columns(3)
        c1.metric("Est. 24h arbitrage revenue", f"${res.revenue:,.0f}")
        c2.metric("Round-trip efficiency", f"{battery.round_trip_efficiency:.0%}")
        c3.metric("Energy / Power", f"{battery.energy_mwh:.0f} MWh / {battery.power_mw:.0f} MW")

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
    q = st.text_input("Question", "What's the optimal dispatch in SA1 tomorrow and how much could I make?")
    if st.button("Ask"):
        with st.spinner("Agent thinking (calling tools)…"):
            try:
                from src.bess.agent import ask

                st.markdown(ask(q))
            except Exception as e:  # noqa: BLE001
                st.error(f"Agent unavailable: {e}\n\nSet OPENAI_API_KEY to enable the advisor.")

with tab3:
    chart = ROOT / "docs" / "money_chart.png"
    if chart.exists():
        st.image(str(chart), caption="Day-ahead backtest, settled at actual prices.")
    else:
        st.info("Run scripts/run_backtest.py to generate the money chart.")
    st.markdown(
        "**Reading it:** the gap between *perfect* and *forecast* is the cost of "
        "forecast error; *forecast* vs *naive* is the value the LSTM adds. Most of "
        "perfect-foresight's lead comes from a few unpredictable price-spike days."
    )
