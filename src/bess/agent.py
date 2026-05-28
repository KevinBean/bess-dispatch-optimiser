"""LangGraph advisor agent over the optimiser + forecaster + RAG store.

A tool-calling agent built as an explicit ``StateGraph`` (not a hidden loop): an
LLM node decides which tool to call, a ToolNode runs it, control returns to the
LLM until it has enough to answer. Three tools expose the rest of the system:

    forecast_prices    — PyTorch LSTM next-24h price forecast for a region
    recommend_dispatch — forecast → MILP optimise → plain-language schedule + $
    search_market_docs — Chroma RAG over the NEM/BESS knowledge corpus (cited)

The graph shape (assistant ⇄ tools, terminating when the LLM stops calling tools)
is the canonical LangGraph pattern and is what makes this an *agent* rather than a
fixed pipeline — the model chooses the tool sequence per question.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

from . import data
from .backtest import build_features  # noqa: F401  (kept for parity of feature path)
from .config import BatteryConfig, MarketConfig
from .optimiser import optimise_dispatch
from .rag import RagStore

ROOT = Path(__file__).resolve().parents[2]

SYSTEM_PROMPT = """You are a battery dispatch advisor for the Australian NEM.
You help an operator decide when to charge and discharge a grid-scale battery to
maximise energy-arbitrage revenue.

Rules:
- For any question about future prices, optimal dispatch, or expected revenue, you
  MUST call a tool. Never invent numbers.
- For questions about market mechanics, rules, efficiency, tariffs, or FCAS, call
  search_market_docs and cite the source document and section in your answer.
- Be concrete and quantitative. Report times in the region's local interval terms.
- State clearly that forecasts are uncertain and revenue is an estimate.
- If a tool returns an error, explain what is missing rather than guessing.
"""


@lru_cache(maxsize=8)
def _load_region(region: str):
    """Cache (prices df, forecaster) per region; forecaster optional."""
    df = data.load_processed(region)
    fc = None
    model_path = ROOT / "models" / f"forecaster_{region}.pt"
    if model_path.exists():
        from .model import Forecaster

        fc = Forecaster().load(str(model_path))
    return df, fc


@lru_cache(maxsize=1)
def _rag() -> RagStore:
    return RagStore()


# --- Tools ------------------------------------------------------------------

@tool
def forecast_prices(region: str = "SA1") -> str:
    """Forecast the next 24 hours of NEM spot prices ($/MWh) for a region
    (one of NSW1, QLD1, SA1, TAS1, VIC1) using the trained PyTorch model."""
    df, fc = _load_region(region)
    if fc is None:
        return f"No trained forecaster for {region}. Train one with scripts/train_forecast.py."
    preds = fc.predict_next(df.iloc[-fc.cfg.lookback :])
    dt_h = MarketConfig(region=region).interval_hours
    peak_i = int(np.argmax(preds))
    trough_i = int(np.argmin(preds))
    return (
        f"{region} next-24h forecast ({len(preds)} x {dt_h*60:.0f}-min intervals): "
        f"mean ${preds.mean():.0f}/MWh, min ${preds.min():.0f} at +{trough_i*dt_h:.1f}h, "
        f"max ${preds.max():.0f} at +{peak_i*dt_h:.1f}h."
    )


@tool
def recommend_dispatch(region: str = "SA1") -> str:
    """Recommend the optimal next-24h battery charge/discharge schedule for a
    region and estimate arbitrage revenue, by forecasting prices and solving the
    dispatch MILP. Returns a plain-language plan."""
    df, fc = _load_region(region)
    market = MarketConfig(region=region)
    battery = BatteryConfig()
    if fc is None:
        return f"No trained forecaster for {region}."
    preds = fc.predict_next(df.iloc[-fc.cfg.lookback :])
    res = optimise_dispatch(preds, battery, market)
    dt_h = market.interval_hours
    s = res.schedule
    charge_h = float((s.charge > 1).sum()) * dt_h
    dis_h = float((s.discharge > 1).sum()) * dt_h
    ci = np.where(s.charge.to_numpy() > 1)[0]
    di = np.where(s.discharge.to_numpy() > 1)[0]
    chg_win = f"+{ci.min()*dt_h:.1f}h..+{ci.max()*dt_h:.1f}h" if len(ci) else "none"
    dis_win = f"+{di.min()*dt_h:.1f}h..+{di.max()*dt_h:.1f}h" if len(di) else "none"
    return (
        f"{region} day-ahead plan for a {battery.power_mw:.0f} MW / {battery.energy_mwh:.0f} MWh "
        f"battery: charge ~{charge_h:.1f}h (mostly {chg_win}), discharge ~{dis_h:.1f}h "
        f"(mostly {dis_win}). Estimated arbitrage revenue ${res.revenue:,.0f} "
        f"(forecast-based estimate; actual depends on realised prices)."
    )


@tool
def search_market_docs(query: str) -> str:
    """Search the NEM/BESS knowledge base (market rules, efficiency, FCAS,
    tariffs, loss factors) and return relevant passages with their sources."""
    cites = _rag().query(query, k=3)
    if not cites:
        return "No relevant passages found."
    return "\n\n".join(
        f"[{c.source} :: {c.section}]\n{c.text}" for c in cites
    )


TOOLS = [forecast_prices, recommend_dispatch, search_market_docs]


# --- Graph ------------------------------------------------------------------

def build_agent(model: str = "gpt-4o-mini", temperature: float = 0.0):
    """Compile the assistant ⇄ tools StateGraph."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    llm = ChatOpenAI(model=model, temperature=temperature).bind_tools(TOOLS)

    def assistant(state: MessagesState):
        return {"messages": [llm.invoke(state["messages"])]}

    g = StateGraph(MessagesState)
    g.add_node("assistant", assistant)
    g.add_node("tools", ToolNode(TOOLS))
    g.add_edge(START, "assistant")
    g.add_conditional_edges("assistant", tools_condition)  # -> "tools" or END
    g.add_edge("tools", "assistant")
    return g.compile()


def ask(question: str, *, model: str = "gpt-4o-mini") -> str:
    """One-shot convenience wrapper: run the agent and return the final answer."""
    agent = build_agent(model=model)
    result = agent.invoke(
        {"messages": [SystemMessage(SYSTEM_PROMPT), HumanMessage(question)]}
    )
    return result["messages"][-1].content
