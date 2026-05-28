"""AEMO NEM price + demand loader.

Primary source: AEMO's public **Aggregated Price and Demand** monthly CSVs, the
canonical free dataset for NEM historical 30-minute RRP (Regional Reference Price)
and operational demand:

    https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_{YYYYMM}_{REGION}.csv

Columns: REGION, SETTLEMENTDATE, TOTALDEMAND, RRP, PERIODTYPE.

Fetched files are cached to ``data/raw/`` and the cleaned, concatenated frame is
written to ``data/processed/{region}.parquet``. A deterministic synthetic
generator (``synthetic_series``) backs offline development and unit tests so the
rest of the pipeline never depends on network availability.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import MarketConfig

AEMO_URL = (
    "https://aemo.com.au/aemo/data/nem/priceanddemand/"
    "PRICE_AND_DEMAND_{yyyymm}_{region}.csv"
)
REGIONS = ("NSW1", "QLD1", "SA1", "TAS1", "VIC1")

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"


def _month_range(start: str, end: str) -> list[str]:
    """Inclusive list of 'YYYYMM' strings between two 'YYYY-MM' bounds."""
    months = pd.period_range(start=start, end=end, freq="M")
    return [p.strftime("%Y%m") for p in months]


def fetch_month(yyyymm: str, region: str, *, cache: bool = True) -> pd.DataFrame:
    """Fetch one AEMO monthly price-and-demand file, with on-disk caching."""
    region = region.upper()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / f"PRICE_AND_DEMAND_{yyyymm}_{region}.csv"

    if cache and cache_path.exists():
        raw = cache_path.read_text(encoding="utf-8")
    else:
        url = AEMO_URL.format(yyyymm=yyyymm, region=region)
        resp = requests.get(url, timeout=30, headers={"User-Agent": "bess-optimiser/0.1"})
        resp.raise_for_status()
        raw = resp.text
        if cache:
            cache_path.write_text(raw, encoding="utf-8")

    df = pd.read_csv(io.StringIO(raw))
    return df


def load(
    region: str = "SA1",
    start: str = "2024-01",
    end: str = "2024-12",
    *,
    cache: bool = True,
    freq: str | None = "30min",
    market: MarketConfig | None = None,
) -> pd.DataFrame:
    """Load + clean a multi-month NEM series for one region.

    The public AEMO files are now 5-minute dispatch intervals. ``freq`` resamples
    to a coarser trading interval (mean price + mean demand) to keep MILP horizons
    tractable; pass ``freq=None`` to keep native 5-minute resolution. The returned
    frame's ``.attrs['interval_hours']`` records the resulting interval length so
    callers can build a matching :class:`MarketConfig`.

    Returns a frame indexed by timezone-naive settlement datetime (NEM time) with
    columns ``price`` ($/MWh) and ``demand`` (MW), sorted, de-duplicated, and
    price-clipped to the market band.
    """
    market = market or MarketConfig(region=region)
    frames = []
    for yyyymm in _month_range(start, end):
        frames.append(fetch_month(yyyymm, region, cache=cache))
    df = pd.concat(frames, ignore_index=True)

    df = df.rename(
        columns={"SETTLEMENTDATE": "datetime", "RRP": "price", "TOTALDEMAND": "demand"}
    )
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = (
        df[["datetime", "price", "demand"]]
        .dropna(subset=["datetime"])
        .drop_duplicates(subset="datetime")
        .sort_values("datetime")
        .set_index("datetime")
    )
    df["price"] = df["price"].clip(market.price_floor, market.price_cap)

    if freq is not None:
        df = df.resample(freq).mean().dropna(how="all")

    df.attrs["interval_hours"] = _infer_interval_hours(df.index)
    return df


def _infer_interval_hours(index: pd.DatetimeIndex) -> float:
    """Median spacing of a datetime index, in hours (robust to gaps/DST)."""
    if len(index) < 2:
        return 0.5
    deltas = pd.Series(index).diff().dropna()
    return float(deltas.median().total_seconds() / 3600.0)


def save_processed(df: pd.DataFrame, region: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / f"{region.upper()}.parquet"
    df.to_parquet(path)
    return path


def load_processed(region: str = "SA1") -> pd.DataFrame:
    path = PROCESSED_DIR / f"{region.upper()}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No cached parquet at {path}. Run scripts/fetch_data.py first."
        )
    return pd.read_parquet(path)


def synthetic_series(
    n_days: int = 60,
    *,
    interval_hours: float = 0.5,
    seed: int = 42,
    start: str = "2024-01-01",
) -> pd.DataFrame:
    """Deterministic, realistic-ish NEM-like series for offline dev + tests.

    Captures the features that matter for arbitrage: a daily double-peak demand
    shape, price roughly tracking demand with a convex markup, an evening price
    spike, occasional volatility, and the odd negative-price midday interval from
    rooftop-solar oversupply (so the optimiser learns to charge when paid to).

    This is NOT real data — it exists only so the optimiser/forecast/backtest can
    be exercised without a network round-trip. The shipped demo uses ``load()``.
    """
    rng = np.random.default_rng(seed)
    periods = int(n_days * 24 / interval_hours)
    idx = pd.date_range(start=start, periods=periods, freq=f"{int(interval_hours * 60)}min")

    hour = idx.hour + idx.minute / 60.0
    # Double-peak demand: morning ~08:00, evening ~18:30.
    morning = np.exp(-((hour - 8.0) ** 2) / (2 * 2.0**2))
    evening = np.exp(-((hour - 18.5) ** 2) / (2 * 2.5**2))
    daily = 0.45 * morning + 1.0 * evening
    weekday = idx.dayofweek < 5
    demand = 1200 + 600 * daily + np.where(weekday, 120, 0)
    demand += rng.normal(0, 40, periods)

    # Price: convex in demand + sharp evening scarcity spike + midday solar dip.
    base = 30 + 0.00008 * (demand - 1100) ** 2
    spike = 600 * np.exp(-((hour - 18.5) ** 2) / (2 * 1.0**2)) * rng.binomial(1, 0.25, periods)
    solar_dip = -40 * np.exp(-((hour - 12.5) ** 2) / (2 * 2.0**2))
    noise = rng.normal(0, 12, periods)
    price = base + spike + solar_dip + noise
    # Occasional negative-price intervals (rooftop solar oversupply).
    price = np.where(rng.binomial(1, 0.02, periods) == 1, rng.uniform(-80, -5, periods), price)
    price = np.clip(price, -1000, 17500)

    return pd.DataFrame({"price": price, "demand": demand}, index=idx).rename_axis("datetime")
