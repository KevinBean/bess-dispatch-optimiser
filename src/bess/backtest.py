"""Receding-horizon (day-ahead) backtest — the honesty engine.

Frames the problem the way a real BESS operator faces it: each day you commit a
dispatch *before* you know the prices, using a forecast; you then get *settled at
the actual prices*. We compare three operating policies over the same test period:

    perfect   — optimise each day against the realised prices   (upper bound)
    forecast  — optimise each day against the LSTM forecast      (what you'd run)
    naive     — optimise each day against seasonal-naive prices  (cheap baseline)

All three settle revenue at ACTUAL prices. The gap between ``forecast`` and
``perfect`` is the cost of forecast error; the gap between ``forecast`` and
``naive`` is the value the LSTM adds over doing nothing clever. Reporting both
keeps the project honest — a forecast that barely beats naive is a finding, not
a failure to hide.

Each day is solved independently with terminal SoC pinned back to the initial
SoC (standard day-ahead bidding), so days don't leak value into each other.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import BatteryConfig, MarketConfig
from .forecast import build_features
from .optimiser import optimise_dispatch


@dataclass
class PolicyResult:
    name: str
    realised_revenue: float        # settled at ACTUAL prices ($)
    schedule: pd.DataFrame         # datetime index; price(actual), charge, discharge, net
    captured_frac_of_perfect: float = 0.0


def _settle(schedule_net: np.ndarray, charge: np.ndarray, discharge: np.ndarray,
            actual_prices: np.ndarray, battery: BatteryConfig, dt: float) -> float:
    return float(
        np.sum(actual_prices * schedule_net * dt)
        - battery.degradation_cost_per_mwh * np.sum((charge + discharge) * dt)
    )


def backtest(
    df: pd.DataFrame,
    forecaster,                       # bess.model.Forecaster (or None to skip 'forecast')
    *,
    test_start: str | pd.Timestamp,
    battery: BatteryConfig | None = None,
    market: MarketConfig | None = None,
    plan_hours: float = 24.0,
) -> dict[str, PolicyResult]:
    """Run the three policies over df[test_start:], returning a result per policy."""
    battery = battery or BatteryConfig()
    market = market or MarketConfig()
    dt = market.interval_hours
    K = int(round(plan_hours / dt))            # intervals per day-ahead block

    df = build_features(df)
    test_start = pd.Timestamp(test_start)
    test_pos = df.index.searchsorted(test_start)

    block_starts = list(range(test_pos, len(df) - K + 1, K))
    soc0 = battery.soc_init_mwh

    rows = {k: [] for k in ("perfect", "forecast", "naive")}
    rev = {k: 0.0 for k in rows}

    season = getattr(getattr(forecaster, "cfg", None), "season", 48)

    for bs in block_starts:
        sl = slice(bs, bs + K)
        actual = df["price"].to_numpy()[sl]
        idx = df.index[sl]

        plans: dict[str, np.ndarray] = {"perfect": actual}

        # seasonal-naive plan: prices one season (day) earlier
        naive = df["price"].to_numpy()[bs - season : bs - season + K]
        plans["naive"] = naive if len(naive) == K else actual

        # LSTM forecast plan (needs lookback history ending at block start)
        if forecaster is not None and getattr(forecaster, "model", None) is not None:
            L = forecaster.cfg.lookback
            if bs >= L and forecaster.cfg.horizon >= K:
                hist = df.iloc[bs - L : bs]
                fc = forecaster.predict_next(hist)[:K]
                plans["forecast"] = fc

        for name, planned in plans.items():
            res = optimise_dispatch(
                planned, battery, market,
                soc_init_mwh=soc0, terminal_soc_mwh=soc0, index=idx,
            )
            charge = res.schedule.charge.to_numpy()
            discharge = res.schedule.discharge.to_numpy()
            net = discharge - charge
            realised = _settle(net, charge, discharge, actual, battery, dt)
            rev[name] += realised
            rows[name].append(
                pd.DataFrame({"price": actual, "charge": charge,
                              "discharge": discharge, "net": net}, index=idx)
            )

    out: dict[str, PolicyResult] = {}
    perfect_rev = rev["perfect"] or 1.0
    for name in rows:
        if not rows[name]:
            continue
        sched = pd.concat(rows[name]).rename_axis("datetime")
        out[name] = PolicyResult(
            name=name,
            realised_revenue=rev[name],
            schedule=sched,
            captured_frac_of_perfect=rev[name] / perfect_rev,
        )
    return out
