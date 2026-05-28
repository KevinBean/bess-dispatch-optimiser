"""MILP battery dispatch optimiser (PuLP + HiGHS).

Formulation
-----------
Decision variables per settlement interval ``t`` (Δt hours each):

    charge_t     >= 0   grid-side charge power (MW), metered/settled
    discharge_t  >= 0   grid-side discharge power (MW), metered/settled
    u_t          ∈ {0,1} charge/discharge mutex flag  ← the integer part (MILP)
    soc_t        >= 0   state of charge at end of interval t (MWh)

Objective — maximise arbitrage profit:

    max  Σ_t  price_t · (discharge_t − charge_t) · Δt
             − c_deg · (charge_t + discharge_t) · Δt

Constraints:

    soc_t = soc_{t-1} + η_c·charge_t·Δt − (1/η_d)·discharge_t·Δt   (SoC dynamics, losses)
    soc_min ≤ soc_t ≤ soc_max
    0 ≤ charge_t    ≤ P · u_t                                       (mutex via Big-M = P)
    0 ≤ discharge_t ≤ P · (1 − u_t)
    soc_0  given;  soc_T ≥ terminal_soc (optional, fair receding-horizon rollover)

Efficiency placement: you *store* η_c of every MWh bought and must *withdraw*
1/η_d from storage per MWh sold, so round-trip loss = η_c·η_d shows up correctly
in the SoC balance rather than smeared into the price.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pulp

from .config import BatteryConfig, MarketConfig


@dataclass
class DispatchResult:
    schedule: pd.DataFrame  # index=datetime; cols: price, charge, discharge, soc, net
    revenue: float          # total arbitrage profit ($) net of degradation
    status: str             # solver status, e.g. "Optimal"
    solver: str             # which backend actually solved it


def _select_solver(msg: bool = False) -> tuple[pulp.LpSolver, str]:
    """Prefer HiGHS (modern open solver); fall back to bundled CBC."""
    try:
        solver = pulp.HiGHS(msg=msg)
        if solver.available():
            return solver, "HiGHS"
    except Exception:
        pass
    return pulp.PULP_CBC_CMD(msg=msg), "CBC"


def optimise_dispatch(
    prices: np.ndarray | pd.Series,
    battery: BatteryConfig | None = None,
    market: MarketConfig | None = None,
    *,
    soc_init_mwh: float | None = None,
    terminal_soc_mwh: float | None = None,
    index: pd.DatetimeIndex | None = None,
    msg: bool = False,
) -> DispatchResult:
    """Solve the perfect-foresight arbitrage MILP for a price horizon.

    ``prices`` are $/MWh per interval. ``terminal_soc_mwh`` defaults to the
    initial SoC so the battery can't cheat by selling its starting energy and
    leaving the horizon empty — essential for an honest receding-horizon backtest.
    """
    battery = battery or BatteryConfig()
    market = market or MarketConfig()
    dt = market.interval_hours

    if isinstance(prices, pd.Series):
        index = index if index is not None else prices.index
        prices = prices.to_numpy(dtype=float)
    prices = np.asarray(prices, dtype=float)
    T = len(prices)
    if T == 0:
        raise ValueError("prices is empty")

    soc0 = battery.soc_init_mwh if soc_init_mwh is None else soc_init_mwh
    soc_T_min = soc0 if terminal_soc_mwh is None else terminal_soc_mwh
    P = battery.power_mw
    c_deg = battery.degradation_cost_per_mwh

    prob = pulp.LpProblem("bess_arbitrage", pulp.LpMaximize)

    charge = [pulp.LpVariable(f"c_{t}", lowBound=0, upBound=P) for t in range(T)]
    discharge = [pulp.LpVariable(f"d_{t}", lowBound=0, upBound=P) for t in range(T)]
    u = [pulp.LpVariable(f"u_{t}", cat="Binary") for t in range(T)]
    soc = [
        pulp.LpVariable(f"soc_{t}", lowBound=battery.soc_min_mwh, upBound=battery.soc_max_mwh)
        for t in range(T)
    ]

    # Objective.
    prob += pulp.lpSum(
        prices[t] * (discharge[t] - charge[t]) * dt
        - c_deg * (charge[t] + discharge[t]) * dt
        for t in range(T)
    )

    # SoC dynamics + mutex.
    for t in range(T):
        prev = soc0 if t == 0 else soc[t - 1]
        prob += soc[t] == prev + battery.eta_charge * charge[t] * dt - (
            1.0 / battery.eta_discharge
        ) * discharge[t] * dt
        prob += charge[t] <= P * u[t]
        prob += discharge[t] <= P * (1 - u[t])

    prob += soc[T - 1] >= soc_T_min

    solver, solver_name = _select_solver(msg=msg)
    prob.solve(solver)

    c = np.array([charge[t].value() or 0.0 for t in range(T)])
    d = np.array([discharge[t].value() or 0.0 for t in range(T)])
    s = np.array([soc[t].value() or 0.0 for t in range(T)])
    revenue = float(np.sum(prices * (d - c) * dt - c_deg * (c + d) * dt))

    sched = pd.DataFrame(
        {"price": prices, "charge": c, "discharge": d, "soc": s, "net": d - c}
    )
    if index is not None and len(index) == T:
        sched.index = index
        sched = sched.rename_axis("datetime")

    return DispatchResult(
        schedule=sched,
        revenue=revenue,
        status=pulp.LpStatus[prob.status],
        solver=solver_name,
    )
