"""Invariants for the MILP dispatch optimiser.

These are economic + physical sanity checks, not numeric goldens — they pin the
behaviours a reviewer would challenge: no free money, losses respected, mutex
holds, SoC stays in band.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bess.config import BatteryConfig, MarketConfig  # noqa: E402
from src.bess.optimiser import optimise_dispatch  # noqa: E402


def test_flat_price_yields_zero_profit():
    """With a constant price there is no spread to arbitrage; degradation makes
    any cycling strictly unprofitable, so the optimiser should sit idle."""
    prices = np.full(48, 50.0)
    r = optimise_dispatch(prices, BatteryConfig(), MarketConfig())
    assert r.status == "Optimal"
    assert r.revenue == pytest.approx(0.0, abs=1.0)
    assert r.schedule.charge.sum() == pytest.approx(0.0, abs=1e-3)
    assert r.schedule.discharge.sum() == pytest.approx(0.0, abs=1e-3)


def test_no_simultaneous_charge_discharge():
    rng = np.random.default_rng(0)
    prices = rng.normal(60, 40, 96)
    r = optimise_dispatch(prices, BatteryConfig(), MarketConfig())
    overlap = np.minimum(r.schedule.charge.to_numpy(), r.schedule.discharge.to_numpy())
    assert overlap.max() == pytest.approx(0.0, abs=1e-6)


def test_soc_stays_in_band():
    rng = np.random.default_rng(1)
    prices = rng.normal(60, 40, 96)
    b = BatteryConfig()
    r = optimise_dispatch(prices, b, MarketConfig())
    soc = r.schedule.soc.to_numpy()
    assert soc.min() >= b.soc_min_mwh - 1e-6
    assert soc.max() <= b.soc_max_mwh + 1e-6


def test_round_trip_losses_respected():
    """Energy charged should exceed energy discharged by the round-trip factor
    when the battery ends where it started (terminal SoC = initial)."""
    rng = np.random.default_rng(2)
    prices = rng.normal(60, 50, 96)
    b = BatteryConfig(degradation_cost_per_mwh=0.0)
    m = MarketConfig()
    r = optimise_dispatch(prices, b, m)
    charged = (r.schedule.charge * m.interval_hours).sum()
    discharged = (r.schedule.discharge * m.interval_hours).sum()
    if charged > 1.0:  # only meaningful when it actually cycled
        assert discharged / charged == pytest.approx(b.round_trip_efficiency, rel=0.02)


def test_profit_positive_with_spread():
    """A clean day/night spread must produce positive arbitrage profit."""
    prices = np.tile(np.concatenate([np.full(24, 20.0), np.full(24, 120.0)]), 2)
    r = optimise_dispatch(prices, BatteryConfig(), MarketConfig())
    assert r.revenue > 0
