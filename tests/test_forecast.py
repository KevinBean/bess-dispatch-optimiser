"""Fast checks on the leakage-relevant forecasting logic (no training)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bess.forecast import (  # noqa: E402
    make_windows,
    seasonal_naive,
    skill_score,
    time_split,
)


def test_time_split_is_ordered_and_disjoint():
    sp = time_split(1000, 0.7, 0.15)
    assert sp.train == slice(0, 700)
    assert sp.val == slice(700, 850)
    assert sp.test == slice(850, 1000)
    # strictly increasing, no overlap
    assert sp.train.stop == sp.val.start and sp.val.stop == sp.test.start


def test_make_windows_targets_are_strictly_future():
    feats = np.arange(100).reshape(-1, 1).astype(np.float32)
    prices = np.arange(100).astype(np.float32)
    L, H = 10, 5
    X, y = make_windows(feats, prices, L, H)
    # first target must be the value immediately after the first window
    assert y[0, 0] == prices[L]
    # window never includes its own targets
    assert X[0].max() < y[0].min()
    assert len(X) == len(y) == 100 - L - H + 1


def test_seasonal_naive_aligns_with_windows():
    prices = np.arange(200).astype(np.float32)
    L, H, season = 48, 24, 48
    _, y = make_windows(prices.reshape(-1, 1), prices, L, H)
    base = seasonal_naive(prices, season, L, H)
    assert base.shape == y.shape
    # on a perfectly periodic ramp shifted by season, naive = truth - season
    assert np.allclose(base[0], y[0] - season)


def test_skill_score_sign():
    truth = np.array([10.0, 20.0, 30.0])
    good = truth + 1
    bad = truth + 50
    baseline = truth + 10
    assert skill_score(truth, good, baseline) > 0   # beats baseline
    assert skill_score(truth, bad, baseline) < 0    # worse than baseline
