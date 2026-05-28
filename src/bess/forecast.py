"""Short-term NEM price forecaster (PyTorch).

A direct multi-horizon LSTM: given a lookback window of past price/demand plus
calendar features, predict the next ``horizon`` price intervals in one shot
(no recursive roll-out, so no compounding error).

Leakage discipline — the three places forecasts usually cheat, and how we don't:

1. **Split before anything else.** Train/val/test are contiguous time blocks
   (``time_split``); val/test are strictly *after* train. No shuffling.
2. **Scale on train statistics only.** The StandardScaler is fit on the train
   slice and applied to val/test — test-set mean/std never touch the model.
3. **Targets are strictly future.** Window ends at t; targets are prices
   [t+1 .. t+H]. Calendar features (hour, day-of-week) are genuinely known in
   advance, so feeding them is forecasting, not leakage.

Baseline: a seasonal-naive predictor (price = same interval one day ago). The
LSTM has to *beat* it on a held-out test block or we report the shortfall
honestly — a model that can't beat seasonal-naive on NEM prices is not useful.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# --- Feature engineering ----------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclical calendar features. Input needs 'price' (+ optional 'demand')."""
    out = df.copy()
    idx = out.index
    hour = idx.hour + idx.minute / 60.0
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7.0)
    if "demand" not in out.columns:
        out["demand"] = 0.0
    return out


FEATURE_COLS = ["price", "demand", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]


@dataclass
class SplitIndex:
    train: slice
    val: slice
    test: slice


def time_split(n: int, train_frac: float = 0.7, val_frac: float = 0.15) -> SplitIndex:
    """Contiguous, time-ordered split. Test is the most recent block."""
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return SplitIndex(
        train=slice(0, n_train),
        val=slice(n_train, n_train + n_val),
        test=slice(n_train + n_val, n),
    )


def make_windows(
    feats: np.ndarray,
    prices: np.ndarray,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows. X[i] = feats[i:i+L]; y[i] = prices[i+L : i+L+H]."""
    X, y = [], []
    n = len(feats)
    for i in range(n - lookback - horizon + 1):
        X.append(feats[i : i + lookback])
        y.append(prices[i + lookback : i + lookback + horizon])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


# --- Metrics ----------------------------------------------------------------

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def skill_score(y_true: np.ndarray, y_pred: np.ndarray, y_baseline: np.ndarray) -> float:
    """Forecast skill vs a reference: 1 − MAE(model)/MAE(baseline).

    >0 means the model beats the baseline; 0 ties; <0 is worse.
    """
    e_model = mae(y_true, y_pred)
    e_base = mae(y_true, y_baseline)
    return float(1.0 - e_model / e_base) if e_base > 0 else 0.0


def seasonal_naive(prices: np.ndarray, season: int, lookback: int, horizon: int) -> np.ndarray:
    """Seasonal-naive multi-horizon forecast aligned to make_windows targets.

    Prediction for target step (i+L+h) = price at (i+L+h − season). Built per
    window so it lines up 1:1 with the y from make_windows.
    """
    preds = []
    n = len(prices)
    for i in range(n - lookback - horizon + 1):
        base = []
        for h in range(horizon):
            t = i + lookback + h
            src = t - season
            base.append(prices[src] if src >= 0 else prices[t])
        preds.append(base)
    return np.asarray(preds, dtype=np.float32)


# --- Model (torch imported lazily so config/data work without it) ----------

@dataclass
class ForecastConfig:
    lookback: int = 96          # 48 h of 30-min intervals
    horizon: int = 48           # forecast next 24 h
    season: int = 48            # one day, for the seasonal-naive baseline
    hidden: int = 64
    layers: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    epochs: int = 40
    batch_size: int = 128
    patience: int = 6           # early-stopping patience on val loss
    seed: int = 42
    feature_cols: list[str] = field(default_factory=lambda: list(FEATURE_COLS))


@dataclass
class ForecastReport:
    test_mae: float
    test_rmse: float
    baseline_mae: float
    baseline_rmse: float
    skill: float
    history: list[dict]

    def summary(self) -> str:
        return (
            f"LSTM  MAE={self.test_mae:7.2f}  RMSE={self.test_rmse:7.2f}\n"
            f"naive MAE={self.baseline_mae:7.2f}  RMSE={self.baseline_rmse:7.2f}\n"
            f"skill={self.skill:+.3f}  "
            f"({'beats' if self.skill > 0 else 'loses to'} seasonal-naive)"
        )
