"""PyTorch LSTM trainer for the price forecaster.

Separated from ``forecast.py`` (feature engineering / splits / metrics) so that
the data + optimiser pipeline imports without torch present. This module owns the
``nn.Module`` and the train loop with early stopping.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .forecast import (
    ForecastConfig,
    ForecastReport,
    build_features,
    make_windows,
    mae,
    rmse,
    seasonal_naive,
    skill_score,
    time_split,
)


class PriceLSTM(nn.Module):
    """Encoder LSTM → dense head emitting all ``horizon`` steps at once."""

    def __init__(self, n_features: int, cfg: ForecastConfig):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=cfg.hidden,
            num_layers=cfg.layers,
            dropout=cfg.dropout if cfg.layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)          # (B, L, H)
        last = out[:, -1, :]           # final timestep encoding
        return self.head(last)         # (B, horizon)


class Forecaster:
    """Fit/predict wrapper holding the trained model + the train-fit scalers.

    The price scaler is kept separately so predictions can be inverse-transformed
    back to $/MWh, and so the optimiser receives real prices, not z-scores.
    """

    def __init__(self, cfg: ForecastConfig | None = None):
        self.cfg = cfg or ForecastConfig()
        self.feat_scaler = StandardScaler()
        self.price_scaler = StandardScaler()
        self.model: PriceLSTM | None = None

    # -- internal helpers ---------------------------------------------------

    def _frame_to_arrays(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        feats = build_features(df)
        X = feats[self.cfg.feature_cols].to_numpy(dtype=np.float32)
        price = feats["price"].to_numpy(dtype=np.float32)
        return X, price

    # -- public API ---------------------------------------------------------

    def fit(self, df: pd.DataFrame, *, verbose: bool = True) -> ForecastReport:
        torch.manual_seed(self.cfg.seed)
        np.random.seed(self.cfg.seed)

        feats, price = self._frame_to_arrays(df)
        n = len(feats)
        sp = time_split(n, 0.7, 0.15)

        # Scalers fit on TRAIN ONLY — the load-bearing anti-leakage step.
        self.feat_scaler.fit(feats[sp.train])
        self.price_scaler.fit(price[sp.train].reshape(-1, 1))
        feats_s = self.feat_scaler.transform(feats).astype(np.float32)
        price_s = self.price_scaler.transform(price.reshape(-1, 1)).ravel().astype(np.float32)

        L, H = self.cfg.lookback, self.cfg.horizon
        # Build windows per split on the scaled features but real-price targets in
        # z-space; we invert to $/MWh only at evaluation time.
        def windows(s: slice):
            return make_windows(feats_s[s], price_s[s], L, H)

        Xtr, ytr = windows(sp.train)
        Xva, yva = windows(sp.val)
        Xte, yte = windows(sp.test)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = PriceLSTM(len(self.cfg.feature_cols), self.cfg).to(device)
        opt = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
        loss_fn = nn.SmoothL1Loss()  # robust to NEM price spikes vs plain MSE

        tr_loader = DataLoader(
            TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
            batch_size=self.cfg.batch_size,
            shuffle=True,
        )
        Xva_t = torch.from_numpy(Xva).to(device)
        yva_t = torch.from_numpy(yva).to(device)

        best_val = float("inf")
        best_state = None
        bad = 0
        history: list[dict] = []
        for epoch in range(self.cfg.epochs):
            self.model.train()
            tr_loss = 0.0
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()
                tr_loss += loss.item() * len(xb)
            tr_loss /= len(tr_loader.dataset)

            self.model.eval()
            with torch.no_grad():
                val_loss = loss_fn(self.model(Xva_t), yva_t).item()
            history.append({"epoch": epoch, "train": tr_loss, "val": val_loss})
            if verbose:
                print(f"  epoch {epoch:3d}  train={tr_loss:.4f}  val={val_loss:.4f}")

            if val_loss < best_val - 1e-5:
                best_val, best_state, bad = val_loss, {
                    k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()
                }, 0
            else:
                bad += 1
                if bad >= self.cfg.patience:
                    if verbose:
                        print(f"  early stop at epoch {epoch}")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)

        # --- Test-set evaluation in real $/MWh ---
        self.model.eval()
        with torch.no_grad():
            pred_s = self.model(torch.from_numpy(Xte).to(device)).cpu().numpy()
        y_pred = self.price_scaler.inverse_transform(pred_s)
        y_true = self.price_scaler.inverse_transform(yte)

        # Baseline windows over the *test* prices (in real $/MWh).
        price_test = price[sp.test]
        y_base = seasonal_naive(price_test, self.cfg.season, L, H)

        report = ForecastReport(
            test_mae=mae(y_true, y_pred),
            test_rmse=rmse(y_true, y_pred),
            baseline_mae=mae(y_true, y_base),
            baseline_rmse=rmse(y_true, y_base),
            skill=skill_score(y_true, y_pred, y_base),
            history=history,
        )
        return report

    def predict_next(self, recent: pd.DataFrame) -> np.ndarray:
        """Forecast the next ``horizon`` prices ($/MWh) given the latest window.

        ``recent`` must contain at least ``lookback`` rows ending at 'now'.
        """
        assert self.model is not None, "call fit() first"
        feats, _ = self._frame_to_arrays(recent)
        window = feats[-self.cfg.lookback :]
        if len(window) < self.cfg.lookback:
            raise ValueError(f"need >= {self.cfg.lookback} rows, got {len(window)}")
        x = self.feat_scaler.transform(window).astype(np.float32)[None, ...]
        self.model.eval()
        with torch.no_grad():
            pred_s = self.model(torch.from_numpy(x)).cpu().numpy()
        return self.price_scaler.inverse_transform(pred_s).ravel()

    def save(self, path: str) -> None:
        assert self.model is not None
        torch.save(
            {
                "cfg": self.cfg.__dict__,
                "model": self.model.state_dict(),
                "feat_scaler": self.feat_scaler,
                "price_scaler": self.price_scaler,
            },
            path,
        )

    def load(self, path: str) -> "Forecaster":
        ckpt = torch.load(path, weights_only=False)
        self.cfg = ForecastConfig(**ckpt["cfg"])
        self.feat_scaler = ckpt["feat_scaler"]
        self.price_scaler = ckpt["price_scaler"]
        self.model = PriceLSTM(len(self.cfg.feature_cols), self.cfg)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        return self
