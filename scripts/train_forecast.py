"""Train the PyTorch price forecaster on cached NEM data and report test skill."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bess import data  # noqa: E402
from src.bess.forecast import ForecastConfig  # noqa: E402
from src.bess.model import Forecaster  # noqa: E402

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="SA1")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lookback", type=int, default=96)
    ap.add_argument("--horizon", type=int, default=48)
    args = ap.parse_args()

    df = data.load_processed(args.region)
    print(f"loaded {len(df)} intervals for {args.region}")

    cfg = ForecastConfig(epochs=args.epochs, lookback=args.lookback, horizon=args.horizon)
    fc = Forecaster(cfg)
    report = fc.fit(df, verbose=True)

    print("\n=== TEST REPORT ===")
    print(report.summary())

    MODELS_DIR.mkdir(exist_ok=True)
    out = MODELS_DIR / f"forecaster_{args.region}.pt"
    fc.save(str(out))
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
