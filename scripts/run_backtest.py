"""Backtest the three dispatch policies and render the money chart.

Loads the trained forecaster + cached prices, runs the day-ahead backtest over the
test period, prints a revenue table, and saves docs/money_chart.png.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bess import data  # noqa: E402
from src.bess.backtest import backtest  # noqa: E402
from src.bess.config import BatteryConfig, MarketConfig  # noqa: E402
from src.bess.model import Forecaster  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="SA1")
    ap.add_argument("--test-start", default="2024-10-15", help="day-ahead backtest start")
    args = ap.parse_args()

    df = data.load_processed(args.region)
    fc = Forecaster().load(str(ROOT / "models" / f"forecaster_{args.region}.pt"))

    res = backtest(df, fc, test_start=args.test_start,
                   battery=BatteryConfig(), market=MarketConfig(region=args.region))

    print(f"\n=== Day-ahead backtest  {args.region}  from {args.test_start} ===")
    print(f"{'policy':<10}{'realised $':>14}{'% of perfect':>14}")
    for name in ("perfect", "forecast", "naive"):
        if name in res:
            r = res[name]
            print(f"{name:<10}{r.realised_revenue:>14,.0f}{r.captured_frac_of_perfect:>13.1%}")

    # --- money chart: cumulative realised revenue over the test period ---
    fig, ax = plt.subplots(figsize=(10, 5))
    colours = {"perfect": "#2ca02c", "forecast": "#1f77b4", "naive": "#7f7f7f"}
    for name in ("perfect", "forecast", "naive"):
        if name not in res:
            continue
        s = res[name].schedule
        dt = MarketConfig(region=args.region).interval_hours
        deg = BatteryConfig().degradation_cost_per_mwh
        per = s["price"] * s["net"] * dt - deg * (s["charge"] + s["discharge"]) * dt
        ax.plot(s.index, per.cumsum(), label=f"{name} (${res[name].realised_revenue:,.0f})",
                color=colours[name], lw=2 if name == "forecast" else 1.5)
    ax.set_title(f"Cumulative BESS arbitrage revenue — {args.region} (day-ahead, settled at actual prices)")
    ax.set_ylabel("Cumulative realised revenue ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    region = args.region.upper()
    out = ROOT / "docs" / f"money_chart_{region}.png"
    fig.savefig(out, dpi=120)
    if region == "SA1":  # keep the canonical name for the README hero image
        fig.savefig(ROOT / "docs" / "money_chart.png", dpi=120)
    # JSON sidecar so the demo can caption with the numbers without recomputing.
    import json

    summary = {
        "region": region,
        "test_start": args.test_start,
        "policies": {
            name: {
                "realised_revenue": round(res[name].realised_revenue, 2),
                "pct_of_perfect": round(res[name].captured_frac_of_perfect, 4),
            }
            for name in ("perfect", "forecast", "naive") if name in res
        },
    }
    (ROOT / "docs" / f"backtest_{region}.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved -> {out} + backtest_{region}.json")


if __name__ == "__main__":
    main()
