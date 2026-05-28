"""Pull a span of AEMO NEM price+demand data and cache it as Parquet.

Usage:
    python scripts/fetch_data.py --region SA1 --start 2024-01 --end 2024-12
    python scripts/fetch_data.py --region NSW1 --start 2024-06 --end 2024-12 --freq 30min
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bess import data  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", default="SA1", choices=data.REGIONS)
    ap.add_argument("--start", default="2024-01", help="YYYY-MM inclusive")
    ap.add_argument("--end", default="2024-12", help="YYYY-MM inclusive")
    ap.add_argument("--freq", default="30min", help="resample interval; 'none' for native 5min")
    args = ap.parse_args()

    freq = None if args.freq.lower() == "none" else args.freq
    print(f"Fetching {args.region} {args.start}..{args.end} (freq={freq}) ...")
    df = data.load(args.region, args.start, args.end, freq=freq)
    path = data.save_processed(df, args.region)

    print(f"  rows={len(df)}  interval_hours={df.attrs.get('interval_hours'):.4f}")
    print(f"  price  $/MWh  min={df.price.min():.1f}  mean={df.price.mean():.1f}  max={df.price.max():.1f}")
    print(f"  demand MW     min={df.demand.min():.0f}  mean={df.demand.mean():.0f}  max={df.demand.max():.0f}")
    print(f"  saved -> {path}")


if __name__ == "__main__":
    main()
