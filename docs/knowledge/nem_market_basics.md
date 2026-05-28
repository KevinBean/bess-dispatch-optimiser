# NEM Market Basics

The National Electricity Market (NEM) is the wholesale electricity market covering
eastern and southern Australia: Queensland, New South Wales (incl. ACT), Victoria,
South Australia, and Tasmania. It is operated by the Australian Energy Market
Operator (AEMO).

## Dispatch and pricing

The NEM is a gross pool, energy-only market. Generators submit offers and AEMO
dispatches the cheapest combination to meet demand every 5 minutes. Since the
5-Minute Settlement reform (commenced 1 October 2021), both dispatch and financial
settlement occur on a 5-minute basis (previously dispatch was 5-minute but
settlement averaged over 30-minute trading intervals).

The **Regional Reference Price (RRP)** is the spot price ($/MWh) at each region's
Regional Reference Node. All energy in a region settles at that region's RRP for
the interval, regardless of which generator was dispatched.

## Price bands

Spot prices are bounded by AEMO-administered limits:

- **Market Price Cap (MPC)** — the maximum spot price. For FY2024-25 it is
  approximately $17,500/MWh (indexed annually).
- **Market Price Floor (MPF)** — the minimum spot price, set at -$1,000/MWh.

Negative prices occur when there is more must-run or zero-marginal-cost generation
(notably rooftop and utility solar around the middle of the day) than demand can
absorb; generators may pay to keep running rather than shut down and restart.

## Why prices are volatile

Because the NEM is energy-only with no capacity market, scarcity is priced through
high spot prices. Evening peaks (after solar sets, while demand is still high)
routinely produce price spikes, while sunny low-demand middays can drive prices to
zero or negative. This intraday spread is precisely what a battery monetises
through arbitrage.

## Public data

AEMO publishes aggregated price and demand data per region as monthly CSV files
(columns: REGION, SETTLEMENTDATE, TOTALDEMAND, RRP, PERIODTYPE). TOTALDEMAND is
operational demand in MW; RRP is the spot price in $/MWh.
