"""BESS Dispatch Optimiser & Advisor.

A grid-scale battery dispatch toolkit built around real NEM (National Electricity
Market) data:

- ``data``      — pull/clean AEMO price + demand time-series
- ``config``    — battery + market parameters
- ``optimiser`` — MILP arbitrage dispatch (PuLP + HiGHS)
- ``forecast``  — PyTorch short-term price forecaster
- ``backtest``  — receding-horizon evaluation (perfect vs forecast vs naive)
"""

__version__ = "0.1.0"
