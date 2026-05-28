# BESS Energy Arbitrage Economics

Energy arbitrage is the simplest battery revenue stream: charge when the spot
price is low, discharge when it is high, and capture the spread net of losses and
wear.

## Round-trip efficiency

A battery does not return all the energy it stores. **Round-trip efficiency (RTE)**
is the ratio of energy discharged to energy charged, typically 85-92% for modern
lithium-ion grid systems. It can be modelled as a charge efficiency η_c and a
discharge efficiency η_d, with RTE = η_c × η_d. Losses mean the price spread must
exceed roughly 1/RTE before a charge-discharge cycle is profitable: at 86% RTE you
need the sell price to be at least ~1.16× the buy price just to break even on
energy, before degradation.

## State of charge limits

Operators rarely cycle 0-100%. A usable SoC window (e.g. 5-95%) protects cell
health and preserves headroom. The usable energy capacity is the nameplate energy
times this window.

## Power and energy ratings

A battery is described by its power rating (MW) and energy rating (MWh). The ratio
energy/power is the **duration** (hours at full power). A "2-hour" 100 MW / 200 MWh
battery can discharge at 100 MW for two hours. Duration determines how much of a
daily price shape the battery can exploit: a 1-hour battery captures only the
sharpest peak, while a 4-hour battery can shift bulk energy across the evening.

## Degradation and cycling cost

Every cycle ages the cells. A practical model charges a marginal degradation cost
($/MWh of throughput) so the optimiser only cycles when the captured spread covers
both losses and wear. Without a degradation term, an optimiser will chase
vanishingly small spreads and over-cycle. Typical assumed values are a few dollars
per MWh, derived from cell replacement cost divided by lifetime throughput.

## Why forecasting matters

Optimal arbitrage requires knowing future prices. With perfect foresight the
optimiser finds the global best schedule (an upper bound on revenue). In practice
the operator commits a day-ahead schedule against a *forecast*, then is settled at
*actual* prices. The revenue gap between perfect-foresight and forecast-driven
dispatch is the economic cost of forecast error, and it is the headline metric for
evaluating a price model in this context — more informative than raw MAE/RMSE.
