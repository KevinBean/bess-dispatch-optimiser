# FCAS and Revenue Stacking

Energy arbitrage is rarely a battery's only revenue stream. Grid-scale BESS
typically "stack" multiple markets to improve returns.

## Frequency Control Ancillary Services (FCAS)

AEMO procures FCAS to keep system frequency near 50 Hz. There are eight FCAS
markets, in two families:

- **Regulation** (raise/lower) — continuous correction of minor frequency
  deviations, dispatched via AGC signals.
- **Contingency** (raise/lower) across three timeframes — 6-second, 60-second, and
  5-minute — responding to sudden events such as a generator or line trip. (The
  earlier "fast/slow/delayed" naming maps onto these timeframes.)

Batteries are well suited to FCAS because they respond in milliseconds. FCAS is
priced per MW of *enablement* per interval, so a battery can earn FCAS revenue on
capacity it holds in reserve while still arbitraging energy with the rest.

## Revenue stacking

A co-optimised battery allocates its power and energy across energy arbitrage and
FCAS to maximise total revenue, subject to physical limits (it cannot simultaneously
promise the same MW to two markets). In many intervals FCAS pays more than energy
arbitrage, particularly in regions and periods with thin FCAS supply. A complete
dispatch optimiser eventually co-optimises energy + FCAS; an energy-only arbitrage
model (as in this project's core) is the foundational first layer and a lower bound
on achievable revenue.

## Other streams

- **Caps and contracts** — financial hedging products that smooth revenue.
- **Capacity / reliability mechanisms** — e.g. the Capacity Investment Scheme
  underwrites revenue for new storage, reducing merchant risk.
- **Network services** — some batteries are contracted for non-network solutions
  (deferring transmission/distribution upgrades).

For a dispatch model, the key point is that arbitrage revenue computed in isolation
understates a real battery's total return; it is the cleanest stream to model and
validate first.
