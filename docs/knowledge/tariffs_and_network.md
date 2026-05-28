# Tariffs, Loss Factors and Network Charges

The spot price (RRP) is not the whole story of what a battery earns or pays per
MWh. Several adjustments sit between the spot price and the cash that reaches the
asset owner.

## Marginal Loss Factors (MLF)

Electricity is lost as heat in transmission. AEMO publishes annual **Marginal Loss
Factors** for each connection point — a multiplier (typically 0.9-1.05) applied to
a generator's output to account for its electrical distance from the Regional
Reference Node. A battery at a location with MLF 0.95 effectively sells at 0.95 ×
RRP and buys adjusted similarly. MLFs change yearly and materially affect project
economics; a declining MLF (from new nearby generation) is a known merchant risk.

## Transmission Use of System (TUOS) and connection

Network charges (TUOS/DUOS) and connection costs apply depending on the connection
voltage and whether the asset is transmission- or distribution-connected. Storage
has an unusual position because it both consumes and produces; the treatment of
storage under network tariffs has been the subject of ongoing rule changes.

## Settlement vs offer

A battery participates as both a scheduled load (when charging) and a scheduled
generator (when discharging). It is settled at RRP for energy in each direction.
Modelling arbitrage on raw RRP — as this project's core optimiser does — gives the
gross energy margin; subtracting MLF effects, network charges, and degradation
gives the net merchant margin.

## Practical modelling guidance

For a first-pass arbitrage model, using raw RRP and a degradation cost captures the
dominant economics. Refinements, in rough order of materiality:

1. Apply the connection point's MLF to both charge and discharge legs.
2. Add a marginal degradation/throughput cost (already in the core model).
3. Net off any fixed network charges (these don't change the dispatch decision,
   only the bottom line).
4. Co-optimise with FCAS for total revenue (changes the dispatch itself).
