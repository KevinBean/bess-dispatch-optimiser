"""Battery + market parameters.

One dataclass for the physical battery, one for market/settlement conventions.
Kept dependency-free (pydantic only) so it imports without torch/pulp present.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class BatteryConfig(BaseModel):
    """Physical + economic parameters for a grid-scale BESS.

    Defaults model a representative ~2-hour utility battery (100 MW / 200 MWh),
    the same scale as the Enerven-class projects this repo is benchmarked on.
    """

    power_mw: float = Field(100.0, gt=0, description="Max charge/discharge power (MW).")
    energy_mwh: float = Field(200.0, gt=0, description="Usable energy capacity (MWh).")

    # Round-trip efficiency split symmetrically into charge/discharge legs.
    eta_charge: float = Field(0.93, gt=0, le=1, description="Charge efficiency.")
    eta_discharge: float = Field(0.93, gt=0, le=1, description="Discharge efficiency.")

    soc_min_frac: float = Field(0.05, ge=0, lt=1, description="Min state-of-charge fraction.")
    soc_max_frac: float = Field(0.95, gt=0, le=1, description="Max state-of-charge fraction.")
    soc_init_frac: float = Field(0.50, ge=0, le=1, description="Initial SoC fraction.")

    # Marginal degradation cost charged per MWh of throughput ($/MWh). Models the
    # economic penalty of cycling; set to 0 to disable. A non-zero value stops the
    # optimiser from chasing tiny spreads that don't cover wear.
    degradation_cost_per_mwh: float = Field(2.0, ge=0)

    @property
    def round_trip_efficiency(self) -> float:
        return self.eta_charge * self.eta_discharge

    @property
    def soc_min_mwh(self) -> float:
        return self.soc_min_frac * self.energy_mwh

    @property
    def soc_max_mwh(self) -> float:
        return self.soc_max_frac * self.energy_mwh

    @property
    def soc_init_mwh(self) -> float:
        return self.soc_init_frac * self.energy_mwh


class MarketConfig(BaseModel):
    """NEM settlement conventions.

    The NEM settles on 5-minute dispatch but the public Price-and-Demand files are
    30-minute trading intervals — so the default interval is 0.5 h.
    """

    interval_hours: float = Field(0.5, gt=0, description="Settlement interval length (h).")
    region: str = Field("SA1", description="NEM region id (NSW1/QLD1/SA1/TAS1/VIC1).")

    # Market Price Cap / Floor (AEMO administered, FY25 values, $/MWh). Prices are
    # clipped to this band on ingest to keep the optimiser numerically sane.
    price_cap: float = Field(17_500.0)
    price_floor: float = Field(-1_000.0)
