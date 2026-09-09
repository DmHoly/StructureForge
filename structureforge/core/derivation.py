"""How a `Length` was actually reached on the tool, not just what it came out to. A 5 nm GaN
layer is one number, but it's produced by *some* combination of a growth rate and a duration -
and the rate itself might be a flat, calibrated number or something that shifts with temperature
via an Arrhenius law. `Length.derivation` is where that explanation lives, as a small recursive
tree: a `LengthDerivation` resolves to nanometres, and can itself lean on a `RateDerivation`
(nm/s) which is its own little tree. Nesting bottoms out whenever a node is a plain literal
(`ConstantRate`) instead of one that unfolds further (`ArrheniusRate`, `MultiStageGrowth`).

None of this is required - a `Length` can always just carry a literal `value`/`unit` instead, the
same as before this module existed. `derivation` is for when *how* a parameter was reached is
worth recording alongside *what* it came out to.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

_BOLTZMANN_EV_PER_K = 8.617333262e-5


class ConstantRate(BaseModel):
    """A flat, calibrated rate - the recursion's base case: nothing further explains it."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["constant_rate"] = "constant_rate"
    nm_per_s: float

    def resolve_nm_per_s(self) -> float:
        return self.nm_per_s


class ArrheniusRate(BaseModel):
    """Temperature-activated rate: ``rate = prefactor * exp(-Ea / (kB * T))``, the standard
    model for how a growth or etch rate responds to temperature. Doubling `prefactor_nm_per_s`
    doubles the rate at any temperature; raising `activation_energy_eV` makes the rate more
    sensitive to temperature swings.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["arrhenius_rate"] = "arrhenius_rate"
    prefactor_nm_per_s: float
    activation_energy_eV: float
    temperature_K: float

    def resolve_nm_per_s(self) -> float:
        return self.prefactor_nm_per_s * math.exp(
            -self.activation_energy_eV / (_BOLTZMANN_EV_PER_K * self.temperature_K)
        )


RateDerivation = Annotated[Union[ConstantRate, ArrheniusRate], Field(discriminator="kind")]


class GrowthAtRate(BaseModel):
    """``thickness = rate * duration``. `rate` is itself a `RateDerivation`, which is what makes
    this recursive: it can bottom out immediately (`ConstantRate`) or unfold through a further
    sub-process (`ArrheniusRate`, and later additions).
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["growth_at_rate"] = "growth_at_rate"
    rate: RateDerivation
    duration_s: float

    def resolve_nm(self) -> float:
        return self.rate.resolve_nm_per_s() * self.duration_s


class MultiStageGrowth(BaseModel):
    """Several growth stages back to back (e.g. a temperature ramp approximated as discrete
    holds) - the total thickness is the sum of each stage's own resolved thickness. Each stage is
    itself a full `LengthDerivation`, so this is where the tree can actually branch rather than
    just chain.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["multi_stage_growth"] = "multi_stage_growth"
    stages: list["LengthDerivation"]

    def resolve_nm(self) -> float:
        return sum(stage.resolve_nm() for stage in self.stages)


LengthDerivation = Annotated[Union[GrowthAtRate, MultiStageGrowth], Field(discriminator="kind")]

MultiStageGrowth.model_rebuild()
