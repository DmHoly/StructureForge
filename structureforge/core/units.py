"""A minimal length quantity, so recipe/material/step specs can carry a unit without pulling in
`follow` or a general-purpose units library (StructureForge is independent from `follow` by
design - see `structureforge.adapters.follow_adapter` for the optional bridge). The geometry
engine itself always works in nanometres internally; `Length` is only the user-facing/interchange
form.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

Unit = Literal["nm", "um", "mm", "A"]

_TO_NM = {
    "A": 0.1,
    "nm": 1.0,
    "um": 1_000.0,
    "mm": 1_000_000.0,
}


class Length(BaseModel):
    """A length with an explicit unit. Immutable, like `follow.Quantity` - convert once via
    `to_nm()` rather than carrying mixed units through the simulation.
    """

    model_config = ConfigDict(frozen=True)

    value: float
    unit: Unit = "nm"

    def to_nm(self) -> float:
        return self.value * _TO_NM[self.unit]

    @classmethod
    def nm(cls, value: float) -> "Length":
        return cls(value=value, unit="nm")

    def __str__(self) -> str:
        return f"{self.value:g} {self.unit}"
