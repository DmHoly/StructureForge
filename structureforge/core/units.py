"""A minimal length quantity, so recipe/material/step specs can carry a unit without pulling in
`follow` or a general-purpose units library (StructureForge is independent from `follow` by
design - see `structureforge.adapters.follow_adapter` for the optional bridge). The geometry
engine itself always works in nanometres internally; `Length` is only the user-facing/interchange
form.

A `Length` can also carry a `derivation` instead of a literal `value` - see
`structureforge.core.derivation` for how a thickness like "5 nm of GaN" unfolds into *how* it was
grown (a rate and a duration, the rate itself possibly temperature-dependent).
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from .derivation import LengthDerivation

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

    Exactly one of `value` or `derivation` is set: either a plain literal (the common case), or a
    `LengthDerivation` tree explaining how the length was reached (see
    `structureforge.core.derivation`). `to_nm()` doesn't care which - it resolves either form to
    the same flat number for the geometry engine.
    """

    model_config = ConfigDict(frozen=True)

    value: float | None = None
    unit: Unit = "nm"
    derivation: LengthDerivation | None = None

    @model_validator(mode="after")
    def _check_exactly_one_source(self) -> "Length":
        if (self.value is None) == (self.derivation is None):
            raise ValueError("Length needs exactly one of value or derivation")
        return self

    def to_nm(self) -> float:
        if self.derivation is not None:
            return self.derivation.resolve_nm()
        return self.value * _TO_NM[self.unit]

    @classmethod
    def nm(cls, value: float) -> "Length":
        return cls(value=value, unit="nm")

    @classmethod
    def derived(cls, derivation: LengthDerivation) -> "Length":
        """A length whose value comes entirely from `derivation` - see
        `structureforge.core.derivation` for the available process trees.
        """
        return cls(derivation=derivation)

    def __str__(self) -> str:
        if self.derivation is not None:
            return f"{self.to_nm():g} nm (derived)"
        return f"{self.value:g} {self.unit}"
