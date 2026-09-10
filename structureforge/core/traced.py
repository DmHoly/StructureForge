"""A single generic wrapper for "this parameter's value, and - optionally - how it was
reached" - the same two-level shape `Length`/`LengthDerivation` already use for a growth
thickness (see `structureforge.core.derivation`), but detached from any one parameter family
so it can be reused for anything a `Layer` might eventually carry: a growth rate, a doping
concentration, a measured roughness, and whatever else shows up later.

`Traced` never grows new fields to keep up with new *kinds* of parameter - it stays exactly
`{value, derivation}` forever. What varies is the *shape* of `derivation`:

- a resolvable formula: a typed model with its own resolve semantics (e.g. `LengthDerivation`
  - rate x duration, Arrhenius...), where `value` is whatever that formula resolved to;
- a declared-but-uncomputed set of process parameters: a plain dict recording what was used
  (e.g. a doping recipe's precursor/flow/duration), even though nothing here computes `value`
  *from* them - useful for traceability alone;
- nothing at all (`derivation=None`): the value was simply given or measured, no "how" worth
  recording.

`derivation` is deliberately typed `Any` rather than a fixed union: it can hold either shape,
or a plain dict, without `Traced` itself ever needing to change - see the three constructors
below, one per case.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class Traced(BaseModel):
    """A parameter's resolved `value`, plus an optional `derivation` explaining how it was
    reached. Use `literal`/`declared`/`computed` rather than the constructor directly - they
    name which of the three cases above you're in.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    value: Any
    derivation: Any = None

    @classmethod
    def literal(cls, value: Any) -> "Traced":
        """Just the value - nothing is known (or worth recording) about how it was reached."""
        return cls(value=value)

    @classmethod
    def declared(cls, value: Any, **obtention_params: Any) -> "Traced":
        """The value, plus the process parameters that produced it - recorded for
        traceability even though nothing here computes `value` *from* them (e.g. a doping
        concentration alongside the precursor/flow/duration used, with no solubility model
        wired up to invert one from the other).
        """
        return cls(value=value, derivation=dict(obtention_params))

    @classmethod
    def computed(cls, value: Any, derivation: Any) -> "Traced":
        """The value together with the formula that actually produced it (a model exposing
        its own `resolve_*()`, e.g. `LengthDerivation`) - by convention `value` is whatever
        that formula resolved to, not an independently-supplied number.
        """
        return cls(value=value, derivation=derivation)
