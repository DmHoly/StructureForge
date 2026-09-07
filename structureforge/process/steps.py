"""The elementary process bricks. Each step names a material/recipe from the libraries by string
key (resolved at simulate() time, not at construction time) so a step list is a plain, JSON-
serializable recipe independent of which library instance ends up interpreting it - the same
spirit as `follow.Structure` staying agnostic of anything outside its own fields.

`ChemicalStep` is the one kind that never touches the geometry: it exists purely so a cleaning,
anneal, or surface treatment shows up in the process history (and, via the optional follow
adapter, in the resulting experiment's protocol) even though it leaves no visible trace on the
cross-section.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..core.units import Length


class GrowthOrientation(str, Enum):
    """Crystal orientation governing how the epitaxial film grows in 2D cross-section.

    c_plane   — [0001], growth strictly upward (the common case for c-plane GaN/AlGaN/InGaN).
    m_plane   — {10-10}, lateral growth on vertical sidewalls (core-shell / ELO regrowth).
    semi_polar — tilted facet; the exact angle is carried by `EpitaxialGrowth.angle_deg`
                 (e.g. 28° for {10-11}, 32° for {11-22}, measured from the c-axis = from vertical).
    """

    c_plane = "c_plane"
    m_plane = "m_plane"
    semi_polar = "semi_polar"


class Deposition(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["deposition"] = "deposition"
    name: str
    material: str
    recipe: str
    thickness: Length


class Etch(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["etch"] = "etch"
    name: str
    recipe: str
    depth: Length


class Planarization(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["planarization"] = "planarization"
    name: str
    target_level: Length | None = None
    stop_material: str | None = None

    @model_validator(mode="after")
    def _check_exactly_one_target(self) -> "Planarization":
        if (self.target_level is None) == (self.stop_material is None):
            raise ValueError("planarization needs exactly one of target_level or stop_material")
        return self


class ChemicalStep(BaseModel):
    """A step with no geometric effect (cleaning, anneal, surface treatment...) - recorded for
    traceability only.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["chemical"] = "chemical"
    name: str
    description: str | None = None
    parameters: dict[str, Length] = Field(default_factory=dict)


class Lithography(BaseModel):
    """Deposit a patterned resist: a blanket conformal coat of `resist_material`, kept only in
    the given `openings` (x-ranges, in nm, where the mask is *open* - i.e. resist ends up
    everywhere else). Exposure/development physics aren't modelled; the openings are the
    already-developed pattern.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["lithography"] = "lithography"
    name: str
    resist_material: str
    thickness: Length
    openings: list[tuple[float, float]]

    @model_validator(mode="after")
    def _check_openings(self) -> "Lithography":
        for x0, x1 in self.openings:
            if x0 >= x1:
                raise ValueError(f"opening ({x0}, {x1}) is not a valid x-range: start must be < end")
        return self


class ResistStrip(BaseModel):
    kind: Literal["resist_strip"] = "resist_strip"
    name: str
    material: str = "Photoresist"


class Flip(BaseModel):
    """Turn the wafer over to process its backside (thinning, through-substrate vias, backside
    contacts): the current front is bonded face-down to a temporary carrier, and what was the
    wafer's untouched bulk floor becomes the new front, ready for ordinary process steps to
    continue on. See `structureforge.geometry.engine.Geometry.flip` for why this requires the
    current front to be flat across the whole domain width first (a real temporary bond needs a
    flat surface to adhere to).
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["flip"] = "flip"
    name: str


class FacetedGrowth(BaseModel):
    """Multi-facet epitaxial growth driven by relative per-plane growth rates.

    Three families of crystal planes each advance strictly along their own outward normal, at
    their own rate, independently of the other two:

    - c-plane {0001}    → rate_c  (grows strictly upward)
    - m-plane {10-10}   → rate_m  (lateral sidewalls)
    - semi-polar facets → rate_sp (inclined planes, e.g. {10-11} or {11-22};
                                    angle from c-axis = semi_polar_angle_deg)

    `thickness` is the nominal growth increment for the c-plane (rate_c * thickness nm of
    material are added above flat c-plane surfaces); m-plane and semi-polar faces advance by
    rate_m * thickness and rate_sp * thickness respectively. A rate of 0 pins that facet's own
    line - it never moves, however fast the other two are growing (see
    `Geometry.deposit_faceted`'s docstring for exactly how a corner behaves when only some
    facets are active, and the rate_c vs. rate_sp*cos(angle) relation that decides whether a
    c-plane facet flanked by semi-polar ones widens or closes at each step).

    Repeatedly applying this step (one thin layer per MQW period) wraps each layer conformally
    around the shape as it develops - `rate_c` vs. `rate_sp`/`rate_m` decide whether that shape
    stays a flat-topped pencil or narrows into a sharp point.

    `material_c`/`material_m`/`material_sp` let each facet family incorporate a different
    material - typically the same alloy at a different composition, matching real facet-
    dependent incorporation (e.g. more indium on the c-plane than on the semi-polar sidewalls).
    Left unset (the default), each falls back to `material`; when all three resolve to the same
    name this step adds one layer, otherwise one layer per distinct material, each holding only
    the area that actually grew from that family's own facets.

    `seed_materials` works exactly like in `EpitaxialGrowth` (SAG selectivity).
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["faceted_growth"] = "faceted_growth"
    name: str
    material: str
    thickness: Length
    rate_c: float = 1.0
    rate_m: float = 0.3
    rate_sp: float = 0.6
    semi_polar_angle_deg: float = 30.0
    seed_materials: list[str] = Field(default_factory=list)
    material_c: str | None = None
    material_m: str | None = None
    material_sp: str | None = None

    @model_validator(mode="after")
    def _check_rates(self) -> "FacetedGrowth":
        if self.rate_c < 0 or self.rate_m < 0 or self.rate_sp < 0:
            raise ValueError("growth rates must be >= 0")
        if self.rate_c == 0 and self.rate_m == 0 and self.rate_sp == 0:
            raise ValueError("at least one growth rate must be > 0")
        if not (0.0 < self.semi_polar_angle_deg < 90.0):
            raise ValueError(f"semi_polar_angle_deg must be in (0, 90), got {self.semi_polar_angle_deg}")
        return self


class EpitaxialGrowth(BaseModel):
    """Selective-area epitaxial growth (homo- or hetero-epitaxy) for III-N and related systems.

    `thickness` is the growth thickness measured along the growth-front normal.

    `orientation` picks the growth mode (see `GrowthOrientation`).  For `semi_polar`, set
    `angle_deg` to the tilt of the growth direction from the c-axis ([0001] = vertical):
    0° collapses back to c-plane, 90° to m-plane; typical semi-polar values are 28–58°.

    `seed_materials` drives Selective-Area Growth (SAG): the film only nucleates on exposed
    surfaces made of one of those materials.  A SiO2 or Si3N4 mask left in place will block
    growth wherever it covers the seed.  Leave `seed_materials` empty to grow on every exposed
    surface (no selectivity — useful for blanket buffer/template layers).
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["epitaxial_growth"] = "epitaxial_growth"
    name: str
    material: str
    thickness: Length
    orientation: GrowthOrientation = GrowthOrientation.c_plane
    angle_deg: float = 0.0
    seed_materials: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_semi_polar_angle(self) -> "EpitaxialGrowth":
        if self.orientation is GrowthOrientation.semi_polar and not (0.0 < self.angle_deg < 90.0):
            raise ValueError(
                f"semi_polar orientation requires 0 < angle_deg < 90 (got {self.angle_deg})"
            )
        return self


ProcessStep = Annotated[
    Union[
        Deposition,
        Etch,
        Planarization,
        ChemicalStep,
        Lithography,
        ResistStrip,
        EpitaxialGrowth,
        FacetedGrowth,
        Flip,
    ],
    Field(discriminator="kind"),
]
