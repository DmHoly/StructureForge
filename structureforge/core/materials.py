"""The materials library: what a layer is made of, independent of how it got there or how it
etches (that's the recipe's job - see `structureforge.core.recipes`). Selectivity lives on the
*recipe*, not the material, because the same oxide etches differently under a wet HF dip than
under a fluorine dry etch; the material only carries the properties that don't depend on the
process step being applied to it.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class MaterialCategory(str, Enum):
    """The coarse bucket a recipe's selectivity table can key on, when a per-material override
    isn't worth writing out (e.g. "this etch attacks oxides" rather than naming every oxide).
    """

    substrate = "substrate"
    semiconductor = "semiconductor"
    dielectric = "dielectric"
    metal = "metal"
    resist = "resist"
    other = "other"


class Material(BaseModel):
    """A material entry in the library. `color` is a CSS color used by the GUI/renderer only -
    it carries no physical meaning.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    category: MaterialCategory
    color: str = "#9e9e9e"
    density_g_cm3: float | None = None
    refractive_index: float | None = None
    notes: str | None = None


class MaterialLibrary(BaseModel):
    """A named collection of materials, keyed by `Material.name`. Custom domains extend this
    (`default_library().with_materials(...)`) rather than editing the built-in one in place.
    """

    materials: dict[str, Material] = Field(default_factory=dict)

    def get(self, name: str) -> Material:
        try:
            return self.materials[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown material {name!r} - known materials: {sorted(self.materials)!r}"
            ) from exc

    def with_materials(self, *materials: Material) -> "MaterialLibrary":
        merged = dict(self.materials)
        for material in materials:
            merged[material.name] = material
        return MaterialLibrary(materials=merged)

    def __contains__(self, name: str) -> bool:
        return name in self.materials

    def __iter__(self):
        return iter(self.materials.values())


def default_library() -> MaterialLibrary:
    """A broad starter library spanning a classic planar CMOS-ish flow (Si/SiO2/Si3N4/HfO2...), a
    III-N/III-V quantum-well/nanowire stack (GaN/AlGaN/InGaN/AlN plus the wider III-V family and
    their native/foreign growth substrates), common metals and a transparent conductive oxide,
    and four resists (standard photoresist, PMMA and HSQ for e-beam lithography, SU-8 for thick/
    high-aspect-ratio masks). Extend it for a real domain rather than editing it in place.
    """

    materials = [
        # -- substrates --------------------------------------------------
        Material(name="Si", category=MaterialCategory.substrate, color="#5b5f66", density_g_cm3=2.33, refractive_index=3.48),
        Material(name="SiC", category=MaterialCategory.substrate, color="#4a5259", density_g_cm3=3.21, refractive_index=2.65, notes="Common growth substrate for GaN."),
        Material(name="Sapphire", category=MaterialCategory.substrate, color="#dbe4ee", density_g_cm3=3.95, refractive_index=1.76, notes="Al2O3 single crystal - the usual low-cost GaN growth substrate."),
        Material(name="InP", category=MaterialCategory.substrate, color="#8a8fa3", density_g_cm3=4.81, refractive_index=3.1, notes="Native substrate for InP/InGaAsP telecom photonics."),
        # -- dielectrics ---------------------------------------------------
        Material(name="SiO2", category=MaterialCategory.dielectric, color="#8ecae6", density_g_cm3=2.2, refractive_index=1.46),
        Material(name="Si3N4", category=MaterialCategory.dielectric, color="#588157", density_g_cm3=3.17, refractive_index=2.0),
        Material(name="Al2O3", category=MaterialCategory.dielectric, color="#a3cef1", density_g_cm3=3.95, refractive_index=1.76),
        Material(name="HfO2", category=MaterialCategory.dielectric, color="#6d9dc5", density_g_cm3=9.68, refractive_index=2.1, notes="High-k gate dielectric."),
        Material(name="ZrO2", category=MaterialCategory.dielectric, color="#7ba3c9", density_g_cm3=5.68, refractive_index=2.1, notes="High-k dielectric, also an optical coating material."),
        Material(name="Ta2O5", category=MaterialCategory.dielectric, color="#5f7d9e", density_g_cm3=8.2, refractive_index=2.1, notes="High-k / high-index optical dielectric."),
        Material(name="TiO2", category=MaterialCategory.dielectric, color="#4f7396", density_g_cm3=4.23, refractive_index=2.4, notes="High-index optical coating."),
        Material(name="SiON", category=MaterialCategory.dielectric, color="#6f9a6e", density_g_cm3=2.7, refractive_index=1.7, notes="Index-tunable SiO2/Si3N4 blend, common anti-reflection/passivation layer."),
        Material(name="MgF2", category=MaterialCategory.dielectric, color="#c9dced", density_g_cm3=3.15, refractive_index=1.38, notes="Low-index optical coating."),
        Material(name="Polyimide", category=MaterialCategory.dielectric, color="#c98a3d", density_g_cm3=1.42, refractive_index=1.7, notes="Flexible passivation/planarization dielectric, also a packaging material."),
        Material(name="BCB", category=MaterialCategory.dielectric, color="#d9a86c", density_g_cm3=1.05, refractive_index=1.54, notes="Benzocyclobutene - low-k planarization dielectric, common in photonics/MMIC."),
        # -- semiconductors ------------------------------------------------
        Material(name="Poly-Si", category=MaterialCategory.semiconductor, color="#adb5bd", density_g_cm3=2.32, refractive_index=3.6),
        Material(name="Ge", category=MaterialCategory.semiconductor, color="#8d99ae", density_g_cm3=5.32, refractive_index=4.0),
        Material(name="SiGe", category=MaterialCategory.semiconductor, color="#9aa5b1", density_g_cm3=4.0, refractive_index=4.0, notes="Composition-dependent; values shown are representative, not a fixed alloy fraction."),
        Material(name="GaAs", category=MaterialCategory.semiconductor, color="#7c8aa3", density_g_cm3=5.32, refractive_index=3.5),
        Material(name="AlAs", category=MaterialCategory.semiconductor, color="#93a0b8", density_g_cm3=3.76, refractive_index=3.0),
        Material(name="GaP", category=MaterialCategory.semiconductor, color="#87947f", density_g_cm3=4.14, refractive_index=3.3),
        Material(name="GaSb", category=MaterialCategory.semiconductor, color="#6d7a99", density_g_cm3=5.61, refractive_index=3.8),
        Material(name="InAs", category=MaterialCategory.semiconductor, color="#5c6b8a", density_g_cm3=5.67, refractive_index=3.5),
        Material(name="InSb", category=MaterialCategory.semiconductor, color="#4d5c7a", density_g_cm3=5.78, refractive_index=4.0),
        Material(name="ZnO", category=MaterialCategory.semiconductor, color="#a8b89a", density_g_cm3=5.61, refractive_index=2.0, notes="Wide-gap oxide semiconductor, transparent electronics/UV optoelectronics."),
        Material(name="GaN", category=MaterialCategory.semiconductor, color="#7b6d8d", density_g_cm3=6.15, refractive_index=2.4),
        Material(name="AlN", category=MaterialCategory.semiconductor, color="#b0a3c9", density_g_cm3=3.26, refractive_index=2.1, notes="Wide-gap III-N, common nucleation/buffer layer on foreign substrates."),
        Material(name="AlGaN", category=MaterialCategory.semiconductor, color="#9c89b8", density_g_cm3=5.0, refractive_index=2.3),
        Material(name="InGaN", category=MaterialCategory.semiconductor, color="#5e548e", density_g_cm3=6.9, refractive_index=2.5),
        # -- metals (and one transparent conductive oxide) ------------------
        Material(name="Al", category=MaterialCategory.metal, color="#ced4da", density_g_cm3=2.7),
        Material(name="W", category=MaterialCategory.metal, color="#495057", density_g_cm3=19.3),
        Material(name="Ti", category=MaterialCategory.metal, color="#6c757d", density_g_cm3=4.5),
        Material(name="TiN", category=MaterialCategory.metal, color="#7d6608", density_g_cm3=5.22, notes="Diffusion barrier / gate metal."),
        Material(name="Cu", category=MaterialCategory.metal, color="#d08159", density_g_cm3=8.96),
        Material(name="Au", category=MaterialCategory.metal, color="#e6c260", density_g_cm3=19.3, notes="Ohmic/bond-pad metal, common on III-N and III-V contacts."),
        Material(name="Ag", category=MaterialCategory.metal, color="#c9c9d1", density_g_cm3=10.49),
        Material(name="Ni", category=MaterialCategory.metal, color="#8a8478", density_g_cm3=8.91, notes="Common Schottky/ohmic contact metal."),
        Material(name="Pt", category=MaterialCategory.metal, color="#b8b8c0", density_g_cm3=21.45),
        Material(name="Pd", category=MaterialCategory.metal, color="#a9a9b3", density_g_cm3=12.02),
        Material(name="Mo", category=MaterialCategory.metal, color="#6d7278", density_g_cm3=10.28),
        Material(name="Ta", category=MaterialCategory.metal, color="#5c6066", density_g_cm3=16.65, notes="Diffusion barrier metal."),
        Material(name="ITO", category=MaterialCategory.metal, color="#bcd4d8", density_g_cm3=7.12, refractive_index=1.9, notes="Indium tin oxide - transparent conductive oxide, used as a transparent electrode; chemically an oxide but grouped here for its conductive/electrode role, not oxide-selective etch behaviour."),
        # -- resists --------------------------------------------------------
        Material(name="Photoresist", category=MaterialCategory.resist, color="#f4a261", density_g_cm3=1.2),
        Material(name="PMMA", category=MaterialCategory.resist, color="#f6bd60", density_g_cm3=1.18, notes="Positive-tone e-beam lithography resist."),
        Material(name="HSQ", category=MaterialCategory.resist, color="#f7e0ad", density_g_cm3=1.4, notes="Negative-tone e-beam resist (hydrogen silsesquioxane) - high resolution, common for nanowire/nanostructure masks."),
        Material(name="SU-8", category=MaterialCategory.resist, color="#e8a33d", density_g_cm3=1.2, notes="Thick negative photoresist for high-aspect-ratio/MEMS masks."),
    ]
    return MaterialLibrary(materials={m.name: m for m in materials})


def _lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def _lerp_color(color_a: str, color_b: str, t: float) -> str:
    """Linearly blend two "#rrggbb" colors channel-by-channel at t in [0, 1]."""
    a, b = color_a.lstrip("#"), color_b.lstrip("#")
    channels = (
        round(_lerp(int(a[i : i + 2], 16), int(b[i : i + 2], 16), t)) for i in (0, 2, 4)
    )
    return "#" + "".join(f"{c:02x}" for c in channels)


def _spectral_color(t: float, stops: list[tuple[float, str]]) -> str:
    """Piecewise-linear blend across `stops` (sorted, ascending `t` in [0, 1]) - a multi-point
    generalization of `_lerp_color`'s single blend, used to sweep a composition fraction across
    a whole visible-spectrum palette instead of just two end colors.
    """
    if t <= stops[0][0]:
        return stops[0][1]
    if t >= stops[-1][0]:
        return stops[-1][1]
    for (t0, c0), (t1, c1) in zip(stops, stops[1:]):
        if t0 <= t <= t1:
            local_t = (t - t0) / (t1 - t0)
            return _lerp_color(c0, c1, local_t)
    return stops[-1][1]  # pragma: no cover - unreachable, t is within [stops[0][0], stops[-1][0]]


# Reference points for the two ternary III-N alloys the library parameterizes - not full
# `Material` entries of their own (the library only stocks the binaries it actually needs
# elsewhere, GaN and AlN; InN would exist solely to anchor this interpolation). Density and
# refractive index follow Vegard's law (linear in composition) - a first-order approximation,
# fine for a schematic cross-section but not for quantitative optics (real InGaN/AlGaN bow).
# InN's color is the conventional deep red/orange used in epitaxy schematics for "high indium",
# chosen to be visually far from GaN's own violet-grey so a composition series reads at a glance.
_GaN_REF = {"color": "#7b6d8d", "density_g_cm3": 6.15, "refractive_index": 2.4}
_AlN_REF = {"color": "#b0a3c9", "density_g_cm3": 3.26, "refractive_index": 2.1}
_InN_REF = {"color": "#c1352e", "density_g_cm3": 6.81, "refractive_index": 2.9}

# Indium content read as a sweep across the visible spectrum (near-UV/violet at x=0 -> red at
# x=1), echoing the real physics of InGaN: increasing indium content shrinks the bandgap, which
# red-shifts the corresponding emission/absorption from GaN's own near-UV edge towards visible
# red. Stops go in actual spectral (wavelength) order - violet, blue, cyan, green, yellow,
# orange, red - rather than any particular verbal listing of the colors, so the gradient reads
# as a physically coherent rainbow instead of a jumbled hue cycle.
_INDIUM_SPECTRUM: list[tuple[float, str]] = [
    (0.0, "#4b0082"),  # near-UV / violet - pure GaN
    (1 / 6, "#0033cc"),  # blue
    (2 / 6, "#00b4d8"),  # cyan
    (3 / 6, "#2ecc71"),  # green
    (4 / 6, "#f1c40f"),  # yellow
    (5 / 6, "#e67e22"),  # orange
    (1.0, "#e63946"),  # red - pure InN
]


def _ternary_nitride(
    symbol: str,
    param_name: str,
    fraction: float,
    other_ref: dict,
    *,
    category: MaterialCategory,
    notes: str | None,
    color_fn=None,
) -> Material:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"{param_name} must be in [0, 1], got {fraction}")
    name = f"{symbol}{fraction:.2f}Ga{1 - fraction:.2f}N"
    color = color_fn(fraction) if color_fn is not None else _lerp_color(_GaN_REF["color"], other_ref["color"], fraction)
    return Material(
        name=name,
        category=category,
        color=color,
        density_g_cm3=_lerp(_GaN_REF["density_g_cm3"], other_ref["density_g_cm3"], fraction),
        refractive_index=_lerp(_GaN_REF["refractive_index"], other_ref["refractive_index"], fraction),
        notes=notes or f"{name} - linear Vegard's-law estimate between GaN and {symbol}N.",
    )


def indium_gan(
    indium_fraction: float, *, category: MaterialCategory = MaterialCategory.semiconductor, notes: str | None = None
) -> Material:
    """In_x Ga_(1-x) N, `indium_fraction` = x from 0 (pure GaN) to 1 (pure InN).

    Named `In{x:.2f}Ga{1-x:.2f}N`, so two calls with the same fraction (rounded to 2 decimals)
    produce an identically-named `Material` - safe to register into a `MaterialLibrary` once per
    call site (e.g. once per MQW period) without colliding or needing to de-duplicate by hand.
    Color sweeps the visible spectrum from near-UV/violet (x=0, pure GaN) through blue, cyan,
    green, yellow, orange to red (x=1, pure InN) - see `_INDIUM_SPECTRUM` - so a stack of
    increasing In content reads, at a glance, as a physically-ordered rainbow: the color *is*
    the composition, not just a label for it.
    """
    return _ternary_nitride(
        "In",
        "indium fraction",
        indium_fraction,
        _InN_REF,
        category=category,
        notes=notes,
        color_fn=lambda x: _spectral_color(x, _INDIUM_SPECTRUM),
    )


def aluminum_gan(
    aluminum_fraction: float, *, category: MaterialCategory = MaterialCategory.semiconductor, notes: str | None = None
) -> Material:
    """Al_y Ga_(1-y) N, symmetric to `indium_gan` - blends toward AlN (a pale violet) instead of
    InN, so higher Al content reads as lighter/cooler rather than red-shifted.
    """
    return _ternary_nitride("Al", "aluminum fraction", aluminum_fraction, _AlN_REF, category=category, notes=notes)


def _fraction_series(start: float, end: float, steps: int) -> list[float]:
    if steps < 2:
        raise ValueError(f"steps must be >= 2 to span a range, got {steps}")
    return [start + (end - start) * i / (steps - 1) for i in range(steps)]


def indium_gan_gradient(indium_fraction_start: float, indium_fraction_end: float, steps: int) -> list[Material]:
    """`steps` `indium_gan` materials evenly spaced from `indium_fraction_start` to
    `indium_fraction_end` (both ends included) - e.g. one call per sub-layer of a graded-index
    or digitally-graded InGaN stack. `steps` must be >= 2.
    """
    return [indium_gan(x) for x in _fraction_series(indium_fraction_start, indium_fraction_end, steps)]


def aluminum_gan_gradient(aluminum_fraction_start: float, aluminum_fraction_end: float, steps: int) -> list[Material]:
    """`steps` `aluminum_gan` materials evenly spaced from `aluminum_fraction_start` to
    `aluminum_fraction_end` (both ends included). `steps` must be >= 2.
    """
    return [aluminum_gan(y) for y in _fraction_series(aluminum_fraction_start, aluminum_fraction_end, steps)]
