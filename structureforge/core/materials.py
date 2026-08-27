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
