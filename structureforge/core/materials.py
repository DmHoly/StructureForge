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
    """A small library covering the common cases from the prompt: Si/SiO2/Si3N4 for a classic
    planar flow, GaN/AlGaN/InGaN for a III-N quantum-well/nanowire stack, a couple of metals, and
    a photoresist. Extend it for a real domain rather than editing it in place.
    """

    materials = [
        Material(name="Si", category=MaterialCategory.substrate, color="#5b5f66", density_g_cm3=2.33, refractive_index=3.48),
        Material(name="SiO2", category=MaterialCategory.dielectric, color="#8ecae6", density_g_cm3=2.2, refractive_index=1.46),
        Material(name="Si3N4", category=MaterialCategory.dielectric, color="#588157", density_g_cm3=3.17, refractive_index=2.0),
        Material(name="Al2O3", category=MaterialCategory.dielectric, color="#a3cef1", density_g_cm3=3.95, refractive_index=1.76),
        Material(name="Poly-Si", category=MaterialCategory.semiconductor, color="#adb5bd", density_g_cm3=2.32, refractive_index=3.6),
        Material(name="GaN", category=MaterialCategory.semiconductor, color="#7b6d8d", density_g_cm3=6.15, refractive_index=2.4),
        Material(name="AlGaN", category=MaterialCategory.semiconductor, color="#9c89b8", density_g_cm3=5.0, refractive_index=2.3),
        Material(name="InGaN", category=MaterialCategory.semiconductor, color="#5e548e", density_g_cm3=6.9, refractive_index=2.5),
        Material(name="Al", category=MaterialCategory.metal, color="#ced4da", density_g_cm3=2.7),
        Material(name="W", category=MaterialCategory.metal, color="#495057", density_g_cm3=19.3),
        Material(name="Ti", category=MaterialCategory.metal, color="#6c757d", density_g_cm3=4.5),
        Material(name="Photoresist", category=MaterialCategory.resist, color="#f4a261", density_g_cm3=1.2),
    ]
    return MaterialLibrary(materials={m.name: m for m in materials})
