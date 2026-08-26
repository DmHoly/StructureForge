"""The recipe library: *how* a deposition or etch behaves, independent of the material it's
applied to. Selectivity - "this etch attacks oxides at the nominal rate and everything else at
0.8x" - lives here, on the recipe, keyed by material name (specific override) or
`MaterialCategory` (broad rule), with `default_factor` for anything not listed. A recipe's
`angle_deg` is measured from the surface normal (vertical): 0 deg is normal incidence (etches
straight down / deposits straight up from a source directly overhead), positive tilts the beam
towards +x. Conformal deposition and isotropic etch ignore the angle entirely - they act equally
in every direction, which is exactly what makes them conformal/isotropic.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from .materials import Material, MaterialCategory


class DepositionMode(str, Enum):
    conformal = "conformal"  #: CVD/ALD-like - uniform thickness on every exposed surface, any angle.
    directional = "directional"  #: PVD/evaporation-like - arrives from one direction, tilt-dependent coverage.


class DepositionRecipe(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    mode: DepositionMode
    angle_deg: float = 0.0
    notes: str | None = None


class EtchMode(str, Enum):
    isotropic = "isotropic"  #: wet-etch-like - removes material equally in every direction from the exposed surface.
    directional = "directional"  #: RIE/ion-milling-like - removes material along one direction, tilt-dependent.


class EtchRecipe(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    mode: EtchMode
    angle_deg: float = 0.0
    selectivity_by_material: dict[str, float] = Field(default_factory=dict)
    selectivity_by_category: dict[MaterialCategory, float] = Field(default_factory=dict)
    default_factor: float = 1.0
    notes: str | None = None

    def factor_for(self, material: Material) -> float:
        """The relative etch-rate multiplier this recipe applies to `material` (1.0 = the
        recipe's nominal/reference rate, 0 = doesn't etch it at all - a mask/stop layer).
        """
        if material.name in self.selectivity_by_material:
            return self.selectivity_by_material[material.name]
        if material.category in self.selectivity_by_category:
            return self.selectivity_by_category[material.category]
        return self.default_factor


class RecipeLibrary(BaseModel):
    deposition: dict[str, DepositionRecipe] = Field(default_factory=dict)
    etch: dict[str, EtchRecipe] = Field(default_factory=dict)

    def get_deposition(self, name: str) -> DepositionRecipe:
        try:
            return self.deposition[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown deposition recipe {name!r} - known recipes: {sorted(self.deposition)!r}"
            ) from exc

    def get_etch(self, name: str) -> EtchRecipe:
        try:
            return self.etch[name]
        except KeyError as exc:
            raise KeyError(f"unknown etch recipe {name!r} - known recipes: {sorted(self.etch)!r}") from exc

    def with_recipes(
        self,
        *,
        deposition: list[DepositionRecipe] | None = None,
        etch: list[EtchRecipe] | None = None,
    ) -> "RecipeLibrary":
        merged_deposition = dict(self.deposition)
        for recipe in deposition or []:
            merged_deposition[recipe.name] = recipe
        merged_etch = dict(self.etch)
        for recipe in etch or []:
            merged_etch[recipe.name] = recipe
        return RecipeLibrary(deposition=merged_deposition, etch=merged_etch)


def default_recipes() -> RecipeLibrary:
    """A small starter library. `"Dry Oxide Etch"` is exactly the example from the spec: attacks
    oxides at the nominal rate, everything else at 0.8x.
    """

    deposition = [
        DepositionRecipe(name="ALD Conformal", mode=DepositionMode.conformal, notes="Perfect step coverage, any topology."),
        DepositionRecipe(name="CVD Conformal", mode=DepositionMode.conformal, notes="Good but imperfect step coverage, modelled as fully conformal."),
        DepositionRecipe(
            name="PVD Sputter (tilted)",
            mode=DepositionMode.directional,
            angle_deg=15.0,
            notes="Line-of-sight metal deposition, mild tilt.",
        ),
        DepositionRecipe(
            name="Evaporation (normal)",
            mode=DepositionMode.directional,
            angle_deg=0.0,
            notes="Straight line-of-sight from directly overhead - poor sidewall coverage, good for lift-off.",
        ),
    ]
    etch = [
        EtchRecipe(
            name="Dry Oxide Etch",
            mode=EtchMode.isotropic,
            selectivity_by_category={MaterialCategory.dielectric: 1.0},
            default_factor=0.8,
            notes="Attacks oxides at the nominal rate; everything else at 0.8x.",
        ),
        EtchRecipe(
            name="Wet HF Dip",
            mode=EtchMode.isotropic,
            selectivity_by_category={MaterialCategory.dielectric: 1.0},
            default_factor=0.05,
            notes="Highly selective wet etch of oxide over nitride/silicon/metal.",
        ),
        EtchRecipe(
            name="Anisotropic RIE",
            mode=EtchMode.directional,
            angle_deg=0.0,
            selectivity_by_material={"Photoresist": 0.1},
            default_factor=1.0,
            notes="Near-vertical dry etch; resist mask erodes slowly, everything else at the nominal rate.",
        ),
        EtchRecipe(
            name="Ion Mill (tilted)",
            mode=EtchMode.directional,
            angle_deg=30.0,
            default_factor=1.0,
            notes="Tilted physical (ion-milling) etch, roughly material-independent.",
        ),
    ]
    return RecipeLibrary(deposition={r.name: r for r in deposition}, etch={r.name: r for r in etch})
