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
        DepositionRecipe(
            name="MOCVD Epitaxial",
            mode=DepositionMode.conformal,
            notes=(
                "III-N/III-V epitaxial growth (GaN, AlGaN, InGaN...). Modelled as conformal like "
                "any other blanket growth - real epitaxy is crystallographic/faceted, which this "
                "engine doesn't simulate; conformal is the closest built-in approximation for a "
                "layer grown on an already-flat template."
            ),
        ),
        DepositionRecipe(
            name="PECVD Conformal",
            mode=DepositionMode.conformal,
            notes="Plasma-enhanced CVD - lower deposition temperature than thermal CVD, similar (good, not perfect) step coverage.",
        ),
        DepositionRecipe(
            name="Sputter Metal (normal)",
            mode=DepositionMode.directional,
            angle_deg=0.0,
            notes="Normal-incidence physical sputtering - better sidewall coverage than evaporation, still line-of-sight/directional.",
        ),
        DepositionRecipe(
            name="Electroplating (Cu)",
            mode=DepositionMode.conformal,
            notes="Bottom-up electrochemical fill (Cu damascene interconnects) - modelled as conformal; needs a conductive seed layer already present in a real process, which this engine doesn't check for.",
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
        EtchRecipe(
            name="KOH Anisotropic Wet Etch",
            mode=EtchMode.directional,
            angle_deg=54.7,
            selectivity_by_material={"Si": 1.0},
            default_factor=0.02,
            notes=(
                "Crystallographic wet etch of Si (100), self-terminating on {111} planes at "
                "54.7deg from the surface - modelled here as a directional etch at that fixed "
                "angle. Stops almost completely on oxide/nitride masks and anything else."
            ),
        ),
        EtchRecipe(
            name="Cl2 ICP-RIE (III-N)",
            mode=EtchMode.directional,
            angle_deg=0.0,
            selectivity_by_material={"GaN": 1.0, "AlGaN": 1.0, "InGaN": 1.0, "AlN": 1.0},
            default_factor=0.05,
            notes="Near-vertical dry etch of the III-N family (GaN/AlGaN/InGaN/AlN); selective over masks/dielectrics/metals.",
        ),
        EtchRecipe(
            name="TMAH Anisotropic Wet Etch",
            mode=EtchMode.directional,
            angle_deg=54.7,
            selectivity_by_material={"Si": 1.0},
            default_factor=0.02,
            notes=(
                "CMOS-compatible alternative to KOH (no alkali metal contamination) - same {111} "
                "self-terminating angle, but noticeably gentler on exposed Al than KOH is."
            ),
        ),
        EtchRecipe(
            name="SF6 Deep RIE (Si)",
            mode=EtchMode.directional,
            angle_deg=0.0,
            selectivity_by_material={"Si": 1.0, "SiO2": 0.02, "Si3N4": 0.02, "Photoresist": 0.05},
            default_factor=1.0,
            notes="Bosch-style deep silicon etch (through-silicon vias, MEMS) - near-vertical, high-rate, stops well on an oxide/nitride/resist mask.",
        ),
        EtchRecipe(
            name="Wet Metal Etch",
            mode=EtchMode.isotropic,
            selectivity_by_category={MaterialCategory.metal: 1.0},
            default_factor=0.05,
            notes="Generic wet metal patterning etchant - attacks metals at the nominal rate, everything else slowly; undercuts under its mask like any isotropic etch.",
        ),
        EtchRecipe(
            name="GaN V-pit Etch",
            mode=EtchMode.isotropic,
            selectivity_by_material={"GaN": 1.0, "AlGaN": 1.0, "InGaN": 1.0, "AlN": 1.0},
            default_factor=0.05,
            notes=(
                "A real V-pit forms by growth (not etching) that self-terminates on two "
                "symmetric {1-101} semipolar facets, giving a sharp-walled cone; this engine's "
                "directional recipes model a single tilted beam rather than two symmetric "
                "self-terminating facets, so the closest reproducible stand-in for a symmetric "
                "pit is an isotropic undercut under a narrow mask opening - rounded rather than "
                "sharp-faceted, but the right topology (wide at the surface, narrowing towards a "
                "point below) to cut down through an already-grown, otherwise flat III-N "
                "multilayer stack. GaN/AlGaN/InGaN/AlN all etch at the same nominal rate, a mask "
                "stays mostly put."
            ),
        ),
    ]
    return RecipeLibrary(deposition={r.name: r for r in deposition}, etch={r.name: r for r in etch})
