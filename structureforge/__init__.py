"""StructureForge: a 2D cross-section process simulator.

Build device geometries from elementary process bricks (deposition, etch,
planarization, chemical steps) drawn from a materials library and a
recipe library (deposition/etch modes, angle, selectivity). Independent
from `follow` (the experiment-tracking library) - see
`structureforge.adapters.follow_adapter` for an optional bridge that turns
a simulated structure and its process history into a `follow.Structure`
and `follow.core.models.Step` list.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .core.materials import (
    Material,
    MaterialCategory,
    MaterialLibrary,
    aluminum_gan,
    aluminum_gan_gradient,
    default_library,
    indium_gan,
    indium_gan_gradient,
)
from .core.derivation import ArrheniusRate, ConstantRate, GrowthAtRate, LengthDerivation, MultiStageGrowth, RateDerivation
from .core.recipes import DepositionMode, DepositionRecipe, EtchMode, EtchRecipe, RecipeLibrary, default_recipes
from .core.units import Length
from .geometry.engine import Geometry, Layer
from .process.steps import (
    ChemicalStep,
    Deposition,
    EpitaxialGrowth,
    Etch,
    FacetedGrowth,
    Flip,
    GrowthOrientation,
    Lithography,
    Planarization,
    ProcessStep,
    ResistStrip,
)
from .process.simulate import Frame, SimulationError, simulate
from .presentation.svg import frame_to_svg, save_svg

__all__ = [
    "Material",
    "MaterialCategory",
    "MaterialLibrary",
    "default_library",
    "indium_gan",
    "aluminum_gan",
    "indium_gan_gradient",
    "aluminum_gan_gradient",
    "DepositionMode",
    "DepositionRecipe",
    "EtchMode",
    "EtchRecipe",
    "RecipeLibrary",
    "default_recipes",
    "Length",
    "LengthDerivation",
    "RateDerivation",
    "GrowthAtRate",
    "MultiStageGrowth",
    "ConstantRate",
    "ArrheniusRate",
    "Geometry",
    "Layer",
    "ProcessStep",
    "Deposition",
    "Etch",
    "Planarization",
    "ChemicalStep",
    "Lithography",
    "ResistStrip",
    "EpitaxialGrowth",
    "FacetedGrowth",
    "Flip",
    "GrowthOrientation",
    "Frame",
    "simulate",
    "SimulationError",
    "frame_to_svg",
    "save_svg",
]
