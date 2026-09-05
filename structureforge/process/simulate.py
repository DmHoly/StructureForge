"""Applies a `ProcessStep` list to a starting `Geometry` and captures one `Frame` per step, so a
caller (GUI, report, test) can scrub through the process history instead of only seeing the final
result - which is exactly what the multi-view/zoom GUI needs a timeline for.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core.materials import MaterialLibrary
from ..core.recipes import RecipeLibrary
from ..geometry.engine import Geometry, Layer
from .steps import ChemicalStep, Deposition, EpitaxialGrowth, Etch, Lithography, Planarization, ProcessStep, ResistStrip


class SimulationError(RuntimeError):
    """A step failed to apply - wraps the original error with which step (index + kind + name)."""

    def __init__(self, step_index: int, step: ProcessStep, original: Exception):
        self.step_index = step_index
        self.step = step
        self.original = original
        super().__init__(f"step {step_index} ({step.kind} {step.name!r}) failed: {original}")


@dataclass
class Frame:
    step_index: int
    step_kind: str
    step_name: str
    layers: list[Layer]
    domain_width_nm: float

    def to_dict(self) -> dict:
        return {
            "step_index": self.step_index,
            "step_kind": self.step_kind,
            "step_name": self.step_name,
            "domain_width_nm": self.domain_width_nm,
            "layers": [{"material": l.material, "rings": l.rings()} for l in self.layers],
        }


def _snapshot(geometry: Geometry) -> list[Layer]:
    """A frame must freeze *this instant's* polygons - `geometry.layers` holds mutable `Layer`
    objects whose `.polygon` attribute later steps reassign in place, so simply keeping a
    reference to the live list (even a shallow copy of the list itself) would make every earlier
    frame silently show the *final* geometry once `.rings()` is eventually called on it. Copying
    each `Layer` (not its polygon - shapely geometries are themselves immutable under how this
    engine uses them, only the wrapper's attribute gets reassigned) is enough to decouple a
    snapshot from whatever `geometry` does afterwards.
    """
    return [Layer(material=l.material, polygon=l.polygon) for l in geometry.frame_layers()]


def simulate(
    initial: Geometry,
    steps: list[ProcessStep],
    materials: MaterialLibrary,
    recipes: RecipeLibrary,
) -> list[Frame]:
    """Apply `steps` in order to `initial` (mutated in place) and return one `Frame` per step,
    plus an implicit frame 0 for the starting geometry.
    """
    geometry = initial
    frames = [Frame(0, "initial", "Structure de depart", _snapshot(geometry), geometry.domain_width_nm)]

    for i, step in enumerate(steps, start=1):
        try:
            _apply(geometry, step, materials, recipes)
            geometry.compact()
        except Exception as exc:
            raise SimulationError(i, step, exc) from exc
        frames.append(Frame(i, step.kind, step.name, _snapshot(geometry), geometry.domain_width_nm))

    return frames


def _apply(geometry: Geometry, step: ProcessStep, materials: MaterialLibrary, recipes: RecipeLibrary) -> None:
    if isinstance(step, Deposition):
        materials.get(step.material)  # fail fast with a clear message if the material is unknown
        recipe = recipes.get_deposition(step.recipe)
        geometry.deposit(step.material, step.thickness.to_nm(), recipe)
    elif isinstance(step, Etch):
        recipe = recipes.get_etch(step.recipe)
        geometry.etch(recipe, step.depth.to_nm(), materials)
    elif isinstance(step, Planarization):
        target = step.target_level.to_nm() if step.target_level is not None else None
        geometry.planarize(target_level_nm=target, stop_material=step.stop_material)
    elif isinstance(step, ChemicalStep):
        pass  # no geometric effect, by design - recorded for traceability only
    elif isinstance(step, Lithography):
        materials.get(step.resist_material)
        geometry.deposit_conformal_masked(step.resist_material, step.thickness.to_nm(), step.openings)
    elif isinstance(step, ResistStrip):
        geometry.strip_material(step.material)
        geometry.remove_floating_debris()
    elif isinstance(step, EpitaxialGrowth):
        materials.get(step.material)  # fail fast if material unknown
        geometry.deposit_epitaxial(
            step.material,
            step.thickness.to_nm(),
            orientation=step.orientation.value,
            angle_deg=step.angle_deg,
            seed_materials=list(step.seed_materials) if step.seed_materials else None,
        )
    else:  # pragma: no cover - exhaustive over ProcessStep's Union
        raise TypeError(f"unknown step type {type(step).__name__}")
