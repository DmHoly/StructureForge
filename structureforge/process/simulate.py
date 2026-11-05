"""Applies a `ProcessStep` list to a starting `Geometry` and captures one `Frame` per step, so a
caller (GUI, report, test) can scrub through the process history instead of only seeing the final
result - which is exactly what the multi-view/zoom GUI needs a timeline for.

`SimulationCache` is for a caller that resubmits the same flow with small edits (the GUI, on
every change): since a frame is a complete snapshot of the stack, a run that shares its first
steps with an earlier one restarts from that earlier run's frame and replays only the rest.
"""

from __future__ import annotations

import hashlib
import zlib
from dataclasses import dataclass

from ..core.materials import MaterialLibrary
from ..core.recipes import RecipeLibrary
from ..core.traced import Traced
from ..core.units import Length
from ..geometry.engine import Geometry, Layer, LayerProvenance
from .steps import ChemicalStep, Deposition, EpitaxialGrowth, FacetEnvelope, FacetedGrowth, Etch, Flip, Lithography, Planarization, ProcessStep, ResistStrip


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
            "layers": [
                {
                    "material": l.material,
                    "rings": l.rings(),
                    "provenance": l.provenance.model_dump(mode="json") if l.provenance else None,
                }
                for l in self.layers
            ],
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
    return [Layer(material=l.material, polygon=l.polygon, provenance=l.provenance) for l in geometry.frame_layers()]


def simulate(
    initial: Geometry,
    steps: list[ProcessStep],
    materials: MaterialLibrary,
    recipes: RecipeLibrary,
    *,
    preview: bool = False,
) -> list[Frame]:
    """Apply `steps` in order to `initial` (mutated in place) and return one `Frame` per step,
    plus an implicit frame 0 for the starting geometry. `preview` trades accuracy for speed:
    every etch runs a few times fewer substeps (see `Geometry.etch`), which shifts an isotropic
    front by a few nm at most - for a quick look while editing, not for a result to keep.
    """
    frames = [Frame(0, "initial", "Structure de depart", _snapshot(initial), initial.domain_width_nm)]
    return _simulate_from(initial, steps, materials, recipes, frames, preview=preview)


def _simulate_from(
    geometry: Geometry,
    steps: list[ProcessStep],
    materials: MaterialLibrary,
    recipes: RecipeLibrary,
    frames: list[Frame],
    *,
    preview: bool = False,
) -> list[Frame]:
    """Apply `steps` to `geometry` - the state after the last of `frames` - appending one `Frame`
    per step to a copy of `frames`, numbered on from them."""
    frames = list(frames)
    first = len(frames)
    for i, step in enumerate(steps, start=first):
        try:
            _apply(geometry, step, materials, recipes, preview=preview)
            geometry.compact()
        except Exception as exc:
            raise SimulationError(i, step, exc) from exc
        frames.append(Frame(i, step.kind, step.name, _snapshot(geometry), geometry.domain_width_nm))
    return frames


def _restore(frame: Frame, floor_nm: float | None) -> Geometry:
    """The live `Geometry` a frame was taken from, rebuilt from its snapshot: the frame's layers
    are copied (their polygons are never mutated, only reassigned, so sharing them is safe - see
    `_snapshot`), and `floor_nm` is what `Geometry.substrate` set, unchanged by every step (a
    `Flip` mirrors about the stack's own mid-height)."""
    geometry = Geometry(
        frame.domain_width_nm,
        [Layer(material=l.material, polygon=l.polygon, provenance=l.provenance) for l in frame.layers],
    )
    geometry.floor_nm = floor_nm
    return geometry


@dataclass
class _CachedRun:
    key: str
    steps: list[dict]
    frames: list[Frame]


class SimulationCache:
    """`simulate`, remembering the last `capacity` runs so that a flow sharing its first steps
    with one of them is replayed only from the first step that differs - an edit to step 18 of
    20 costs two steps, not twenty. The GUI resubmits the whole flow on every change, and its
    users edit one step at a time, so nearly every run is such a prefix of an earlier one.

    A run is keyed on everything else its frames depend on: the substrate, the material library,
    the recipe library (custom recipes can change between two runs of the same flow) and
    whether it is a preview, so an exact run never resumes from a preview's frames. Steps are
    compared by their serialized form, so two equal steps built separately match. The snapshot
    a run resumes from is a `Frame`, which is immutable by construction (its layers' polygons
    are never mutated in place), so resuming never disturbs the cached run.
    """

    def __init__(self, capacity: int = 8):
        self.capacity = capacity
        self._runs: list[_CachedRun] = []

    @staticmethod
    def _key(
        material: str, domain_width_nm: float, thickness_nm: float, materials: MaterialLibrary, recipes: RecipeLibrary, preview: bool
    ) -> str:
        digest = hashlib.sha256()
        digest.update(repr((material, float(domain_width_nm), float(thickness_nm), bool(preview))).encode())
        digest.update(materials.model_dump_json().encode())
        digest.update(recipes.model_dump_json().encode())
        return digest.hexdigest()

    def simulate(
        self,
        material: str,
        domain_width_nm: float,
        thickness_nm: float,
        steps: list[ProcessStep],
        materials: MaterialLibrary,
        recipes: RecipeLibrary,
        *,
        preview: bool = False,
    ) -> list[Frame]:
        """One `Frame` per step plus the initial one, exactly as `simulate` on
        `Geometry.substrate(material, domain_width_nm, thickness_nm)` would return."""
        key = self._key(material, domain_width_nm, thickness_nm, materials, recipes, preview)
        serialized = [step.model_dump(mode="json") for step in steps]
        best, shared = None, 0
        for run in self._runs:
            if run.key != key:
                continue
            n = 0
            for a, b in zip(run.steps, serialized):
                if a != b:
                    break
                n += 1
            if n > shared:
                best, shared = run, n
        if best is None:
            geometry = Geometry.substrate(material, domain_width_nm, thickness_nm)
            frames = simulate(geometry, steps, materials, recipes, preview=preview)
        else:
            geometry = _restore(best.frames[shared], -thickness_nm)
            frames = _simulate_from(geometry, steps[shared:], materials, recipes, best.frames[: shared + 1], preview=preview)
            self._runs.remove(best)
        self._runs.insert(0, _CachedRun(key, serialized, frames))
        del self._runs[self.capacity :]
        return frames


def _traced_length(length: Length) -> Traced:
    """Bridge `Length`'s own value/derivation duality into a generic `Traced`: a literal
    `Length` becomes a bare `Traced.literal`, and a derived one (rate x duration, Arrhenius...)
    becomes `Traced.computed`, carrying the same `LengthDerivation` tree that produced it.
    """
    if length.derivation is not None:
        return Traced.computed(value=length.to_nm(), derivation=length.derivation.model_dump(mode="json"))
    return Traced.literal(length.to_nm())


def _apply(
    geometry: Geometry, step: ProcessStep, materials: MaterialLibrary, recipes: RecipeLibrary, preview: bool = False
) -> None:
    if isinstance(step, Deposition):
        materials.get(step.material)  # fail fast with a clear message if the material is unknown
        recipe = recipes.get_deposition(step.recipe)
        geometry.deposit(
            step.material,
            step.thickness.to_nm(),
            recipe,
            roughness_nm=step.roughness_nm,
            grain_nm=step.grain_nm,
            seed=zlib.crc32(step.name.encode()),  # stable across runs, unlike hash()
            seed_materials=list(step.seed_materials) if step.seed_materials else None,
        )
    elif isinstance(step, Etch):
        recipe = recipes.get_etch(step.recipe)
        geometry.etch(recipe, step.depth.to_nm(), materials, preview=preview)
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
        provenance = LayerProvenance(
            step_kind=step.kind,
            step_name=step.name,
            parameters={
                "thickness": _traced_length(step.thickness),
                "orientation": Traced.literal(step.orientation.value),
                "angle_deg": Traced.literal(step.angle_deg),
                "seed_materials": Traced.literal(list(step.seed_materials)),
            },
        )
        geometry.deposit_epitaxial(
            step.material,
            step.thickness.to_nm(),
            orientation=step.orientation.value,
            angle_deg=step.angle_deg,
            seed_materials=list(step.seed_materials) if step.seed_materials else None,
            provenance=provenance,
        )
    elif isinstance(step, FacetedGrowth):
        materials.get(step.material)
        for override in (step.material_c, step.material_m, step.material_sp, step.material_sp_inv):
            if override is not None:
                materials.get(override)
        provenance = LayerProvenance(
            step_kind=step.kind,
            step_name=step.name,
            parameters={
                "thickness": _traced_length(step.thickness),
                "rate_c": Traced.literal(step.rate_c),
                "rate_m": Traced.literal(step.rate_m),
                "rate_sp": Traced.literal(step.rate_sp),
                **({"rate_sp_inv": Traced.literal(step.rate_sp_inv)} if step.rate_sp_inv else {}),
                **({"lateral_reach_nm": Traced.literal(step.lateral_reach_nm)} if step.lateral_reach_nm is not None else {}),
                "semi_polar_angle_deg": Traced.literal(step.semi_polar_angle_deg),
                "seed_materials": Traced.literal(list(step.seed_materials)),
            },
        )
        geometry.deposit_faceted(
            step.material,
            step.thickness.to_nm(),
            rate_c=step.rate_c,
            rate_m=step.rate_m,
            rate_sp=step.rate_sp,
            semi_polar_angle_deg=step.semi_polar_angle_deg,
            seed_materials=list(step.seed_materials) if step.seed_materials else None,
            material_c=step.material_c,
            material_m=step.material_m,
            material_sp=step.material_sp,
            provenance=provenance,
            rate_sp_inv=step.rate_sp_inv,
            material_sp_inv=step.material_sp_inv,
            lateral_reach_nm=step.lateral_reach_nm,
        )
    elif isinstance(step, FacetEnvelope):
        materials.get(step.material)
        provenance = LayerProvenance(
            step_kind=step.kind,
            step_name=step.name,
            parameters={
                "c_plane": Traced.literal(step.c_plane),
                "m_plane": Traced.literal(step.m_plane),
                "semi_polar_angle_deg": Traced.literal(step.semi_polar_angle_deg),
                "seed_materials": Traced.literal(list(step.seed_materials)),
                **({"top_level": _traced_length(step.top_level)} if step.top_level is not None else {}),
            },
        )
        geometry.fill_facet_envelope(
            step.material,
            seed_materials=list(step.seed_materials) if step.seed_materials else None,
            c_plane=step.c_plane,
            m_plane=step.m_plane,
            semi_polar_angle_deg=step.semi_polar_angle_deg,
            top_level_nm=step.top_level.to_nm() if step.top_level is not None else None,
            provenance=provenance,
        )
    elif isinstance(step, Flip):
        geometry.flip()
    else:  # pragma: no cover - exhaustive over ProcessStep's Union
        raise TypeError(f"unknown step type {type(step).__name__}")
