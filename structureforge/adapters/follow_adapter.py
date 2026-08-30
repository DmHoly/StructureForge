"""Bridge to `follow` (git-for-experiments): turns a simulated `Geometry` and its `ProcessStep`
history into a `follow.Structure` and a `follow.Step` protocol list, so a completed process flow
can be committed as a Follow experiment - the geometry becomes what's studied, the steps become
how it was made.

StructureForge's engine never imports `follow` - by design (see the project README), the
geometry engine doesn't depend on it. This module is the one place that does: importing it at all
means you want the bridge, so the `import follow` below is not lazy, unlike the rest of the
package. Install the optional dependency first: `pip install structureforge[follow]` (or
`pip install follow` directly).

`ProcessStructure` and `LayerSpec` are defined here, at module level, deliberately - not inside a
function. `follow.Structure.registry_key()` is derived from `__module__` + `__qualname__`, and a
class nested inside a function gets a qualname containing `<locals>`, which - per
`follow.core.structure.Structure`'s own docstring - "resolves fine in-process but breaks the
CLI's `--structure-type module.Class` dotted-path import" and re-defines a fresh, distinct
registry entry on every call. Defining the class once, at import time, is what makes
`ProcessStructure`'s registry key (`structureforge.adapters.follow_adapter.ProcessStructure`)
stable and dotted-path-importable, and what lets `follow.Structure.resolve()` find the same class
every time instead of a new one per export.
"""

from __future__ import annotations

from typing import Any

from ..geometry.engine import Geometry
from ..process.steps import ProcessStep

try:
    import follow
except ImportError as exc:  # pragma: no cover - exercised only when the extra isn't installed
    raise ImportError(
        "structureforge.adapters.follow_adapter needs the optional `follow` dependency - install "
        "it with `pip install structureforge[follow]` (or `pip install follow` directly)"
    ) from exc

from pydantic import BaseModel, ConfigDict


class LayerSpec(BaseModel):
    """One layer's material + polygon rings, flattened to plain JSON-friendly fields - shapely
    geometries aren't Pydantic types, and a Follow `Structure` needs to diff/serialize cleanly.
    """

    model_config = ConfigDict(frozen=True)

    material: str
    rings: list[dict]


class ProcessStructure(follow.Structure):
    """The `follow.Structure` StructureForge exports to: a domain width plus the layer stack at
    export time. Subclass this (rather than editing it) for a domain that wants extra fields
    alongside the geometry - the same guidance `follow.Structure` itself gives.
    """

    domain_width_nm: float
    layers: list[LayerSpec]


def to_structure(geometry: Geometry) -> ProcessStructure:
    """A `ProcessStructure` snapshot of `geometry`'s current layer stack."""
    return ProcessStructure(
        domain_width_nm=geometry.domain_width_nm,
        layers=[LayerSpec(material=l.material, rings=l.rings()) for l in geometry.frame_layers()],
    )


def to_steps(process_steps: list[ProcessStep]) -> list[Any]:
    """`process_steps` as a `follow.Step` protocol list (one Follow step per process step,
    `order` = its 1-based position) - so the recipe that produced a structure becomes the
    Follow experiment's protocol, including `ChemicalStep`s that never touched the geometry.
    """
    out = []
    for order, step in enumerate(process_steps, start=1):
        parameters = {}
        for field_name, value in step.model_dump(mode="json", exclude={"kind", "name", "description"}).items():
            if isinstance(value, dict) and {"value", "unit"} <= value.keys():
                parameters[field_name] = follow.Quantity(value=value["value"], unit=value.get("unit"))
        out.append(
            follow.Step(
                order=order,
                name=f"{step.kind}: {step.name}",
                description=getattr(step, "description", None),
                parameters=parameters,
            )
        )
    return out


def export_experiment(
    repo: "follow.Repository",
    geometry: Geometry,
    process_steps: list[ProcessStep],
    *,
    branch: str,
    title: str,
    intent: str,
    **experiment_kwargs: Any,
) -> "follow.Experiment":
    """Commit `geometry`'s current state and `process_steps`'s history as one Follow experiment
    on `branch` of `repo`: the structure becomes what's studied, the steps become the protocol.
    Extra keyword arguments (`hypothesis`, `references`, `objectives`, `tags`, `author`...) pass
    straight through to `follow.Repository.new`.
    """
    structure = to_structure(geometry)
    steps = to_steps(process_steps)
    builder = repo.new(
        branch=branch,
        structure=structure,
        title=title,
        intent=intent,
        steps=steps,
        **experiment_kwargs,
    )
    return builder.commit()


def build_experiment(
    repo: "follow.Repository",
    geometry: Geometry,
    process_steps: list[ProcessStep],
    *,
    branch: str,
    title: str,
    intent: str,
    **experiment_kwargs: Any,
) -> "follow.ExperimentBuilder":
    """Like `export_experiment`, but returns the uncommitted `ExperimentBuilder` instead of
    committing it - for a caller that needs to set more on the builder first (`metadata`,
    `evidence`, a `commit_form` answer...) before calling `.commit()` itself.
    """
    structure = to_structure(geometry)
    steps = to_steps(process_steps)
    return repo.new(
        branch=branch,
        structure=structure,
        title=title,
        intent=intent,
        steps=steps,
        **experiment_kwargs,
    )


def derive_experiment(
    repo: "follow.Repository",
    ref: str,
    geometry: Geometry,
    process_steps: list[ProcessStep],
    *,
    title: str,
    intent: str,
    new_branch: str | None = None,
    **experiment_kwargs: Any,
) -> "follow.ExperimentBuilder":
    """The evolution equivalent of `build_experiment`: branch off `ref` (see
    `follow.Repository.derive`) with `geometry`/`process_steps` as the new state, rather than
    carrying the parent's structure/steps forward unchanged. Returns the uncommitted builder -
    call `.commit()` once any extra metadata/evidence is set on it.

    `carry_steps` is always overridden to `False`: the new `process_steps` replace the parent's
    protocol outright, the same way `structure` does, since both come from the same re-simulation.
    """
    structure = to_structure(geometry)
    steps = to_steps(process_steps)
    experiment_kwargs.pop("carry_steps", None)
    builder = repo.derive(
        ref,
        title=title,
        intent=intent,
        new_branch=new_branch,
        structure=structure,
        carry_steps=False,
        **experiment_kwargs,
    )
    builder.steps = steps
    return builder
