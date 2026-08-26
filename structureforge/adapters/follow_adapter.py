"""Optional bridge to `follow` (git-for-experiments): turns a simulated `Geometry` and its
`ProcessStep` history into a `follow.Structure` and a `follow.Step` protocol list, so a completed
process flow can be committed as a Follow experiment - the geometry becomes what's studied, the
steps become how it was made.

StructureForge itself never imports `follow` at module load time - by design (see the project
README), the geometry engine doesn't depend on it. This module is the one place that does, and
only lazily: `pip install structureforge[follow]` (or `pip install follow` directly) to use it.
"""

from __future__ import annotations

from typing import Any

from ..geometry.engine import Geometry
from ..process.steps import ProcessStep


def _require_follow():
    try:
        import follow
    except ImportError as exc:
        raise ImportError(
            "the follow adapter needs the optional `follow` dependency - install it with "
            "`pip install structureforge[follow]` (or `pip install follow` directly)"
        ) from exc
    return follow


def to_structure(geometry: Geometry) -> Any:
    """A `follow.Structure` snapshot of `geometry`'s current layer stack: material + polygon
    rings per layer, flattened to plain JSON-friendly fields - shapely geometries aren't Pydantic
    types, and a Follow `Structure` needs to diff/serialize cleanly.
    """
    follow = _require_follow()
    from pydantic import BaseModel, ConfigDict

    class LayerSpec(BaseModel):
        model_config = ConfigDict(frozen=True)
        material: str
        rings: list[dict]

    class ProcessStructure(follow.Structure):
        domain_width_nm: float
        layers: list[LayerSpec]

    return ProcessStructure(
        domain_width_nm=geometry.domain_width_nm,
        layers=[LayerSpec(material=l.material, rings=l.rings()) for l in geometry.frame_layers()],
    )


def to_steps(process_steps: list[ProcessStep]) -> list[Any]:
    """`process_steps` as a `follow.Step` protocol list (one Follow step per process step,
    `order` = its 1-based position) - so the recipe that produced a structure becomes the
    Follow experiment's protocol, including `ChemicalStep`s that never touched the geometry.
    """
    follow = _require_follow()

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
