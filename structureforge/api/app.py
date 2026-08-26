"""The web GUI's backend: serves the materials/recipe libraries so the frontend can populate its
pickers, runs `/api/simulate` (a `ProcessStep` list in, one rendered `Frame` per step out), and
mounts the static frontend. Stateless - the frontend keeps the step list and resubmits the whole
thing each time, so there's no server-side session to manage for a v1 single-user tool.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..core.materials import MaterialLibrary, default_library
from ..core.recipes import RecipeLibrary, default_recipes
from ..core.units import Length
from ..geometry.engine import Geometry
from ..process.simulate import SimulationError, simulate
from ..process.steps import ProcessStep

STATIC_DIR = Path(__file__).parent / "static"


class SubstrateSpec(BaseModel):
    material: str
    domain_width: Length
    thickness: Length


class SimulateRequest(BaseModel):
    substrate: SubstrateSpec
    steps: list[ProcessStep]


class SimulateResponse(BaseModel):
    frames: list[dict]
    material_colors: dict[str, str]


def create_app(materials: MaterialLibrary | None = None, recipes: RecipeLibrary | None = None) -> FastAPI:
    materials = materials or default_library()
    recipes = recipes or default_recipes()

    app = FastAPI(title="StructureForge", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/materials")
    def list_materials() -> list[dict]:
        return [m.model_dump(mode="json") for m in materials]

    @app.get("/api/recipes")
    def list_recipes() -> dict[str, list[dict]]:
        return {
            "deposition": [r.model_dump(mode="json") for r in recipes.deposition.values()],
            "etch": [r.model_dump(mode="json") for r in recipes.etch.values()],
        }

    @app.post("/api/simulate")
    def run_simulation(request: SimulateRequest) -> SimulateResponse:
        try:
            materials.get(request.substrate.material)
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        geometry = Geometry.substrate(
            request.substrate.material,
            request.substrate.domain_width.to_nm(),
            request.substrate.thickness.to_nm(),
        )
        try:
            frames = simulate(geometry, request.steps, materials, recipes)
        except SimulationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        used_materials = {m.name: m.color for m in materials}
        return SimulateResponse(
            frames=[f.to_dict() for f in frames],
            material_colors=used_materials,
        )

    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return app


app = create_app()
