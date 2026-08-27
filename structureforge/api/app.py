"""The web GUI's backend: serves the materials/recipe libraries so the frontend can populate its
pickers, runs `/api/simulate` (a `ProcessStep` list in, one rendered `Frame` per step out), and
mounts the static frontend. The step list/substrate is stateless (the frontend resubmits it each
time), but recipes are not: `/api/recipes/*` reads and writes a `RecipeStore` file on disk, so a
custom recipe added from the GUI's recipe manager survives past one process and is reusable in
later runs - see `structureforge.core.recipe_store`.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..core.materials import MaterialLibrary, default_library
from ..core.recipe_store import RecipeStore
from ..core.recipes import DepositionRecipe, EtchRecipe, RecipeLibrary, default_recipes
from ..core.units import Length
from ..geometry.engine import Geometry
from ..process.simulate import SimulationError, simulate
from ..process.steps import ProcessStep

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_RECIPES_FILE = Path("structureforge_recipes.json")


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


class ExportFollowRequest(BaseModel):
    substrate: SubstrateSpec
    steps: list[ProcessStep]
    repo_path: str
    branch: str = "main"
    title: str
    intent: str


class ExportFollowResponse(BaseModel):
    experiment_id: str
    branch: str
    title: str


def create_app(
    materials: MaterialLibrary | None = None,
    base_recipes: RecipeLibrary | None = None,
    recipes_file: str | Path | None = None,
) -> FastAPI:
    materials = materials or default_library()
    base_recipes = base_recipes or default_recipes()
    recipe_store = RecipeStore(recipes_file or DEFAULT_RECIPES_FILE)

    def current_recipes() -> RecipeLibrary:
        """The base library with every custom (persisted) recipe layered on top - recomputed on
        every call, since the custom set can change at any time via `/api/recipes/*`.
        """
        return recipe_store.combined_with(base_recipes)

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
        custom = recipe_store.load()
        recipes = current_recipes()
        return {
            "deposition": [
                {**r.model_dump(mode="json"), "is_custom": r.name in custom.deposition}
                for r in recipes.deposition.values()
            ],
            "etch": [
                {**r.model_dump(mode="json"), "is_custom": r.name in custom.etch}
                for r in recipes.etch.values()
            ],
        }

    @app.post("/api/recipes/deposition")
    def upsert_deposition_recipe(recipe: DepositionRecipe) -> dict[str, list[dict]]:
        recipe_store.upsert_deposition(recipe)
        return list_recipes()

    @app.delete("/api/recipes/deposition/{name}")
    def delete_deposition_recipe(name: str) -> dict[str, list[dict]]:
        recipe_store.remove_deposition(name)
        return list_recipes()

    @app.post("/api/recipes/etch")
    def upsert_etch_recipe(recipe: EtchRecipe) -> dict[str, list[dict]]:
        recipe_store.upsert_etch(recipe)
        return list_recipes()

    @app.delete("/api/recipes/etch/{name}")
    def delete_etch_recipe(name: str) -> dict[str, list[dict]]:
        recipe_store.remove_etch(name)
        return list_recipes()

    def _build_and_simulate(substrate: SubstrateSpec, steps: list[ProcessStep]) -> tuple[Geometry, list]:
        try:
            materials.get(substrate.material)
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        geometry = Geometry.substrate(substrate.material, substrate.domain_width.to_nm(), substrate.thickness.to_nm())
        try:
            frames = simulate(geometry, steps, materials, current_recipes())
        except SimulationError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return geometry, frames

    @app.post("/api/simulate")
    def run_simulation(request: SimulateRequest) -> SimulateResponse:
        _geometry, frames = _build_and_simulate(request.substrate, request.steps)
        used_materials = {m.name: m.color for m in materials}
        return SimulateResponse(
            frames=[f.to_dict() for f in frames],
            material_colors=used_materials,
        )

    @app.post("/api/export_follow")
    def export_follow(request: ExportFollowRequest) -> ExportFollowResponse:
        try:
            import follow

            from ..adapters import follow_adapter
        except ImportError as exc:
            raise HTTPException(
                status_code=501,
                detail=(
                    "l'export vers Follow necessite la dependance optionnelle 'follow' - "
                    "installez-la avec `pip install structureforge[follow]`"
                ),
            ) from exc

        geometry, _frames = _build_and_simulate(request.substrate, request.steps)
        repo = follow.Repository(request.repo_path)
        try:
            experiment = follow_adapter.export_experiment(
                repo,
                geometry,
                request.steps,
                branch=request.branch,
                title=request.title,
                intent=request.intent,
            )
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return ExportFollowResponse(experiment_id=experiment.id, branch=experiment.branch, title=experiment.title)

    if STATIC_DIR.exists():
        app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")

    return app


app = create_app(recipes_file=os.environ.get("STRUCTUREFORGE_RECIPES_FILE") or None)
