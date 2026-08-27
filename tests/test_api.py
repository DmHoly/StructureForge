import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from structureforge.api.app import create_app  # noqa: E402


@pytest.fixture
def client(tmp_path):
    return TestClient(create_app(recipes_file=tmp_path / "recipes.json"))


def test_list_materials(client):
    response = client.get("/api/materials")
    assert response.status_code == 200
    names = {m["name"] for m in response.json()}
    assert "SiO2" in names and "Si" in names


def test_list_recipes(client):
    response = client.get("/api/recipes")
    assert response.status_code == 200
    body = response.json()
    assert {r["name"] for r in body["etch"]} >= {"Dry Oxide Etch", "Anisotropic RIE"}
    assert all(r["is_custom"] is False for r in body["etch"])  # nothing custom saved yet


def test_add_list_and_delete_a_custom_deposition_recipe(client):
    recipe = {"name": "Mon depot maison", "mode": "conformal", "angle_deg": 0.0, "notes": "test"}

    response = client.post("/api/recipes/deposition", json=recipe)
    assert response.status_code == 200
    deposition = {r["name"]: r for r in response.json()["deposition"]}
    assert deposition["Mon depot maison"]["is_custom"] is True

    listed = client.get("/api/recipes").json()
    assert any(r["name"] == "Mon depot maison" for r in listed["deposition"])

    response = client.delete("/api/recipes/deposition/Mon depot maison")
    assert response.status_code == 200
    assert not any(r["name"] == "Mon depot maison" for r in response.json()["deposition"])


def test_custom_recipe_persists_across_app_instances(tmp_path):
    recipes_file = tmp_path / "recipes.json"
    first = TestClient(create_app(recipes_file=recipes_file))
    first.post(
        "/api/recipes/etch",
        json={"name": "Ma gravure maison", "mode": "isotropic", "default_factor": 0.42},
    )

    second = TestClient(create_app(recipes_file=recipes_file))
    listed = second.get("/api/recipes").json()
    custom = next(r for r in listed["etch"] if r["name"] == "Ma gravure maison")
    assert custom["default_factor"] == 0.42
    assert custom["is_custom"] is True


def test_custom_recipe_can_override_a_built_in_by_name(client):
    client.post(
        "/api/recipes/deposition",
        json={"name": "ALD Conformal", "mode": "conformal", "notes": "overridden"},
    )

    listed = client.get("/api/recipes").json()
    ald = next(r for r in listed["deposition"] if r["name"] == "ALD Conformal")
    assert ald["notes"] == "overridden"
    assert ald["is_custom"] is True


def test_simulate_end_to_end(client):
    body = {
        "substrate": {"material": "Si", "domain_width": {"value": 200, "unit": "nm"}, "thickness": {"value": 50, "unit": "nm"}},
        "steps": [
            {"kind": "deposition", "name": "Oxyde", "material": "SiO2", "recipe": "CVD Conformal", "thickness": {"value": 30, "unit": "nm"}},
            {"kind": "lithography", "name": "Masque", "resist_material": "Photoresist", "thickness": {"value": 5, "unit": "nm"}, "openings": [[80, 120]]},
            {"kind": "etch", "name": "Gravure", "recipe": "Anisotropic RIE", "depth": {"value": 40, "unit": "nm"}},
            {"kind": "resist_strip", "name": "Retrait"},
        ],
    }
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 200
    data = response.json()
    assert len(data["frames"]) == 5  # initial + 4 steps
    last_materials = {layer["material"] for layer in data["frames"][-1]["layers"]}
    assert last_materials == {"Si", "SiO2"}  # resist stripped away


def test_simulate_unknown_material_is_a_422(client):
    body = {
        "substrate": {"material": "Unobtainium", "domain_width": {"value": 100, "unit": "nm"}, "thickness": {"value": 10, "unit": "nm"}},
        "steps": [],
    }
    response = client.post("/api/simulate", json=body)
    assert response.status_code == 422
