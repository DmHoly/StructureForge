import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from structureforge.api.app import create_app  # noqa: E402


@pytest.fixture
def client():
    return TestClient(create_app())


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
