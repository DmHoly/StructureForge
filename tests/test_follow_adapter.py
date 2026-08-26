import pytest

follow = pytest.importorskip("follow")

from structureforge.adapters import follow_adapter  # noqa: E402
from structureforge.core.units import Length  # noqa: E402
from structureforge.geometry.engine import Geometry  # noqa: E402
from structureforge.process.steps import ChemicalStep, Deposition, Etch  # noqa: E402


def _flow():
    return [
        Deposition(name="Oxyde", material="SiO2", recipe="CVD Conformal", thickness=Length.nm(20)),
        ChemicalStep(name="Nettoyage", description="HF dip"),
        Etch(name="Gravure", recipe="Anisotropic RIE", depth=Length.nm(10)),
    ]


def test_registry_key_is_stable_and_module_level():
    """A class defined inside a function gets a `<locals>` qualname that follow.Structure's own
    docstring warns breaks dotted-path resolution and re-registers a fresh entry every call.
    """
    key = follow_adapter.ProcessStructure.registry_key()
    assert key == "structureforge.adapters.follow_adapter.ProcessStructure"
    assert "<locals>" not in key
    assert follow.Structure.resolve(key) is follow_adapter.ProcessStructure


def test_to_structure_captures_the_layer_stack(materials, recipes):
    geometry = Geometry.substrate("Si", domain_width_nm=100, thickness_nm=20)
    geometry.deposit_conformal("SiO2", 10)

    structure = follow_adapter.to_structure(geometry)

    assert structure.domain_width_nm == 100.0
    assert [l.material for l in structure.layers] == ["Si", "SiO2"]
    assert structure.layers[1].rings  # at least one polygon ring recorded


def test_to_steps_preserves_order_and_parameters():
    steps = follow_adapter.to_steps(_flow())

    assert [s.order for s in steps] == [1, 2, 3]
    assert steps[0].parameters["thickness"] == follow.Quantity(value=20.0, unit="nm")
    assert steps[1].description == "HF dip"
    assert steps[2].parameters["depth"] == follow.Quantity(value=10.0, unit="nm")


def test_export_experiment_commits_and_round_trips(tmp_path, materials, recipes):
    from structureforge.process.simulate import simulate

    geometry = Geometry.substrate("Si", domain_width_nm=100, thickness_nm=20)
    flow = _flow()
    simulate(geometry, flow, materials, recipes)

    repo = follow.Repository(tmp_path / "mon_labo")
    experiment = follow_adapter.export_experiment(
        repo, geometry, flow, branch="main", title="Essai", intent="Verifier l'export"
    )

    assert experiment.branch == "main"
    assert len(experiment.steps) == 3

    reloaded = repo.get(experiment.id)
    cls = follow.Structure.resolve(reloaded.structure_type)
    rehydrated = cls.model_validate(reloaded.structure)
    assert rehydrated.domain_width_nm == 100.0
    assert {l.material for l in rehydrated.layers} == {"Si", "SiO2"}

    # a second repository instance opened from the same path sees the same committed experiment
    reopened = follow.Repository(tmp_path / "mon_labo")
    assert [e.id for e in reopened.log("main")] == [experiment.id]
