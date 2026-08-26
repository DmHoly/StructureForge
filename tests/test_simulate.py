import pytest

from structureforge.core.units import Length
from structureforge.geometry.engine import Geometry
from structureforge.process.simulate import SimulationError, simulate
from structureforge.process.steps import ChemicalStep, Deposition, Etch, ResistStrip


def _flow():
    return [
        Deposition(name="Oxyde", material="SiO2", recipe="CVD Conformal", thickness=Length.nm(20)),
        ChemicalStep(name="Nettoyage", description="HF dip"),
        Etch(name="Gravure", recipe="Anisotropic RIE", depth=Length.nm(10)),
    ]


def test_simulate_returns_one_frame_per_step_plus_the_initial_one(materials, recipes):
    geometry = Geometry.substrate("Si", domain_width_nm=100, thickness_nm=30)
    frames = simulate(geometry, _flow(), materials, recipes)

    assert len(frames) == len(_flow()) + 1
    assert frames[0].step_kind == "initial"
    assert [f.step_kind for f in frames[1:]] == ["deposition", "chemical", "etch"]


def test_frames_are_independent_snapshots_not_aliases_of_the_live_geometry(materials, recipes):
    """Regression test: Frame used to store references to the live, mutable Layer objects, so
    every earlier frame silently ended up showing the *final* geometry once anything called
    .rings() on it (which normally happens only when serialising the whole list at the end).
    """
    geometry = Geometry.substrate("Si", domain_width_nm=100, thickness_nm=30)
    frames = simulate(geometry, _flow(), materials, recipes)

    initial_frame = frames[0]
    assert len(initial_frame.layers) == 1
    assert initial_frame.layers[0].polygon.area == pytest.approx(100 * 30)
    # the final frame has more material and a trench - the initial frame must not have changed
    final_frame = frames[-1]
    assert sum(l.polygon.area for l in final_frame.layers) != sum(l.polygon.area for l in initial_frame.layers)
    assert initial_frame.layers[0].polygon.area == pytest.approx(100 * 30)


def test_simulation_error_names_the_failing_step(materials, recipes):
    geometry = Geometry.substrate("Si", domain_width_nm=100, thickness_nm=30)
    bad_flow = [Deposition(name="Oups", material="Unobtainium", recipe="CVD Conformal", thickness=Length.nm(5))]

    with pytest.raises(SimulationError) as excinfo:
        simulate(geometry, bad_flow, materials, recipes)

    assert excinfo.value.step_index == 1
    assert "Oups" in str(excinfo.value)


def test_resist_strip_triggers_lift_off_within_simulate(materials, recipes):
    from structureforge.process.steps import Deposition as Dep
    from structureforge.process.steps import Lithography

    geometry = Geometry.substrate("Si", domain_width_nm=100, thickness_nm=10)
    flow = [
        Lithography(name="Masque", resist_material="Photoresist", thickness=Length.nm(20), openings=[(40, 60)]),
        Dep(name="Metal", material="Al", recipe="Evaporation (normal)", thickness=Length.nm(10)),
        ResistStrip(name="Lift-off"),
    ]
    frames = simulate(geometry, flow, materials, recipes)
    final_layers = {l.material: l for l in frames[-1].layers}

    assert "Photoresist" not in final_layers
    assert final_layers["Al"].polygon.area == pytest.approx(20 * 10, abs=1.0)
