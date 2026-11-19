"""A deposition on the wafer's backside, without a `Flip`."""

from structureforge import (
    Deposition,
    DepositionSide,
    Flip,
    Length,
    SimulationCache,
    default_library,
    default_recipes,
    simulate,
)
from structureforge.geometry.engine import Geometry


def _run(steps, cache=None):
    if cache is not None:
        return cache.simulate("GaN", 1000.0, 500.0, steps, default_library(), default_recipes())
    return simulate(Geometry.substrate("GaN", 1000.0, 500.0), steps, default_library(), default_recipes())


def _recipe():
    return next(iter(default_recipes().deposition))


def _layer(frame, material):
    return [l for l in frame.layers if l.material == material]


def test_front_is_the_default_and_serializes_as_before():
    step = Deposition(name="d", material="SiO2", recipe=_recipe(), thickness=Length(value=10, unit="nm"))
    assert step.side is DepositionSide.front
    assert "side" not in step.model_dump(mode="json")
    old = {"kind": "deposition", "name": "d", "material": "SiO2", "recipe": _recipe(), "thickness": {"value": 10, "unit": "nm"}}
    assert Deposition.model_validate(old).side is DepositionSide.front


def test_back_and_both_round_trip():
    for side in ("back", "both"):
        step = Deposition(name="d", material="SiO2", recipe=_recipe(), thickness=Length(value=10, unit="nm"), side=side)
        data = step.model_dump(mode="json")
        assert data["side"] == side
        assert Deposition.model_validate(data) == step


def test_backside_film_lies_under_the_substrate():
    step = Deposition(name="d", material="SiO2", recipe=_recipe(), thickness=Length(value=20, unit="nm"), side="back")
    frames = _run([step])
    (film,) = _layer(frames[-1], "SiO2")
    x0, y0, x1, y1 = film.polygon.bounds
    assert (x0, x1) == (0.0, 1000.0)
    assert abs(y1 - -500.0) < 1e-9 and abs(y0 - -520.0) < 1e-9


def test_both_sides_and_front_unchanged():
    front = Deposition(name="d", material="SiO2", recipe=_recipe(), thickness=Length(value=20, unit="nm"))
    both = front.model_copy(update={"side": DepositionSide.both})
    front_film = _layer(_run([front])[-1], "SiO2")
    both_films = _layer(_run([both])[-1], "SiO2")
    assert len(front_film) == 1 and len(both_films) == 2
    assert both_films[0].polygon.equals(front_film[0].polygon)
    assert both_films[1].polygon.bounds[3] <= -500.0 + 1e-9


def test_later_front_deposition_does_not_wrap_the_backside_film():
    back = Deposition(name="b", material="SiO2", recipe=_recipe(), thickness=Length(value=20, unit="um"), side="back")
    front = Deposition(name="f", material="Si3N4", recipe=_recipe(), thickness=Length(value=50, unit="nm"))
    (film,) = _layer(_run([back, front])[-1], "Si3N4")
    assert film.polygon.bounds[1] >= -1e-6


def test_flip_brings_the_backside_film_on_top():
    back = Deposition(name="b", material="SiO2", recipe=_recipe(), thickness=Length(value=20, unit="nm"), side="back")
    frames = _run([back, Flip(name="flip")])
    (film,) = _layer(frames[-1], "SiO2")
    (gan,) = _layer(frames[-1], "GaN")
    assert film.polygon.bounds[1] >= gan.polygon.bounds[3] - 1e-6


def test_cache_resumes_after_a_backside_deposition():
    back = Deposition(name="b", material="SiO2", recipe=_recipe(), thickness=Length(value=20, unit="um"), side="back")
    front = Deposition(name="f", material="Si3N4", recipe=_recipe(), thickness=Length(value=50, unit="nm"))
    cache = SimulationCache()
    _run([back], cache)
    (film,) = _layer(_run([back, front], cache)[-1], "Si3N4")
    assert film.polygon.bounds[1] >= -1e-6


def test_selective_backside_only_on_a_seed_bottom():
    step = Deposition(name="d", material="SiO2", recipe=_recipe(), thickness=Length(value=20, unit="nm"), side="back", seed_materials=["Si"])
    assert not _layer(_run([step])[-1], "SiO2")
    step = step.model_copy(update={"seed_materials": ["GaN"]})
    assert _layer(_run([step])[-1], "SiO2")
