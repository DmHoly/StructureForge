"""An etch on the wafer's backside, without a `Flip`."""

from structureforge import (
    Deposition,
    DepositionSide,
    Etch,
    Flip,
    Length,
    Lithography,
    SimulationCache,
    default_library,
    default_recipes,
    simulate,
)
from structureforge.geometry.engine import Geometry

_RIE = "Anisotropic RIE"
_OXIDE = "Dry Oxide Etch"


def _nm(v):
    return Length(value=v, unit="nm")


def _run(steps, cache=None):
    if cache is not None:
        return cache.simulate("GaN", 1000.0, 500.0, steps, default_library(), default_recipes())
    return simulate(Geometry.substrate("GaN", 1000.0, 500.0), steps, default_library(), default_recipes())


def _dep_recipe():
    return next(iter(default_recipes().deposition))


def _layer(frame, material):
    return [l for l in frame.layers if l.material == material]


def _bounds(frame):
    from shapely.ops import unary_union

    return unary_union([l.polygon for l in frame.layers if not l.polygon.is_empty]).bounds


def test_front_is_the_default_and_serializes_as_before():
    step = Etch(name="e", recipe=_RIE, depth=_nm(10))
    assert step.side is DepositionSide.front
    assert "side" not in step.model_dump(mode="json")
    old = {"kind": "etch", "name": "e", "recipe": _RIE, "depth": {"value": 10, "unit": "nm"}}
    assert Etch.model_validate(old).side is DepositionSide.front


def test_back_and_both_round_trip():
    for side in ("back", "both"):
        step = Etch(name="e", recipe=_RIE, depth=_nm(10), side=side)
        data = step.model_dump(mode="json")
        assert data["side"] == side
        assert Etch.model_validate(data) == step


def test_backside_etch_thins_the_substrate_from_below():
    frames = _run([Etch(name="e", recipe=_RIE, depth=_nm(50), side="back")])
    x0, y0, x1, y1 = _bounds(frames[-1])
    assert (x0, x1) == (0.0, 1000.0)
    assert abs(y1 - 0.0) < 1e-6  # the front is untouched
    assert abs(y0 - -450.0) < 0.5


def test_backside_etch_removes_a_backside_film_and_leaves_the_front_pattern():
    oxide = Deposition(name="ox", material="SiO2", recipe=_dep_recipe(), thickness=_nm(30), side="both")
    litho = Lithography(name="l", resist_material="Photoresist", thickness=_nm(100), openings=[(400.0, 600.0)])
    front = Etch(name="front", recipe=_OXIDE, depth=_nm(30))
    back = Etch(name="back", recipe=_OXIDE, depth=_nm(30), side="back")
    frames = _run([oxide, litho, front, back])
    before, after = frames[-2], frames[-1]
    # the front (resist + opened oxide) comes out as it went in
    for material in ("Photoresist",):
        (a,) = [l.polygon for l in _layer(before, material)]
        (b,) = [l.polygon for l in _layer(after, material)]
        assert a.symmetric_difference(b).area < 1.0
    # the backside oxide is gone, the substrate under it barely touched (oxide etch, GaN at 0.8x
    # but the depth is the oxide's own thickness)
    oxide_after = sum(l.polygon.area for l in _layer(after, "SiO2"))
    oxide_before = sum(l.polygon.area for l in _layer(before, "SiO2"))
    assert oxide_after < oxide_before - 1000.0 * 29.0


def test_both_sides_etch_front_then_back():
    front = Etch(name="e", recipe=_RIE, depth=_nm(20))
    both = front.model_copy(update={"side": DepositionSide.both})
    _, y0f, _, y1f = _bounds(_run([front])[-1])
    _, y0b, _, y1b = _bounds(_run([both])[-1])
    assert abs(y1f - y1b) < 1e-6 and abs(y1b - -20.0) < 0.5
    assert abs(y0f - -500.0) < 1e-6 and abs(y0b - -480.0) < 0.5


def test_a_later_front_etch_never_reaches_the_backside_and_the_cache_keeps_the_floor():
    back = Etch(name="b", recipe=_RIE, depth=_nm(50), side="back")
    iso = Etch(name="i", recipe="Wet HF Dip", depth=_nm(20))
    plain = _run([back, iso])
    cache = SimulationCache()
    _run([back], cache)
    cached = _run([back, iso], cache)  # restored from the first run's last frame
    for frames in (plain, cached):
        _, y0, _, _ = _bounds(frames[-1])
        assert abs(y0 - -450.0) < 0.5
    assert abs(_bounds(plain[-1])[3] - _bounds(cached[-1])[3]) < 1e-6


def test_flip_after_a_backside_etch():
    frames = _run([Etch(name="b", recipe=_RIE, depth=_nm(50), side="back"), Flip(name="f")])
    _, y0, _, y1 = _bounds(frames[-1])
    assert abs((y1 - y0) - 450.0) < 0.5
