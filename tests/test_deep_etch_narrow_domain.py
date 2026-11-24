"""A directional etch much deeper than the domain is wide - a micrometre-deep etch back on a
single-wire domain - used to stop with GEOS's "Clipping rectangle must be non-empty": the x-range
the thin-part probe measures in, shrunk by the etch's reach on each side, came out inverted."""

from shapely.geometry import LineString

from structureforge import Deposition, Etch, Geometry, Length, Lithography, default_library, default_recipes, simulate


def _oxide_top(frame, x: float) -> float:
    oxide = next(l for l in frame.layers if l.material == "SiO2").polygon
    return oxide.intersection(LineString([(x, -1e5), (x, 1e5)])).bounds[3]


def test_etch_far_deeper_than_the_domain_is_wide():
    materials, recipes = default_library(), default_recipes()
    start = Geometry.substrate("Si", domain_width_nm=200, thickness_nm=50)
    steps = [
        Deposition(name="oxide", material="SiO2", recipe="CVD Conformal", thickness=Length.nm(8000)),
        Lithography(name="mask", resist_material="Photoresist", thickness=Length.nm(500), openings=[(60, 140)]),
        Etch(name="etch back", recipe="Anisotropic RIE", depth=Length(value=7.5, unit="um")),
    ]
    final = simulate(start, steps, materials, recipes)[-1]
    # in the opening: 8000 - 7500 nm of oxide left; under the mask (resist at 0.1x, gone after
    # 5000 nm of nominal etch): about 8000 - 2500 nm, give or take a substep of the resist's end
    assert abs(_oxide_top(final, 100) - 500) < 1
    assert abs(_oxide_top(final, 20) - 5500) < 100


def test_measure_window_never_inverted():
    from structureforge.geometry.engine import _measure_window

    assert _measure_window(-50, 250, 10, 200) == (-40, 240)  # a usual reach: shrunk as before
    assert _measure_window(-200, 400, 7500, 200) == (0.0, 200)  # past half the padded width: the domain
