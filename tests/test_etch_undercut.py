"""Selective isotropic etch under a mask, and growth into the cavity it leaves: the cases behind
the ripples, stalled undercuts and half-filled cavities seen on SAG nanowire structures."""

import math

import numpy as np
import pytest
from shapely.geometry import Polygon, box

from structureforge.core.materials import default_library
from structureforge.core.recipes import EtchMode, EtchRecipe, default_recipes
from structureforge.geometry.engine import Geometry, Layer, _has_interior


def _oxide_selective_wet_etch(default_factor: float = 0.05) -> EtchRecipe:
    """A BOE-like isotropic etch: oxide at the nominal rate, everything else slowly or not at all."""
    return EtchRecipe(
        name="oxide wet etch",
        mode=EtchMode.isotropic,
        selectivity_by_material={"SiO2": 1.0},
        default_factor=default_factor,
    )


def _sag_mask_stack(default_factor: float = 0.05) -> Geometry:
    """GaN wafer, 50 nm oxide, 80 nm nitride, one 100 nm opening etched dry through the nitride
    and then wet through the oxide (51 nm nominal: just past breakthrough), resist stripped."""
    materials, recipes = default_library(), default_recipes()
    g = Geometry.substrate("GaN", domain_width_nm=500, thickness_nm=50)
    g.deposit_conformal("SiO2", 50)
    g.deposit_conformal("Si3N4", 80)
    g.deposit_conformal_masked("Photoresist", 100, open_x_ranges=[(200, 300)])
    g.etch(recipes.get_etch("Anisotropic RIE"), 83, materials)
    g.etch(_oxide_selective_wet_etch(default_factor), 51, materials)
    g.strip_material("Photoresist")
    return g


def _layer(g: Geometry, material: str):
    return next(l for l in g.layers if l.material == material)


def test_isotropic_etch_keeps_undercutting_after_it_reaches_the_substrate(materials):
    """Reaching a slower-etching layer underneath must not stop the faster one beside it. The
    oxide under the mask used to freeze the moment the opening's floor hit the substrate (the
    substrate was padded "infinitely" upward, swallowing the whole domain)."""
    g = Geometry.substrate("Si", domain_width_nm=200, thickness_nm=20)
    g.deposit_conformal("SiO2", 20)
    g.deposit_conformal_masked("Photoresist", 5, open_x_ranges=[(90, 110)])
    g.etch(default_recipes().get_etch("Wet HF Dip"), 40, materials)

    oxide = _layer(g, "SiO2").polygon
    # 40 nm of reach from the mask corners (90, 20) / (110, 20) clears the floor out to x ~ 55 / 145.
    assert oxide.intersection(box(62, 0, 138, 2)).is_empty
    assert oxide.intersection(box(0, 0, 50, 20)).area == pytest.approx(50 * 20, abs=1.0)
    # the substrate, at 0.05x, loses only the ~1 nm it is exposed to after breakthrough
    assert 200 * 20 - 100 < _layer(g, "Si").polygon.area < 200 * 20


def test_isotropic_etch_leaves_the_domain_edges_alone(materials):
    """The domain's left/right edges are symmetry boundaries. A vertex a rounding error off x=0
    used to break the mirror union, turning the edge into an "exposed" surface eaten by every
    substep - and making the substrate look exposed, which guarded the whole domain."""
    g = _sag_mask_stack()
    for material in ("GaN", "SiO2", "Si3N4"):
        poly = _layer(g, material).polygon
        assert poly.bounds[0] == 0.0, material
        assert poly.bounds[2] == 500.0, material
    # the substrate's top is flat under the mask, right up to the edge
    gan = _layer(g, "GaN").polygon
    assert gan.intersection(box(0, -1, 150, 1)).area == pytest.approx(150, abs=0.01)


def test_isotropic_etch_undercut_is_round_and_symmetric():
    g = _sag_mask_stack()
    oxide = _layer(g, "SiO2").polygon
    parts = sorted(oxide.geoms, key=lambda p: p.bounds[0])
    assert len(parts) == 2
    left, right = parts
    # mirror-symmetric about the opening's centre, to well under a nanometre
    assert left.bounds[2] == pytest.approx(500 - right.bounds[0], abs=0.6)
    # the undercut front is one smooth curve: x recedes monotonically with height and, walking
    # up it, consecutive edges turn gently and always the same way - no zigzag, no scallops
    front = sorted((y, x) for x, y in left.exterior.coords if 149 < x < 200 and 0.5 < y < 49.5)
    assert len(front) >= 6
    xs = [x for _, x in front]
    assert all(b <= a + 1e-6 for a, b in zip(xs, xs[1:]))
    pts = [(x, y) for y, x in front]
    for (ax, ay), (bx, by), (cx, cy) in zip(pts, pts[1:], pts[2:]):
        e1, e2 = (bx - ax, by - ay), (cx - bx, cy - by)
        turn = math.degrees(math.atan2(e1[0] * e2[1] - e1[1] * e2[0], e1[0] * e2[0] + e1[1] * e2[1]))
        assert -30 < turn <= 1e-6


def test_isotropic_etch_smooths_the_staircase_on_the_uncovered_mask_underside():
    """The nitride underside is uncovered a little further by every substep and etched (slowly)
    from then on: time-stepping draws it as a staircase / row of scallops of one step's height.
    It must come out as one straight slope."""
    g = _sag_mask_stack(default_factor=0.05)
    nitride = _layer(g, "Si3N4").polygon
    left = min(nitride.geoms, key=lambda p: p.bounds[0])
    underside = sorted((x, y) for x, y in left.exterior.coords if 145 < x < 199 and y < 60)
    assert len(underside) >= 2
    (x0, y0), (x1, y1) = underside[0], underside[-1]
    assert y1 > y0  # eaten more near the opening, where it was uncovered first
    for x, y in underside[1:-1]:
        on_line = y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        assert abs(y - on_line) < 0.05
    assert len(underside) <= 4


def test_a_thin_mask_still_shields_the_oxide_under_it(materials):
    """Reaching round a mask is fine, through it is not: a resist thinner than one substep's
    reach must still protect the oxide beneath it."""
    g = Geometry.substrate("Si", domain_width_nm=200, thickness_nm=20)
    g.deposit_conformal("SiO2", 20)
    g.deposit_conformal_masked("Photoresist", 0.5, open_x_ranges=[(90, 110)])
    g.etch(_oxide_selective_wet_etch(default_factor=0.0), 10, materials, steps=2)  # 5 nm substeps
    oxide = _layer(g, "SiO2").polygon
    assert oxide.intersection(box(0, 0, 70, 20)).area == pytest.approx(70 * 20, abs=1.0)
    assert oxide.intersection(box(130, 0, 200, 20)).area == pytest.approx(70 * 20, abs=1.0)


def test_faceted_growth_fills_the_cavity_it_seals_under_the_mask():
    """Once the crystal overflows the opening and touches the nitride, the undercut beneath is
    an enclosed void. It used to be treated as a hole in the solid - filled by `solid()`,
    ignored by the facet offset - so nothing ever grew in it and it stayed empty, bounded by
    whatever ragged wall the growth had when it sealed. Its walls must keep growing until the
    void is full."""
    g = _sag_mask_stack()
    g.deposit_faceted("GaN", 400, rate_c=1, rate_m=0.3, rate_sp=0.7, semi_polar_angle_deg=45, seed_materials=["GaN"])
    solid = g.solid()
    assert not _has_interior(solid)
    grown = g.layers[-1].polygon
    cavity = box(170, 15, 195, 45)  # well inside the undercut, left of the opening
    assert grown.intersection(cavity).area == pytest.approx(cavity.area, rel=0.01)
    assert solid.intersection(box(150, 0, 350, 52)).area == pytest.approx(200 * 52, rel=0.01)


def test_faceted_growth_in_a_cavity_stops_at_what_it_can_reach():
    """With too little lateral growth to fill it, the void stays - as air, not as material."""
    g = _sag_mask_stack()
    g.deposit_faceted("GaN", 200, rate_c=1, rate_m=0.1, rate_sp=0.7, semi_polar_angle_deg=45, seed_materials=["GaN"])
    solid = g.solid()
    assert _has_interior(solid)
    void = max((Polygon(ring) for ring in solid.interiors), key=lambda p: p.area)
    assert void.area > 100
    # the cavity wall is one straight m-plane, not stepped
    grown = g.layers[-1].polygon
    wall = [(x, y) for x, y in grown.exterior.coords if x < 198 and 15 < y < 48]  # above the oxide toe
    xs = [x for x, _ in wall]
    assert max(xs) - min(xs) < 0.5


def test_conformal_deposit_does_not_coat_a_sealed_void():
    g = _sag_mask_stack()
    g.deposit_faceted("GaN", 200, rate_c=1, rate_m=0.1, rate_sp=0.7, semi_polar_angle_deg=45, seed_materials=["GaN"])
    void_before = max((Polygon(r) for r in g.solid().interiors), key=lambda p: p.area)
    g.deposit_conformal("Al2O3", 5)
    film = g.layers[-1].polygon
    assert film.intersection(void_before).is_empty
    assert film.area > 0


def test_isotropic_etch_does_not_reach_into_a_sealed_void(materials):
    g = _sag_mask_stack()
    g.deposit_faceted("GaN", 200, rate_c=1, rate_m=0.1, rate_sp=0.7, semi_polar_angle_deg=45, seed_materials=["GaN"])
    oxide_before = _layer(g, "SiO2").polygon.area
    g.etch(_oxide_selective_wet_etch(default_factor=0.0), 10, materials)
    assert _layer(g, "SiO2").polygon.area == pytest.approx(oxide_before, abs=0.5)


def test_faceted_growth_first_strip_over_a_shallow_floor_ramp_is_flat():
    """An etched floor can end in a shallow ramp (a few angstroms over ~20 nm, from the etch
    reaching it later near the mask). The ramp grows as c-plane - but bounding the long flat
    neighbour with the ramp's own tilted line shaved a wedge off its far end."""
    g = Geometry.substrate("GaN", domain_width_nm=400, thickness_nm=50)
    g.layers[0].polygon = Polygon([
        (0, -50), (400, -50), (400, 0), (250, 0), (250, -0.2), (168, -0.2), (150, 0), (0, 0),
    ])
    g.layers.append(Layer(material="SiO2", polygon=box(0, 0, 150, 50)))
    g.layers.append(Layer(material="SiO2", polygon=box(250, 0, 400, 50)))
    g.deposit_faceted("GaN", 10, rate_c=1, rate_m=0.1, rate_sp=0.7, semi_polar_angle_deg=45, seed_materials=["GaN"])
    film = g.layers[-1].polygon
    assert film.bounds[3] < 10.1
    # the top stays within 0.1 nm of 9.8 (floor at -0.2 plus 10) over the flat part of the floor
    assert film.intersection(box(170, 9.7, 245, 11)).area == pytest.approx(75 * 0.1, abs=1.5)


def test_faceted_growth_vicinal_sidewall_bounds_the_facet_nucleating_above_it():
    """A sidewall a fraction of a degree off vertical grows as m-plane, so it must also stop
    the semi-polar strip nucleating at its top corner, exactly as a true m-plane would - or
    that strip pokes several nm past the sidewall and leaves a step once the wall catches up."""
    g = Geometry.substrate("GaN", domain_width_nm=400, thickness_nm=50)
    tilt = 20 * math.tan(math.radians(0.6))
    g.layers.append(Layer(material="GaN", polygon=Polygon([(150, 0), (250, 0), (250, 20), (150 + tilt, 20)])))
    g.layers.append(Layer(material="SiO2", polygon=box(0, 0, 150, 2)))
    g.layers.append(Layer(material="SiO2", polygon=box(250, 0, 400, 2)))
    g.deposit_faceted("GaN", 10, rate_c=1, rate_m=0.1, rate_sp=0.7, semi_polar_angle_deg=45, seed_materials=["GaN"])
    film = g.layers[-1].polygon
    # 1 nm of m-plane advance, nothing poking further out (the strip used to reach ~4 nm past)
    assert film.bounds[0] > 150 - 1.0 - 0.3
    assert film.bounds[2] < 250 + 1.0 + 0.3
