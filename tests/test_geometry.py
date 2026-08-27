import pytest
from shapely.geometry import Point, box

from structureforge.geometry.engine import Geometry, Layer


def test_conformal_deposit_on_a_flat_substrate_is_exactly_width_times_thickness():
    g = Geometry.substrate("Si", domain_width_nm=200, thickness_nm=50)
    g.deposit_conformal("SiO2", 20)

    film = next(l for l in g.layers if l.material == "SiO2")
    assert film.polygon.area == pytest.approx(200 * 20, abs=1e-6)
    assert g.bounds() == pytest.approx((0.0, -50.0, 200.0, 20.0))


def test_wafer_floor_stops_deposition_and_etch_from_touching_the_backside():
    """A conformal deposit or isotropic etch acts on every exposed edge of the solid - without
    the floor, both would just as happily grow from/eat into the substrate's bottom edge as its
    real (top) surface.
    """
    g = Geometry.substrate("Si", domain_width_nm=100, thickness_nm=30)
    g.deposit_conformal("SiO2", 10)
    assert g.bounds()[1] == -30.0  # bottom untouched, only the top grew


def test_masked_anisotropic_etch_protects_oxide_under_a_thin_resist_mask(materials, recipes):
    """The scenario that originally exposed the "thin mask tunnelled through" bug: a resist mask
    thinner than a single substep's depth for the faster (oxide/Si) rate must still fully protect
    what's underneath it for the entire etch, not just until it gets numerically thin.
    """
    g = Geometry.substrate("Si", domain_width_nm=200, thickness_nm=50)
    g.deposit_conformal("SiO2", 30)
    g.deposit_conformal_masked("Photoresist", 5, open_x_ranges=[(80, 120)])

    recipe = recipes.get_etch("Anisotropic RIE")
    g.etch(recipe, depth_nm=40, materials=materials)

    si = next(l for l in g.layers if l.material == "Si")
    oxide = next((l for l in g.layers if l.material == "SiO2"), None)
    resist = next((l for l in g.layers if l.material == "Photoresist"), None)

    # open window (width 40) cleared 30nm of oxide + 10nm into Si; masked area (width 160)
    # fully protected; resist itself recedes at its own 0.1x rate for the full 40nm reference.
    assert si.polygon.area == pytest.approx(200 * 50 - 40 * 10, abs=1.0)
    assert oxide.polygon.area == pytest.approx(160 * 30, abs=1.0)
    assert resist.polygon.area == pytest.approx(160 * (5 - 40 * 0.1), abs=1.0)


def test_isotropic_wet_etch_undercuts_the_mask(materials, recipes):
    """Isotropic etch attacks sideways too: oxide should lose more than just the opening's own
    footprint (some undercut under the mask edges), while the mask and substrate stay ~intact
    (Wet HF Dip is highly selective to oxide).
    """
    g = Geometry.substrate("Si", domain_width_nm=200, thickness_nm=20)
    g.deposit_conformal("SiO2", 20)
    g.deposit_conformal_masked("Photoresist", 5, open_x_ranges=[(90, 110)])

    recipe = recipes.get_etch("Wet HF Dip")
    g.etch(recipe, depth_nm=20, materials=materials)

    si = next(l for l in g.layers if l.material == "Si")
    oxide = next((l for l in g.layers if l.material == "SiO2"), None)

    opening_only_area = 200 * 20 - 20 * 20  # oxide area if only the 20nm-wide opening cleared
    assert oxide.polygon.area < opening_only_area  # undercut removed extra area under the mask
    assert si.polygon.area == pytest.approx(200 * 20, abs=1.0)  # Si ~untouched (factor 0.05)


def test_planarize_stop_material_cuts_flat_at_that_material_s_top():
    """A flat W layer everywhere, overcoated with oxide (also flat, since a flat top has no
    topology to be conformal *around*): planarizing to W's own top must remove the oxide
    entirely - it sits, uniformly, above the stop level.
    """
    g = Geometry.substrate("Si", domain_width_nm=100, thickness_nm=20)
    g.layers.append(Layer(material="W", polygon=box(0, 0, 100, 15)))
    g.deposit_conformal("SiO2", 10)

    g.planarize(stop_material="W")

    oxide = next((l for l in g.layers if l.material == "SiO2"), None)
    assert oxide is None or oxide.polygon.area == pytest.approx(0.0, abs=1e-6)
    tungsten = next(l for l in g.layers if l.material == "W")
    assert tungsten.polygon.bounds[3] == pytest.approx(15.0)


def test_planarize_target_level_keeps_only_a_bump_above_the_flat_fill():
    """A W bump in the middle, overcoated with a *directional* (angle 0 - no sidewall coverage,
    so the geometry stays exact rectangles) deposit: planarizing at the bump's own top must
    remove its oxide cap entirely while leaving the flanking oxide (entirely below that level)
    untouched - the case a naive "remove this material's layer entirely" shortcut would get wrong.
    """
    g = Geometry.substrate("Si", domain_width_nm=100, thickness_nm=20)
    g.layers.append(Layer(material="W", polygon=box(40, 0, 60, 15)))
    g.deposit_directional("SiO2", 10, angle_deg=0)

    g.planarize(target_level_nm=15.0)

    oxide = next(l for l in g.layers if l.material == "SiO2")
    tungsten = next(l for l in g.layers if l.material == "W")
    assert tungsten.polygon.bounds[3] == pytest.approx(15.0)
    # flanking oxide (either side of the bump, 80nm total width, only ever 10nm thick) untouched;
    # the 20x10 cap that sat on top of the bump (all above y=15) is gone.
    assert oxide.polygon.area == pytest.approx((100 - 20) * 10, abs=1.0)


def test_planarize_requires_exactly_one_target():
    g = Geometry.substrate("Si", domain_width_nm=50, thickness_nm=10)
    with pytest.raises(ValueError, match="exactly one"):
        g.planarize()
    with pytest.raises(ValueError, match="exactly one"):
        g.planarize(target_level_nm=5, stop_material="Si")


def test_directional_deposit_shadows_a_mesas_own_leeward_face():
    """A tilted beam (source up-and-left) should coat the mesa's illuminated (left) side and its
    top, leave its leeward (right) face bare, and leave a shadowed gap on the substrate just
    behind it before coating resumes further out where the shadow ends.
    """
    g = Geometry.substrate("Si", domain_width_nm=200, thickness_nm=20)
    g.layers.append(Layer(material="W", polygon=box(80, 0, 120, 40)))
    g.deposit_directional("Al", 15, angle_deg=30)

    al = next(l for l in g.layers if l.material == "Al").polygon
    assert al.contains(Point(75, 5))  # illuminated (left) side: coated
    assert al.contains(Point(100, 45))  # mesa top: coated
    assert not al.contains(Point(125, 5))  # leeward side, in the mesa's own shadow: bare
    assert al.contains(Point(160, 5))  # far enough right that the shadow has ended: coated


def test_directional_etch_protects_a_mesas_leeward_face_and_casts_a_shadow(materials, recipes):
    """The dual of the deposit case: an illuminated face/substrate should recede normally, the
    mesa's own leeward face should stay exactly where it was, and a shelf of un-etched substrate
    should survive in its shadow before the normal etch depth resumes further out.
    """
    g = Geometry.substrate("Si", domain_width_nm=200, thickness_nm=20)
    g.layers.append(Layer(material="W", polygon=box(80, 0, 120, 40)))
    recipe = recipes.get_etch("Ion Mill (tilted)")
    g.etch(recipe, depth_nm=15, materials=materials)

    solid = g.solid()
    assert not solid.contains(Point(75, -5))  # illuminated left: etched away
    assert solid.contains(Point(122, -5))  # shadowed shelf right behind the mesa: still solid
    assert not solid.contains(Point(160, -5))  # far right, beyond the shadow: etched away

    tungsten = next(l for l in g.layers if l.material == "W").polygon
    assert tungsten.bounds[2] == pytest.approx(120.0)  # leeward (right) face unmoved


def test_lift_off_removes_metal_deposited_on_top_of_stripped_resist():
    g = Geometry.substrate("Si", domain_width_nm=100, thickness_nm=10)
    g.deposit_conformal_masked("Photoresist", 20, open_x_ranges=[(40, 60)])
    g.deposit_directional("Al", 10, angle_deg=0)  # blanket metal over resist + the opening

    g.strip_material("Photoresist")
    g.remove_floating_debris()
    g.compact()

    al = next(l for l in g.layers if l.material == "Al")
    assert al.polygon.area == pytest.approx(20 * 10, abs=1.0)  # only the opening's metal survives
    assert not any(l.material == "Photoresist" for l in g.layers)
