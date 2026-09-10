import math

import pytest
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union

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


def test_flip_mirrors_the_stack_and_keeps_the_same_bounds():
    g = Geometry.substrate("Si", domain_width_nm=100, thickness_nm=30)
    g.deposit_conformal("SiO2", 20)
    bounds_before = g.bounds()

    g.flip()

    assert g.bounds() == pytest.approx(bounds_before)
    # the oxide grown on top is now at the bottom (bonded to the carrier), the substrate on top
    oxide = next(l for l in g.layers if l.material == "SiO2")
    si = next(l for l in g.layers if l.material == "Si")
    assert oxide.polygon.bounds[1] == pytest.approx(bounds_before[1])
    assert si.polygon.bounds[3] == pytest.approx(bounds_before[3])


def test_flip_twice_returns_to_the_original_geometry():
    g = Geometry.substrate("Si", domain_width_nm=100, thickness_nm=30)
    g.deposit_conformal("SiO2", 20)
    original_areas = {l.material: l.polygon.area for l in g.layers}

    g.flip()
    g.flip()

    assert {l.material: pytest.approx(l.polygon.area, abs=1e-6) for l in g.layers} == pytest.approx(original_areas)


def test_flip_reverses_layer_order_so_layers_0_is_the_new_bottom():
    """`remove_floating_debris` always anchors on `layers[0]`, meaning "whatever is bonded down"
    - after a flip that must be whatever just became the new bottom (the former top, now bonded
    to the carrier), not the original substrate, which floats freely at the new top like
    everything else once flipped.
    """
    g = Geometry.substrate("Si", domain_width_nm=100, thickness_nm=30)
    g.deposit_conformal("Au", 15)  # Au is now the front surface, on top of Si

    g.flip()

    assert g.layers[0].material == "Au"  # the former top is now the anchor at the bottom
    assert g.layers[-1].material == "Si"  # the original substrate now floats freely at the top


def test_flip_rejects_a_non_flat_front_surface():
    """An isolated raised feature narrower than the domain (e.g. a lift-off metal plot) leaves
    gaps beside it at the top - flipping that wouldn't actually bond down anything in those gaps,
    so it must be rejected rather than silently mirroring a shape nothing could physically
    produce.
    """
    g = Geometry.substrate("Si", domain_width_nm=200, thickness_nm=30)
    g.layers.append(Layer(material="Au", polygon=box(80, 0, 120, 15)))

    with pytest.raises(ValueError, match="flat"):
        g.flip()


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


def test_faceted_growth_with_only_c_plane_active_does_not_widen_the_sidewalls():
    """`deposit_faceted` offsets each facet along its own outward normal independently, so a rate
    of 0 pins that facet's line outright rather than leaking growth from whichever other rate's
    shared growth vector used to dominate that direction. rate_c alone should grow straight up by
    exactly rate_c * thickness and leave the sidewalls untouched.
    """
    g = Geometry(domain_width_nm=100)
    g.layers.append(Layer(material="GaN", polygon=box(40, 0, 60, 50)))
    g.deposit_faceted("GaN", thickness_nm=5.0, rate_c=1.0, rate_m=0.0, rate_sp=0.0)

    solid = g.solid()
    assert solid.bounds[0] == pytest.approx(40.0)
    assert solid.bounds[2] == pytest.approx(60.0)
    assert solid.bounds[3] == pytest.approx(55.0)


def test_faceted_growth_with_only_m_plane_active_does_not_raise_the_top():
    """Mirror of the c-only case: rate_m alone should widen both sidewalls by exactly
    rate_m * thickness and leave the flat top's height untouched.
    """
    g = Geometry(domain_width_nm=100)
    g.layers.append(Layer(material="GaN", polygon=box(40, 0, 60, 50)))
    g.deposit_faceted("GaN", thickness_nm=5.0, rate_c=0.0, rate_m=1.0, rate_sp=0.0)

    solid = g.solid()
    assert solid.bounds[0] == pytest.approx(35.0)
    assert solid.bounds[2] == pytest.approx(65.0)
    assert solid.bounds[3] == pytest.approx(50.0)


def test_faceted_growth_pure_sp_extends_an_already_pointed_tip_without_touching_the_sidewalls():
    """The actual motivating case: a crystal that has already come to a sharp point (no c-plane or
    m-plane facet left to pin) should be extendable by growing *only* the SP facets - the reported
    bug was that this always dragged the c-plane/sidewalls along for the ride too. With rate_c =
    rate_m = 0 there's nothing left for the SP facets to advance against but each other, so the
    apex should rise by exactly thickness / cos(angle) each step while the sidewalls stay put.
    """
    angle_deg = 30.0
    theta = math.radians(angle_deg)
    apex_y = 50 + 10 * math.tan(theta)
    seed = unary_union([box(40, 0, 60, 50), Polygon([(40, 50), (60, 50), (50, apex_y)])])

    g = Geometry(domain_width_nm=100)
    g.layers.append(Layer(material="GaN", polygon=seed))
    for _ in range(3):
        g.deposit_faceted(
            "GaN", thickness_nm=2.0, rate_c=0.0, rate_m=0.0, rate_sp=1.0, semi_polar_angle_deg=angle_deg
        )

    solid = g.solid()
    assert solid.geom_type == "Polygon"
    assert solid.bounds[0] == pytest.approx(40.0)
    assert solid.bounds[2] == pytest.approx(60.0)
    assert solid.bounds[3] == pytest.approx(apex_y + 3 * 2.0 / math.cos(theta))


def test_faceted_growth_small_c_rate_is_not_overridden_by_a_much_larger_sp_rate():
    """A small (but nonzero) rate_c must still cap the c-plane's own line at rate_c * thickness,
    no matter how much faster the flanking SP facets are growing - each facet's line moves at its
    own named rate, full stop, not at whatever the fastest neighbour reaches. Checked away from the
    corners (dead centre of the original top edge), since a corner's own height is legitimately
    free to be pulled up by a much faster neighbouring facet meeting a pinned one there - it's the
    facet's own line, not the silhouette's overall bounds, that a rate of 0 (or a small rate)
    guarantees.
    """
    g = Geometry(domain_width_nm=100)
    g.layers.append(Layer(material="GaN", polygon=box(40, 0, 60, 50)))
    g.deposit_faceted("GaN", thickness_nm=5.0, rate_c=0.2, rate_m=0.3, rate_sp=3.0, semi_polar_angle_deg=30.0)

    solid = g.solid()
    c_plane_y = 50.0 + 0.2 * 5.0
    assert solid.contains(Point(50.0, c_plane_y - 0.5))
    assert not solid.contains(Point(50.0, c_plane_y + 0.5))


def test_faceted_growth_mitre_bevels_instead_of_blowing_up_at_a_lopsided_rate_ratio():
    """A mitre join between two very unevenly-advancing facets at a shallow angle can shoot out to
    many times either facet's own distance (the classic miter-join blowup - see `mitre_or_bevel`).
    A tiny rate_c next to a much larger rate_sp is exactly that case; the film must stay within a
    sane multiple of the larger rate's own reach rather than spanning most of the domain.
    """
    g = Geometry(domain_width_nm=100)
    g.layers.append(Layer(material="GaN", polygon=box(40, 0, 60, 50)))
    g.deposit_faceted("GaN", thickness_nm=5.0, rate_c=0.1, rate_m=0.0, rate_sp=5.0, semi_polar_angle_deg=30.0)

    solid = g.solid()
    max_reach = 5.0 * 5.0  # rate_sp * thickness, generously bounding how far any facet should reach
    assert solid.bounds[0] > 40.0 - 3 * max_reach
    assert solid.bounds[2] < 60.0 + 3 * max_reach


def test_faceted_growth_after_a_bevel_does_not_leave_a_hole_at_the_apex():
    """A lopsided rate ratio (see the bevel test above) leaves a short, frozen bevel edge behind
    once the join gets cut instead of mitred - an edge whose normal matches none of c/m/sp, so it
    never grows in later steps. When the *next* layer's c-plane and semi-polar facets both keep
    advancing past that frozen edge's two ends, the corner fan nucleating a fresh c-facet is
    computed independently at each end (see `_offset_named_facets`) and the two fans can fail to
    meet in the middle, leaving a real, spurious triangular hole right at the apex - reported as a
    "void" in the rendered structure. This is the actual failing sequence (three facet layers with
    the growth-rate ratio flipping between them), reduced to a single seed mesa.
    """
    g = Geometry(domain_width_nm=300)
    g.layers.append(Layer(material="GaN", polygon=box(65, 0, 215, 50)))
    g.deposit_faceted("GaN", thickness_nm=115.3, rate_c=0.95, rate_m=0.0, rate_sp=0.5, semi_polar_angle_deg=30.0)
    g.deposit_faceted("GaN", thickness_nm=36.3, rate_c=0.1, rate_m=0.0, rate_sp=0.65, semi_polar_angle_deg=30.0)
    g.deposit_faceted("GaN", thickness_nm=48.6, rate_c=0.4, rate_m=0.0, rate_sp=0.15, semi_polar_angle_deg=30.0)

    solid = g.solid()
    assert solid.geom_type == "Polygon"
    assert len(solid.interiors) == 0


def _run_faceted(**overrides):
    g = Geometry(domain_width_nm=100)
    g.layers.append(Layer(material="GaN", polygon=box(40, 0, 60, 50)))
    kwargs = dict(thickness_nm=3.0, rate_c=1.0, rate_m=0.3, rate_sp=0.6, semi_polar_angle_deg=32.0)
    kwargs.update(overrides)
    g.deposit_faceted("GaN", **kwargs)
    return g


def test_faceted_growth_without_per_facet_overrides_still_adds_a_single_layer():
    """No material_c/material_m/material_sp given (the default) must behave exactly as before -
    one Layer named `material`, not one per family.
    """
    g = _run_faceted()
    new_layers = g.layers[1:]
    assert len(new_layers) == 1
    assert new_layers[0].material == "GaN"


def test_faceted_growth_splits_into_one_layer_per_distinct_facet_material():
    """More indium on the c-plane than on the semi-polar facets (the motivating case) - each
    facet family's own share of the film becomes its own Layer, and nothing is double-counted
    or dropped: the three per-family areas must sum to exactly what a single unsplit material
    would have covered.
    """
    reference = _run_faceted()
    reference_area = reference.layers[-1].polygon.area

    g = _run_faceted(material_c="In0.30Ga0.70N", material_m="GaN", material_sp="In0.10Ga0.90N")
    new_layers = g.layers[1:]
    materials = {layer.material for layer in new_layers}
    assert materials == {"In0.30Ga0.70N", "GaN", "In0.10Ga0.90N"}

    total_area = sum(layer.polygon.area for layer in new_layers)
    assert total_area == pytest.approx(reference_area)

    # the c-plane's own share sits at the flat top, the widest single facet at these rates
    c_layer = next(layer for layer in new_layers if layer.material == "In0.30Ga0.70N")
    assert c_layer.polygon.bounds[1] == pytest.approx(50.0)  # starts exactly at the original top


def test_faceted_growth_unset_per_facet_materials_fall_back_to_the_base_material():
    """Only overriding material_c: the m- and sp-grown areas both fall back to `material` and
    merge into one Layer (they're the same name), while the c-plane gets its own.
    """
    g = _run_faceted(material_c="In0.30Ga0.70N")
    new_layers = g.layers[1:]
    materials = {layer.material for layer in new_layers}
    assert materials == {"In0.30Ga0.70N", "GaN"}
