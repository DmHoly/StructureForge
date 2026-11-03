import pytest
from shapely.geometry import box

from structureforge.core.materials import indium_gan
from structureforge.core.units import Length
from structureforge.geometry.engine import Geometry, Layer
from structureforge.process.simulate import SimulationCache, SimulationError, simulate
from structureforge.process.steps import ChemicalStep, Deposition, Etch, FacetedGrowth, Flip, Lithography, ResistStrip


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


def test_flip_step_flips_within_simulate_for_backside_processing(materials, recipes):
    geometry = Geometry.substrate("Si", domain_width_nm=100, thickness_nm=30)
    flow = [
        Deposition(name="Metal avant", material="Au", recipe="Evaporation (normal)", thickness=Length.nm(20)),
        Flip(name="Retournement"),
        Deposition(name="Metal arriere", material="Ti", recipe="CVD Conformal", thickness=Length.nm(10)),
    ]
    frames = simulate(geometry, flow, materials, recipes)

    assert [f.step_kind for f in frames[1:]] == ["deposition", "flip", "deposition"]


def test_flip_step_rejects_a_non_flat_front_surface_within_simulate(materials, recipes):
    geometry = Geometry.substrate("Si", domain_width_nm=200, thickness_nm=30)
    flow = [
        Lithography(name="Masque", resist_material="Photoresist", thickness=Length.nm(20), openings=[(80, 120)]),
        Deposition(name="Plot", material="Au", recipe="Evaporation (normal)", thickness=Length.nm(15)),
        ResistStrip(name="Retrait resine"),
        Flip(name="Retournement"),
    ]
    with pytest.raises(SimulationError):
        simulate(geometry, flow, materials, recipes)


def test_faceted_growth_step_with_per_facet_materials_splits_into_distinct_layers(materials, recipes):
    geometry = Geometry(domain_width_nm=100)
    geometry.layers.append(Layer(material="GaN", polygon=box(40, 0, 60, 50)))
    rich, lean = indium_gan(0.30), indium_gan(0.10)
    lib = materials.with_materials(rich, lean)
    step = FacetedGrowth(
        name="InGaN facet-dependent indium",
        material="GaN",
        thickness=Length.nm(3),
        rate_c=1.0,
        rate_m=0.3,
        rate_sp=0.6,
        material_c=rich.name,
        material_sp=lean.name,
    )
    frames = simulate(geometry, [step], lib, recipes)

    new_layers = frames[-1].layers[1:]
    assert {layer.material for layer in new_layers} == {rich.name, "GaN", lean.name}


def test_faceted_growth_step_rejects_an_unregistered_per_facet_material(materials, recipes):
    geometry = Geometry(domain_width_nm=100)
    geometry.layers.append(Layer(material="GaN", polygon=box(40, 0, 60, 50)))
    step = FacetedGrowth(name="Bad alloy", material="GaN", thickness=Length.nm(3), material_c="Unobtainium")

    with pytest.raises(SimulationError):
        simulate(geometry, [step], materials, recipes)


def test_growth_steps_stamp_a_literal_thickness_as_a_traced_literal(materials, recipes):
    """A plain `Length(value=..., unit=...)` (no derivation) becomes a `Traced.literal` on the
    resulting Layer's provenance - value set, nothing recorded about "how"."""
    geometry = Geometry(domain_width_nm=100)
    geometry.layers.append(Layer(material="GaN", polygon=box(40, 0, 60, 50)))
    step = FacetedGrowth(name="Simple growth", material="GaN", thickness=Length.nm(3), rate_c=1.0, rate_m=0.0, rate_sp=0.0)

    frames = simulate(geometry, [step], materials, recipes)
    new_layer = frames[-1].layers[-1]

    assert new_layer.provenance.step_kind == "faceted_growth"
    assert new_layer.provenance.step_name == "Simple growth"
    thickness = new_layer.provenance.parameters["thickness"]
    assert thickness.value == pytest.approx(3.0)
    assert thickness.derivation is None
    assert new_layer.provenance.parameters["rate_c"].value == 1.0


def test_growth_steps_stamp_a_derived_thickness_as_a_traced_computed_value(materials, recipes):
    """A `Length.derived(...)` thickness carries its `LengthDerivation` tree straight through to
    the resulting Layer's provenance, alongside the resolved nanometre value."""
    from structureforge.core.derivation import ConstantRate, GrowthAtRate

    geometry = Geometry(domain_width_nm=100)
    geometry.layers.append(Layer(material="GaN", polygon=box(40, 0, 60, 50)))
    thickness = Length.derived(GrowthAtRate(rate=ConstantRate(nm_per_s=0.5), duration_s=6.0))
    step = FacetedGrowth(name="Timed growth", material="GaN", thickness=thickness, rate_c=1.0, rate_m=0.0, rate_sp=0.0)

    frames = simulate(geometry, [step], materials, recipes)
    traced_thickness = frames[-1].layers[-1].provenance.parameters["thickness"]

    assert traced_thickness.value == pytest.approx(3.0)  # 0.5 nm/s * 6s
    assert traced_thickness.derivation["kind"] == "growth_at_rate"
    assert traced_thickness.derivation["rate"] == {"kind": "constant_rate", "nm_per_s": 0.5}


def test_frame_to_dict_serializes_provenance_and_omits_it_when_absent(materials, recipes):
    geometry = Geometry.substrate("Si", domain_width_nm=100, thickness_nm=30)
    flow = [Deposition(name="Oxyde", material="SiO2", recipe="CVD Conformal", thickness=Length.nm(20))]
    frames = simulate(geometry, flow, materials, recipes)

    initial_layer_dict = frames[0].to_dict()["layers"][0]
    assert initial_layer_dict["provenance"] is None

    # Deposition steps don't build a LayerProvenance (out of scope for this pass) - still None,
    # not a crash - Layer.provenance stays optional everywhere it isn't explicitly wired up.
    deposited_layer_dict = frames[-1].to_dict()["layers"][-1]
    assert deposited_layer_dict["provenance"] is None


def _slit_edges(polygon):
    """Edges a polygon's rings traverse in both directions - a zero-width slit or flap, drawn as a
    stray line across the layer."""
    from shapely.geometry import MultiPolygon

    parts = list(polygon.geoms) if isinstance(polygon, MultiPolygon) else [polygon]
    found = []
    for part in parts:
        segments = set()
        for ring in [part.exterior, *part.interiors]:
            coords = [(round(x, 4), round(y, 4)) for x, y in ring.coords]
            segments |= {(a, b) for a, b in zip(coords, coords[1:]) if a != b}
        found += [(a, b) for a, b in segments if (b, a) in segments]
    return found


def test_mqw_barriers_on_a_sag_nanowire_have_no_internal_seams(recipes):
    """A real SAG flow (isotropic undercut of the oxide, nucleation overflowing the mask, wide
    faceted core, then wells/barriers): each barrier is assembled from many strips, fans and
    substeps over a surface shaped by the etch, which used to leave zero-width flaps along the
    well/barrier interfaces - drawn as lines across the layers."""
    from structureforge.core.materials import default_library

    materials = default_library().with_materials(indium_gan(0.10))

    def faceted(material, thickness, c, m, sp, seeds):
        return FacetedGrowth(
            name="F", material=material, thickness=Length.nm(thickness), rate_c=c, rate_m=m, rate_sp=sp,
            semi_polar_angle_deg=45, seed_materials=seeds,
        )

    steps = [
        Deposition(name="Oxyde", material="SiO2", recipe="ALD Conformal", thickness=Length.nm(50)),
        Deposition(name="Nitrure", material="Si3N4", recipe="ALD Conformal", thickness=Length.nm(80)),
        Lithography(name="Litho", resist_material="Photoresist", thickness=Length.nm(100), openings=[(200, 300)]),
        Etch(name="Gravure", recipe="Anisotropic RIE", depth=Length.nm(83)),
        Etch(name="Gravure SiO2", recipe="Wet HF Dip", depth=Length.nm(51)),
        ResistStrip(name="Strip", material="Photoresist"),
        faceted("GaN", 400, 1, 0.3, 0.7, ["GaN"]),
        faceted("GaN", 600, 1, 0.05, 0.8, ["GaN"]),
    ]
    for _ in range(4):
        steps += [faceted("In0.10Ga0.90N", 10, 1, 0, 0, ["GaN"]), faceted("GaN", 10, 1, 0, 0.5, ["GaN", "InGaN"])]

    final = simulate(Geometry.substrate("GaN", domain_width_nm=500, thickness_nm=50), steps, materials, recipes)[-1]
    for layer in final.layers:
        assert _slit_edges(layer.polygon) == [], layer.material


def test_faceted_growth_step_with_inverted_facets_grows_a_shell_past_the_sidewalls(materials, recipes):
    geometry = Geometry(domain_width_nm=300)
    geometry.layers.append(Layer(material="GaN", polygon=box(100, 0, 200, 600)))
    step = FacetedGrowth(
        name="Coquille",
        material="InGaN",
        thickness=Length.nm(50),
        rate_c=0.5,
        rate_m=0.0,
        rate_sp=0.5,
        rate_sp_inv=0.5,
        semi_polar_angle_deg=45,
        material_sp_inv="GaN",
    )
    frames = simulate(geometry, [step], materials, recipes)

    layers = frames[-1].layers[1:]
    under = next(layer for layer in layers if layer.material == "GaN")  # the inverted facets' share
    assert under.polygon.bounds[0] < 100 and under.polygon.bounds[1] < 600
    assert layers[0].provenance.parameters["rate_sp_inv"].value == 0.5


def test_faceted_growth_step_with_a_lateral_reach_shades_the_sidewalls_and_traces_it(materials, recipes):
    geometry = Geometry(domain_width_nm=400)
    geometry.layers.append(Layer(material="GaN", polygon=box(150, 0, 250, 600)))
    step = FacetedGrowth(
        name="Coquille",
        material="InGaN",
        thickness=Length.nm(100),
        rate_c=0.0,
        rate_m=0.4,
        rate_sp=0.0,
        lateral_reach_nm=200,
    )
    frames = simulate(geometry, [step], materials, recipes)

    shell = frames[-1].layers[-1]
    assert shell.polygon.bounds[1] == pytest.approx(400.0, abs=0.01)
    assert shell.provenance.parameters["lateral_reach_nm"].value == 200


# -- SimulationCache: replay only the steps that changed ---------------------------------------


def _rings(frames):
    return [[(l.material, l.polygon.wkt) for l in f.layers] for f in frames]


def test_cache_returns_the_same_frames_as_a_fresh_simulation(materials, recipes):
    cache = SimulationCache()
    flow = _flow()
    fresh = simulate(Geometry.substrate("Si", 100, 30), flow, materials, recipes)
    cached = cache.simulate("Si", 100, 30, flow, materials, recipes)
    assert _rings(cached) == _rings(fresh)
    assert [f.step_index for f in cached] == [f.step_index for f in fresh]


def test_cache_replays_only_from_the_first_changed_step(materials, recipes, monkeypatch):
    import structureforge.process.simulate as sim

    applied = []
    real_apply = sim._apply
    monkeypatch.setattr(sim, "_apply", lambda g, step, m, r, **kw: (applied.append(step.name), real_apply(g, step, m, r, **kw)))

    cache = SimulationCache()
    flow = _flow()
    cache.simulate("Si", 100, 30, flow, materials, recipes)
    assert applied == ["Oxyde", "Nettoyage", "Gravure"]

    # the same flow again: nothing to replay
    applied.clear()
    again = cache.simulate("Si", 100, 30, list(flow), materials, recipes)
    assert applied == []
    assert len(again) == len(flow) + 1

    # the last step edited: only it is replayed, from the frame after step 2 - and the result
    # is what a fresh run of the edited flow gives
    edited = flow[:2] + [Etch(name="Gravure", recipe="Anisotropic RIE", depth=Length.nm(15))]
    applied.clear()
    frames = cache.simulate("Si", 100, 30, edited, materials, recipes)
    assert applied == ["Gravure"]
    fresh = simulate(Geometry.substrate("Si", 100, 30), edited, materials, recipes)
    assert _rings(frames) == _rings(fresh)
    assert [f.step_index for f in frames] == [0, 1, 2, 3]

    # a step appended: only it is replayed
    longer = edited + [Deposition(name="Cap", material="Si3N4", recipe="ALD Conformal", thickness=Length.nm(5))]
    applied.clear()
    frames = cache.simulate("Si", 100, 30, longer, materials, recipes)
    assert applied == ["Cap"]
    assert _rings(frames) == _rings(simulate(Geometry.substrate("Si", 100, 30), longer, materials, recipes))

    # the first step edited: everything is replayed
    changed_first = [Deposition(name="Oxyde", material="SiO2", recipe="CVD Conformal", thickness=Length.nm(25))] + longer[1:]
    applied.clear()
    cache.simulate("Si", 100, 30, changed_first, materials, recipes)
    assert applied == ["Oxyde", "Nettoyage", "Gravure", "Cap"]


def test_cache_does_not_let_a_resumed_run_disturb_the_cached_frames(materials, recipes):
    cache = SimulationCache()
    flow = _flow()
    first = cache.simulate("Si", 100, 30, flow, materials, recipes)
    before = _rings(first)
    cache.simulate("Si", 100, 30, flow[:2] + [Etch(name="Gravure", recipe="Anisotropic RIE", depth=Length.nm(15))], materials, recipes)
    assert _rings(first) == before


def test_cache_is_keyed_on_substrate_and_recipes(materials, recipes, monkeypatch):
    import structureforge.process.simulate as sim
    from structureforge.core.recipes import EtchMode, EtchRecipe

    applied = []
    real_apply = sim._apply
    monkeypatch.setattr(sim, "_apply", lambda g, step, m, r, **kw: (applied.append(step.name), real_apply(g, step, m, r, **kw)))
    cache = SimulationCache()
    flow = _flow()
    cache.simulate("Si", 100, 30, flow, materials, recipes)

    applied.clear()
    cache.simulate("Si", 120, 30, flow, materials, recipes)  # a wider domain: a different structure
    assert len(applied) == 3

    applied.clear()
    faster = recipes.with_recipes(
        etch=[EtchRecipe(name="Anisotropic RIE", mode=EtchMode.directional, selectivity_by_material={"Photoresist": 0.1}, default_factor=2.0)]
    )
    cache.simulate("Si", 100, 30, flow, materials, faster)  # an edited recipe: stale frames would be wrong
    assert len(applied) == 3


def test_cache_reports_a_failing_step_with_its_index_in_the_whole_flow(materials, recipes):
    cache = SimulationCache()
    flow = _flow()
    cache.simulate("Si", 100, 30, flow, materials, recipes)
    bad = flow + [Etch(name="Mauvaise", recipe="does-not-exist", depth=Length.nm(5))]
    with pytest.raises(SimulationError) as excinfo:
        cache.simulate("Si", 100, 30, bad, materials, recipes)
    assert excinfo.value.step_index == 4


def test_preview_runs_etches_with_fewer_substeps_and_is_cached_apart(materials, recipes, monkeypatch):
    import structureforge.geometry.engine as engine

    counts = []
    real_steps = engine.Geometry._etch_steps
    monkeypatch.setattr(engine.Geometry, "_etch_steps", lambda self, *a: (counts.append(real_steps(self, *a)), counts[-1])[1])
    flow = [
        Deposition(name="Oxyde", material="SiO2", recipe="CVD Conformal", thickness=Length.nm(40)),
        Lithography(name="Masque", resist_material="Photoresist", thickness=Length.nm(20), openings=[(40, 60)]),
        Etch(name="Gravure humide", recipe="Dry Oxide Etch", depth=Length.nm(30)),
    ]
    exact = simulate(Geometry.substrate("Si", 100, 30), flow, materials, recipes)
    quick = simulate(Geometry.substrate("Si", 100, 30), flow, materials, recipes, preview=True)
    assert counts[0] == 15  # the isotropic default for 30 nm
    oxide_exact = next(l.polygon for l in exact[-1].layers if l.material == "SiO2")
    oxide_quick = next(l.polygon for l in quick[-1].layers if l.material == "SiO2")
    assert oxide_exact.symmetric_difference(oxide_quick).area < 0.05 * oxide_exact.area  # a preview, not a different structure

    cache = SimulationCache()
    cache.simulate("Si", 100, 30, flow, materials, recipes, preview=True)
    applied = []
    real_apply = sim_apply = __import__("structureforge.process.simulate", fromlist=["_apply"])._apply
    monkeypatch.setattr(__import__("structureforge.process.simulate", fromlist=["_apply"]), "_apply", lambda g, step, m, r, **kw: (applied.append(kw.get("preview", False)), real_apply(g, step, m, r, **kw)))
    cache.simulate("Si", 100, 30, flow, materials, recipes)  # exact: never resumed from the preview's frames
    assert applied == [False, False, False]
