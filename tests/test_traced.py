from structureforge.core.derivation import ArrheniusRate, ConstantRate, GrowthAtRate
from structureforge.core.traced import Traced


def test_literal_carries_no_derivation():
    t = Traced.literal(50.0)
    assert t.value == 50.0
    assert t.derivation is None


def test_declared_wraps_the_obtention_parameters_as_a_plain_dict_without_computing_value():
    """`declared` records *what was used* alongside the value, but never computes the value
    from it - a doping concentration and the precursor/flow/duration that produced it are
    independent inputs here, not related by any formula this object knows about.
    """
    t = Traced.declared(2.5e18, precursor="SiH4", flow_sccm=12, duration_s=97.2)
    assert t.value == 2.5e18
    assert t.derivation == {"precursor": "SiH4", "flow_sccm": 12, "duration_s": 97.2}


def test_computed_keeps_the_resolvable_formula_alongside_its_own_result():
    derivation = GrowthAtRate(rate=ConstantRate(nm_per_s=0.5), duration_s=100.0)
    t = Traced.computed(value=derivation.resolve_nm(), derivation=derivation)
    assert t.value == 50.0
    assert t.derivation is derivation


def test_all_three_shapes_serialize_to_plain_json():
    literal = Traced.literal("c_plane")
    declared = Traced.declared(2.5e18, precursor="SiH4")
    computed = Traced.computed(
        value=57.94871807280426,
        derivation=GrowthAtRate(
            rate=ArrheniusRate(prefactor_nm_per_s=1e7, activation_energy_eV=1.9, temperature_K=1323.15),
            duration_s=100.0,
        ),
    )

    assert literal.model_dump(mode="json") == {"value": "c_plane", "derivation": None}
    assert declared.model_dump(mode="json") == {"value": 2.5e18, "derivation": {"precursor": "SiH4"}}
    dumped = computed.model_dump(mode="json")
    assert dumped["value"] == 57.94871807280426
    assert dumped["derivation"]["kind"] == "growth_at_rate"
    assert dumped["derivation"]["rate"]["kind"] == "arrhenius_rate"
