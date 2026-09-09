import math

import pytest
from pydantic import ValidationError

from structureforge.core.derivation import ArrheniusRate, ConstantRate, GrowthAtRate, MultiStageGrowth
from structureforge.core.units import Length


def test_length_needs_exactly_one_of_value_or_derivation():
    with pytest.raises(ValidationError, match="exactly one"):
        Length()
    with pytest.raises(ValidationError, match="exactly one"):
        Length(value=5.0, derivation=GrowthAtRate(rate=ConstantRate(nm_per_s=1.0), duration_s=5.0))

    Length.nm(5.0)
    Length.derived(GrowthAtRate(rate=ConstantRate(nm_per_s=1.0), duration_s=5.0))


def test_growth_at_constant_rate_resolves_to_rate_times_duration():
    length = Length.derived(GrowthAtRate(rate=ConstantRate(nm_per_s=0.5), duration_s=10.0))
    assert length.to_nm() == pytest.approx(5.0)


def test_growth_at_arrhenius_rate_matches_the_formula_directly():
    rate = ArrheniusRate(prefactor_nm_per_s=2.0, activation_energy_eV=1.0, temperature_K=900.0)
    expected_rate = 2.0 * math.exp(-1.0 / (8.617333262e-5 * 900.0))
    assert rate.resolve_nm_per_s() == pytest.approx(expected_rate)

    length = Length.derived(GrowthAtRate(rate=rate, duration_s=3.0))
    assert length.to_nm() == pytest.approx(expected_rate * 3.0)


def test_multi_stage_growth_sums_each_stage_and_can_mix_rate_kinds():
    stage_a = GrowthAtRate(rate=ConstantRate(nm_per_s=1.0), duration_s=2.0)
    stage_b = GrowthAtRate(
        rate=ArrheniusRate(prefactor_nm_per_s=5.0, activation_energy_eV=0.5, temperature_K=1000.0),
        duration_s=4.0,
    )
    ramped = Length.derived(MultiStageGrowth(stages=[stage_a, stage_b]))
    assert ramped.to_nm() == pytest.approx(stage_a.resolve_nm() + stage_b.resolve_nm())


def test_derived_length_survives_a_json_roundtrip():
    original = Length.derived(
        MultiStageGrowth(
            stages=[
                GrowthAtRate(rate=ConstantRate(nm_per_s=0.5), duration_s=10.0),
                GrowthAtRate(
                    rate=ArrheniusRate(prefactor_nm_per_s=3.0, activation_energy_eV=0.8, temperature_K=950.0),
                    duration_s=6.0,
                ),
            ]
        )
    )
    restored = Length.model_validate_json(original.model_dump_json())
    assert restored.to_nm() == pytest.approx(original.to_nm())


def test_a_process_step_can_carry_a_derived_thickness():
    from structureforge.process.steps import EpitaxialGrowth

    thickness = Length.derived(GrowthAtRate(rate=ConstantRate(nm_per_s=0.5), duration_s=10.0))
    step = EpitaxialGrowth(name="GaN buffer", material="GaN", thickness=thickness)
    assert step.thickness.to_nm() == pytest.approx(5.0)
