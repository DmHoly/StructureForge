import pytest
from pydantic import ValidationError

from structureforge.core.units import Length
from structureforge.process.steps import FacetedGrowth, Lithography, Planarization


def test_planarization_needs_exactly_one_target():
    with pytest.raises(ValidationError, match="exactly one"):
        Planarization(name="CMP")
    with pytest.raises(ValidationError, match="exactly one"):
        Planarization(name="CMP", target_level=Length.nm(10), stop_material="W")

    # each alone is fine
    Planarization(name="CMP", target_level=Length.nm(10))
    Planarization(name="CMP", stop_material="W")


def test_lithography_rejects_a_backwards_opening():
    with pytest.raises(ValidationError, match="not a valid x-range"):
        Lithography(
            name="Masque", resist_material="Photoresist", thickness=Length.nm(5), openings=[(120, 80)]
        )


def test_faceted_growth_rejects_a_negative_inverted_semi_polar_rate():
    with pytest.raises(ValidationError, match=">= 0"):
        FacetedGrowth(name="Coquille", material="InGaN", thickness=Length.nm(5), rate_sp_inv=-0.1)
    # inverted facets alone are a valid growth
    FacetedGrowth(name="Coquille", material="InGaN", thickness=Length.nm(5), rate_c=0, rate_m=0, rate_sp=0, rate_sp_inv=1)


def test_faceted_growth_lateral_reach_is_positive():
    FacetedGrowth(name="Coquille", material="InGaN", thickness=Length.nm(5), lateral_reach_nm=200)
    with pytest.raises(ValidationError, match="lateral_reach_nm"):
        FacetedGrowth(name="Coquille", material="InGaN", thickness=Length.nm(5), lateral_reach_nm=-10)



def test_deposition_roughness_and_grain_must_be_positive_and_grain_needs_roughness():
    from structureforge.process.steps import Deposition

    def dep(**kw):
        return Deposition(name="Gratin", material="SiO2", recipe="CVD Conformal", thickness=Length.nm(20), **kw)

    with pytest.raises(ValidationError, match="roughness_nm must be > 0"):
        dep(roughness_nm=0)
    with pytest.raises(ValidationError, match="grain_nm must be > 0"):
        dep(roughness_nm=2, grain_nm=-1)
    with pytest.raises(ValidationError, match="set roughness_nm too"):
        dep(grain_nm=10)
    assert dep(roughness_nm=2, grain_nm=10).grain_nm == 10
    assert dep().roughness_nm is None
