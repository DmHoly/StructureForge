import pytest

from structureforge.core.materials import Material, MaterialCategory


def test_dry_oxide_etch_matches_the_spec_example(materials, recipes):
    """"attaque selectivement les oxydes avec une selectivite de 0.8 sur le reste"."""
    recipe = recipes.get_etch("Dry Oxide Etch")
    assert recipe.factor_for(materials.get("SiO2")) == 1.0
    assert recipe.factor_for(materials.get("Si")) == 0.8
    assert recipe.factor_for(materials.get("Al")) == 0.8


def test_material_override_wins_over_category(recipes):
    recipe = recipes.get_etch("Anisotropic RIE")
    resist = Material(name="Photoresist", category=MaterialCategory.resist, color="#000")
    other_resist = Material(name="SomeOtherResist", category=MaterialCategory.resist, color="#000")

    assert recipe.factor_for(resist) == 0.1  # material-specific override
    assert recipe.factor_for(other_resist) == 1.0  # falls back to default_factor, no category rule


def test_get_unknown_recipe_raises_with_useful_message(recipes):
    with pytest.raises(KeyError, match="unknown etch recipe 'Nope'"):
        recipes.get_etch("Nope")
    with pytest.raises(KeyError, match="unknown deposition recipe 'Nope'"):
        recipes.get_deposition("Nope")


def test_koh_wet_etch_is_selective_to_silicon_at_the_111_angle(materials, recipes):
    recipe = recipes.get_etch("KOH Anisotropic Wet Etch")
    assert recipe.angle_deg == pytest.approx(54.7)
    assert recipe.factor_for(materials.get("Si")) == 1.0
    assert recipe.factor_for(materials.get("Si3N4")) == pytest.approx(0.02)
    assert recipe.factor_for(materials.get("SiO2")) == pytest.approx(0.02)


def test_cl2_icp_rie_is_selective_to_the_iii_n_family(materials, recipes):
    recipe = recipes.get_etch("Cl2 ICP-RIE (III-N)")
    for name in ["GaN", "AlGaN", "InGaN", "AlN"]:
        assert recipe.factor_for(materials.get(name)) == 1.0
    assert recipe.factor_for(materials.get("PMMA")) == pytest.approx(0.05)
    assert recipe.factor_for(materials.get("Si")) == pytest.approx(0.05)  # not III-N, no override
