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


def test_sf6_deep_rie_stops_on_common_mask_materials(materials, recipes):
    recipe = recipes.get_etch("SF6 Deep RIE (Si)")
    assert recipe.factor_for(materials.get("Si")) == 1.0
    for mask in ["SiO2", "Si3N4"]:
        assert recipe.factor_for(materials.get(mask)) == pytest.approx(0.02)
    assert recipe.factor_for(materials.get("Photoresist")) == pytest.approx(0.05)


def test_wet_metal_etch_is_selective_by_category(materials, recipes):
    recipe = recipes.get_etch("Wet Metal Etch")
    for name in ["Al", "Cu", "Au", "W"]:
        assert recipe.factor_for(materials.get(name)) == 1.0
    assert recipe.factor_for(materials.get("SiO2")) == pytest.approx(0.05)


def test_default_recipe_library_has_the_expected_names(recipes):
    for name in ["ALD Conformal", "CVD Conformal", "PVD Sputter (tilted)", "Evaporation (normal)",
                 "MOCVD Epitaxial", "PECVD Conformal", "Sputter Metal (normal)", "Electroplating (Cu)"]:
        assert name in recipes.deposition
    for name in ["Dry Oxide Etch", "Wet HF Dip", "Anisotropic RIE", "Ion Mill (tilted)",
                 "KOH Anisotropic Wet Etch", "Cl2 ICP-RIE (III-N)", "TMAH Anisotropic Wet Etch",
                 "SF6 Deep RIE (Si)", "Wet Metal Etch"]:
        assert name in recipes.etch
