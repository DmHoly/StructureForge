import pytest

from structureforge.core.materials import (
    Material,
    MaterialCategory,
    MaterialLibrary,
    aluminum_gan,
    aluminum_gan_gradient,
    indium_gan,
    indium_gan_gradient,
)


def test_default_library_has_the_common_materials(materials):
    for name in [
        # substrates
        "Si", "SiC", "Sapphire", "InP",
        # dielectrics
        "SiO2", "Si3N4", "Al2O3", "HfO2", "ZrO2", "Ta2O5", "TiO2", "SiON", "MgF2", "Polyimide", "BCB",
        # semiconductors
        "Poly-Si", "Ge", "SiGe", "GaAs", "AlAs", "GaP", "GaSb", "InAs", "InSb", "ZnO",
        "GaN", "AlN", "AlGaN", "InGaN",
        # metals (+ ITO, a transparent conductive oxide grouped with metals)
        "Al", "W", "Ti", "TiN", "Cu", "Au", "Ag", "Ni", "Pt", "Pd", "Mo", "Ta", "ITO",
        # resists
        "Photoresist", "PMMA", "HSQ", "SU-8",
    ]:
        assert name in materials
    assert materials.get("SiO2").category is MaterialCategory.dielectric
    assert materials.get("GaN").category is MaterialCategory.semiconductor
    assert materials.get("Sapphire").category is MaterialCategory.substrate
    assert materials.get("PMMA").category is MaterialCategory.resist
    assert materials.get("ITO").category is MaterialCategory.metal


def test_get_unknown_material_raises_with_useful_message(materials):
    with pytest.raises(KeyError, match="unknown material 'Unobtainium'"):
        materials.get("Unobtainium")


def test_with_materials_extends_without_mutating_the_original(materials):
    custom = Material(name="pzGaN", category=MaterialCategory.semiconductor, color="#123456")
    extended = materials.with_materials(custom)

    assert "pzGaN" in extended
    assert "pzGaN" not in materials
    assert "Si" in extended  # still carries the base library forward


def test_library_is_keyed_by_name_last_write_wins():
    lib = MaterialLibrary().with_materials(
        Material(name="X", category=MaterialCategory.other, color="#000"),
        Material(name="X", category=MaterialCategory.metal, color="#fff"),
    )
    assert lib.get("X").category is MaterialCategory.metal


def test_indium_gan_endpoints_match_visible_spectrum_ends():
    pure_gan = indium_gan(0.0)
    pure_inn = indium_gan(1.0)
    assert pure_gan.name == "In0.00Ga1.00N"
    assert pure_inn.name == "In1.00Ga0.00N"
    assert pure_gan.color == "#4b0082"  # near-UV/violet spectrum stop at x=0
    assert pure_inn.color == "#e63946"  # red spectrum stop at x=1
    assert pure_gan.category is MaterialCategory.semiconductor


def test_indium_gan_density_increases_monotonically_with_fraction():
    low, mid, high = indium_gan(0.1), indium_gan(0.3), indium_gan(0.6)
    assert low.density_g_cm3 < mid.density_g_cm3 < high.density_g_cm3
    assert low.refractive_index < mid.refractive_index < high.refractive_index


def test_indium_gan_color_sweeps_the_visible_spectrum_with_fraction():
    # violet -> blue -> cyan -> green -> yellow -> orange -> red: distinct colors throughout,
    # and by the high-indium end the color should have swung solidly into the red.
    stops = [indium_gan(x).color for x in (0.0, 1 / 6, 2 / 6, 3 / 6, 4 / 6, 5 / 6, 1.0)]
    assert len(set(stops)) == len(stops)
    red = lambda m: int(m.color[1:3], 16)
    blue = lambda m: int(m.color[5:7], 16)
    assert red(indium_gan(0.95)) > red(indium_gan(0.05))
    assert blue(indium_gan(0.05)) > blue(indium_gan(0.95))


def test_indium_gan_same_fraction_is_deterministically_named():
    assert indium_gan(0.15).name == indium_gan(0.15).name == "In0.15Ga0.85N"
    assert indium_gan(0.15) == indium_gan(0.15)


def test_indium_gan_rejects_out_of_range_fraction():
    with pytest.raises(ValueError, match=r"indium fraction must be in \[0, 1\]"):
        indium_gan(1.5)
    with pytest.raises(ValueError, match=r"indium fraction must be in \[0, 1\]"):
        indium_gan(-0.1)


def test_aluminum_gan_blends_toward_aln_not_inn():
    algan = aluminum_gan(0.4)
    assert algan.name == "Al0.40Ga0.60N"
    ingan = indium_gan(0.4)
    assert algan.color != ingan.color
    assert algan.density_g_cm3 < indium_gan(0.4).density_g_cm3  # AlN is lighter than InN


def test_indium_gan_gradient_spans_the_requested_range_inclusive():
    grad = indium_gan_gradient(0.05, 0.25, 5)
    assert [m.name for m in grad] == [
        "In0.05Ga0.95N", "In0.10Ga0.90N", "In0.15Ga0.85N", "In0.20Ga0.80N", "In0.25Ga0.75N",
    ]


def test_aluminum_gan_gradient_spans_the_requested_range_inclusive():
    grad = aluminum_gan_gradient(0.1, 0.3, 3)
    assert [m.name for m in grad] == ["Al0.10Ga0.90N", "Al0.20Ga0.80N", "Al0.30Ga0.70N"]


def test_gradient_requires_at_least_two_steps():
    with pytest.raises(ValueError, match="steps must be >= 2"):
        indium_gan_gradient(0.0, 0.2, 1)


def test_gradient_materials_register_cleanly_into_a_library(materials):
    lib = materials.with_materials(*indium_gan_gradient(0.05, 0.30, 6))
    for name in ["In0.05Ga0.95N", "In0.10Ga0.90N", "In0.15Ga0.85N", "In0.20Ga0.80N", "In0.25Ga0.75N", "In0.30Ga0.70N"]:
        assert name in lib
    assert "GaN" in lib  # base library still present
