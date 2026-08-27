import pytest

from structureforge.core.materials import Material, MaterialCategory, MaterialLibrary


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
