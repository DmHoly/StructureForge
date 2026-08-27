import pytest

from structureforge.core.materials import Material, MaterialCategory, MaterialLibrary


def test_default_library_has_the_common_materials(materials):
    for name in [
        "Si", "SiC", "Sapphire", "SiO2", "Si3N4", "HfO2",
        "GaN", "AlN", "AlGaN", "InGaN", "GaAs", "Ge",
        "Al", "W", "Ti", "TiN", "Cu",
        "Photoresist", "PMMA",
    ]:
        assert name in materials
    assert materials.get("SiO2").category is MaterialCategory.dielectric
    assert materials.get("GaN").category is MaterialCategory.semiconductor
    assert materials.get("Sapphire").category is MaterialCategory.substrate
    assert materials.get("PMMA").category is MaterialCategory.resist


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
