"""Selective-Area Growth (SAG) of III-N heterostructures — three orientations.

Three independent scenarios in one file:

1. c-plane SAG (plan C) — GaN grown in SiO2 mask windows, followed by an InGaN QW cap.
   The SiO2 blocks nucleation everywhere except the open windows.  Demonstrates homo-
   epitaxy (GaN/GaN) and hetero-epitaxy (InGaN/GaN) on c-plane.

2. m-plane shell (plan M) — AlGaN shell grown laterally on the sidewalls of a GaN pillar.
   The mask is stripped before regrowth so GaN sidewalls are the only exposed seed.

3. semi-polar facets (plan SP) — InGaN film grown at 32° from the c-axis ({11-22}-like)
   on a bare GaN template with no mask, mimicking planar semi-polar growth on a miscut
   or a patterned substrate.  angle_deg = 32° gives ~4.24 nm of vertical rise for 5 nm
   along the growth-front normal.

Run: python examples/epitaxial_growth_sag.py
"""

from __future__ import annotations

from pathlib import Path

from structureforge import (
    EpitaxialGrowth,
    Etch,
    Geometry,
    GrowthOrientation,
    Length,
    Lithography,
    ResistStrip,
    default_library,
    default_recipes,
    save_svg,
    simulate,
)


def scenario_cplane_sag(mat, rec) -> tuple:
    """c-plane SAG: GaN mesa + InGaN QW, blocked by SiO2 mask."""
    domain = 200.0
    geo = Geometry.substrate("GaN", domain_width_nm=domain, thickness_nm=30)
    steps = [
        Lithography(
            name="Masque SiO2 SAG",
            resist_material="SiO2",
            thickness=Length.nm(30),
            openings=[(80.0, 120.0)],
        ),
        EpitaxialGrowth(
            name="GaN c-plan SAG (homo)",
            material="GaN",
            thickness=Length.nm(60),
            orientation=GrowthOrientation.c_plane,
            seed_materials=["GaN"],
        ),
        EpitaxialGrowth(
            name="InGaN puits QW (hetero)",
            material="InGaN",
            thickness=Length.nm(3),
            orientation=GrowthOrientation.c_plane,
            seed_materials=["GaN"],
        ),
        EpitaxialGrowth(
            name="GaN cap p-type",
            material="GaN",
            thickness=Length.nm(15),
            orientation=GrowthOrientation.c_plane,
            seed_materials=["GaN", "InGaN"],
        ),
    ]
    return geo, steps


def scenario_mplane_shell(mat, rec) -> tuple:
    """m-plane shell: AlGaN shell on GaN pillar sidewalls."""
    domain = 200.0
    geo = Geometry.substrate("GaN", domain_width_nm=domain, thickness_nm=60)
    steps = [
        Lithography(
            name="Masque HSQ pilier",
            resist_material="HSQ",
            thickness=Length.nm(40),
            openings=[(0.0, 80.0), (120.0, 200.0)],
        ),
        Etch(
            name="Gravure ICP-RIE pilier",
            recipe="Cl2 ICP-RIE (III-N)",
            depth=Length.nm(60),
        ),
        ResistStrip(name="Retrait HSQ", material="HSQ"),
        EpitaxialGrowth(
            name="AlGaN shell m-plan (hetero)",
            material="AlGaN",
            thickness=Length.nm(8),
            orientation=GrowthOrientation.m_plane,
            seed_materials=["GaN"],
        ),
        EpitaxialGrowth(
            name="GaN cap m-plan (homo)",
            material="GaN",
            thickness=Length.nm(5),
            orientation=GrowthOrientation.m_plane,
            seed_materials=["GaN", "AlGaN"],
        ),
    ]
    return geo, steps


def scenario_semipolar(mat, rec) -> tuple:
    """Semi-polar growth at 32° from c-axis ({11-22}-like facet)."""
    domain = 200.0
    geo = Geometry.substrate("GaN", domain_width_nm=domain, thickness_nm=20)
    steps = [
        EpitaxialGrowth(
            name="GaN template semi-polaire",
            material="GaN",
            thickness=Length.nm(30),
            orientation=GrowthOrientation.semi_polar,
            angle_deg=32.0,
        ),
        EpitaxialGrowth(
            name="InGaN puits semi-polaire (hetero)",
            material="InGaN",
            thickness=Length.nm(4),
            orientation=GrowthOrientation.semi_polar,
            angle_deg=32.0,
            seed_materials=["GaN"],
        ),
        EpitaxialGrowth(
            name="GaN barrierre semi-polaire",
            material="GaN",
            thickness=Length.nm(10),
            orientation=GrowthOrientation.semi_polar,
            angle_deg=32.0,
            seed_materials=["GaN", "InGaN"],
        ),
    ]
    return geo, steps


def main() -> None:
    mat = default_library()
    rec = default_recipes()
    material_colors = {m.name: m.color for m in mat}

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    scenarios = [
        ("cplane_sag", scenario_cplane_sag),
        ("mplane_shell", scenario_mplane_shell),
        ("semipolar", scenario_semipolar),
    ]

    for tag, builder in scenarios:
        print(f"\n=== {tag} ===")
        geo, steps = builder(mat, rec)
        frames = simulate(geo, steps, mat, rec)
        for f in frames:
            areas = ", ".join(f"{l.material}={l.polygon.area:.0f}nm²" for l in f.layers)
            print(f"  [{f.step_index:02d}] {f.step_kind:20s} {f.step_name:35s} {areas}")
        save_svg(str(out_dir / f"epitaxy_{tag}_final.svg"), frames[-1], material_colors)
        print(f"  -> saved epitaxy_{tag}_final.svg")


if __name__ == "__main__":
    main()
