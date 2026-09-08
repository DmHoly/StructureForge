"""Nanofil semipolaire avec un puits quantique InGaN gradué en indium - chaque période a sa
propre fraction d'indium (`structureforge.core.materials.indium_gan`), et sa couleur en découle
directement : la couleur code la composition, ce n'est plus une étiquette arbitraire.

Même germe et même pointe que `nanowire_pencil.py` (rate_c=1.0/rate_m=0.8/rate_sp=0.6). Au lieu
d'un unique "InGaN" à couleur fixe, les puits successifs prennent des fractions d'indium
croissantes (5% à 45%, via `indium_gan_gradient`) - `indium_gan(x)` renvoie un `Material` nommé
et coloré à partir de x, donc chaque période est un matériau distinct dans la pile de couches,
avec sa propre couleur interpolée entre celle du GaN et un rouge profond "haute teneur en indium".

Run : python examples/nanowire_ingan_gradient.py
SVG : examples/output/nanowire_ingan_gradient.svg
"""

from __future__ import annotations

from pathlib import Path

import nanowire_pencil as pencil
from structureforge.core.materials import Material, indium_gan_gradient
from structureforge.geometry.engine import Geometry

DOMAIN_NM = pencil.DOMAIN_NM
SP_ANGLE_DEG = pencil.SP_ANGLE_DEG

CORE_RATE_C, CORE_RATE_M, CORE_RATE_SP = 1.0, 0.8, 0.6
CORE_STEPS = [3.0] * 7

QW_T = 2.5   # épaisseur de chaque puits InGaN
QB_T = 5.0   # épaisseur de chaque barrière GaN
INDIUM_FRACTION_START, INDIUM_FRACTION_END, N_WELLS = 0.05, 0.45, 5


def build(wells: list[Material]) -> Geometry:
    g = pencil._seed_geometry()
    for thickness_nm in CORE_STEPS:
        g.deposit_faceted(
            "GaN", thickness_nm=thickness_nm, rate_c=CORE_RATE_C, rate_m=CORE_RATE_M, rate_sp=CORE_RATE_SP,
            semi_polar_angle_deg=SP_ANGLE_DEG, seed_materials=["GaN"],
        )
    for well in wells:
        g.deposit_faceted(
            well.name, thickness_nm=QW_T, rate_c=1.0, rate_m=1.0, rate_sp=1.0,
            semi_polar_angle_deg=SP_ANGLE_DEG, seed_materials=["GaN", well.name],
        )
        g.deposit_faceted(
            "GaN", thickness_nm=QB_T, rate_c=1.0, rate_m=1.0, rate_sp=1.0,
            semi_polar_angle_deg=SP_ANGLE_DEG, seed_materials=["GaN", well.name],
        )
    return g


def main() -> None:
    wells = indium_gan_gradient(INDIUM_FRACTION_START, INDIUM_FRACTION_END, N_WELLS)
    print("Puits InGaN à déposer (fraction d'indium -> couleur) :")
    for well in wells:
        print(f"  {well.name:16s} color={well.color}")

    print("Cœur GaN semipolaire, puis les puits gradués …")
    g = build(wells)

    # nanowire_pencil.COLORS is read by _frame_paths at module scope - swap it for one matching
    # exactly what this structure uses, so each indium_gan material gets its own interpolated
    # color (and the legend doesn't carry the unused flat "InGaN" entry from that other example).
    pencil.COLORS = {"GaN": pencil.COLORS["GaN"], "SiO2": pencil.COLORS["SiO2"]}
    pencil.COLORS.update({well.name: well.color for well in wells})

    label = f"InGaN gradué {INDIUM_FRACTION_START:.0%} -> {INDIUM_FRACTION_END:.0%} (couleur = teneur en indium)"
    frame = pencil.geometry_to_frame(g, label)
    scenarios = [pencil.Scenario(label=frame.step_name, frame=frame)]
    svg = pencil.build_combined_svg(scenarios)
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    svg_path = out_dir / "nanowire_ingan_gradient.svg"
    svg_path.write_text(svg, encoding="utf-8")
    print(f"\nSVG enregistré : {svg_path}")


if __name__ == "__main__":
    main()
