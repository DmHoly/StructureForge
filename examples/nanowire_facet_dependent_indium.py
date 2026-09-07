"""Nanofil semipolaire avec incorporation d'indium dépendante de la facette - plus d'indium
sur le plan C, moins sur les plans M et semi-polaires, dans une seule et même période InGaN.

`Geometry.deposit_faceted(material_c=..., material_m=..., material_sp=...)` fait croître un
film dont chaque facette peut recevoir un matériau différent - typiquement le même alliage à
une composition différente. C'est le pendant physique bien connu de la croissance III-N réelle :
le taux d'incorporation de l'indium dépend fortement de l'orientation du plan de croissance (plus
fort en plan C, plus faible en plan M/semi-polaire), ce qui donne des longueurs d'onde d'émission
différentes selon la facette dans un même puits quantique de nanofil.

Même germe et même pointe que `nanowire_pencil.py`. Trois périodes InGaN identiques en taux de
croissance (rate_c=1.0/rate_m=0.5/rate_sp=0.5) mais chacune avec `material_c` nettement plus
riche en indium que `material_m`/`material_sp` - visuellement, chaque anneau InGaN doit montrer
une teinte différente sur son sommet (plan C) et sur ses flancs (plan M/SP), au lieu d'une seule
couleur uniforme.

Run : python examples/nanowire_facet_dependent_indium.py
SVG : examples/output/nanowire_facet_dependent_indium.svg
"""

from __future__ import annotations

from pathlib import Path

import nanowire_pencil as pencil
from structureforge.core.materials import indium_gan
from structureforge.geometry.engine import Geometry

DOMAIN_NM = pencil.DOMAIN_NM
SP_ANGLE_DEG = pencil.SP_ANGLE_DEG

CORE_RATE_C, CORE_RATE_M, CORE_RATE_SP = 1.0, 0.8, 0.6
CORE_STEPS = [3.0] * 7

QW_T = 2.5   # épaisseur de chaque puits InGaN
QB_T = 5.0   # épaisseur de chaque barrière GaN
N_PERIODS = 3
C_PLANE_INDIUM = 0.35   # fraction d'indium incorporée sur le plan C (sommet)
SIDE_INDIUM = 0.05      # fraction d'indium incorporée sur les plans M et semi-polaires (flancs)


def build() -> Geometry:
    c_rich = indium_gan(C_PLANE_INDIUM)
    side_lean = indium_gan(SIDE_INDIUM)
    seed_materials = ["GaN", c_rich.name, side_lean.name]

    g = pencil._seed_geometry()
    for thickness_nm in CORE_STEPS:
        g.deposit_faceted(
            "GaN", thickness_nm=thickness_nm, rate_c=CORE_RATE_C, rate_m=CORE_RATE_M, rate_sp=CORE_RATE_SP,
            semi_polar_angle_deg=SP_ANGLE_DEG, seed_materials=["GaN"],
        )
    for _ in range(N_PERIODS):
        g.deposit_faceted(
            "GaN", thickness_nm=QW_T, rate_c=1.0, rate_m=0.5, rate_sp=0.5,
            semi_polar_angle_deg=SP_ANGLE_DEG, seed_materials=seed_materials,
            material_c=c_rich.name, material_m=side_lean.name, material_sp=side_lean.name,
        )
        g.deposit_faceted(
            "GaN", thickness_nm=QB_T, rate_c=1.0, rate_m=1.0, rate_sp=1.0,
            semi_polar_angle_deg=SP_ANGLE_DEG, seed_materials=seed_materials,
        )
    return g, c_rich, side_lean


def main() -> None:
    print(f"Plan C : {C_PLANE_INDIUM:.0%} d'indium — plans M/SP : {SIDE_INDIUM:.0%} d'indium")
    g, c_rich, side_lean = build()

    new_materials = {layer.material for layer in g.layers[3:]}
    print("Matériaux présents dans la pile déposée :", sorted(new_materials))

    pencil.COLORS = {"GaN": pencil.COLORS["GaN"], "SiO2": pencil.COLORS["SiO2"]}
    pencil.COLORS[c_rich.name] = c_rich.color
    pencil.COLORS[side_lean.name] = side_lean.color

    label = f"Indium facette-dépendant — plan C={C_PLANE_INDIUM:.0%}, plans M/SP={SIDE_INDIUM:.0%}"
    frame = pencil.geometry_to_frame(g, label)
    scenarios = [pencil.Scenario(label=frame.step_name, frame=frame)]
    svg = pencil.build_combined_svg(scenarios)
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    svg_path = out_dir / "nanowire_facet_dependent_indium.svg"
    svg_path.write_text(svg, encoding="utf-8")
    print(f"\nSVG enregistré : {svg_path}")


if __name__ == "__main__":
    main()
