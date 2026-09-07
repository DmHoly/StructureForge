"""Nanofil à méplat — un facette C qui naît et grandit sur une pointe semipolaire, puis un
second matériau plus plan-C referme le sommet, sans jamais faire croître le plan M.

Trois étapes, un seul germe (substrat + masque SiO2 SAG + pilier GaN), toutes sans rate_m :

  1. Cœur GaN     : la même pointe aiguë que `nanowire_pencil.py` (rate_m y est actif, pour
                    façonner le cône évasé) - le germe semipolaire sur lequel tout le reste pousse.
  2. InGaN bulk   : rate_sp dominant, rate_m=0 - croissance "sur le semipolaire". Le rate_c
                    modeste mais non nul fait naître une petite facette C au sommet (le coin
                    entre les deux facettes SP couvre déjà la direction (0,1) dès qu'il est
                    parfaitement aigu), et rate_sp > rate_c*cos(angle) la fait grandir à chaque
                    pas au lieu de se refermer.
  3. GaN plan-C   : rate_c dominant, rate_sp plus lent que celui de l'InGaN, rate_m=0 toujours.
                    Ici rate_sp < rate_c*cos(angle) : le méplat se referme à nouveau,
                    progressivement, formant un capuchon qui rétrécit sur le bulge InGaN.

Le sens du rétrécissement/élargissement d'une facette C flanquée de facettes SP ne dépend que
de rate_c, rate_sp et l'angle - jamais de rate_m (qui ne fait qu'évaser les flancs, voir
`nanowire_pencil.build_pencil`) :

    rate_sp > rate_c * cos(angle)  ->  la facette C s'élargit à chaque pas
    rate_sp < rate_c * cos(angle)  ->  la facette C se referme à chaque pas

Chaque étape utilise des incréments volontairement petits : un grand pas de croissance
appliqué sur une facette encore étroite peut faire dépasser le point d'onglet loin de l'autre
côté de la facette (le même risque de "miter join" qu'ailleurs dans le moteur), ce qui produit
un pincement en double-losange au lieu d'un élargissement net.

Run : python examples/nanowire_meplat.py
SVG : examples/output/nanowire_meplat.svg
"""

from __future__ import annotations

from pathlib import Path

from shapely.geometry import LineString
from shapely.ops import unary_union

import nanowire_pencil as pencil
from structureforge.geometry.engine import Geometry

DOMAIN_NM = pencil.DOMAIN_NM
SP_ANGLE_DEG = pencil.SP_ANGLE_DEG

# Cœur GaN : reprend exactement la "pointe aiguë" de nanowire_pencil.py.
CORE_RATE_C, CORE_RATE_M, CORE_RATE_SP = 1.0, 0.8, 0.6
CORE_STEPS = [3.0] * 7

# InGaN bulk : semipolaire dominant (sur le SP), rate_m=0, rate_c modeste mais suffisant pour
# que rate_sp > rate_c*cos(angle) - la facette C naît et grandit à chaque pas.
INGAN_RATE_C, INGAN_RATE_SP = 0.3, 1.0
INGAN_STEPS = [2.0] * 4

# Capuchon GaN "plan-C" : rate_c dominant, rate_sp plus lent que celui de l'InGaN, rate_m=0 -
# rate_sp < rate_c*cos(angle) cette fois, donc la facette C se referme à nouveau.
CAP_RATE_C, CAP_RATE_SP = 1.0, 0.3
CAP_STEPS = [1.0] * 5


def build_core() -> Geometry:
    g = pencil._seed_geometry()
    for thickness_nm in CORE_STEPS:
        g.deposit_faceted(
            "GaN", thickness_nm=thickness_nm, rate_c=CORE_RATE_C, rate_m=CORE_RATE_M, rate_sp=CORE_RATE_SP,
            semi_polar_angle_deg=SP_ANGLE_DEG, seed_materials=["GaN"],
        )
    return g


def meplat_width(g: Geometry) -> tuple[float, float]:
    """(top y, exposed width just below it) - a quick read on how wide the topmost facet is."""
    solid = unary_union([layer.polygon for layer in g.layers[2:]])
    top_y = solid.bounds[3]
    line = LineString([(0.0, top_y - 0.02), (DOMAIN_NM, top_y - 0.02)])
    inter = solid.intersection(line)
    return top_y, (inter.length if not inter.is_empty else 0.0)


def main() -> None:
    print("Cœur GaN (pointe aiguë semipolaire) …")
    g = build_core()
    print(f"  top={meplat_width(g)[0]:.1f}nm, méplat naissant={meplat_width(g)[1]:.2f}nm")

    print("InGaN bulk sur le semipolaire (rate_sp dominant, rate_m=0) …")
    for i, thickness_nm in enumerate(INGAN_STEPS):
        g.deposit_faceted(
            "InGaN", thickness_nm=thickness_nm, rate_c=INGAN_RATE_C, rate_m=0.0, rate_sp=INGAN_RATE_SP,
            semi_polar_angle_deg=SP_ANGLE_DEG, seed_materials=["GaN", "InGaN"],
        )
        top_y, width = meplat_width(g)
        print(f"  pas {i + 1}/{len(INGAN_STEPS)}: top={top_y:.1f}nm, méplat={width:.2f}nm")
    frame_after_ingan = pencil.geometry_to_frame(g, "Après InGaN bulk — méplat naissant")

    print("Capuchon GaN plan-C (rate_c dominant, rate_sp plus lent, rate_m=0) …")
    for i, thickness_nm in enumerate(CAP_STEPS):
        g.deposit_faceted(
            "GaN", thickness_nm=thickness_nm, rate_c=CAP_RATE_C, rate_m=0.0, rate_sp=CAP_RATE_SP,
            semi_polar_angle_deg=SP_ANGLE_DEG, seed_materials=["GaN", "InGaN"],
        )
        top_y, width = meplat_width(g)
        print(f"  pas {i + 1}/{len(CAP_STEPS)}: top={top_y:.1f}nm, méplat={width:.2f}nm")
    frame_final = pencil.geometry_to_frame(g, "Après capuchon GaN — méplat refermé")

    print(f"\nSolide final : {g.solid().geom_type}")

    scenarios = [
        pencil.Scenario(label=frame_after_ingan.step_name, frame=frame_after_ingan),
        pencil.Scenario(label=frame_final.step_name, frame=frame_final),
    ]
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    svg_path = out_dir / "nanowire_meplat.svg"
    svg_path.write_text(pencil.build_combined_svg(scenarios), encoding="utf-8")
    print(f"\nSVG enregistré : {svg_path}")


if __name__ == "__main__":
    main()
