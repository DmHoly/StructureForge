"""Nanowire en crayon — vue en coupe 2D (III-N nitride, plan C), simulé via le vrai moteur.

Deux géométries de pointe côte à côte dans un seul SVG, obtenues en appliquant
`Geometry.deposit_faceted` en petits incréments répétés (comme le ferait un pas
`FacetedGrowth` réel dans un pipeline de simulation) sur un germe identique (substrat +
masque SiO2 SAG + court pilier GaN) - la forme du crayon est un résultat de la simulation,
pas une formule géométrique écrite à la main :

  • Pointe plate  : rate_m modeste face à rate_c → le sommet C reste large, à peine chanfreiné.
  • Pointe aiguë  : rate_m nettement plus grand, rate_c > rate_sp*cos(θ) → les flancs s'évasent
                    vite pendant que le sommet C, qui n'avance pas aussi vite que l'exigerait
                    la facette SP, se referme en pointe.

Les puits quantiques (MQW) sont ajoutés par les mêmes incréments de `deposit_faceted`, avec
les trois taux égaux (croissance conforme) et sélectifs au GaN/InGaN déjà exposé (SAG) - ils
suivent donc fidèlement la forme du crayon, plate ou pointue, telle qu'elle a émergé.

Run : python examples/nanowire_pencil.py
SVG : examples/output/nanowire_pencil.svg
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from shapely.geometry import box
from shapely.ops import unary_union

from structureforge.geometry.engine import Geometry, Layer
from structureforge.process.simulate import Frame

# ---------------------------------------------------------------------------
# Paramètres communs
# ---------------------------------------------------------------------------
DOMAIN_NM = 140.0
SUBSTRATE_H = 15.0      # épaisseur substrat GaN
MASK_H = 40.0           # épaisseur masque SiO2
PILLAR_SEED_H = 40.0    # germe de pilier au-dessus du masque, avant croissance à facettes
W = 20.0                # largeur de la fenêtre SAG / du germe
CX = DOMAIN_NM / 2.0    # centre = 70 nm

SP_ANGLE_DEG = 32.0      # angle semi-polaire depuis c-axe (degrés)

N_QW = 3                 # périodes MQW
QW_T = 2.5               # épaisseur InGaN QW
QB_T = 5.0               # épaisseur GaN barrière

# Couleurs (proches des valeurs par défaut de la librairie StructureForge)
COLORS = {
    "GaN":   "#4e8ec5",
    "SiO2":  "#c8a96e",
    "InGaN": "#6cbe6c",
}

# ---------------------------------------------------------------------------
# Construction de la géométrie crayon (simulation réelle, deposit_faceted)
# ---------------------------------------------------------------------------


def _seed_geometry() -> Geometry:
    """Substrat + masque SiO2 SAG (ouverture de largeur W) + court germe de pilier GaN."""
    g = Geometry(domain_width_nm=DOMAIN_NM)
    g.layers.append(Layer(material="GaN", polygon=box(0.0, -SUBSTRATE_H, DOMAIN_NM, 0.0)))
    mask = unary_union([
        box(0.0, 0.0, CX - W / 2, MASK_H),
        box(CX + W / 2, 0.0, DOMAIN_NM, MASK_H),
    ])
    g.layers.append(Layer(material="SiO2", polygon=mask))
    g.layers.append(Layer(material="GaN", polygon=box(CX - W / 2, 0.0, CX + W / 2, PILLAR_SEED_H)))
    return g


def build_pencil(rate_c: float, rate_m: float, rate_sp: float, step_thicknesses: list[float]) -> Geometry:
    """Fait croître le crayon depuis le germe via `deposit_faceted`, un appel par épaisseur de
    `step_thicknesses`. `rate_m` ne fait qu'évaser les flancs (il ne change pas la vitesse à
    laquelle le sommet se referme) ; ce qui décide si le sommet C rétrécit ou s'élargit à chaque
    pas est uniquement le rapport rate_c / rate_sp : tant que rate_c > rate_sp*cos(θ), la facette
    SP avance verticalement moins vite que le plan C, donc le coin où elle rencontre le plan C se
    rapproche du centre à chaque pas - un `rate_m` généreux donne alors un évasement net suivi
    d'une pointe franche. Si cette condition s'inverse (rate_sp*cos(θ) > rate_c), le sommet C
    s'élargit au lieu de rétrécir. La sélectivité SAG (`seed_materials`) confine toute la
    croissance au GaN déjà exposé, comme le ferait un vrai masque SiO2.

    Volontairement peu d'incréments, mais chacun plus épais, plutôt que beaucoup de pas fins :
    `_offset_named_facets` mitre chaque nouveau coin à partir du contour déjà déformé par le pas
    précédent, et un très grand nombre de pas fins laisse s'accumuler de minuscules artefacts
    numériques (auto-intersections quasi dégénérées) le long du contour - inoffensifs pour la
    physique simulée, mais visibles à l'écran. Peu de pas, plus francs, donne un contour propre.
    """
    g = _seed_geometry()
    for thickness_nm in step_thicknesses:
        g.deposit_faceted(
            "GaN",
            thickness_nm=thickness_nm,
            rate_c=rate_c,
            rate_m=rate_m,
            rate_sp=rate_sp,
            semi_polar_angle_deg=SP_ANGLE_DEG,
            seed_materials=["GaN"],
        )

    # MQW : incréments conformes (taux égaux sur les trois familles de facettes) pour envelopper
    # fidèlement la forme du crayon telle qu'elle a émergé, plate ou pointue.
    for _ in range(N_QW):
        g.deposit_faceted(
            "InGaN",
            thickness_nm=QW_T,
            rate_c=1.0,
            rate_m=1.0,
            rate_sp=1.0,
            semi_polar_angle_deg=SP_ANGLE_DEG,
            seed_materials=["GaN", "InGaN"],
        )
        g.deposit_faceted(
            "GaN",
            thickness_nm=QB_T,
            rate_c=1.0,
            rate_m=1.0,
            rate_sp=1.0,
            semi_polar_angle_deg=SP_ANGLE_DEG,
            seed_materials=["GaN", "InGaN"],
        )
    return g


def _merge_consecutive_same_material(layers: list[Layer]) -> list[Layer]:
    """Collapse runs of consecutive same-material layers into one - each `deposit_faceted` call
    above appends its own thin increment as a separate layer (so the process history stays
    accurate), but for a picture of the *final* crystal, a stack of 6-11 GaN slivers should read
    as one continuous shape, not one stroked outline per increment.
    """
    merged: list[Layer] = []
    for layer in layers:
        if merged and merged[-1].material == layer.material:
            merged[-1] = Layer(material=layer.material, polygon=unary_union([merged[-1].polygon, layer.polygon]))
        else:
            merged.append(layer)
    return merged


def geometry_to_frame(g: Geometry, label: str) -> Frame:
    layers = _merge_consecutive_same_material([layer for layer in g.layers if not layer.polygon.is_empty])
    return Frame(
        step_index=0,
        step_kind="pencil",
        step_name=label,
        layers=layers,
        domain_width_nm=DOMAIN_NM,
    )


# ---------------------------------------------------------------------------
# SVG combiné : deux panneaux côte à côte
# ---------------------------------------------------------------------------

_MARGIN     = 10.0
_GAP        = 28.0
_SCALE_BAR  = 20.0
_LABEL_FONT = 6.5
_TICK_FONT  = 4.5
_GRID_STEP  = 20.0
_GRID_COLOR = "#e4e7ee"
_AXIS_COLOR = "#9aa1ab"
_BG         = "#ffffff"
_LABEL_CLR  = "#1a1d23"
_SP_COLOR   = "#e07b39"


def _ring_d(ring: dict) -> str:
    def seg(pts):
        return "M " + " L ".join(f"{x:.2f},{y:.2f}" for x, y in pts) + " Z"
    d = seg(ring["exterior"])
    for h in ring["holes"]:
        d += " " + seg(h)
    return d


def _frame_paths(frame: Frame) -> list[str]:
    paths = []
    for layer in frame.layers:
        rings = layer.rings()
        if not rings:
            continue
        d = " ".join(_ring_d(r) for r in rings)
        color = COLORS.get(layer.material, "#aaa")
        paths.append(
            f'<path d="{d}" fill="{color}" fill-rule="evenodd" '
            f'stroke="rgba(0,0,0,0.18)" stroke-width="0.5">'
            f"<title>{layer.material}</title></path>"
        )
    return paths


def _grid_lines(x0, x1, y0, y1) -> list[str]:
    lines = []
    for i in range(int(x0 / _GRID_STEP) - 1, int(x1 / _GRID_STEP) + 2):
        x = i * _GRID_STEP
        if x0 <= x <= x1:
            cls = "axis" if abs(x) < 0.01 else "grid"
            lines.append(f'<line class="{cls}" x1="{x:.1f}" y1="{-y0:.1f}" x2="{x:.1f}" y2="{-y1:.1f}"/>')
    for j in range(int(y0 / _GRID_STEP) - 1, int(y1 / _GRID_STEP) + 2):
        y = j * _GRID_STEP
        if y0 <= y <= y1:
            cls = "axis" if abs(y) < 0.01 else "grid"
            lines.append(f'<line class="{cls}" x1="{x0:.1f}" y1="{-y:.1f}" x2="{x1:.1f}" y2="{-y:.1f}"/>')
    return lines


def _ytick_labels(y0, y1, svg_ox, svg_oy) -> list[str]:
    labels = []
    for j in range(int(y0 / _GRID_STEP) - 1, int(y1 / _GRID_STEP) + 2):
        y = j * _GRID_STEP
        if y0 <= y <= y1 and j != 0:
            svg_y = svg_oy - y
            labels.append(
                f'<text x="{svg_ox - 2:.1f}" y="{svg_y + _TICK_FONT * 0.4:.1f}" '
                f'font-size="{_TICK_FONT}" text-anchor="end" fill="{_AXIS_COLOR}">{y:g}</text>'
            )
    return labels


@dataclass
class Scenario:
    label: str
    frame: Frame


def build_combined_svg(scenarios: list[Scenario]) -> str:
    # compute y bounds from all layers
    all_y = []
    for sc in scenarios:
        for layer in sc.frame.layers:
            if not layer.polygon.is_empty:
                all_y += [layer.polygon.bounds[1], layer.polygon.bounds[3]]
    y0 = min(all_y) - _MARGIN
    y1 = max(all_y) + _MARGIN * 2

    panel_w = DOMAIN_NM + 2 * _MARGIN
    panel_h = y1 - y0
    ax_off  = 36.0   # space for y-axis labels

    n      = len(scenarios)
    svg_w  = ax_off + n * panel_w + (n - 1) * _GAP + 4
    svg_h  = panel_h + 30     # for title above + legend below

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{svg_w * 4:.0f}" height="{svg_h * 4:.0f}" '
        f'viewBox="0 0 {svg_w:.1f} {svg_h:.1f}">',
        "<defs><style>",
        f"  .grid {{ stroke: {_GRID_COLOR}; stroke-width: 0.4; }}",
        f"  .axis {{ stroke: {_AXIS_COLOR}; stroke-width: 0.7; }}",
        f"  text {{ font-family: -apple-system, 'Segoe UI', sans-serif; }}",
        "</style></defs>",
        f'<rect width="{svg_w:.1f}" height="{svg_h:.1f}" fill="{_BG}"/>',
    ]

    for idx, sc in enumerate(scenarios):
        ox = ax_off + idx * (panel_w + _GAP)   # SVG x of panel left edge
        oy = 16.0                                # SVG y of panel top
        # SVG y when nm y = 0
        y_origin_svg = oy + y1

        # -- grid (in flipped group) -----------------------------------------
        lines.append(f'<g transform="translate({ox + _MARGIN:.1f},{oy:.1f}) scale(1,-1)">')
        for gl in _grid_lines(-_MARGIN, DOMAIN_NM + _MARGIN, y0 - oy, y1 - oy):
            lines.append(gl)
        lines.append("</g>")

        # -- geometry (flip y) ------------------------------------------------
        g_tx = ox + _MARGIN
        g_ty = y_origin_svg
        lines.append(f'<g transform="translate({g_tx:.1f},{g_ty:.1f}) scale(1,-1)">')
        lines += _frame_paths(sc.frame)
        lines.append("</g>")

        # -- SP angle annotation on right side of tip --------------------------
        theta = math.radians(SP_ANGLE_DEG)
        tip_xR = CX + W / 2          # right edge of pillar
        tip_yB = PILLAR_SEED_H       # bottom of tip region (top of the straight seed)
        arrow_len = 18.0
        ax0s = g_tx + tip_xR
        ay0s = g_ty - tip_yB
        ax1s = ax0s + arrow_len * math.sin(theta)
        ay1s = ay0s - arrow_len * math.cos(theta)
        lines += [
            f'<line x1="{ax0s:.1f}" y1="{ay0s:.1f}" x2="{ax1s:.1f}" y2="{ay1s:.1f}" '
            f'stroke="{_SP_COLOR}" stroke-width="1.0" stroke-dasharray="2,1.5"/>',
            f'<text x="{ax1s + 1:.1f}" y="{ay1s - 1:.1f}" font-size="{_TICK_FONT}" '
            f'fill="{_SP_COLOR}">{SP_ANGLE_DEG:.0f}°</text>',
        ]

        # -- panel title -------------------------------------------------------
        cx_label = ox + panel_w / 2
        lines.append(
            f'<text x="{cx_label:.1f}" y="{oy - 3:.1f}" '
            f'font-size="{_LABEL_FONT}" font-weight="700" text-anchor="middle" '
            f'fill="{_LABEL_CLR}">{sc.label}</text>'
        )

        # -- y-axis tick labels (first panel only) ----------------------------
        if idx == 0:
            for lbl in _ytick_labels(y0, y1, ox + _MARGIN, y_origin_svg):
                lines.append(lbl)
            lines.append(
                f'<text x="{ox - _MARGIN * 1.5:.1f}" y="{oy + panel_h / 2:.1f}" '
                f'font-size="{_TICK_FONT}" fill="{_AXIS_COLOR}" text-anchor="middle" '
                f'transform="rotate(-90,{ox - _MARGIN * 1.5:.1f},{oy + panel_h / 2:.1f})">y (nm)</text>'
            )

    # -- legend ---------------------------------------------------------------
    leg_x = ax_off
    leg_y = 16.0 + panel_h + 6.0
    box_s = 5.0
    gap   = 2.0
    x_cur = leg_x
    for mat, color in COLORS.items():
        lines.append(
            f'<rect x="{x_cur:.1f}" y="{leg_y:.1f}" width="{box_s}" height="{box_s}" '
            f'fill="{color}" stroke="rgba(0,0,0,0.3)" stroke-width="0.4"/>'
        )
        lines.append(
            f'<text x="{x_cur + box_s + gap:.1f}" y="{leg_y + box_s * 0.85:.1f}" '
            f'font-size="{_TICK_FONT}" fill="{_LABEL_CLR}">{mat}</text>'
        )
        x_cur += box_s + gap + len(mat) * _TICK_FONT * 0.62 + 8.0

    # -- scale bar ------------------------------------------------------------
    bar_x = svg_w - 8 - _SCALE_BAR
    bar_y = leg_y + 1.0
    lines += [
        f'<line x1="{bar_x:.1f}" y1="{bar_y:.1f}" x2="{bar_x + _SCALE_BAR:.1f}" y2="{bar_y:.1f}" '
        f'stroke="{_LABEL_CLR}" stroke-width="1.2"/>',
        f'<line x1="{bar_x:.1f}" y1="{bar_y - 2:.1f}" x2="{bar_x:.1f}" y2="{bar_y + 2:.1f}" '
        f'stroke="{_LABEL_CLR}" stroke-width="1.2"/>',
        f'<line x1="{bar_x + _SCALE_BAR:.1f}" y1="{bar_y - 2:.1f}" '
        f'x2="{bar_x + _SCALE_BAR:.1f}" y2="{bar_y + 2:.1f}" '
        f'stroke="{_LABEL_CLR}" stroke-width="1.2"/>',
        f'<text x="{bar_x + _SCALE_BAR / 2:.1f}" y="{bar_y + 6:.1f}" '
        f'font-size="{_TICK_FONT}" text-anchor="middle" fill="{_LABEL_CLR}">{_SCALE_BAR:.0f} nm</text>',
    ]

    lines.append("</svg>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Pointe plate  : rate_c dominant (1.0) face à rate_sp modeste (0.7) -> le sommet C reste
    #                 large, seulement chanfreiné aux coins par la facette SP.
    # Pointe aiguë  : rate_m/rate_sp dominants (1.0) face à rate_c modeste (0.3) -> les facettes
    #                 SP referment le sommet en pointe avant que le plan C ne puisse suivre.
    print("Croissance pointe plate  (rate_c=1.0, rate_m=0.4, rate_sp=0.7) …")
    g_flat = build_pencil(rate_c=1.0, rate_m=0.4, rate_sp=0.7, step_thicknesses=[8.0, 8.0])

    print("Croissance pointe aiguë  (rate_c=1.0, rate_m=0.8, rate_sp=0.6) …")
    g_sharp = build_pencil(rate_c=1.0, rate_m=0.8, rate_sp=0.6, step_thicknesses=[3.0] * 7)

    label_flat  = "Pointe plate — c=1.0 / m=0.4 / sp=0.7"
    label_sharp = "Pointe aiguë — c=1.0 / m=0.8 / sp=0.6"
    frame_flat  = geometry_to_frame(g_flat, label_flat)
    frame_sharp = geometry_to_frame(g_sharp, label_sharp)

    for label, frame in [("Plate", frame_flat), ("Aiguë", frame_sharp)]:
        top_y = max(l.polygon.bounds[3] for l in frame.layers if not l.polygon.is_empty)
        mats  = ", ".join(f"{l.material}({l.polygon.area:.0f}nm²)" for l in frame.layers)
        print(f"  [{label}] top={top_y:.1f}nm | {mats}")

    scenarios = [
        Scenario(label=frame_flat.step_name,  frame=frame_flat),
        Scenario(label=frame_sharp.step_name, frame=frame_sharp),
    ]

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    svg_path = out_dir / "nanowire_pencil.svg"
    svg_path.write_text(build_combined_svg(scenarios), encoding="utf-8")
    print(f"\nSVG enregistré : {svg_path}")


if __name__ == "__main__":
    main()
