"""Nanowire en crayon — vue en coupe 2D (III-N nitride, plan C).

Deux géométries de pointe côte à côte dans un seul SVG :

  • Pointe plate  : plan C rapide → le sommet reste plat, petits chanfreins SP aux coins.
  • Pointe aiguë : plan SP rapide → les facettes SP convergent en pyramide avant que
                   le plan C ait le temps de « rattraper ».

La géométrie du crayon est construite directement par calcul de Wulff analytique depuis
le bord supérieur du pilier (surface exposée), plus réaliste qu'un Minkowski sum depuis
toute la hauteur du pilier.  Les puits quantiques (MQW) sont ajoutés par buffer Shapely
pour être conformes à la forme du crayon.

Run : python examples/nanowire_pencil.py
SVG : examples/output/nanowire_pencil.svg
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from structureforge.geometry.engine import Layer
from structureforge.process.simulate import Frame

# ---------------------------------------------------------------------------
# Paramètres communs
# ---------------------------------------------------------------------------
DOMAIN_NM = 140.0
SUBSTRATE_H = 15.0       # épaisseur substrat GaN
MASK_H = 40.0            # épaisseur masque SiO2
PILLAR_H = 80.0          # hauteur du corps rectangulaire (m-plane walls)
W = 20.0                 # largeur de la fenêtre SAG / du pilier
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
# Construction de la géométrie crayon (analytique, Wulff depuis bord supérieur)
# ---------------------------------------------------------------------------

def _pillar_top_y() -> float:
    return PILLAR_H   # y du sommet du corps rectangulaire (y=0 = haut du substrat)


def _pencil_tip(sp_chamfer_nm: float, flat_top_nm: float) -> Polygon:
    """Polygone de la pointe du crayon.

    Les facettes SP naissent aux coins supérieurs du pilier et penchent vers
    l'intérieur à `SP_ANGLE_DEG` depuis l'axe c.

    sp_chamfer_nm : avancée latérale (inward) des facettes SP.
                    Si sp_chamfer_nm >= W/2 → pointe aiguë (triangle).
    flat_top_nm   : hauteur supplémentaire du plan C après les SP (0 = pointe aiguë).
    """
    theta = math.radians(SP_ANGLE_DEG)
    Y0 = _pillar_top_y()
    xL = CX - W / 2
    xR = CX + W / 2
    sp_chamfer_nm = min(sp_chamfer_nm, W / 2)
    dh_sp = sp_chamfer_nm / math.tan(theta)   # hauteur verticale du chanfrein SP

    if sp_chamfer_nm >= W / 2 - 1e-6 and flat_top_nm <= 0:
        # Pointe aiguë : triangle
        h_apex = (W / 2) / math.tan(theta)
        return Polygon([(xL, Y0), (xR, Y0), (CX, Y0 + h_apex)])
    else:
        xL_top = xL + sp_chamfer_nm
        xR_top = xR - sp_chamfer_nm
        y_sp   = Y0 + dh_sp
        y_top  = y_sp + flat_top_nm
        return Polygon([
            (xL,     Y0),
            (xR,     Y0),
            (xR_top, y_sp),
            (xR_top, y_top),
            (xL_top, y_top),
            (xL_top, y_sp),
        ])


def build_pencil_layers(sp_chamfer_nm: float, flat_top_nm: float
                        ) -> list[tuple[str, object]]:
    """Construit la liste (material, polygon) du crayon + MQW.

    Returns list de (material_name, Shapely polygon).
    """
    Y0 = _pillar_top_y()

    # -- Structure de base ---------------------------------------------------
    substrate = box(0.0, -SUBSTRATE_H, DOMAIN_NM, 0.0)
    mask_l    = box(0.0,  0.0, CX - W / 2, MASK_H)
    mask_r    = box(CX + W / 2, 0.0, DOMAIN_NM, MASK_H)
    pillar    = box(CX - W / 2, 0.0, CX + W / 2, Y0)
    tip       = _pencil_tip(sp_chamfer_nm, flat_top_nm)

    gan_solid = unary_union([pillar, tip])

    # -- MQW : buffer conformel, clipé au-dessus du masque ------------------
    # Only wrap MQW on the exposed pillar surface above the SAG mask level
    clip_above = box(-1000.0, MASK_H, DOMAIN_NM + 1000.0, 1e6)
    # On wrappe uniquement la surface exposée du crayon (pas dans le masque)
    layers: list[tuple[str, object]] = [
        ("GaN",  substrate),
        ("SiO2", unary_union([mask_l, mask_r])),
        ("GaN",  gan_solid),
    ]

    current_solid = unary_union([substrate, mask_l, mask_r, gan_solid])
    current_gan   = gan_solid

    for _ in range(N_QW):
        # InGaN QW
        qw_shell = current_gan.buffer(QW_T, join_style=1).intersection(clip_above)
        qw_film  = qw_shell.difference(current_solid)
        layers.append(("InGaN", qw_film))
        current_solid = unary_union([current_solid, qw_film])
        current_gan   = qw_film

        # GaN barrière
        qb_shell = current_gan.buffer(QB_T, join_style=1).intersection(clip_above)
        qb_film  = qb_shell.difference(current_solid)
        layers.append(("GaN", qb_film))
        current_solid = unary_union([current_solid, qb_film])
        current_gan   = qb_film

    return layers


def layers_to_frame(mat_layers: list[tuple[str, object]], label: str) -> Frame:
    sf_layers = [Layer(material=m, polygon=p) for m, p in mat_layers if not p.is_empty]
    return Frame(
        step_index=0,
        step_kind="pencil",
        step_name=label,
        layers=sf_layers,
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
        tip_yB = _pillar_top_y()     # bottom of tip
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
    # Pointe plate : petits chanfreins SP (3 nm) + plan C bien plat (8 nm)
    # Pointe aiguë : facettes SP qui ferment jusqu'à l'apex (triangle)
    print("Construction pointe plate  (chanfrein SP=3 nm, plan C=8 nm) …")
    layers_flat = build_pencil_layers(sp_chamfer_nm=3.0, flat_top_nm=8.0)

    print("Construction pointe aiguë  (chanfrein SP=10 nm = fermeture totale) …")
    layers_sharp = build_pencil_layers(sp_chamfer_nm=10.0, flat_top_nm=0.0)

    label_flat  = f"Pointe plate   SP chamfer=3 nm / C=8 nm / {SP_ANGLE_DEG:.0f}°"
    label_sharp = f"Pointe aiguë   SP {SP_ANGLE_DEG:.0f}° → apex (triangle)"
    frame_flat  = layers_to_frame(layers_flat,  label_flat)
    frame_sharp = layers_to_frame(layers_sharp, label_sharp)

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
