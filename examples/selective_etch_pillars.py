"""Gravure sélective d'un oxyde pour découvrir le sommet d'un pilier GaN - trois formes de pilier.

Un pilier de GaN isolé (rien autour) est défini par masque de lithographie + gravure RIE Cl2
ICP (aspect ratio 3:1 pour un pilier droit : 120 nm de haut pour 40 nm de large), puis on lui
donne l'une de trois formes de sommet / de flanc :

  • pointu           : pilier droit coiffé d'une pointe aiguë (facettes semi-polaires raides),
  • trapèze          : tout le pilier est conique (flancs inclinés, gravure RIE à flancs évasés),
                       sommet plat plus étroit que la base,
  • pyramide aplatie : pilier droit coiffé d'une pyramide tronquée par un plan c.

Chaque pilier subit ensuite le même flow de « révélation du sommet » (damascène) :

  1. remplissage SiO2 conforme, plus épais que la hauteur du pilier (il l'ensevelit),
  2. CMP : l'oxyde est ramené plan, juste au-dessus du sommet du pilier,
  3. gravure sélective de l'oxyde (HF-vapeur modélisé par une recette sur mesure : SiO2 attaqué,
     GaN à ~1 %) jusqu'à ce que seul le sommet dépasse - la profondeur de gravure est calculée
     pour dégager un même `REVEAL_NM` de pilier quelle que soit sa forme.

Seules les facettes du sommet (pointe, pyramide) et les flancs coniques sont construits à la
main (`add_frustum_cap`) : le moteur ne sait pas graver un flanc symétrique incliné (ses
recettes directionnelles décalent les deux flancs ensemble) - même limite que
`nanowire_semipolar_tip.py`.

Run : python examples/selective_etch_pillars.py
SVG : examples/output/selective_etch_pillars.svg (grille formes x étapes) + un SVG final par forme
"""

from __future__ import annotations

import math
from pathlib import Path

from shapely.geometry import Polygon, box

from structureforge import (
    ChemicalStep,
    Deposition,
    Etch,
    EtchMode,
    EtchRecipe,
    Geometry,
    Length,
    Lithography,
    Planarization,
    ResistStrip,
    default_library,
    default_recipes,
    save_svg,
    simulate,
)
from structureforge.process.simulate import Frame

DOMAIN_NM = 200.0
CX = DOMAIN_NM / 2.0
SUBSTRATE_NM = 20.0
PILLAR_H = 120.0           # hauteur du pilier droit
PILLAR_W = 40.0            # largeur du pilier droit -> aspect ratio 3:1
REVEAL_NM = 40.0           # hauteur de pilier dégagée au final, quelle que soit la forme
CMP_MARGIN_NM = 15.0       # oxyde laissé au-dessus du sommet après CMP

# Gravure sélective de l'oxyde : attaque SiO2/Si3N4 à plein régime, le GaN à 1 % seulement.
# (C'est la sélectivité qui compte ici : l'erosion résiduelle du GaN est visible sur le sommet.)
SELECTIVE_OXIDE_ETCH = EtchRecipe(
    name="HF vapeur (sélectif SiO2/GaN)",
    mode=EtchMode.isotropic,
    selectivity_by_material={"SiO2": 1.0, "Si3N4": 0.3},
    default_factor=0.01,
    notes="Gravure humide/vapeur de l'oxyde, quasi inerte sur le GaN.",
)


def build_vertical_pillar(pillar_half_width: float) -> list:
    """GaN épitaxié à plat, masque de résine ne laissant que le pilier, RIE Cl2, retrait résine."""
    return [
        Deposition(name="GaN (couche épitaxiée)", material="GaN", recipe="MOCVD Epitaxial", thickness=Length.nm(PILLAR_H)),
        Lithography(
            name="Masque du pilier",
            resist_material="Photoresist",
            thickness=Length.nm(80),
            openings=[(0.0, CX - pillar_half_width), (CX + pillar_half_width, DOMAIN_NM)],
        ),
        Etch(name="Gravure RIE du pilier", recipe="Cl2 ICP-RIE (III-N)", depth=Length.nm(PILLAR_H)),
        ResistStrip(name="Retrait résine"),
    ]


def build_reveal_flow(top_y: float, cmp_level: float) -> list:
    """Remplissage oxyde -> CMP -> gravure sélective qui découvre `REVEAL_NM` de pilier sous le sommet."""
    etch_depth = cmp_level - (top_y - REVEAL_NM)
    return [
        Deposition(name="Remplissage SiO2", material="SiO2", recipe="CVD Conformal", thickness=Length.nm(top_y + 20.0)),
        Planarization(name="CMP oxyde", target_level=Length.nm(cmp_level)),
        Etch(name="Gravure sélective SiO2", recipe=SELECTIVE_OXIDE_ETCH.name, depth=Length.nm(etch_depth)),
    ]


# -- formes ------------------------------------------------------------------


def add_frustum_cap(g: Geometry, base_half_width: float, tip_half_width: float, facet_angle_deg: float) -> float:
    """Coiffe le pilier droit d'un tronc de pyramide : flancs symétriques inclinés de
    `facet_angle_deg` par rapport à l'horizontale, convergeant vers un sommet de demi-largeur
    `tip_half_width` (0.x nm = pointe, plus large = pyramide tronquée). Construit à la main : le
    moteur n'a pas de brique pour deux flancs symétriques convergents (`FacetEnvelope` remplit
    toute la région exposée depuis son point le plus bas, donc ferait une pyramide depuis la
    base du pilier, pas depuis son sommet). Renvoie la hauteur du sommet."""
    base_y = _pillar_top(g)
    height = (base_half_width - tip_half_width) * math.tan(math.radians(facet_angle_deg))
    cap = Polygon([
        (CX - base_half_width, base_y), (CX + base_half_width, base_y),
        (CX + tip_half_width, base_y + height), (CX - tip_half_width, base_y + height),
    ])
    for layer in g.layers:
        if layer.material == "GaN":
            layer.polygon = layer.polygon.union(cap)
    return base_y + height


def shape_pointed(g: Geometry, materials, recipes) -> tuple[list[Frame], float]:
    """Pilier droit + pointe aiguë (facettes à 75 degrés, sommet de 2 nm de large)."""
    frames = simulate(g, build_vertical_pillar(PILLAR_W / 2), materials, recipes)
    top = add_frustum_cap(g, PILLAR_W / 2, tip_half_width=1.0, facet_angle_deg=75.0)
    frames += simulate(g, [ChemicalStep(name="Pointe aiguë (facettes 75°)")], materials, recipes)[1:]
    return frames, top


def shape_flattened_pyramid(g: Geometry, materials, recipes) -> tuple[list[Frame], float]:
    """Pilier droit + pyramide semi-polaire (facettes à 62 degrés) tronquée par un plan c."""
    frames = simulate(g, build_vertical_pillar(PILLAR_W / 2), materials, recipes)
    top = add_frustum_cap(g, PILLAR_W / 2, tip_half_width=9.0, facet_angle_deg=62.0)
    frames += simulate(g, [ChemicalStep(name="Pyramide aplatie (facettes 62°)")], materials, recipes)[1:]
    return frames, top


def shape_trapezoid(g: Geometry, materials, recipes) -> tuple[list[Frame], float]:
    """Pilier entièrement conique : base 56 nm, sommet 24 nm. Construit à la main (le moteur ne
    sait pas faire converger deux flancs symétriques par un seul etch directionnel)."""
    base_hw, top_hw = 28.0, 12.0
    pillar = Polygon([(CX - base_hw, 0.0), (CX + base_hw, 0.0), (CX + top_hw, PILLAR_H), (CX - top_hw, PILLAR_H)])
    g.layers.append(type(g.layers[0])(material="GaN", polygon=pillar))
    frame = Frame(0, "carve", "Pilier trapézoïdal (RIE à flancs évasés)", [type(l)(l.material, l.polygon) for l in g.layers], DOMAIN_NM)
    return [frame], PILLAR_H


def _pillar_top(g: Geometry) -> float:
    return max(l.polygon.bounds[3] for l in g.layers if l.material == "GaN")


def run_shape(name: str, shape_fn, materials, recipes) -> list[Frame]:
    g = Geometry.substrate("Sapphire", domain_width_nm=DOMAIN_NM, thickness_nm=SUBSTRATE_NM)
    frames, top_y = shape_fn(g, materials, recipes)
    flow = build_reveal_flow(top_y, cmp_level=top_y + CMP_MARGIN_NM)
    reveal_frames = simulate(g, flow, materials, recipes)[1:]
    all_frames = frames + reveal_frames
    print(f"\n{name}: sommet du pilier à {top_y:.0f} nm, CMP à {top_y + CMP_MARGIN_NM:.0f} nm")
    for i, f in enumerate(all_frames):
        print(f"  [{i}] {f.step_kind:14s} {f.step_name}")
    return all_frames


# -- SVG en grille -----------------------------------------------------------

COLUMN_TITLES = ["Pilier GaN", "Remplissage SiO2", "CMP", "Gravure sélective"]


def _panel(frame: Frame, colors: dict[str, str], y_min: float, y_max: float) -> str:
    from structureforge.presentation.svg import _ring_path

    paths = []
    for layer in frame.layers:
        rings = layer.rings()
        if not rings:
            continue
        d = " ".join(_ring_path(r) for r in rings)
        paths.append(
            f'<path d="{d}" fill="{colors.get(layer.material, "#999")}" fill-rule="evenodd" '
            f'stroke="rgba(0,0,0,0.3)" stroke-width="0.5"><title>{layer.material}</title></path>'
        )
    return f'<g transform="translate(0,{y_max}) scale(1,-1)">{"".join(paths)}</g>'


def build_grid_svg(rows: list[tuple[str, list[Frame]]], colors: dict[str, str]) -> str:
    pw, ph = DOMAIN_NM, 240.0          # panel (nm)
    gap, left, top = 10.0, 60.0, 30.0
    y_min, y_max = -SUBSTRATE_NM, ph - SUBSTRATE_NM
    n_cols = 4
    width = left + n_cols * (pw + gap)
    height = top + len(rows) * (ph + gap) + 40
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:.0f} {height:.0f}" width="{width * 2:.0f}" height="{height * 2:.0f}" '
        f'font-family="-apple-system,Segoe UI,sans-serif">',
        f'<rect width="{width:.0f}" height="{height:.0f}" fill="#fff"/>',
    ]
    for c, title in enumerate(COLUMN_TITLES):
        out.append(f'<text x="{left + c * (pw + gap) + pw / 2:.0f}" y="18" font-size="11" font-weight="700" text-anchor="middle" fill="#1a1d23">{title}</text>')
    for r, (label, frames) in enumerate(rows):
        oy = top + r * (ph + gap)
        out.append(
            f'<text transform="translate(14,{oy + ph / 2:.0f}) rotate(-90)" font-size="11" font-weight="700" text-anchor="middle" fill="#1a1d23">{label}</text>'
        )
        picks = [-4, -3, -2, -1]   # pilier fini, remplissage, CMP, gravure sélective
        for c, idx in enumerate(picks):
            ox = left + c * (pw + gap)
            out.append(f'<rect x="{ox:.0f}" y="{oy:.0f}" width="{pw:.0f}" height="{ph:.0f}" fill="#f6f7fa" stroke="#d7dbe3"/>')
            out.append(f'<svg x="{ox:.0f}" y="{oy:.0f}" width="{pw:.0f}" height="{ph:.0f}" viewBox="0 0 {pw:.0f} {ph:.0f}">{_panel(frames[idx], colors, y_min, y_max)}</svg>')
    leg_y = height - 22
    x = left
    for mat in ("GaN", "SiO2", "Sapphire"):
        out.append(f'<rect x="{x:.0f}" y="{leg_y:.0f}" width="10" height="10" fill="{colors[mat]}" stroke="rgba(0,0,0,0.3)"/>')
        out.append(f'<text x="{x + 14:.0f}" y="{leg_y + 9:.0f}" font-size="9" fill="#1a1d23">{mat}</text>')
        x += 70
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    materials = default_library()
    recipes = default_recipes().with_recipes(etch=[SELECTIVE_OXIDE_ETCH])
    colors = {m.name: m.color for m in materials}

    shapes = [
        ("Pointu", shape_pointed),
        ("Trapèze", shape_trapezoid),
        ("Pyramide aplatie", shape_flattened_pyramid),
    ]
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    rows = []
    for name, fn in shapes:
        frames = run_shape(name, fn, materials, recipes)
        rows.append((name, frames))
        slug = name.lower().replace(" ", "_").replace("è", "e")
        save_svg(str(out_dir / f"selective_etch_{slug}.svg"), frames[-1], colors)

    (out_dir / "selective_etch_pillars.svg").write_text(build_grid_svg(rows, colors), encoding="utf-8")
    print(f"\nSVG : {out_dir / 'selective_etch_pillars.svg'}")


if __name__ == "__main__":
    main()
