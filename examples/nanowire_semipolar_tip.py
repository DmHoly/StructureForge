"""A selective-area-grown GaN nanowire with a facetted tip: a vertical m-plane stem, tapering
inward on the same two symmetric {1-101} semi-polar facets a V-pit has - an "anti-V-pit", the same
crystal planes on a convex mesa growing up instead of a concave notch growing down - up to a
narrow c-plane top, where growth (a thin InGaN quantum well, a GaN cap, an ITO contact) continues
flat again exactly as it would on any other c-plane surface.

Like `examples/vpit_led.py`, the one thing this engine's process bricks can't reproduce is the
facet itself: a *symmetric*, precisely angled sidewall (see `add_semipolar_tip` below). Everything
else - the blanket GaN grown flat, the RIE etch that turns it into a freestanding pillar, the QW
and cap grown on the shrunken c-plane top - is ordinary Deposition/Etch/Lithography, the same
bricks `examples/nanowire_pzgan.py` uses for its (unfacetted, vertical-walled) nanowire array.

Run: python examples/nanowire_semipolar_tip.py
"""

from __future__ import annotations

import math
from pathlib import Path

from shapely.geometry import Polygon

from structureforge import (
    Deposition,
    Etch,
    Geometry,
    Length,
    Lithography,
    ResistStrip,
    default_library,
    default_recipes,
    save_svg,
    simulate,
)
from structureforge.process.simulate import Frame


def add_semipolar_tip(
    geometry: Geometry,
    cx: float,
    base_half_width_nm: float,
    tip_half_width_nm: float,
    facet_angle_from_horizontal_deg: float,
) -> None:
    """Cap a vertical pillar with a symmetric frustum tapering in on two straight facets - the
    mirror image of `vpit_led.py`'s `carve_v_pit_nucleus`, and for the same reason: this engine's
    directional recipes model a single tilted beam (both sidewalls end up parallel, shifted
    sideways together rather than converging to a point), so a *symmetric*, precisely angled
    facet has to be built directly out of the two straight lines it would trace, rather than grown
    with a recipe. Ordinary conformal deposition on the narrower c-plane top left at its apex
    behaves exactly as it would on any other flat surface - it's only this one transition that
    needs constructing by hand.
    """
    height_nm = (base_half_width_nm - tip_half_width_nm) * math.tan(math.radians(facet_angle_from_horizontal_deg))
    base_y = max(l.polygon.bounds[3] for l in geometry.layers)
    frustum = Polygon(
        [
            (cx - base_half_width_nm, base_y),
            (cx + base_half_width_nm, base_y),
            (cx + tip_half_width_nm, base_y + height_nm),
            (cx - tip_half_width_nm, base_y + height_nm),
        ]
    )
    for layer in geometry.layers:
        if layer.polygon.bounds[3] >= base_y - 1e-6:
            layer.polygon = layer.polygon.union(frustum)


def build_stem(cx: float, pillar_half_width_nm: float, pillar_height_nm: float) -> list:
    return [
        Deposition(name="Tampon AlN", material="AlN", recipe="MOCVD Epitaxial", thickness=Length.nm(10)),
        Deposition(name="GaN (precurseur)", material="GaN", recipe="MOCVD Epitaxial", thickness=Length.nm(pillar_height_nm)),
        Lithography(
            name="Masque du nanofil",
            resist_material="Photoresist",
            thickness=Length.nm(80),
            openings=[(0.0, cx - pillar_half_width_nm), (cx + pillar_half_width_nm, cx * 2)],
        ),
        Etch(name="Gravure RIE du nanofil", recipe="Cl2 ICP-RIE (III-N)", depth=Length.nm(pillar_height_nm)),
        ResistStrip(name="Retrait resine"),
    ]


def build_cap() -> list:
    return [
        Deposition(name="Puits quantique InGaN", material="InGaN", recipe="MOCVD Epitaxial", thickness=Length.nm(3)),
        Deposition(name="Capot GaN", material="GaN", recipe="MOCVD Epitaxial", thickness=Length.nm(8)),
        Deposition(name="Contact ITO", material="ITO", recipe="Sputter Metal (normal)", thickness=Length.nm(15)),
    ]


def main() -> None:
    materials = default_library()
    recipes = default_recipes()

    domain_width = 300.0
    cx = domain_width / 2
    pillar_half_width = 30.0
    geometry = Geometry.substrate("Sapphire", domain_width_nm=domain_width, thickness_nm=20)

    frames_stem = simulate(geometry, build_stem(cx, pillar_half_width, pillar_height_nm=60.0), materials, recipes)
    add_semipolar_tip(geometry, cx=cx, base_half_width_nm=pillar_half_width, tip_half_width_nm=8.0, facet_angle_from_horizontal_deg=60.0)
    frames_cap = simulate(geometry, build_cap(), materials, recipes)
    # frames_cap[0] is simulate()'s own starting snapshot for the second call - relabel it to what
    # it actually shows (the facet tip just added) instead of a second "Structure de depart".
    tip_frame = frames_cap[0]
    frames_cap[0] = Frame(tip_frame.step_index, "carve", "Facette semi-polaire", tip_frame.layers, tip_frame.domain_width_nm)
    frames = frames_stem + frames_cap

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    material_colors = {m.name: m.color for m in materials}
    save_svg(str(out_dir / "nanowire_semipolar_tip_final.svg"), frames[-1], material_colors)

    for i, frame in enumerate(frames):
        print(f"[{i}] {frame.step_kind:14s} {frame.step_name}")

    print(f"final GaN bounds: {geometry.bounds()}")


if __name__ == "__main__":
    main()
