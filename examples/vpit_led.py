"""A III-N LED epitaxial stack with a V-defect ("V-pit"): a threading dislocation nucleates a tiny
notch on two symmetric, straight {1-101} semipolar facets, and every layer grown after that -
superlattice periods, multi-quantum wells, the electron-blocking layer (EBL), p-GaN - drapes
conformally over that sidewall instead of growing flat, so each thin layer traces the same facet
line and the notch narrows a little further with each period. A thicker "V-pit capping layer"
(VCL) finally coalesces over what's left, and growth continues flat above it.

The facet angle (~58-62deg from the c-plane in real GaN, ~60deg here) is the one thing this
engine's process bricks can't reproduce directly: its directional recipes model a single tilted
beam, and its isotropic recipes are round, not straight - neither gives a *symmetric, precisely
angled* facet (see `carve_v_pit_nucleus` below for what that single small geometric step stands in
for). Every layer above the nucleus, though, is grown with the ordinary conformal "MOCVD
Epitaxial" recipe used everywhere else in this stack: conformal deposition drapes each layer over
the existing sidewall and preserves its angle exactly (a parallel offset of a straight line is the
same line), which is what produces the nested, narrowing facet lines through the MQW stack that a
real V-pit cross-section shows.

Run: python examples/vpit_led.py
"""

from __future__ import annotations

import math
from pathlib import Path

from shapely.geometry import Polygon

from structureforge import (
    Deposition,
    Geometry,
    Length,
    default_library,
    default_recipes,
    save_svg,
    simulate,
)
from structureforge.process.simulate import Frame


def carve_v_pit_nucleus(geometry: Geometry, cx: float, half_width_nm: float, facet_angle_from_horizontal_deg: float) -> None:
    """Cut a small, symmetric, straight-walled V notch into the current top surface at `cx` -
    the one non-process-brick step in this flow, standing in for a threading dislocation
    spontaneously nucleating a V-pit. No combination of this engine's deposition/etch recipes
    produces a *symmetric* facet at a *chosen* angle (directional recipes model a single tilted
    beam - both walls end up parallel, shifted sideways rather than converging; isotropic recipes
    are round, with no single facet angle at all) - so the nucleus itself is built directly out of
    the two straight lines a real V-pit's facets would trace, and every layer grown on top of it
    (all ordinary conformal deposition, see below) carries that same angle upward untouched.
    """
    depth_nm = half_width_nm / math.tan(math.radians(90.0 - facet_angle_from_horizontal_deg))
    top_y = max(l.polygon.bounds[3] for l in geometry.layers)
    notch = Polygon([(cx - half_width_nm, top_y), (cx + half_width_nm, top_y), (cx, top_y - depth_nm)])
    for layer in geometry.layers:
        if layer.polygon.bounds[3] <= top_y - depth_nm:
            continue  # entirely below the notch - untouched
        carved = layer.polygon.difference(notch)
        if not carved.is_empty and carved.area > 1e-6:
            layer.polygon = carved


def build_below_nucleus() -> list:
    # Bottom superlattice (SLs) - flat, grown before the dislocation nucleates a pit.
    steps: list = []
    for i in range(3):
        steps.append(Deposition(name=f"SL bas AlGaN {i + 1}", material="AlGaN", recipe="MOCVD Epitaxial", thickness=Length.nm(2)))
        steps.append(Deposition(name=f"SL bas GaN {i + 1}", material="GaN", recipe="MOCVD Epitaxial", thickness=Length.nm(3)))
    return steps


def build_above_nucleus() -> list:
    steps: list = []
    # Multi-quantum wells (MQWs) - conformal growth drapes each barrier/well over the pit's
    # sidewall, narrowing it a little further with every period, just like a real V-pit's QW
    # lines do in cross-section.
    for i in range(8):
        steps.append(Deposition(name=f"MQW barriere GaN {i + 1}", material="GaN", recipe="MOCVD Epitaxial", thickness=Length.nm(4)))
        steps.append(Deposition(name=f"MQW puits InGaN {i + 1}", material="InGaN", recipe="MOCVD Epitaxial", thickness=Length.nm(1)))

    steps.append(Deposition(name="EBL AlGaN", material="AlGaN", recipe="MOCVD Epitaxial", thickness=Length.nm(8)))
    steps.append(Deposition(name="p-GaN", material="GaN", recipe="MOCVD Epitaxial", thickness=Length.nm(10)))

    # V-pit capping layer (VCL) - thick enough to coalesce over what's left of the notch - then
    # flat growth continues above it, exactly like the region below the pit.
    steps.append(Deposition(name="VCL (capping)", material="AlGaN", recipe="MOCVD Epitaxial", thickness=Length.nm(35)))
    steps.append(Deposition(name="SL haut AlGaN", material="AlGaN", recipe="MOCVD Epitaxial", thickness=Length.nm(2)))
    steps.append(Deposition(name="SL haut GaN", material="GaN", recipe="MOCVD Epitaxial", thickness=Length.nm(3)))
    steps.append(Deposition(name="p-GaN haut", material="GaN", recipe="MOCVD Epitaxial", thickness=Length.nm(15)))
    return steps


def main() -> None:
    materials = default_library()
    recipes = default_recipes()

    domain_width = 300.0
    cx = domain_width / 2
    geometry = Geometry.substrate("GaN", domain_width_nm=domain_width, thickness_nm=20)

    frames_below = simulate(geometry, build_below_nucleus(), materials, recipes)
    carve_v_pit_nucleus(geometry, cx=cx, half_width_nm=15.0, facet_angle_from_horizontal_deg=60.0)
    frames_above = simulate(geometry, build_above_nucleus(), materials, recipes)
    # frames_above[0] is simulate()'s own starting snapshot for the second call - relabel it to
    # what it actually shows (the nucleus just carved) instead of a second "Structure de depart".
    nucleus_frame = frames_above[0]
    frames_above[0] = Frame(nucleus_frame.step_index, "carve", "Nucleation V-pit", nucleus_frame.layers, nucleus_frame.domain_width_nm)
    frames = frames_below + frames_above

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    material_colors = {m.name: m.color for m in materials}
    save_svg(str(out_dir / "vpit_led_final.svg"), frames[-1], material_colors)

    for i, frame in enumerate(frames):
        print(f"[{i}] {frame.step_kind:14s} {frame.step_name}")

    print(f"final GaN-family bounds: {geometry.bounds()}")


if __name__ == "__main__":
    main()
