"""A III-N LED epitaxial stack with a V-defect ("V-pit"): a threading dislocation nucleates a tiny
notch, and every layer grown after that - superlattice periods, multi-quantum wells, the
electron-blocking layer (EBL), p-GaN - drapes conformally over its sidewall instead of growing
flat, so each thin layer traces the slope and the notch narrows a little further with each period.
A thicker "V-pit capping layer" (VCL) finally coalesces over what's left, and growth continues
flat above it.

A real V-pit's sidewall is a pair of symmetric, self-terminating {1-101} semipolar facets that
grow far slower than the c-plane; this engine's directional recipes model a single tilted beam
rather than two symmetric facets (see `GaN V-pit Etch`'s notes), so nucleating the notch that way
isn't reproducible here. Instead, the flow below seeds one small, symmetric (isotropic) notch
right where the dislocation would sit, and lets plain conformal deposition - the same "MOCVD
Epitaxial" recipe used for every flat layer in this stack - drape each subsequent layer over it:
conformal growth naturally narrows a concave notch a little with every layer, which is what
produces the nested, narrowing facet lines through the MQW stack that a real V-pit cross-section
shows, without needing a dedicated "growing wider" mechanism this engine doesn't have.

Run: python examples/vpit_led.py
"""

from __future__ import annotations

from pathlib import Path

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


def build_flow(cx: float, nucleation_half_width: float, nucleation_depth_nm: float) -> list:
    steps: list = []

    # Bottom superlattice (SLs) - flat, grown before the dislocation nucleates a pit.
    for i in range(3):
        steps.append(Deposition(name=f"SL bas AlGaN {i + 1}", material="AlGaN", recipe="MOCVD Epitaxial", thickness=Length.nm(2)))
        steps.append(Deposition(name=f"SL bas GaN {i + 1}", material="GaN", recipe="MOCVD Epitaxial", thickness=Length.nm(3)))

    # Nucleate the V-pit: a single small, localized notch at the dislocation's position.
    steps.append(
        Lithography(
            name="Masque nucleation V-pit",
            resist_material="Photoresist",
            thickness=Length.nm(5),
            openings=[(cx - nucleation_half_width, cx + nucleation_half_width)],
        )
    )
    steps.append(Etch(name="Nucleation V-pit", recipe="GaN V-pit Etch", depth=Length.nm(nucleation_depth_nm)))
    steps.append(ResistStrip(name="Retrait resine"))

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
    steps.append(Deposition(name="VCL (capping)", material="AlGaN", recipe="MOCVD Epitaxial", thickness=Length.nm(30)))
    steps.append(Deposition(name="SL haut AlGaN", material="AlGaN", recipe="MOCVD Epitaxial", thickness=Length.nm(2)))
    steps.append(Deposition(name="SL haut GaN", material="GaN", recipe="MOCVD Epitaxial", thickness=Length.nm(3)))
    steps.append(Deposition(name="p-GaN haut", material="GaN", recipe="MOCVD Epitaxial", thickness=Length.nm(15)))
    return steps


def main() -> None:
    materials = default_library()
    recipes = default_recipes()

    domain_width = 300.0
    geometry = Geometry.substrate("GaN", domain_width_nm=domain_width, thickness_nm=20)
    steps = build_flow(cx=domain_width / 2, nucleation_half_width=20.0, nucleation_depth_nm=6.0)

    frames = simulate(geometry, steps, materials, recipes)

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    material_colors = {m.name: m.color for m in materials}
    save_svg(str(out_dir / "vpit_led_final.svg"), frames[-1], material_colors)

    nucleation_index = next(i for i, f in enumerate(frames) if f.step_name == "Retrait resine")
    save_svg(str(out_dir / "vpit_led_nucleation.svg"), frames[nucleation_index], material_colors)

    for frame in frames:
        print(f"[{frame.step_index}] {frame.step_kind:14s} {frame.step_name}")

    print(f"final GaN-family bounds: {geometry.bounds()}")


if __name__ == "__main__":
    main()
