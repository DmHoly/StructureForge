"""A III-N LED epitaxial stack with a V-defect ("V-pit"): a nucleation site left by a threading
dislocation, where the {1-101} semipolar facets that bound it grow much slower than the c-plane,
so the pit stays open and widens as more layers are grown over it - superlattice, multi-quantum
wells, electron-blocking layer, p-GaN - until a thicker "V-pit capping layer" (VCL) finally closes
it and the growth continues flat above.

This engine's directional recipes model a single tilted beam, not the two symmetric
self-terminating facets a real V-pit has (see `GaN V-pit Etch`'s notes) - so rather than growing
an ever-widening void step by step, the flow below grows the whole multilayer stack flat first and
then cuts the pit's cross-section in one masked, symmetric (isotropic) undercut, narrowing towards
where the dislocation would sit. The end result - a cone through many QW periods, capped flat above
by the VCL - looks the same either way.

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


def build_flow(cx: float, opening_half_width: float, pit_depth_nm: float) -> list:
    steps: list = []

    # Bottom superlattice (SLs) - flat, grown before the dislocation opens a pit.
    for i in range(3):
        steps.append(Deposition(name=f"SL bas AlGaN {i + 1}", material="AlGaN", recipe="MOCVD Epitaxial", thickness=Length.nm(2)))
        steps.append(Deposition(name=f"SL bas GaN {i + 1}", material="GaN", recipe="MOCVD Epitaxial", thickness=Length.nm(3)))

    # Multi-quantum wells (MQWs) - the periods whose QW lines cross the pit's sidewall.
    for i in range(8):
        steps.append(Deposition(name=f"MQW barriere GaN {i + 1}", material="GaN", recipe="MOCVD Epitaxial", thickness=Length.nm(4)))
        steps.append(Deposition(name=f"MQW puits InGaN {i + 1}", material="InGaN", recipe="MOCVD Epitaxial", thickness=Length.nm(1)))

    steps.append(Deposition(name="EBL AlGaN", material="AlGaN", recipe="MOCVD Epitaxial", thickness=Length.nm(8)))
    steps.append(Deposition(name="p-GaN", material="GaN", recipe="MOCVD Epitaxial", thickness=Length.nm(10)))

    # Cut the V-pit's cross-section down through the (so far flat) stack above.
    steps.append(
        Lithography(
            name="Masque V-pit",
            resist_material="Photoresist",
            thickness=Length.nm(8),
            openings=[(cx - opening_half_width, cx + opening_half_width)],
        )
    )
    steps.append(Etch(name="Gravure V-pit", recipe="GaN V-pit Etch", depth=Length.nm(pit_depth_nm)))
    steps.append(ResistStrip(name="Retrait resine"))

    # V-pit capping layer (VCL) - thick enough to coalesce over the remaining opening - then flat
    # growth continues above it, exactly like the region below the pit.
    steps.append(Deposition(name="VCL (capping)", material="AlGaN", recipe="MOCVD Epitaxial", thickness=Length.nm(55)))
    steps.append(Deposition(name="SL haut AlGaN", material="AlGaN", recipe="MOCVD Epitaxial", thickness=Length.nm(2)))
    steps.append(Deposition(name="SL haut GaN", material="GaN", recipe="MOCVD Epitaxial", thickness=Length.nm(3)))
    steps.append(Deposition(name="p-GaN haut", material="GaN", recipe="MOCVD Epitaxial", thickness=Length.nm(15)))
    return steps


def main() -> None:
    materials = default_library()
    recipes = default_recipes()

    domain_width = 300.0
    geometry = Geometry.substrate("GaN", domain_width_nm=domain_width, thickness_nm=20)
    steps = build_flow(cx=domain_width / 2, opening_half_width=31.0, pit_depth_nm=40.0)

    frames = simulate(geometry, steps, materials, recipes)

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    material_colors = {m.name: m.color for m in materials}
    save_svg(str(out_dir / "vpit_led_final.svg"), frames[-1], material_colors)

    pit_cut_index = next(i for i, f in enumerate(frames) if f.step_name == "Retrait resine")
    save_svg(str(out_dir / "vpit_led_pit_open.svg"), frames[pit_cut_index], material_colors)

    for frame in frames:
        print(f"[{frame.step_index}] {frame.step_kind:14s} {frame.step_name}")

    print(f"final GaN-family bounds: {geometry.bounds()}")


if __name__ == "__main__":
    main()
