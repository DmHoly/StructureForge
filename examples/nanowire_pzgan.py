"""A III-N nanowire array etched from a planar quantum-well stack - the kind of multi-scale
structure (nanometre epitaxial layers, tens-of-nanometre EBL-defined pillars) this project was
started for: a GaN buffer, an AlGaN barrier, a thin InGaN quantum well, a GaN cap, an
EBL-patterned resist leaving narrow pillars, and a directional dry etch turning the planar stack
into a nanowire array with a buried quantum well in each wire.

Run: python examples/nanowire_pzgan.py
"""

from __future__ import annotations

from pathlib import Path

from structureforge import (
    ChemicalStep,
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


def main() -> None:
    materials = default_library()
    recipes = default_recipes()

    domain_width = 300.0
    geometry = Geometry.substrate("GaN", domain_width_nm=domain_width, thickness_nm=50)

    pillar_width = 40.0
    pillar_pitch = 100.0
    pillars = [(50.0, 90.0), (150.0, 190.0), (250.0, 290.0)]
    openings = []
    cursor = 0.0
    for start, end in pillars:
        if start > cursor:
            openings.append((cursor, start))
        cursor = end
    if cursor < domain_width:
        openings.append((cursor, domain_width))

    steps = [
        Deposition(name="Barriere AlGaN", material="AlGaN", recipe="ALD Conformal", thickness=Length.nm(15)),
        Deposition(name="Puits quantique InGaN", material="InGaN", recipe="ALD Conformal", thickness=Length.nm(3)),
        Deposition(name="Capot GaN", material="GaN", recipe="ALD Conformal", thickness=Length.nm(10)),
        Lithography(
            name="Masque EBL nanofils",
            resist_material="Photoresist",
            thickness=Length.nm(80),
            openings=openings,
        ),
        Etch(name="Gravure ICP-RIE des nanofils", recipe="Ion Mill (tilted)", depth=Length.nm(60)),
        ResistStrip(name="Retrait resine"),
        ChemicalStep(name="Passivation de surface", description="Traitement de surface (S)2- ou similaire"),
    ]

    frames = simulate(geometry, steps, materials, recipes)

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    material_colors = {m.name: m.color for m in materials}
    save_svg(str(out_dir / "nanowire_pzgan_final.svg"), frames[-1], material_colors)
    save_svg(str(out_dir / "nanowire_pzgan_stack.svg"), frames[3], material_colors)

    for frame in frames:
        areas = ", ".join(f"{l.material}={l.polygon.area:.0f}nm2" for l in frame.layers)
        print(f"[{frame.step_index}] {frame.step_kind:14s} {frame.step_name:30s} {areas}")

    print(f"pillars: {pillars} (width {pillar_width}nm, pitch {pillar_pitch}nm) - openings: {openings}")


if __name__ == "__main__":
    main()
