"""A classic silicon-micromachining cavity: a nitride mask, an opening patterned into it, and a
KOH anisotropic wet etch that self-terminates on the {111} planes at 54.7deg from the surface -
here just a directional etch at that fixed angle, highly selective to Si over the nitride mask.
Etched long enough, the two sidewalls of a narrow-enough opening meet at a point for a clean V;
this script stops short of that (a flat-bottomed trapezoid) - the near-degenerate geometry right
at closure is a known rough edge, tracked separately from this demo.

Run: python examples/koh_v_groove.py
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


def build_flow(domain_width: float, opening: tuple[float, float], koh_depth_nm: float) -> list:
    return [
        Deposition(name="Masque nitrure", material="Si3N4", recipe="CVD Conformal", thickness=Length.nm(20)),
        Lithography(name="Ouverture du masque", resist_material="Photoresist", thickness=Length.nm(5), openings=[opening]),
        Etch(name="Gravure du nitrure (ouverture)", recipe="Anisotropic RIE", depth=Length.nm(20)),
        ResistStrip(name="Retrait resine"),
        ChemicalStep(name="Nettoyage pre-KOH", description="Dip RCA / desoxydation native"),
        Etch(name="Gravure KOH anisotrope", recipe="KOH Anisotropic Wet Etch", depth=Length.nm(koh_depth_nm)),
    ]


def main() -> None:
    materials = default_library()
    recipes = default_recipes()
    domain_width = 200.0
    opening = (60.0, 140.0)

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    material_colors = {m.name: m.color for m in materials}

    # Two etch times, both short of where the two {111} sidewalls would meet: a shallow cavity
    # and a deeper one approaching a V, without quite reaching the fully-closed point.
    for koh_depth_nm, label in [(20.0, "shallow"), (45.0, "deep")]:
        geometry = Geometry.substrate("Si", domain_width_nm=domain_width, thickness_nm=60)
        simulate(geometry, build_flow(domain_width, opening, koh_depth_nm), materials, recipes)
        save_svg(str(out_dir / f"koh_v_groove_{label}.svg"), _last_frame(geometry), material_colors)
        print(f"{label}: depth nominal {koh_depth_nm}nm -> Si bounds {geometry.bounds()}")


def _last_frame(geometry: Geometry):
    from structureforge.process.simulate import Frame

    return Frame(0, "final", "final", geometry.frame_layers(), geometry.domain_width_nm)


if __name__ == "__main__":
    main()
