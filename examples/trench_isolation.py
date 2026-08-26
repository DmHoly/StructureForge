"""Shallow trench isolation (STI): pad oxide + nitride stop layer, a lithography-defined trench,
an anisotropic etch through both into the substrate, a gap-fill oxide, and a CMP planarization
that stops on the nitride - a classic planar-CMOS flow exercising every step kind except a
directional deposit.

Run: python examples/trench_isolation.py
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
    Planarization,
    default_library,
    default_recipes,
    save_svg,
    simulate,
)


def main() -> None:
    materials = default_library()
    recipes = default_recipes()

    geometry = Geometry.substrate("Si", domain_width_nm=400, thickness_nm=100)
    steps = [
        Deposition(name="Oxyde tampon", material="SiO2", recipe="ALD Conformal", thickness=Length.nm(8)),
        Deposition(name="Nitrure d'arret", material="Si3N4", recipe="CVD Conformal", thickness=Length.nm(20)),
        Lithography(
            name="Masque de tranchee STI",
            resist_material="Photoresist",
            thickness=Length.nm(200),
            openings=[(150.0, 250.0)],
        ),
        Etch(name="Gravure de la tranchee", recipe="Anisotropic RIE", depth=Length.nm(120)),
        ChemicalStep(name="Nettoyage post-gravure", description="Dip HF dilue + rincage"),
        Deposition(name="Remplissage oxyde", material="SiO2", recipe="CVD Conformal", thickness=Length.nm(150)),
        Planarization(name="CMP oxyde", stop_material="Si3N4"),
    ]

    frames = simulate(geometry, steps, materials, recipes)

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    material_colors = {m.name: m.color for m in materials}
    save_svg(str(out_dir / "trench_isolation_final.svg"), frames[-1], material_colors)
    save_svg(str(out_dir / "trench_isolation_after_etch.svg"), frames[4], material_colors)

    for frame in frames:
        areas = ", ".join(f"{l.material}={l.polygon.area:.0f}nm2" for l in frame.layers)
        print(f"[{frame.step_index}] {frame.step_kind:14s} {frame.step_name:28s} {areas}")


if __name__ == "__main__":
    main()
