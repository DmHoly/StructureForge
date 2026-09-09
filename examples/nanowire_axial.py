"""A single axial III-N nanowire: one freestanding pillar in which the heterostructure changes
*along the growth axis* (buffer, n-GaN stem, an InGaN/GaN multi-quantum-well active region, an
AlGaN electron-blocking layer, a p-GaN top segment, then a metal contact) - as opposed to a
*radial/core-shell* nanowire, where composition instead changes outward from a fixed core.

This engine has no true selective-area-growth deposition mode (no "nucleates only on this seed
material through a mask opening, not on the mask itself" - see the `MOCVD Epitaxial` recipe's own
notes in `structureforge/core/recipes.py`). So, exactly like `examples/nanowire_pzgan.py` and
`examples/nanowire_semipolar_tip.py`, the axial stack is built the other way around: grow every
segment as an ordinary blanket epitaxial `Deposition` (bottom to top = growth order = the wire's
future axis), *then* pattern a single narrow opening and dry-etch a lithography mask down through
the whole stack in one anisotropic step, leaving one freestanding pillar that carries every axial
segment intact inside it. The one-liner mental model: axial heterostructure = deposition order;
the pillar's outline = one mask + one directional etch, not per-segment masking.

The bottom two segments (nucleation buffer, n-GaN stem) use a derived `Length` instead of a
literal one - see `structureforge.core.derivation` / `examples/derived_gan_growth.py` - to show
it's a drop-in in a real multi-step flow: `total_height` below sums `.to_nm()` over every segment
exactly as before, unaware that two of them were reached via rate x duration rather than typed in
directly.

Run: python examples/nanowire_axial.py
"""

from __future__ import annotations

from pathlib import Path

from structureforge import (
    ArrheniusRate,
    ConstantRate,
    Deposition,
    Etch,
    Geometry,
    GrowthAtRate,
    Length,
    Lithography,
    ResistStrip,
    default_library,
    default_recipes,
    save_svg,
    simulate,
)


def build_axial_stack() -> list:
    """The heterostructure, one `Deposition` per axial segment, in growth order (bottom to top).
    Each stays a full-width blanket layer at this point - the wire doesn't exist yet, only a flat
    epitaxial stack does; `build_mask_and_etch` is what actually carves it into a pillar.
    """
    aln_buffer_thickness = Length.derived(
        GrowthAtRate(rate=ConstantRate(nm_per_s=0.2), duration_s=40.0)  # 0.2nm/s * 40s = 8nm
    )
    n_gan_stem_thickness = Length.derived(
        GrowthAtRate(
            # Hotter growth than the AlN buffer above, held long enough to reach 60nm.
            rate=ArrheniusRate(prefactor_nm_per_s=2.06e8, activation_energy_eV=1.8, temperature_K=1053.0),
            duration_s=120.0,
        )
    )
    steps = [
        Deposition(name="Tampon de nucleation AlN", material="AlN", recipe="MOCVD Epitaxial", thickness=aln_buffer_thickness),
        Deposition(name="Tige n-GaN", material="GaN", recipe="MOCVD Epitaxial", thickness=n_gan_stem_thickness),
        Deposition(name="Barriere axiale AlGaN", material="AlGaN", recipe="MOCVD Epitaxial", thickness=Length.nm(8)),
    ]
    for i in range(1, 4):
        steps.append(Deposition(name=f"MQW barriere GaN {i}", material="GaN", recipe="MOCVD Epitaxial", thickness=Length.nm(5)))
        steps.append(Deposition(name=f"MQW puits InGaN {i}", material="InGaN", recipe="MOCVD Epitaxial", thickness=Length.nm(2)))
    steps.append(Deposition(name="Blocage electrons AlGaN (EBL)", material="AlGaN", recipe="MOCVD Epitaxial", thickness=Length.nm(8)))
    steps.append(Deposition(name="Segment p-GaN", material="GaN", recipe="MOCVD Epitaxial", thickness=Length.nm(20)))
    return steps


def build_mask_and_etch(cx: float, half_width_nm: float, domain_width_nm: float, total_height_nm: float) -> list:
    """Pattern one narrow opening (the future wire, `[cx - half_width, cx + half_width]` stays
    covered) and dry-etch straight down through the *entire* axial stack in a single anisotropic
    step - deep enough to clear even the bottom-most segment (the AlN nucleation buffer), stopping
    on the (non-III-N) substrate thanks to `Cl2 ICP-RIE (III-N)`'s selectivity.
    """
    return [
        Lithography(
            name="Masque HSQ du nanofil",
            resist_material="HSQ",
            thickness=Length.nm(60),
            openings=[(0.0, cx - half_width_nm), (cx + half_width_nm, domain_width_nm)],
        ),
        Etch(name="Gravure Cl2 ICP-RIE (pilier axial)", recipe="Cl2 ICP-RIE (III-N)", depth=Length.nm(total_height_nm)),
        ResistStrip(name="Retrait masque HSQ", material="HSQ"),
    ]


def main() -> None:
    materials = default_library()
    recipes = default_recipes()

    domain_width = 120.0
    cx = domain_width / 2
    half_width = 20.0
    geometry = Geometry.substrate("Sapphire", domain_width_nm=domain_width, thickness_nm=20)

    axial_stack = build_axial_stack()
    total_height = sum(step.thickness.to_nm() for step in axial_stack)

    steps = [
        *axial_stack,
        *build_mask_and_etch(cx, half_width, domain_width, total_height),
        Deposition(name="Contact p (Ni/Au)", material="Ni", recipe="Sputter Metal (normal)", thickness=Length.nm(10)),
        Deposition(name="Capot d'or", material="Au", recipe="Sputter Metal (normal)", thickness=Length.nm(15)),
    ]

    frames = simulate(geometry, steps, materials, recipes)

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    material_colors = {m.name: m.color for m in materials}
    save_svg(str(out_dir / "nanowire_axial_stack.svg"), frames[len(axial_stack)], material_colors)
    save_svg(str(out_dir / "nanowire_axial_final.svg"), frames[-1], material_colors)

    for frame in frames:
        areas = ", ".join(f"{l.material}={l.polygon.area:.0f}nm2" for l in frame.layers)
        print(f"[{frame.step_index}] {frame.step_kind:14s} {frame.step_name:32s} {areas}")

    print(f"pillar: x in [{cx - half_width}, {cx + half_width}] nm, axial height {total_height:.0f}nm")
    print(f"final structure bounds: {geometry.bounds()}")


if __name__ == "__main__":
    main()
