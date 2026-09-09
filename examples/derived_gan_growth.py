"""The question that motivated `structureforge.core.derivation`: a 5nm GaN layer is one number,
but there are many ways to reach it - grow longer at a modest rate, or grow faster (hotter) for
less time. `Length` normally just carries that number (`Length.nm(5)`); `Length.derived(...)`
instead carries a small recursive tree explaining *how* it was reached - a `GrowthAtRate` (rate x
duration), whose `rate` is itself either a flat `ConstantRate` or a temperature-dependent
`ArrheniusRate` (Arrhenius law: rate = prefactor * exp(-Ea / kB*T)) - and `MultiStageGrowth` chains
several such stages (e.g. a temperature ramp approximated as discrete holds) into one thickness.

Four levels, matching the structure/step/param/process shape this module exists for: a
`ProcessStep` (here `EpitaxialGrowth`) has a `thickness` *param* (a `Length`), which can itself
unfold into a `derivation` *process* explaining how that param's value was actually reached.

None of this changes what gets simulated - `Length.to_nm()` resolves either form to the same flat
number the geometry engine consumes, so a derived `Length` is a drop-in replacement anywhere a
literal one was used (see `examples/nanowire_axial.py` for that in a full device flow).

Run: python examples/derived_gan_growth.py
"""

from __future__ import annotations

from pathlib import Path

from structureforge import (
    ArrheniusRate,
    ConstantRate,
    EpitaxialGrowth,
    Geometry,
    GrowthAtRate,
    Length,
    MultiStageGrowth,
    default_library,
    default_recipes,
    save_svg,
    simulate,
)


def main() -> None:
    # Same 5nm, three ways: a flat literal, a flat rate held for a given duration (numerically
    # identical - `Length.derived` just also records *how*), and a temperature-dependent rate
    # that happens to land close to it at a chosen growth temperature.
    literal = Length.nm(5.0)

    flat_rate = Length.derived(GrowthAtRate(rate=ConstantRate(nm_per_s=0.5), duration_s=10.0))

    thermal_rate = Length.derived(
        GrowthAtRate(
            rate=ArrheniusRate(prefactor_nm_per_s=2.06e8, activation_energy_eV=1.8, temperature_K=1053.0),
            duration_s=10.0,
        )
    )

    # A temperature ramp approximated as three discrete holds - each stage is a full
    # `LengthDerivation` in its own right, summed by `MultiStageGrowth`.
    ramped = Length.derived(
        MultiStageGrowth(
            stages=[
                GrowthAtRate(
                    rate=ArrheniusRate(prefactor_nm_per_s=2.06e8, activation_energy_eV=1.8, temperature_K=1000.0),
                    duration_s=4.0,
                ),
                GrowthAtRate(
                    rate=ArrheniusRate(prefactor_nm_per_s=2.06e8, activation_energy_eV=1.8, temperature_K=1050.0),
                    duration_s=4.0,
                ),
                GrowthAtRate(
                    rate=ArrheniusRate(prefactor_nm_per_s=2.06e8, activation_energy_eV=1.8, temperature_K=1100.0),
                    duration_s=4.0,
                ),
            ]
        )
    )

    for label, length in [("literal", literal), ("flat_rate", flat_rate), ("thermal_rate", thermal_rate), ("ramped", ramped)]:
        print(f"{label:12s} -> {length.to_nm():.3f} nm")

    # A derived Length is a drop-in for a literal one anywhere a step expects a Length - simulate
    # a single GaN buffer grown at the thermal (Arrhenius) rate above.
    materials = default_library()
    recipes = default_recipes()
    geometry = Geometry.substrate("Sapphire", domain_width_nm=100, thickness_nm=20)
    steps = [
        EpitaxialGrowth(name="Tampon GaN (vitesse Arrhenius)", material="GaN", thickness=thermal_rate),
    ]
    frames = simulate(geometry, steps, materials, recipes)

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)
    save_svg(str(out_dir / "derived_gan_growth.svg"), frames[-1], {m.name: m.color for m in materials})
    print(f"simulated GaN buffer: {thermal_rate.to_nm():.3f}nm, final bounds {geometry.bounds()}")


if __name__ == "__main__":
    main()
