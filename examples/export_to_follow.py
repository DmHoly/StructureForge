"""Simulate a process flow, then commit its final structure and its step-by-step history as a
`follow` experiment: the geometry becomes what's studied, the process becomes the protocol.

Needs the optional `follow` dependency: `pip install structureforge[follow]`.

Run: python examples/export_to_follow.py [chemin_du_depot]
"""

from __future__ import annotations

import sys

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
    simulate,
)
from structureforge.adapters import follow_adapter


def main() -> None:
    import follow

    repo_path = sys.argv[1] if len(sys.argv) > 1 else "mon_labo_structureforge"

    materials = default_library()
    recipes = default_recipes()

    geometry = Geometry.substrate("Si", domain_width_nm=400, thickness_nm=100)
    process = [
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

    simulate(geometry, process, materials, recipes)  # mutates geometry to its final state

    repo = follow.Repository(repo_path)
    experiment = follow_adapter.export_experiment(
        repo,
        geometry,
        process,
        branch="main",
        title="Isolation par tranchee peu profonde (STI)",
        intent="Verifier le flow STI simule avant de le lancer en salle blanche",
    )

    print(f"Commite : {experiment.id} sur la branche '{experiment.branch}' ({repo_path})")
    print()
    print(follow.render_fiche(experiment, repo))


if __name__ == "__main__":
    main()
