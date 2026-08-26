# StructureForge

StructureForge simule un **process 2D en coupe** (façon Silvaco Athena / Victory Process, en
beaucoup plus simple) : on dessine une structure de dispositif en enchaînant des briques
élémentaires - dépôt, gravure, planarisation, étape chimique - piochées dans une bibliothèque de
matériaux et une bibliothèque de recettes (avec sélectivité), et on obtient la géométrie 2D
résultante, image par image (une par étape), avec une interface graphique multi-vues
(vue d'ensemble + vue zoomée sur la zone active).

Le projet est **indépendant de [Follow](https://github.com/DmHoly/Follow)** (le système de
tracking façon git pour expériences scientifiques) : StructureForge simule et dessine ; Follow
trace et versionne. Un adaptateur optionnel (`structureforge.adapters.follow_adapter`) convertit
une géométrie simulée et son historique de process en `follow.Structure` / `follow.Step`, pour
committer un flow de process comme une expérience Follow - sans que le moteur géométrique ne
dépende de Follow pour fonctionner.

## Installation

```bash
git clone <ce depot> && cd StructureForge
pip install -e .                 # moteur seul (pydantic, shapely)
pip install -e ".[api]"          # + interface web (FastAPI, uvicorn)
pip install -e ".[follow]"       # + adaptateur Follow
pip install -e ".[dev]"          # + tests (pytest, httpx)
```

Python ≥ 3.11 requis.

## Démarrage rapide (sans GUI)

```python
from structureforge import (
    Deposition, Etch, Lithography, ResistStrip, Geometry, Length,
    default_library, default_recipes, simulate, save_svg,
)

materials = default_library()
recipes = default_recipes()

geometry = Geometry.substrate("Si", domain_width_nm=200, thickness_nm=50)
steps = [
    Deposition(name="Oxyde de champ", material="SiO2", recipe="CVD Conformal", thickness=Length.nm(30)),
    Lithography(name="Masque tranchee", resist_material="Photoresist", thickness=Length.nm(5), openings=[(80, 120)]),
    Etch(name="Gravure tranchee", recipe="Anisotropic RIE", depth=Length.nm(40)),
    ResistStrip(name="Retrait resine"),
]

frames = simulate(geometry, steps, materials, recipes)   # une Frame par etape + l'etat initial
save_svg("trench.svg", frames[-1], {m.name: m.color for m in materials})
```

Voir `examples/trench_isolation.py` (flow STI planaire complet : pad oxide, nitrure d'arrêt,
masque, gravure, remplissage, CMP) et `examples/nanowire_pzgan.py` (empilement GaN/AlGaN/InGaN/GaN
gravé en réseau de nanofils par lithographie EBL + gravure directionnelle inclinée - le cas
multi-échelle qui a motivé le projet).

## Interface graphique web

```bash
pip install -e ".[api]"
structureforge --port 8000
# ou : uvicorn structureforge.api.app:app --reload
```

Ouvrir `http://127.0.0.1:8000`. On choisit le substrat, on ajoute des étapes une par une
(matériau/recette/épaisseur pour un dépôt, recette/profondeur pour une gravure, etc.), on clique
**Simuler**, puis on parcourt l'historique du process avec le curseur - les deux vues SVG (vue
d'ensemble sur tout le domaine, vue zoomée sur une zone qu'on définit numériquement ou via
« Cadrer sur la structure ») se mettent à jour ensemble, à chaque étape.

## Concepts

| Concept | Rôle |
|---|---|
| `Material` / `MaterialLibrary` | Une entrée de bibliothèque : nom, catégorie (substrat, semiconducteur, diélectrique, métal, résine...), couleur (rendu), densité, indice optique. Ne porte **pas** de sélectivité - voir plus bas. |
| `DepositionRecipe` | Un mode de dépôt : `conformal` (CVD/ALD - épaisseur uniforme dans toutes les directions) ou `directional` (PVD/évaporation - dépôt en ligne de vue depuis un angle, `angle_deg` mesuré depuis la normale). |
| `EtchRecipe` | Un mode de gravure : `isotropic` (attaque uniforme dans toutes les directions - sous-gravure sous un masque) ou `directional` (RIE/usinage ionique, angle réglable), plus une table de sélectivité (`factor_for(material)`). |
| `Geometry` / `Layer` | La coupe 2D elle-même : un empilement de polygones (shapely), un par couche, dans l'ordre de création (qui fait aussi office d'ordre en z). |
| `ProcessStep` | Une brique élémentaire : `Deposition`, `Etch`, `Planarization`, `Lithography` (dépôt de résine motif via des ouvertures), `ResistStrip`, ou `ChemicalStep` (aucun effet géométrique - nettoyage, recuit... juste tracé pour l'historique). |
| `simulate()` | Applique une liste de `ProcessStep` à une `Geometry` de départ, renvoie une `Frame` par étape (dont l'état initial) pour l'historique/le défilement. |

### L'exemple de sélectivité de la spec

> une gravure dry qui attaque sélectivement les oxydes avec une sélectivité de 0.8 sur le reste

```python
from structureforge.core.recipes import EtchRecipe, EtchMode
from structureforge.core.materials import MaterialCategory

EtchRecipe(
    name="Dry Oxide Etch",
    mode=EtchMode.isotropic,
    selectivity_by_category={MaterialCategory.dielectric: 1.0},
    default_factor=0.8,
)
```

C'est exactement `default_recipes()["Dry Oxide Etch"]`. `selectivity_by_material` prend le pas sur
`selectivity_by_category`, qui prend le pas sur `default_factor` - voir `EtchRecipe.factor_for`.

## Le moteur géométrique (limites v1 documentées)

`structureforge/geometry/engine.py` implémente dépôt/gravure/planarisation par opérations
booléennes sur des polygones shapely plutôt que sur un champ de hauteur - choix fait pour bien
représenter angles, conformité et cavités. Trois simplifications assumées pour cette v1, chacune
documentée dans le code à l'endroit exact où elle s'applique :

- **Pas d'ombrage entre motifs distincts.** Une gravure/dépôt directionnel gère correctement
  l'auto-ombrage d'une même topologie (le flanc d'une tranchée s'ombrage bien lui-même), mais pas
  le fait qu'un motif voisin plus haut puisse en masquer un autre plus bas. Un vrai ray-tracing
  serait l'extension naturelle.
- **Les bords du domaine sont des frontières de symétrie**, pas des bords libres : tout
  buffer/balayage est calculé sur la géométrie reflétée en x=0 et x=largeur puis recadré, pour
  éviter un arrondi/une érosion artificiels pile sur le bord. Garder le domaine assez large pour
  que les motifs d'intérêt n'y collent pas.
- **La gravure sélective avance par sous-étapes.** Une sous-étape plus lente (masque, couche
  d'arrêt), tant qu'elle est elle-même exposée à la surface, est temporairement "gonflée" vers le
  haut le temps du calcul d'une autre sous-étape plus rapide - sinon, une fois plus fine qu'une
  seule sous-étape de la gravure rapide, elle serait traversée d'un coup au lieu de protéger ce
  qu'il y a dessous jusqu'à sa consommation complète. Le résultat retenu est toujours recalculé
  contre les polygones réels (non gonflés), donc ce mécanisme ne fuit jamais dans le résultat - il
  ne fait que borner jusqu'où une sous-étape peut aller.

Le lift-off (matière déposée sur une résine, qui part avec elle) est une conséquence émergente de
`Geometry.remove_floating_debris()` (appelé après `ResistStrip`) plutôt qu'un modèle physique
dédié : tout ce qui n'est plus connecté au substrat après le retrait de la résine est retiré.

## Structure du projet

```
structureforge/
  core/           unites (Length), materiaux (Material/MaterialLibrary), recettes (DepositionRecipe/EtchRecipe)
  geometry/       le moteur (Geometry/Layer, operations booleennes shapely)
  process/        les briques de process (ProcessStep) et simulate()
  presentation/   export SVG d'une Frame (script/notebook, sans la GUI)
  adapters/       pont optionnel vers follow (import paresseux, extra [follow])
  api/            backend FastAPI + frontend statique (vanilla JS/SVG, extra [api])
examples/         flows de process complets et executables (STI planaire, nanofils III-N)
tests/            suite pytest (materiaux, recettes, moteur geometrique, simulate, API)
```

## Développer

```bash
pip install -e ".[dev,api]"
pytest
```
