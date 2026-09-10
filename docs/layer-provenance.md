# Provenance d'une couche : valeur + comment on y arrive

Ce document décrit `Traced` et `LayerProvenance` (`structureforge/core/traced.py`,
`structureforge/geometry/engine.py`) : le mécanisme qui permet à un `Layer` de porter, en plus
de `material`/`polygon`, un nombre **ouvert et imprévisible** de paramètres process - une
épaisseur aujourd'hui, un dopage demain, une rugosité mesurée après - sans jamais avoir à
retoucher la structure de données pour en faire de la place.

## Le problème que ça résout

Un `Layer` (`structureforge/geometry/engine.py`) ne porte que sa géométrie :

```python
@dataclass
class Layer:
    material: str
    polygon: BaseGeometry
```

Le `ProcessStep` qui l'a créé (`structureforge/process/steps.py`) porte, lui, tous les
paramètres utilisés (`thickness`, `rate_c`, `seed_materials`...) - mais cette information
n'est jamais recopiée sur le `Layer` résultant. Une fois `simulate()` passé, la géométrie ne
sait plus "comment" elle a été atteinte, sauf à remonter, en dehors du résultat, jusqu'à la
liste `steps` d'origine.

`LayerProvenance` referme cette boucle : c'est un enregistrement optionnel, attaché à un
`Layer`, qui dit *quel step* l'a produit et *avec quels paramètres* - chacun d'eux pouvant
individuellement porter sa propre traçabilité.

## `Traced` : le conteneur à deux niveaux

```python
class Traced(BaseModel):
    value: Any
    derivation: Any = None
```

`Traced` ne grossit jamais pour accueillir un nouveau type de paramètre - il reste exactement
`{value, derivation}` pour toujours. Ce qui change, c'est la **forme** de `derivation`. Trois
façons de le construire, une par cas d'usage :

### 1. `Traced.literal(value)` — juste la valeur

Rien n'est su (ou digne d'être noté) sur la façon dont on y est arrivé.

```python
from structureforge.core.traced import Traced

roughness = Traced.literal(0.8)
# Traced(value=0.8, derivation=None)
```

Cas d'usage typique : une rugosité RMS mesurée à l'AFM après coup - il n'y a pas de "façon de
l'obtenir" au sens process, juste une mesure.

### 2. `Traced.declared(value, **obtention_params)` — valeur + paramètres déclarés, non calculés

La valeur est donnée directement (mesurée, connue par ailleurs), et on **note en plus** les
paramètres process qui l'ont produite - mais rien ici ne calcule `value` à partir d'eux.

```python
doping = Traced.declared(2.5e18, precursor="SiH4", flow_sccm=12, duration_s=97.2)
# Traced(value=2.5e18, derivation={"precursor": "SiH4", "flow_sccm": 12, "duration_s": 97.2})
```

Cas d'usage typique : un dopage - la concentration réelle vient d'une mesure SIMS (ou d'une
calibration empirique), mais on veut quand même garder trace du flux de précurseur et de la
durée utilisés, même si personne n'a (encore) écrit le modèle de solubilité qui inverserait
l'un depuis l'autre.

### 3. `Traced.computed(value, derivation)` — valeur calculée depuis une formule exécutable

`derivation` est ici un modèle qui sait se résoudre lui-même (il expose son propre
`resolve_*()`) - `value` est censé être exactement ce que cette formule a produit, pas un
nombre fourni indépendamment.

```python
from structureforge.core.derivation import ArrheniusRate, GrowthAtRate

derivation = GrowthAtRate(
    rate=ArrheniusRate(prefactor_nm_per_s=1e7, activation_energy_eV=1.9, temperature_K=1323.15),
    duration_s=100.0,
)
thickness = Traced.computed(value=derivation.resolve_nm(), derivation=derivation)
# Traced(value=57.9487..., derivation=GrowthAtRate(rate=ArrheniusRate(...), duration_s=100.0))
```

C'est exactement ce que fait déjà `Length`/`LengthDerivation` (`core/derivation.py`) pour une
épaisseur - `Traced` ne fait que rendre ce même pattern réutilisable pour n'importe quel autre
paramètre.

Les trois se sérialisent en JSON de la même façon (`.model_dump(mode="json")`), `derivation`
valant simplement `null`, un dict libre, ou l'arbre du modèle selon le cas :

```json
{"value": 0.8, "derivation": null}
{"value": 2.5e+18, "derivation": {"precursor": "SiH4", "flow_sccm": 12, "duration_s": 97.2}}
{"value": 57.9487..., "derivation": {"kind": "growth_at_rate", "rate": {"kind": "arrhenius_rate", ...}, "duration_s": 100.0}}
```

## `LayerProvenance` : la table ouverte de paramètres

```python
class LayerProvenance(BaseModel):
    step_kind: str
    step_name: str
    parameters: dict[str, Traced]
```

`parameters` est une simple table `nom -> Traced`, jamais une liste de champs fixes. C'est ce
qui permet exactement le scénario visé : aujourd'hui une couche n'a que `thickness`, demain on
lui ajoute `doping_cm3`, après-demain `roughness_rms_nm` - à chaque fois une nouvelle clé dans
le dict, jamais une modification de `LayerProvenance` ou de `Layer` lui-même.

```python
@dataclass
class Layer:
    material: str
    polygon: BaseGeometry
    provenance: LayerProvenance | None = None   # optionnel, défaut None - zéro impact sur l'existant
```

## Exemple complet, vérifié en exécution réelle

### Couche A - juste une valeur mesurée

```python
from shapely.geometry import box
from structureforge.core.traced import Traced
from structureforge.geometry.engine import Layer, LayerProvenance

layer_a = Layer(
    material="GaN",
    polygon=box(0, 0, 100, 50),
    provenance=LayerProvenance(
        step_kind="measurement", step_name="AFM post-croissance",
        parameters={"roughness_rms_nm": Traced.literal(0.8)},
    ),
)
```
```json
{"step_kind": "measurement", "step_name": "AFM post-croissance",
 "parameters": {"roughness_rms_nm": {"value": 0.8, "derivation": null}}}
```

### Couche B - valeur + paramètres d'obtention déclarés (non calculés)

```python
layer_b = Layer(
    material="GaN",
    polygon=box(0, 0, 100, 20),
    provenance=LayerProvenance(
        step_kind="chemical", step_name="Dopage n",
        parameters={"doping_cm3": Traced.declared(2.5e18, precursor="SiH4", flow_sccm=12, duration_s=97.2)},
    ),
)
```
```json
{"step_kind": "chemical", "step_name": "Dopage n",
 "parameters": {"doping_cm3": {"value": 2.5e+18,
   "derivation": {"precursor": "SiH4", "flow_sccm": 12, "duration_s": 97.2}}}}
```

### Couche C - valeur calculée depuis une formule, branchée automatiquement par `simulate()`

```python
from structureforge.core.derivation import ArrheniusRate, GrowthAtRate
from structureforge.core.units import Length
from structureforge.process.steps import FacetedGrowth
from structureforge.process.simulate import simulate

thickness = Length.derived(GrowthAtRate(
    rate=ArrheniusRate(prefactor_nm_per_s=1e7, activation_energy_eV=1.9, temperature_K=1323.15),
    duration_s=100.0,
))
step = FacetedGrowth(name="Croissance GaN", material="GaN", thickness=thickness,
                      rate_c=1.0, rate_m=0.0, rate_sp=0.0)

frames = simulate(geometry, [step], materials, recipes)
layer_c = frames[-1].layers[-1]
```
```json
{
  "step_kind": "faceted_growth", "step_name": "Croissance GaN",
  "parameters": {
    "thickness": {
      "value": 57.94871807280426,
      "derivation": {
        "kind": "growth_at_rate",
        "rate": {"kind": "arrhenius_rate", "prefactor_nm_per_s": 10000000.0,
                  "activation_energy_eV": 1.9, "temperature_K": 1323.15},
        "duration_s": 100.0
      }
    },
    "rate_c": {"value": 1.0, "derivation": null},
    "rate_m": {"value": 0.0, "derivation": null},
    "rate_sp": {"value": 0.0, "derivation": null},
    "semi_polar_angle_deg": {"value": 30.0, "derivation": null},
    "seed_materials": {"value": [], "derivation": null}
  }
}
```

Personne n'a construit `LayerProvenance` à la main pour la couche C : `_apply()`
(`structureforge/process/simulate.py`) le fait automatiquement pour tout step
`EpitaxialGrowth`/`FacetedGrowth`, via l'aide `_traced_length()` qui pont l'épaisseur
`Length`/`LengthDerivation` déjà existante vers un `Traced` - littéral si `Length(value=...)`,
calculé si `Length.derived(...)`.

## Ce qui est câblé aujourd'hui, et ce qui ne l'est pas

| Step | `provenance` posée automatiquement par `simulate()` |
|---|---|
| `EpitaxialGrowth` | ✅ (`thickness`, `orientation`, `angle_deg`, `seed_materials`) |
| `FacetedGrowth` | ✅ (`thickness`, `rate_c`, `rate_m`, `rate_sp`, `semi_polar_angle_deg`, `seed_materials`) |
| `Deposition`, `Etch`, `Lithography`, `Planarization`, `Flip`, `ResistStrip`, `ChemicalStep` | ❌ - `Layer.provenance` reste `None` |

`Geometry.deposit_epitaxial()` et `Geometry.deposit_faceted()` acceptent un paramètre
`provenance: LayerProvenance | None = None` (défaut `None`, donc zéro impact si on ne le passe
pas) et le posent tel quel sur chaque `Layer` qu'ils créent - y compris, pour
`deposit_faceted`, sur chacune des couches d'un éventuel éclatement multi-matériaux
(`material_c`/`material_m`/`material_sp`). Étendre la couverture à un autre `deposit_*`
(conformal, directional...) ou à un autre step (un futur `Doping`, par exemple) suit exactement
le même patron : ajouter le paramètre `provenance` à la méthode `Geometry` concernée, puis le
construire et le passer depuis `_apply()`.

## Pourquoi pas de champs nommés directement sur `Layer`

L'alternative la plus simple - coller `thickness`, `rate_c`, `doping`... directement comme
attributs de `Layer` - casse dès qu'un même step produit plusieurs `Layer` distincts (le cas
`material_c`/`material_m`/`material_sp` de `deposit_faceted` : chaque morceau n'a été façonné
que par **sa propre** famille de facette, pas par les trois), et oblige à modifier `Layer`
chaque fois qu'un nouveau paramètre process apparaît. La table ouverte `parameters: dict[str,
Traced]` évite les deux : chaque `Layer` ne porte que ce qui l'a réellement concerné, et
`Layer`/`LayerProvenance` n'ont plus jamais besoin de changer de forme.
