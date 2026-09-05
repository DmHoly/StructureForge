"""The 2D cross-section geometry engine.

Layers are shapely polygons kept in construction order. Deposition/etch/planarization are all
implemented as boolean operations on that stack. See `Geometry`'s docstring for the v1
simplifications (hard-silhouette directional shadowing, domain edges treated as symmetry
boundaries, substep-based selective etch) - they're deliberate scope cuts for a first version,
not oversights.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from shapely.affinity import scale, translate
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from ..core.materials import MaterialLibrary
from ..core.recipes import DepositionMode, DepositionRecipe, EtchMode, EtchRecipe

_EPS_AREA = 1e-6  # nm^2 - polygons smaller than this are numerical noise, dropped
_MARGIN = 1.0  # nm of slack padding used around bounding boxes for directional etch
_GUARD_MARGIN = 1.0e4  # nm - "effectively infinite" padding for other-factor layers during etch, see Geometry.etch
_BULK_MARGIN = 1.0e4  # nm - "effectively infinite" downward extension standing in for the wafer's bulk, see floor_nm
_SIMPLIFY_TOL = 0.02  # nm - keeps vertex count from growing unboundedly over many substeps


def _clean(geom: BaseGeometry) -> BaseGeometry:
    """Fix up sliver/self-intersection artifacts left by repeated boolean ops, and cap vertex
    growth (each buffer/union call can otherwise add vertices indefinitely over many substeps).
    """
    if geom.is_empty:
        return geom
    return geom.buffer(0).simplify(_SIMPLIFY_TOL, preserve_topology=True)


def _drop_tiny(geom: BaseGeometry) -> BaseGeometry:
    if geom.is_empty:
        return geom
    if isinstance(geom, MultiPolygon):
        kept = [g for g in geom.geoms if g.area > _EPS_AREA]
        if not kept:
            return Polygon()
        return kept[0] if len(kept) == 1 else MultiPolygon(kept)
    return geom if geom.area > _EPS_AREA else Polygon()


def sweep_union(geom: BaseGeometry, vector: tuple[float, float]) -> BaseGeometry:
    """The exact Minkowski sum of `geom` with the segment from (0,0) to `vector` - this is what
    "directional" deposition/etch actually mean: a uniform offset in one direction, rather than
    every direction like `.buffer()`.

    Computed directly rather than by sampling many intermediate translations: the union of `geom`,
    `geom` translated by `vector`, and - for every edge of every ring of every part of `geom` - the
    parallelogram that edge sweeps out along `vector`, is exactly the swept region (sweeping a
    rigid shape along a straight line traces out its own two end positions plus, along the way,
    the quad each edge carries with it). Besides being exact rather than an approximation, this
    sidesteps a real GEOS robustness trap the earlier sampled version had: many translated copies
    at regular, exactly-aligned offsets is a textbook way to trigger a spurious "side location
    conflict" TopologyException in `unary_union` (axis-parallel edges landing exactly on top of
    each other at various samples) - a handful of edge-quads doesn't have that failure mode.
    """
    if geom.is_empty or (vector[0] == 0 and vector[1] == 0):
        return geom
    parts = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
    pieces = [geom, translate(geom, vector[0], vector[1])]
    for part in parts:
        for ring in [part.exterior, *part.interiors]:
            coords = list(ring.coords)
            for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
                pieces.append(
                    Polygon([(x1, y1), (x2, y2), (x2 + vector[0], y2 + vector[1]), (x1 + vector[0], y1 + vector[1])])
                )
    return _clean(unary_union(pieces))


def beam_vector(angle_deg: float, length: float) -> tuple[float, float]:
    """Direction a beam travels into the structure (source above, travelling down), for a recipe
    tilted `angle_deg` from the surface normal (0 = straight down, positive tilts towards +x),
    scaled to `length`.
    """
    angle = math.radians(angle_deg)
    return (length * math.sin(angle), -length * math.cos(angle))


@dataclass
class Layer:
    """One material region of the stack, kept in construction order (oldest first) - that order
    doubles as z-order, since a layer was, by construction, deposited on top of whatever existed
    when it was added. `Geometry.exposed` relies on this to say what's currently on the surface.
    """

    material: str
    polygon: BaseGeometry

    def rings(self) -> list[dict]:
        """This layer's polygon(s) as plain coordinate lists, for JSON/SVG rendering."""
        geoms = list(self.polygon.geoms) if isinstance(self.polygon, MultiPolygon) else [self.polygon]
        out = []
        for g in geoms:
            if g.is_empty:
                continue
            out.append(
                {
                    "exterior": [list(pt) for pt in g.exterior.coords],
                    "holes": [[list(pt) for pt in interior.coords] for interior in g.interiors],
                }
            )
        return out


class Geometry:
    """The evolving 2D cross-section: a fixed-width domain and an ordered stack of `Layer`.

    Known v1 simplifications, documented once here rather than scattered through the methods:

    - **Directional shadowing is a hard, single-bounce silhouette test, not a ray tracer.**
      `_shadow` sweeps the current solid forward along the beam direction, far enough to span the
      structure, and subtracts that from a directional deposit/etch's raw result - this is what
      makes a mesa's own leeward face, and a shorter feature standing behind a taller one, stay
      untouched instead of being coated/etched as if the beam passed straight through solid
      matter. It's still a simplification: no partial/soft shadows (a beam is either fully blocked
      or not - real sources aren't perfect points), and no secondary effects (reflection,
      redeposition of sputtered material). Isotropic processes have no direction to shadow along,
      so they ignore this entirely, as they should.
    - **Domain edges are symmetry boundaries, not free edges.** Every buffer/sweep operates on the
      geometry mirrored across `x=0` and `x=domain_width_nm` and is cropped back afterwards, so a
      feature near the edge behaves as if the pattern repeats past the boundary instead of
      rounding off oddly. Keep the domain wide enough that features of interest aren't hugging it.
    - **Selective etch runs in substeps.** Each substep erodes/sweeps the solid once per distinct
      rate factor present, with every *other*-factor layer temporarily padded to an effectively
      infinite thickness so that factor's erosion can only ever consume its own material - never
      tunnel through a thinner, slower-etching layer (a resist mask, a stop layer) into whatever
      sits beneath it. What's actually removed is intersected back against the real, unpadded
      per-layer polygons, so the padding never leaks into the result - only into how far a given
      substep's erosion is allowed to reach. This gets multi-material selectivity, masked
      undercut, and etch-through-once-a-mask-is-fully-consumed all correct without a full
      level-set solver, at the residual cost of a finite-step-size error bounded by roughly one
      substep's depth (see `etch`'s `steps` parameter) - the same kind of discretisation error any
      explicit time-stepping scheme has, not a structural limitation.
    """

    def __init__(self, domain_width_nm: float, layers: list[Layer] | None = None):
        self.domain_width_nm = domain_width_nm
        self.layers: list[Layer] = layers if layers is not None else []
        # Below this y, the substrate is treated as infinite bulk: never coated, never eroded -
        # otherwise a conformal deposit or isotropic etch (which act on *every* exposed edge of
        # the solid) would just as happily grow from / eat into the wafer's backside as its front.
        # Set by `substrate()`; a `Geometry` built by hand has none, i.e. no floor is enforced.
        self.floor_nm: float | None = None

    @classmethod
    def substrate(cls, material: str, domain_width_nm: float, thickness_nm: float) -> "Geometry":
        """A flat starting wafer: one layer of `material`, spanning the whole domain, top surface
        at y=0 - everything simulated afterwards grows upward from there. Its bottom edge becomes
        `floor_nm`: the boundary below which the wafer is inexhaustible bulk (see `__init__`).
        """
        poly = box(0.0, -thickness_nm, domain_width_nm, 0.0)
        geometry = cls(domain_width_nm, [Layer(material=material, polygon=poly)])
        geometry.floor_nm = -thickness_nm
        return geometry

    # -- stack queries --------------------------------------------------

    def solid(self) -> BaseGeometry:
        polys = [l.polygon for l in self.layers if not l.polygon.is_empty]
        if not polys:
            return Polygon()
        return _clean(unary_union(polys))

    def bounds(self) -> tuple[float, float, float, float]:
        solid = self.solid()
        if solid.is_empty:
            return (0.0, 0.0, self.domain_width_nm, 0.0)
        return solid.bounds

    # -- domain-edge handling --------------------------------------------

    def _mirror_pad(self, geom: BaseGeometry) -> BaseGeometry:
        if geom.is_empty:
            return geom
        left = scale(geom, xfact=-1, yfact=1, origin=(0.0, 0.0))
        right = scale(geom, xfact=-1, yfact=1, origin=(self.domain_width_nm, 0.0))
        return _clean(unary_union([geom, left, right]))

    def _bulk_pad(self, geom: BaseGeometry) -> BaseGeometry:
        """Extend `geom` far downward past `floor_nm`, so a buffer/sweep never sees the wafer's
        true bottom edge (see the `floor_nm` note in `__init__`). No-op if there's no floor.
        """
        if self.floor_nm is None or geom.is_empty:
            return geom
        bulk = box(0.0, self.floor_nm - _BULK_MARGIN, self.domain_width_nm, self.floor_nm)
        return _clean(unary_union([geom, bulk]))

    def _pad(self, geom: BaseGeometry) -> BaseGeometry:
        """Domain-edge handling for any buffer/sweep: extend past the wafer floor, then mirror
        across both vertical domain edges (see the class docstring's simplifications).
        """
        return self._mirror_pad(self._bulk_pad(geom))

    def _clamp_floor(self, y: float) -> float:
        return y if self.floor_nm is None else max(y, self.floor_nm)

    def _guard_extend_up(self, geom: BaseGeometry) -> BaseGeometry:
        """`geom` (a layer's polygon) extended far upward *within each of its own connected
        components' x-span* - not a uniform outward buffer, which would balloon sideways and
        swallow unrelated features. Used by `etch()` to make a slower-etching layer (a mask) act
        as effectively bottomless for one substep's faster erosion, without changing its footprint
        or affecting whatever sits *beneath* it.
        """
        if geom.is_empty:
            return geom
        parts = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
        extended = []
        for part in parts:
            minx, _, maxx, maxy = part.bounds
            above = box(minx, maxy, maxx, maxy + _GUARD_MARGIN)
            extended.append(unary_union([part, above]))
        return _clean(unary_union(extended))

    def _shadow(self, solid: BaseGeometry, angle_deg: float, y_min: float, y_max: float) -> BaseGeometry:
        """The region shadowed by `solid` for a beam tilted `angle_deg`: everywhere "behind"
        existing material, continuing in the beam's own forward direction, out to a length that
        safely spans the current structure. A directional process (deposition or etch) can't
        affect anything back there - which is what makes a feature's own leeward face, and a
        shorter feature standing behind a taller one, correctly stay untouched instead of being
        coated/etched as if the beam went straight through solid matter to reach them.

        Only mirror-padded (domain edges), deliberately *not* bulk-padded (`floor_nm`): the shadow
        sweep only ever extends a silhouette further along the beam direction, so there's no false
        "wafer backside" boundary to worry about here the way `deposit`/`etch` need to - and
        skipping it keeps this sweep's coordinates at the structure's own scale rather than the
        bulk pad's ~1e7 nm extension, which paired badly with `_UNION_GRID`'s precision.
        """
        if solid.is_empty:
            return solid
        padded = self._mirror_pad(solid)
        span = (y_max - y_min) + self.domain_width_nm + _MARGIN
        dx, dy = beam_vector(angle_deg, span)
        swept = sweep_union(padded, (dx, dy))
        return swept.difference(padded)

    def _domain_box(self, y_min: float, y_max: float) -> BaseGeometry:
        return box(0.0, y_min, self.domain_width_nm, y_max)

    def _crop(self, geom: BaseGeometry, y_min: float, y_max: float) -> BaseGeometry:
        return _drop_tiny(_clean(geom.intersection(self._domain_box(self._clamp_floor(y_min), y_max))))

    # -- process operations ----------------------------------------------

    def deposit_conformal(self, material: str, thickness_nm: float) -> None:
        """Uniform-thickness coat over every exposed surface, in every direction (CVD/ALD-like)."""
        self.deposit_conformal_masked(material, thickness_nm, open_x_ranges=[])

    def deposit_conformal_masked(
        self, material: str, thickness_nm: float, open_x_ranges: list[tuple[float, float]]
    ) -> None:
        """Like `deposit_conformal`, but the film is removed again wherever it falls within one of
        `open_x_ranges` (x-ranges, in nm) - used to lay down a patterned resist in one step
        (`structureforge.process.steps.Lithography`): coat conformally, then keep only outside the
        mask openings. The opening's edge is a vertical cut, not shaped to the topology beneath it
        - a deliberate simplification (real exposure/development isn't modelled).
        """
        if thickness_nm <= 0:
            return
        solid = self.solid()
        y_min, y_max = (solid.bounds[1], solid.bounds[3]) if not solid.is_empty else (0.0, 0.0)
        padded = self._pad(solid)
        grown = padded.buffer(thickness_nm, quad_segs=12)
        film = self._crop(grown.difference(padded), y_min - thickness_nm, y_max + thickness_nm)
        if open_x_ranges and not film.is_empty:
            y0, y1 = y_min - thickness_nm - 10.0, y_max + thickness_nm + 10.0
            openings = unary_union([box(x0, y0, x1, y1) for x0, x1 in open_x_ranges])
            film = _drop_tiny(_clean(film.difference(openings)))
        if not film.is_empty:
            self.layers.append(Layer(material=material, polygon=film))

    def deposit_directional(self, material: str, thickness_nm: float, angle_deg: float) -> None:
        """Line-of-sight deposition from one direction (PVD/evaporation-like): grows the solid by
        sweeping it towards the source (the beam's arrival direction, reversed), then removes
        whatever growth would have landed in another feature's shadow (including a feature's own
        leeward face) - see `_shadow`.
        """
        if thickness_nm <= 0:
            return
        solid = self.solid()
        y_min, y_max = (solid.bounds[1], solid.bounds[3]) if not solid.is_empty else (0.0, 0.0)
        dx, dy = beam_vector(angle_deg, thickness_nm)
        padded = self._pad(solid)
        swept = sweep_union(padded, (-dx, -dy))
        film_raw = swept.difference(padded)
        shadow = self._shadow(solid, angle_deg, y_min, y_max)
        film = self._crop(film_raw.difference(shadow), y_min - thickness_nm, y_max + thickness_nm)
        if not film.is_empty:
            self.layers.append(Layer(material=material, polygon=film))

    def deposit(self, material: str, thickness_nm: float, recipe: DepositionRecipe) -> None:
        if recipe.mode is DepositionMode.conformal:
            self.deposit_conformal(material, thickness_nm)
        else:
            self.deposit_directional(material, thickness_nm, recipe.angle_deg)

    def deposit_epitaxial(
        self,
        material: str,
        thickness_nm: float,
        orientation: str = "c_plane",
        angle_deg: float = 0.0,
        seed_materials: list[str] | None = None,
    ) -> None:
        """Selective-area epitaxial growth — three orientations, optional SAG selectivity.

        c_plane   — strictly upward growth along [0001]: new material rises above every exposed
                    seed surface.  Models blanket or SAG c-plane GaN/AlGaN/InGaN regrowth.
        m_plane   — lateral growth on {10-10} sidewalls: the film spreads horizontally in both
                    ±x directions from every exposed seed surface.  Models shell growth on a
                    pillar, or lateral ELO over a mask.
        semi_polar — growth at `angle_deg` from the c-axis (from vertical): the film fans out
                    both left-tilted and right-tilted from the seed surface, mimicking symmetric
                    facet growth on a mesa or V-groove.

        If `seed_materials` is non-empty the film nucleates *only* where one of those materials
        is the topmost exposed surface (every non-seed layer covering the seed blocks growth
        there — SAG selectivity).  Pass an empty list / None to grow on all exposed surfaces.
        """
        if thickness_nm <= 0 or not self.layers:
            return
        solid = self.solid()
        if solid.is_empty:
            return

        y_min, y_max = solid.bounds[1], solid.bounds[3]

        # --- resolve growth origin (with or without SAG selectivity) ---
        if seed_materials:
            seed_polys = [l.polygon for l in self.layers if l.material in seed_materials and not l.polygon.is_empty]
            if not seed_polys:
                return
            seed_union = _clean(unary_union(seed_polys))
            non_seed_polys = [l.polygon for l in self.layers if l.material not in seed_materials and not l.polygon.is_empty]
            if non_seed_polys:
                # A non-seed layer blocks the seed underneath its full x-footprint even though
                # it sits above the seed in y and doesn't geometrically overlap it. Project each
                # non-seed polygon downward (toward -∞) so the difference correctly removes the
                # seed region hidden under a mask.
                shadows = []
                for p in non_seed_polys:
                    parts = list(p.geoms) if isinstance(p, MultiPolygon) else [p]
                    for part in parts:
                        minx, miny, maxx, _ = part.bounds
                        shadows.append(box(minx, miny - _GUARD_MARGIN, maxx, miny))
                blocking = _clean(unary_union(non_seed_polys + shadows))
                exposed_seed = _clean(seed_union.difference(blocking))
            else:
                exposed_seed = seed_union
            if exposed_seed.is_empty:
                return
            growth_base = exposed_seed
        else:
            growth_base = solid

        padded = self._pad(growth_base)

        # --- sweep the growth base along the growth direction ---
        if orientation == "c_plane":
            film_raw = sweep_union(padded, (0.0, thickness_nm))
        elif orientation == "m_plane":
            # Symmetric lateral expansion on both sidewalls
            film_raw = _clean(unary_union([
                sweep_union(padded, (thickness_nm, 0.0)),
                sweep_union(padded, (-thickness_nm, 0.0)),
            ]))
        elif orientation == "semi_polar":
            # Symmetric tilted facets: ±x tilt from vertical by angle_deg
            rad = math.radians(angle_deg)
            dx = thickness_nm * math.sin(rad)
            dy = thickness_nm * math.cos(rad)
            film_raw = _clean(unary_union([
                sweep_union(padded, (dx, dy)),
                sweep_union(padded, (-dx, dy)),
            ]))
        else:  # pragma: no cover
            raise ValueError(f"unknown epitaxial orientation {orientation!r}")

        film = self._crop(_clean(film_raw.difference(padded)), y_min - thickness_nm, y_max + thickness_nm)
        film = _drop_tiny(_clean(film.difference(solid)))

        if not film.is_empty:
            self.layers.append(Layer(material=material, polygon=film))

    def etch(
        self,
        recipe: EtchRecipe,
        depth_nm: float,
        materials: MaterialLibrary,
        steps: int | None = None,
    ) -> None:
        """Remove material along `recipe`'s direction, `depth_nm` deep for the recipe's reference
        (factor 1.0) material - other materials recede slower/faster per `recipe.factor_for`.
        Runs in substeps so a mixed-rate recipe advances each material's own front correctly,
        including etching through a thin masking layer into whatever sits below it.

        Each substep computes one erosion/sweep per distinct rate factor present. Any *slower*
        layer that is itself currently exposed to the true surface (a resist mask, a stop layer -
        checked against the padded solid's boundary, so the wafer floor and domain edges don't
        count) is temporarily extended far upward (`_GUARD_MARGIN`) before a *faster* factor's
        erosion is computed - otherwise, once that slower layer gets thinner than one substep's
        depth at the faster rate, the faster erosion would tunnel straight through it into
        whatever sits beneath, ignoring the mask. A slower layer that isn't itself exposed (e.g. a
        slow-etching substrate buried under a faster top layer) is left alone: it isn't in the way
        of anything this substep, and extending it upward would incorrectly swallow whatever
        faster material sits above it. What's actually removed is always intersected back against
        the real, unextended per-layer polygons, so the padding never leaks into the result itself
        - it only prevents a given substep's erosion from reaching further than it should.
        """
        if depth_nm <= 0 or not self.layers:
            return
        if steps is None:
            steps = min(40, max(10, round(depth_nm / 2)))
        substep = depth_nm / steps

        for _ in range(steps):
            solid = self.solid()
            if solid.is_empty:
                break
            y_min, y_max = solid.bounds[1], solid.bounds[3]
            # Padded (mirrored + bulk-extended) once per substep: used below to test whether a
            # layer is exposed to the *true* surface, without the wafer floor or the domain's
            # left/right edges - which are real edges of the raw `solid` polygon but not real
            # exposure - registering as false positives. Also the basis for this substep's
            # shadow (directional mode only) - computed from the *plain* solid, never the
            # per-factor guard (which deliberately makes a mask look near-infinitely thick and
            # would cast a wildly oversized false shadow if used here).
            padded_plain = self._pad(solid)
            padded_solid_boundary = padded_plain.boundary
            shadow = (
                self._shadow(solid, recipe.angle_deg, y_min, y_max)
                if recipe.mode is EtchMode.directional
                else None
            )

            factor_by_index: dict[int, float] = {}
            for i, layer in enumerate(self.layers):
                if layer.polygon.is_empty:
                    continue
                factor_by_index[i] = recipe.factor_for(materials.get(layer.material))
            distinct_factors = sorted(set(factor_by_index.values()))

            ring_by_factor: dict[float, BaseGeometry] = {}
            for factor in distinct_factors:
                if factor <= 0:
                    ring_by_factor[factor] = Polygon()
                    continue
                # Only guard against *slower* other layers that are themselves currently exposed
                # to the true surface: a mask sitting on top of this factor's own material could
                # otherwise be tunnelled through once it's thinner than this substep's depth. A
                # slower layer that isn't exposed at all (e.g. a slow-etching substrate buried
                # under a faster top layer) isn't "in the way" of anything this substep and would
                # only corrupt the guard if extended upward regardless (see the etch() docstring).
                slower_layers = [
                    self.layers[i].polygon
                    for i, f in factor_by_index.items()
                    if f < factor
                    and not self.layers[i].polygon.is_empty
                    and self.layers[i].polygon.intersects(padded_solid_boundary)
                ]
                guard = solid if not slower_layers else _clean(
                    unary_union([solid] + [self._guard_extend_up(p) for p in slower_layers])
                )
                padded = self._pad(guard)
                if recipe.mode is EtchMode.isotropic:
                    eroded = padded.buffer(-substep * factor, quad_segs=12)
                    ring = padded.difference(eroded)
                else:
                    dx, dy = beam_vector(recipe.angle_deg, substep * factor)
                    px0, _, px1, _ = padded.bounds
                    air_bbox = box(px0 - 1.0, y_min - substep * factor - _MARGIN, px1 + 1.0, y_max + _MARGIN)
                    air = air_bbox.difference(padded)
                    swept_air = sweep_union(air, (dx, dy))
                    ring = padded.intersection(swept_air).difference(shadow)
                ring_by_factor[factor] = self._crop(ring, y_min - substep - _MARGIN, y_max + _MARGIN)

            removed_parts = [
                self.layers[i].polygon.intersection(ring_by_factor[factor])
                for i, factor in factor_by_index.items()
                if factor > 0
            ]
            removed_parts = [p for p in removed_parts if not p.is_empty]
            if not removed_parts:
                continue
            total_removed = _clean(unary_union(removed_parts))
            new_solid = _clean(solid.difference(total_removed))
            for layer in self.layers:
                if not layer.polygon.is_empty:
                    layer.polygon = _drop_tiny(_clean(layer.polygon.intersection(new_solid)))

    def planarize(self, target_level_nm: float | None = None, stop_material: str | None = None) -> None:
        """Cut the stack flat at `target_level_nm`, or - given `stop_material` instead - at the
        current top of that material's layer(s) (CMP-style "polish until the stop layer"). Give
        exactly one of the two.
        """
        if (target_level_nm is None) == (stop_material is None):
            raise ValueError("planarize needs exactly one of target_level_nm or stop_material")
        solid = self.solid()
        if solid.is_empty:
            return
        y_min = solid.bounds[1]
        if stop_material is not None:
            regions = [l.polygon for l in self.layers if l.material == stop_material and not l.polygon.is_empty]
            if not regions:
                raise ValueError(f"planarize: no layer of material {stop_material!r} exists yet to stop on")
            target_level_nm = unary_union(regions).bounds[3]
        new_solid = _clean(solid.intersection(self._domain_box(y_min - 1.0, target_level_nm)))
        for layer in self.layers:
            if not layer.polygon.is_empty:
                layer.polygon = _drop_tiny(_clean(layer.polygon.intersection(new_solid)))

    def remove_floating_debris(self) -> None:
        """Drop any part of the solid not connected down to the substrate (`layers[0]`). Called
        after a resist strip so material deposited on top of resist lifts off with it instead of
        floating in place - an emergent, connectivity-only approximation of lift-off, not a
        physical adhesion/mechanical model.
        """
        if not self.layers or self.layers[0].polygon.is_empty:
            return
        solid = self.solid()
        if solid.is_empty:
            return
        anchor = self.layers[0].polygon
        components = list(solid.geoms) if isinstance(solid, MultiPolygon) else [solid]
        kept = [c for c in components if c.intersects(anchor)]
        if len(kept) == len(components):
            return
        new_solid = _clean(unary_union(kept)) if kept else Polygon()
        for layer in self.layers:
            if not layer.polygon.is_empty:
                layer.polygon = _drop_tiny(_clean(layer.polygon.intersection(new_solid)))

    def strip_material(self, material: str) -> None:
        """Remove every layer of `material` entirely (e.g. a resist strip)."""
        for layer in self.layers:
            if layer.material == material:
                layer.polygon = Polygon()

    def compact(self) -> None:
        """Drop layers that have become empty (stripped, or fully consumed by an etch)."""
        self.layers = [l for l in self.layers if not l.polygon.is_empty]

    def frame_layers(self) -> list[Layer]:
        """Non-empty layers, oldest first - one frame of the process history for rendering."""
        return [l for l in self.layers if not l.polygon.is_empty]
