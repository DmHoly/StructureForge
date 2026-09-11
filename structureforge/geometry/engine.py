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

from pydantic import BaseModel, ConfigDict, Field
from shapely.affinity import scale, translate
from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from ..core.materials import MaterialLibrary
from ..core.recipes import DepositionMode, DepositionRecipe, EtchMode, EtchRecipe
from ..core.traced import Traced

_EPS_AREA = 1e-6  # nm^2 - polygons smaller than this are numerical noise, dropped
_MARGIN = 1.0  # nm of slack padding used around bounding boxes for directional etch
_GUARD_MARGIN = 1.0e4  # nm - "effectively infinite" padding for other-factor layers during etch, see Geometry.etch
_BULK_MARGIN = 1.0e4  # nm - "effectively infinite" downward extension standing in for the wafer's bulk, see floor_nm
_SIMPLIFY_TOL = 0.02  # nm - keeps vertex count from growing unboundedly over many substeps
_TOUCH_EPS = 1e-6  # nm - closes a single-point/zero-width seam between sibling polygon parts;
                    # far below any real feature size, so it never bridges a genuine gap.


def _clean(geom: BaseGeometry) -> BaseGeometry:
    """Fix up sliver/self-intersection artifacts left by repeated boolean ops, and cap vertex
    growth (each buffer/union call can otherwise add vertices indefinitely over many substeps).
    """
    if geom.is_empty:
        return geom
    return geom.buffer(0).simplify(_SIMPLIFY_TOL, preserve_topology=True)


def _merge_touching(geom: BaseGeometry) -> BaseGeometry:
    """Merge sibling polygon parts of a `MultiPolygon` that only touch at a single point or a
    zero-width seam into one polygon each, instead of leaving them as separate parts that share a
    boundary but never overlap.

    `_offset_named_facets` pivots each corner's mitre/bevel fan from its own vertex independently
    (see its docstring); two fans nucleated from opposite sides of a closing facet - most visibly
    a `deposit_faceted` tip narrowing to a point - can end up touching at exactly one point rather
    than overlapping over a real area. `unary_union`/`.buffer(0)` leave such parts as separate
    `MultiPolygon` members even though they share that point: a real closing tip rendered as two
    separate "pointed" pieces pinched together instead of one continuous shape. A vanishingly
    small closing buffer (dilate then erode by `_TOUCH_EPS`) turns the point-contact into a shared
    area first, merging the parts into one polygon; genuinely separate features - always many
    orders of magnitude further apart than `_TOUCH_EPS` - are unaffected. Deliberately not folded
    into `_clean()`, which runs on every intermediate boolean op in this module and would pay this
    buffer's cost far more often than needed: called instead once per `Geometry.solid()` (the one
    place every layer, however it was built, ends up merged into the single shape everything else
    - rendering, further growth, etch - reads back), and only actually does the extra buffer work
    on the `MultiPolygon` case it exists for.
    """
    if isinstance(geom, MultiPolygon) and len(geom.geoms) > 1:
        geom = _clean(geom.buffer(_TOUCH_EPS, quad_segs=1).buffer(-_TOUCH_EPS, quad_segs=1))
    return geom


def _drop_tiny(geom: BaseGeometry) -> BaseGeometry:
    if geom.is_empty:
        return geom
    if isinstance(geom, MultiPolygon):
        kept = [g for g in geom.geoms if g.area > _EPS_AREA]
        if not kept:
            return Polygon()
        return kept[0] if len(kept) == 1 else MultiPolygon(kept)
    return geom if geom.area > _EPS_AREA else Polygon()


def _fill_holes(geom: BaseGeometry) -> BaseGeometry:
    """Drop every interior ring from `geom`.

    `_offset_named_facets` mitres/bevels each vertex of a facet chain independently: when the
    same emergent facet must nucleate from two separate convex corners flanking a short, already-
    frozen bevel edge (left over from an earlier lopsided-rate step - see `mitre_or_bevel`), each
    corner's fan is pivoted from its own vertex with no knowledge of the other, and the two fans
    can fail to reach each other, leaving a sliver of the tip ungrown - a small, real, spurious
    interior hole in the resulting film, not a physical void (this engine has no mechanism for
    genuine coalescence-over-a-trench voids - every part of `solid` is offset independently and
    then unioned back together). Used only by `deposit_faceted`, the one caller whose offset
    construction can produce this artifact.
    """
    if geom.is_empty:
        return geom
    if isinstance(geom, MultiPolygon):
        return MultiPolygon([Polygon(g.exterior) for g in geom.geoms if not g.is_empty])
    return Polygon(geom.exterior)


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


def _offset_named_facets(
    solid: BaseGeometry, named_directions: list[tuple[float, float, float, str]]
) -> dict[str, BaseGeometry]:
    """Advance each straight edge of `solid`'s exterior ring(s) along its own outward normal by
    the distance named for that exact direction in `named_directions` (nx, ny, distance, family
    label) - 0 (or absent) for any direction not meant to move. Every other edge (any normal not
    in the list at all) is left exactly where it is.

    Returns the *new* area alone (not yet unioned with `solid`, nor differenced against it -
    callers do both), grouped by the family label of whichever named direction produced each
    piece: every edge strip belongs entirely to its own edge's family, and a corner fan spanning
    several named directions is split into one sub-wedge per direction it nucleates, so a caller
    that wants "more indium on the c-plane, less on the semi-polar facets" can hand each family's
    share of the film to a different material instead of one flat composition for the whole layer.

    This is a per-facet offset, not a global one. For each vertex, the two adjacent edges' offset
    lines (plus, at a convex corner, any named direction whose normal cone falls strictly between
    them and isn't already an edge there - inserted as new facet(s), splitting the corner into a
    short chain of mitre points) are intersected pairwise, giving that vertex a `near` point (where
    its incoming edge's line meets the chain) and a `far` point (where the chain meets its outgoing
    edge's line). Two pieces per vertex/edge reconstruct the grown boundary: a mitre-fill polygon
    fanned from the vertex through its whole chain whenever more than one point was needed there,
    and - for every edge with a nonzero distance - a strip from that edge's own two endpoints to
    its neighbours' `far`/`near` points. A zero-distance edge needs neither: it's already part of
    `solid`, unchanged, and its neighbours' own strips/fans butt up against its two endpoints as-is.

    This gives a rate of 0 a real guarantee a shared global growth vector never could: that facet's
    *own line* never moves, no matter how far a neighbouring facet advances - only how much of it
    stays exposed can change (e.g. flanking SP facets eating into a c-plane top as they grow, or a
    facet nucleating from scratch at a bare corner the first time a direction with no prior edge
    there gets a positive rate).

    Reflex (concave) corners never sprout a new facet - the two adjacent offset lines are mitred
    directly, since a concave corner's normal cone points back into the solid, not out into growth
    space.

    A mitre between two very unevenly-advancing facets at a shallow angle can land far outside
    anything either facet actually moved (the classic miter-join blowup); when a candidate mitre
    point would land past `1.5 * (d1 + d2)` from the vertex, `mitre_or_bevel` bevels that corner
    instead - the two facets' plain per-line offsets, joined by a short straight cut - rather than
    chase a point that distant.
    """
    named_directions = [(nx, ny, d, label) for nx, ny, d, label in named_directions if d > 1e-9]
    named = [(nx, ny, d, label, math.atan2(ny, nx)) for nx, ny, d, label in named_directions]

    def classify(nx: float, ny: float) -> tuple[float, str]:
        for dx, dy, d, label, _ang in named:
            if abs(nx - dx) < 1e-4 and abs(ny - dy) < 1e-4:
                return d, label
        return 0.0, ""

    def mitre_or_bevel(
        v: tuple[float, float],
        n1: tuple[float, float],
        d1: float,
        label1: str,
        n2: tuple[float, float],
        d2: float,
        label2: str,
    ) -> list[tuple[tuple[float, float], str]]:
        """The point where offset lines n1@d1 and n2@d2 (both measured from `v`) cross, tagged
        with `label2` (the direction it's arriving at) - or, if that crossing lands unreasonably
        far out (near-tangent normals paired with very unequal distances - e.g. a barely-active
        facet next to a fast one, at a shallow angle between them), two points instead: the plain
        per-line offsets (tagged with their own facet's label), bevelling the corner rather than
        chasing a mitre point that could otherwise land tens of times further out than either
        facet actually advanced.
        """
        p1 = (v[0] + n1[0] * d1, v[1] + n1[1] * d1)
        p2 = (v[0] + n2[0] * d2, v[1] + n2[1] * d2)
        a, b = n1
        c, d = n2
        det = a * d - b * c
        if abs(det) < 1e-9:
            return [(p1, label2)]
        px = (d1 * d - b * d2) / det
        py = (a * d2 - c * d1) / det
        point = (v[0] + px, v[1] + py)
        if math.hypot(point[0] - v[0], point[1] - v[1]) > 1.5 * (d1 + d2):
            return [(p1, label1), (p2, label2)]
        return [(point, label2)]

    def between_ccw(angle: float, lo: float, hi: float) -> bool:
        span = (hi - lo) % (2 * math.pi)
        rel = (angle - lo) % (2 * math.pi)
        return 1e-9 < rel < span - 1e-9

    pieces_by_family: dict[str, list[BaseGeometry]] = {}

    def add_piece(label: str, geom: BaseGeometry) -> None:
        pieces_by_family.setdefault(label, []).append(geom)

    parts = list(solid.geoms) if isinstance(solid, MultiPolygon) else [solid]
    for part in parts:
        part = orient(part, sign=1.0)
        coords = list(part.exterior.coords)[:-1]
        coords = [p for i, p in enumerate(coords) if p != coords[i - 1]]
        n = len(coords)
        if n < 3:
            continue

        normals: list[tuple[float, float]] = []
        dists: list[float] = []
        labels: list[str] = []
        for i in range(n):
            x1, y1 = coords[i]
            x2, y2 = coords[(i + 1) % n]
            length = math.hypot(x2 - x1, y2 - y1)
            nx, ny = (y2 - y1) / length, -(x2 - x1) / length
            normals.append((nx, ny))
            d, label = classify(nx, ny)
            dists.append(d)
            labels.append(label)

        near_pt: list[tuple[float, float]] = [(0.0, 0.0)] * n
        far_pt: list[tuple[float, float]] = [(0.0, 0.0)] * n
        for i in range(n):
            v = coords[i]
            n_in, d_in, lbl_in = normals[i - 1], dists[i - 1], labels[i - 1]
            n_out, d_out, lbl_out = normals[i], dists[i], labels[i]
            edge_in_dir = (-n_in[1], n_in[0])
            edge_out_dir = (-n_out[1], n_out[0])
            convex = edge_in_dir[0] * edge_out_dir[1] - edge_in_dir[1] * edge_out_dir[0] > 1e-9

            chain = [(n_in[0], n_in[1], d_in, lbl_in)]
            if convex:
                ang_in = math.atan2(n_in[1], n_in[0])
                ang_out = math.atan2(n_out[1], n_out[0])
                extras = sorted(
                    ((nx, ny, d, label, ang) for nx, ny, d, label, ang in named if between_ccw(ang, ang_in, ang_out)),
                    key=lambda e: (e[4] - ang_in) % (2 * math.pi),
                )
                chain.extend((nx, ny, d, label) for nx, ny, d, label, _ang in extras)
            chain.append((n_out[0], n_out[1], d_out, lbl_out))

            mitre_points = [
                tagged
                for (nx1, ny1, d1, l1), (nx2, ny2, d2, l2) in zip(chain, chain[1:])
                for tagged in mitre_or_bevel(v, (nx1, ny1), d1, l1, (nx2, ny2), d2, l2)
            ]
            near_pt[i] = mitre_points[0][0]
            far_pt[i] = mitre_points[-1][0]
            for (p_a, label_a), (p_b, _label_b) in zip(mitre_points, mitre_points[1:]):
                add_piece(label_a, Polygon([v, p_a, p_b]).buffer(0))

        for i in range(n):
            if dists[i] <= 0:
                continue
            v_i, v_next = coords[i], coords[(i + 1) % n]
            add_piece(labels[i], Polygon([v_i, v_next, near_pt[(i + 1) % n], far_pt[i]]).buffer(0))

    return {label: _clean(unary_union(pcs)) for label, pcs in pieces_by_family.items()}


def beam_vector(angle_deg: float, length: float) -> tuple[float, float]:
    """Direction a beam travels into the structure (source above, travelling down), for a recipe
    tilted `angle_deg` from the surface normal (0 = straight down, positive tilts towards +x),
    scaled to `length`.
    """
    angle = math.radians(angle_deg)
    return (length * math.sin(angle), -length * math.cos(angle))


class LayerProvenance(BaseModel):
    """Which process step produced a `Layer`, and the process parameters that were active -
    each individually possibly carrying its own `Traced` derivation (see
    `structureforge.core.traced`).

    Deliberately a single open `parameters` table rather than named fields: a `Layer` can end
    up needing an unbounded, unpredictable set of these over time (a thickness today, a doping
    concentration next, a measured surface roughness after that), and `LayerProvenance` must
    never need to change shape just to make room for a new one - only its caller (whichever
    `Geometry.deposit_*` builds it) needs to grow the dict it passes in.
    """

    model_config = ConfigDict(frozen=True)

    step_kind: str
    step_name: str
    parameters: dict[str, Traced] = Field(default_factory=dict)


@dataclass
class Layer:
    """One material region of the stack, kept in construction order (oldest first) - that order
    doubles as z-order, since a layer was, by construction, deposited on top of whatever existed
    when it was added. `Geometry.exposed` relies on this to say what's currently on the surface.

    `provenance` is optional and purely descriptive - the geometry engine itself never reads it
    back for anything - set by whichever `Geometry.deposit_*` call created this layer, when the
    caller (typically `structureforge.process.simulate`) passed one in.
    """

    material: str
    polygon: BaseGeometry
    provenance: LayerProvenance | None = None

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
        return _merge_touching(_clean(unary_union(polys)))

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
        provenance: LayerProvenance | None = None,
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

        `provenance`, if given, is attached to the resulting `Layer` as-is (see
        `LayerProvenance`) — purely descriptive, this method never reads it back.
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

        # SAG: never bulk-pad the seed — the floor extension would span the full domain width
        # and cause growth to appear everywhere.  Only pad for non-selective blanket growth.
        padded = growth_base if seed_materials else self._pad(growth_base)

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
            self.layers.append(Layer(material=material, polygon=film, provenance=provenance))

    def deposit_faceted(
        self,
        material: str,
        thickness_nm: float,
        rate_c: float = 1.0,
        rate_m: float = 0.3,
        rate_sp: float = 0.6,
        semi_polar_angle_deg: float = 30.0,
        seed_materials: list[str] | None = None,
        material_c: str | None = None,
        material_m: str | None = None,
        material_sp: str | None = None,
        provenance: LayerProvenance | None = None,
    ) -> None:
        """Faceted growth: add `material` by advancing three crystal-plane families - c-plane
        {0001} (`rate_c`), m-plane {10-10} sidewalls (`rate_m`), and semi-polar facets at
        `semi_polar_angle_deg` from the c-axis (`rate_sp`) - each strictly along its own outward
        normal, independently of the other two.

        Concretely: every straight edge of the current exposed surface whose outward normal is
        exactly (0,1) [c], (±1,0) [m] or (±sin θ,cos θ) [sp] advances by that facet's own
        rate * thickness_nm; every other edge - including a facet whose own rate is 0 - doesn't
        move at all. Where an existing corner's normal cone spans one of these three directions
        without already having an edge there (e.g. a bare 90 degree corner the first time it
        grows), a brand new facet is inserted exactly at that angle instead of being interpolated
        from its neighbours.

        Setting a rate to 0 therefore *pins* that facet's own line - it never rises, widens, or
        otherwise moves, no matter what the other two rates are doing. Its exposed *extent* can
        still change, though: pure `rate_sp` growth on an already-formed sharp point (rate_c =
        rate_m = 0, no c-plane or sidewall left to pin) extends that point further along the SP
        direction with the sidewalls exactly where they were - the common case once a tip has
        actually formed. Applied instead to a seed that *still has* a flat c-plane top, the pinned
        c-plane and m-plane lines bound how far the flanking SP facets can advance into the
        existing chamfer; since neither line can move to meet them, growing SP alone there erodes
        the chamfer back down (the corners it occupied revert to the pinned lines, so the c-plane's
        exposed width actually grows) rather than sharpening the tip - closing a flat top into a
        point takes the flanking rate (m and/or c) that made it narrower in the first place, not a
        positive rate_sp on its own.

        Repeatedly applying thin layers (small `thickness_nm`) produces conformal MQW stacks that
        faithfully follow the pencil/pyramid shape as it develops.

        `material_c`/`material_m`/`material_sp` let each facet family incorporate a *different*
        material - typically the same alloy at a different composition, e.g. more indium on the
        c-plane than on the semi-polar facets (`material_c=indium_gan(0.30).name,
        material_sp=indium_gan(0.10).name`), matching the well-known facet-dependent indium
        incorporation of real InGaN growth. Any of the three left as None falls back to
        `material`. When they all resolve to the same name (the default - none given) this adds
        exactly one `Layer`, as before; otherwise it adds one `Layer` per distinct material,
        each holding only the area that actually grew from that family's own facets (a corner
        where a new facet nucleates between two different families is split at the nucleation
        point, not blended - there is no in-between composition at a sub-nm sharp edge).

        `seed_materials` enables SAG selectivity (same semantics as `deposit_epitaxial`).

        `provenance`, if given, is attached as-is to every `Layer` this call creates (see
        `LayerProvenance`) - including each per-family split, when `material_c`/`material_m`/
        `material_sp` produce more than one. This method never reads it back.
        """
        if thickness_nm <= 0 or not self.layers:
            return
        solid = self.solid()
        if solid.is_empty:
            return

        y_min, y_max = solid.bounds[1], solid.bounds[3]
        t = thickness_nm
        theta = math.radians(semi_polar_angle_deg)

        # --- SAG selectivity -----------------------------------------------
        if seed_materials:
            seed_polys = [l.polygon for l in self.layers if l.material in seed_materials and not l.polygon.is_empty]
            if not seed_polys:
                return
            seed_union = _clean(unary_union(seed_polys))
            non_seed_polys = [l.polygon for l in self.layers if l.material not in seed_materials and not l.polygon.is_empty]
            if non_seed_polys:
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

        # SAG: never bulk-pad the seed (same reason as deposit_epitaxial).
        padded = growth_base if seed_materials else self._pad(growth_base)

        named_directions = [
            (0.0, 1.0, rate_c * t, "c"),
            (1.0, 0.0, rate_m * t, "m"),
            (-1.0, 0.0, rate_m * t, "m"),
            (math.sin(theta), math.cos(theta), rate_sp * t, "sp"),
            (-math.sin(theta), math.cos(theta), rate_sp * t, "sp"),
        ]
        pieces_by_family = _offset_named_facets(padded, named_directions)
        max_reach = t * (rate_c + rate_m + rate_sp) / max(math.cos(theta), 0.05) + 1.0

        materials_by_family = {"c": material_c or material, "m": material_m or material, "sp": material_sp or material}
        pieces_by_material: dict[str, list[BaseGeometry]] = {}
        if len(set(materials_by_family.values())) == 1:
            # No per-facet override (the common case): keep the single-Layer behaviour exactly
            # as before, rather than needlessly splitting one material into three identical Layers.
            pieces_by_material[material] = list(pieces_by_family.values())
        else:
            for family in ("c", "m", "sp", *sorted(set(pieces_by_family) - {"c", "m", "sp"})):
                if family in pieces_by_family:
                    pieces_by_material.setdefault(materials_by_family.get(family, material), []).append(
                        pieces_by_family[family]
                    )

        for layer_material, new_areas in pieces_by_material.items():
            new_area = unary_union(new_areas)
            grown = _clean(unary_union([solid, new_area]))
            film = self._crop(
                _clean(grown.difference(padded)),
                y_min - t,
                y_max + max_reach,
            )
            film = _fill_holes(_drop_tiny(_clean(film.difference(solid))))
            if not film.is_empty:
                self.layers.append(Layer(material=layer_material, polygon=film, provenance=provenance))

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

    def flip(self) -> None:
        """Turn the wafer over: mirror the whole stack vertically around the mid-height of the
        current solid, so what was the top surface becomes the new bottom (bonded face-down to
        a temporary carrier) and what was the untouched bulk floor becomes the new top, ready
        for further process steps. `floor_nm` and the overall y-span are unchanged by
        construction (the mirror axis is the solid's own mid-height) - only what used to be
        "up" is now "down".

        `self.layers` is reversed at the same time, so `layers[0]` keeps meaning "the anchor
        `remove_floating_debris` treats as attached down" - after a flip that's whatever just
        became the new bottom (bonded to the carrier), not the original substrate, which is now
        floating freely at the new top like everything else.

        Requires the current top surface to be flat across the *entire* domain width, the same
        way a real temporary bond needs a flat surface to adhere to: raises `ValueError`
        otherwise (e.g. an isolated raised feature wouldn't actually make contact with a carrier
        across the gaps beside it - there'd be nothing physically holding those regions in place
        once flipped).
        """
        solid = self.solid()
        if solid.is_empty or self.floor_nm is None:
            raise ValueError("flip: no substrate to flip")

        y_min, y_max = solid.bounds[1], solid.bounds[3]
        probe_height = min(0.5, y_max - y_min)
        if probe_height <= 0:
            raise ValueError("flip: geometry has no height to flip")
        probe = self._crop(solid, y_max - probe_height, y_max)
        expected_area = self.domain_width_nm * probe_height
        flat = not probe.is_empty and abs(probe.area - expected_area) < 0.02 * expected_area
        if not flat:
            raise ValueError(
                "flip: the front surface must be flat across the whole domain width before "
                "flipping (e.g. planarize first) - like bonding the wafer to a temporary carrier"
            )

        mirror_axis = (y_min + y_max) / 2.0
        flipped = []
        for layer in reversed(self.layers):
            polygon = layer.polygon if layer.polygon.is_empty else _clean(scale(layer.polygon, xfact=1.0, yfact=-1.0, origin=(0.0, mirror_axis)))
            flipped.append(Layer(material=layer.material, polygon=polygon, provenance=layer.provenance))
        self.layers = flipped

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
