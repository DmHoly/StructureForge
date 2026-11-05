"""The 2D cross-section geometry engine.

Layers are shapely polygons kept in construction order. Deposition/etch/planarization are all
implemented as boolean operations on that stack. See `Geometry`'s docstring for the v1
simplifications (hard-silhouette directional shadowing, domain edges treated as symmetry
boundaries, substep-based selective etch) - they're deliberate scope cuts for a first version,
not oversights.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import shapely
import shapely.errors

from pydantic import BaseModel, ConfigDict, Field
from shapely.affinity import scale, translate
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from ..core.materials import MaterialLibrary
from ..core.recipes import DepositionMode, DepositionRecipe, EtchMode, EtchRecipe
from ..core.traced import Traced

_EPS_AREA = 1e-6  # nm^2 - polygons smaller than this are numerical noise, dropped
_SEAM_EPS = 0.01  # nm - closing radius that seals seams between pieces of one film, see _close_seams
_FLAP_EPS = 1e-4  # nm - opening radius that removes zero-thickness flaps from a film, see _close_seams
_SLIVER_WIDTH = 0.025  # nm - polygons under 2x this mean thickness (2 * area / perimeter) are noise too:
                      # hairlines a near-vertical flank leaves when swept up, far below one monolayer
_MARGIN = 1.0  # nm of slack padding used around bounding boxes for directional etch
_BULK_MARGIN = 1.0e4  # nm - "effectively infinite" downward extension standing in for the wafer's bulk, see floor_nm
_SIMPLIFY_TOL = 0.02  # nm - keeps vertex count from growing unboundedly over many substeps
_TOUCH_EPS = 1e-6  # nm - closes a single-point/zero-width seam between sibling polygon parts;
                    # far below any real feature size, so it never bridges a genuine gap.
_SPURIOUS_HOLE_AREA = 10.0  # nm^2 - an interior hole smaller than this is a construction artifact, see _fill_holes
_FACET_SNAP_TOL = 1e-2  # normal-vector tolerance (~0.6 deg) for an edge to count as a named crystal facet:
                        # a c-plane top tilted by a sub-nm etch non-uniformity is still the c-plane, while
                        # genuinely distinct facets are always several degrees apart.


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


def _is_real(g: BaseGeometry) -> bool:
    """A polygon worth keeping: not numerical noise, either in area or as a zero-width sliver left
    along a shared boundary by two boolean ops that disagree in the last few bits."""
    return isinstance(g, Polygon) and g.area > _EPS_AREA and g.area > _SLIVER_WIDTH * g.length


def _close_seams(geom: BaseGeometry, solid: BaseGeometry | None = None) -> BaseGeometry:
    """Morphological closing then opening by `_SEAM_EPS`, for a film assembled from many pieces
    (strips, fans, substeps) and differenced against the solid it grew on. Where those boundaries
    disagree in the last few bits, the film otherwise keeps zero-width slits inside one polygon,
    parts a hair apart, or zero-thickness flaps running along an interface - all drawn as stray
    lines across the layer. Closing by `_SEAM_EPS` seals the first two, and a much finer opening
    by `_FLAP_EPS` removes the third - fine enough to keep a thin but real wedge (e.g. the one
    holding an overgrowth front down on a mask). Mitre joins leave real corners where they are. `solid`, if given, is subtracted again afterwards,
    since closing can reach a hair into a concave corner of it.

    GEOS (3.13) can silently drop whole parts when it mitre-dilates a `MultiPolygon` of many
    pieces with zero-width tips - a shaded faceted film over a block of wires comes as hundreds
    of chevrons, and came back with 4 wires out of 15. A dilation never loses area, so one that
    does is redone part by part, which keeps them all."""
    if geom.is_empty:
        return geom
    mitre = {"join_style": "mitre"}
    with np.errstate(divide="ignore", invalid="ignore"):  # GEOS mitre on a degenerate segment - harmless
        closed = geom.buffer(_SEAM_EPS, **mitre)
        if closed.area < geom.area and hasattr(geom, "geoms"):
            closed = shapely.union_all([g.buffer(_SEAM_EPS, **mitre) for g in geom.geoms])
        geom = closed.buffer(-_SEAM_EPS - _FLAP_EPS, **mitre).buffer(_FLAP_EPS, **mitre)
    if solid is not None and not geom.is_empty:
        geom = geom.difference(solid)
    return _clean(geom)


def _drop_tiny(geom: BaseGeometry) -> BaseGeometry:
    if geom.is_empty:
        return geom
    if hasattr(geom, "geoms"):
        # A MultiPolygon - or a GeometryCollection, when a boolean op leaves stray lines/points
        # (an edge merely touching the crop box) next to the real area: only polygons are kept.
        flat = [p for g in geom.geoms for p in (g.geoms if isinstance(g, MultiPolygon) else [g])]
        kept = [g for g in flat if _is_real(g)]
        if not kept:
            return Polygon()
        return kept[0] if len(kept) == 1 else MultiPolygon(kept)
    return geom if _is_real(geom) else Polygon()


def _fill_holes(geom: BaseGeometry, max_area: float = _SPURIOUS_HOLE_AREA, occupied: BaseGeometry | None = None) -> BaseGeometry:
    """Drop every interior ring of `geom` smaller than `max_area`.

    `_offset_named_facets` mitres/bevels each vertex of a facet chain independently: when the
    same emergent facet must nucleate from two separate convex corners flanking a short, already-
    frozen bevel edge (left over from an earlier lopsided-rate step - see `mitre_or_bevel`), each
    corner's fan is pivoted from its own vertex with no knowledge of the other, and the two fans
    can fail to reach each other, leaving a sliver of the tip ungrown - a small, real, spurious
    interior hole, not a physical void. The same independent-fan construction can leave this seam
    either *within* one `deposit_faceted` call's own film (this function is applied there
    directly) or *between* two already-clean layers once every layer is unioned together (see
    `Geometry.solid()`, which calls this too, guarded by `_has_interior`, whenever the merged
    result actually has holes) - both are the same artifact, just caught at a different point in
    the construction.

    A hole of real size is a different thing: a cavity the growth has enclosed (a crystal
    overflowing its mask and sealing the undercut beneath it, two fronts coalescing over a
    trench). That one is kept - as air the growth goes on filling from its walls (see
    `_offset_named_facets`), that an etch or a conformal deposit can't reach (see `_outside_air`).

    `occupied`, if given, keeps a small hole it mostly fills: a speck of another layer the film
    wraps round, not air.
    """
    if geom.is_empty:
        return geom

    def held(ring) -> bool:
        hole = Polygon(ring)
        return occupied is not None and hole.intersection(occupied).area > hole.area / 2

    def keep(poly: Polygon) -> Polygon:
        holes = [ring for ring in poly.interiors if Polygon(ring).area >= max_area or held(ring)]
        return Polygon(poly.exterior, holes) if holes else Polygon(poly.exterior)

    if isinstance(geom, MultiPolygon):
        return MultiPolygon([keep(g) for g in geom.geoms if not g.is_empty])
    return keep(geom)


def _has_interior(geom: BaseGeometry) -> bool:
    if isinstance(geom, MultiPolygon):
        return any(len(p.interiors) > 0 for p in geom.geoms)
    if isinstance(geom, Polygon):
        return len(geom.interiors) > 0
    return False


def _polygonal(geom: BaseGeometry) -> BaseGeometry:
    """Only the polygon parts of `geom`: a clip can leave stray lines/points (an edge lying
    exactly on the clip rectangle) next to the real area."""
    if isinstance(geom, (Polygon, MultiPolygon)) or geom.is_empty:
        return geom
    polys = [p for g in geom.geoms for p in (g.geoms if isinstance(g, MultiPolygon) else [g]) if isinstance(p, Polygon) and not p.is_empty]
    if not polys:
        return Polygon()
    return polys[0] if len(polys) == 1 else MultiPolygon(polys)


def _mitre_tips(eroded: BaseGeometry, d: float) -> np.ndarray:
    """Where a mitre dilation of `eroded` by `d` can poke past the shape it was eroded from: at
    every convex corner, the region between the two offset lines from the corner to their
    mitre tip (or, past GEOS's default mitre limit of 5 offsets, to where the bevel cuts it),
    as an array of polygons. The corners the erosion sharpened or created (the tip of a
    tapering part) are where the dilation overshoots the original outline; a corner of the
    original is restored exactly, and its quad just holds nothing. Corners turning by less
    than `_MITRE_TIP_MIN_TURN` - the vertices of a buffer's arcs, mostly - have tips a few
    percent of `d` past the offset edges, and are left out."""
    quads = []
    for part in (eroded.geoms if isinstance(eroded, MultiPolygon) else [eroded]):
        if part.is_empty:
            continue
        part = orient(part, sign=1.0)  # exterior counter-clockwise, holes clockwise
        for ring, sign in [(part.exterior, 1.0), *((hole, -1.0) for hole in part.interiors)]:
            c = np.asarray(ring.coords)[:-1]
            if len(c) < 3:
                continue
            a = c - np.roll(c, 1, axis=0)  # incoming edge
            b = np.roll(c, -1, axis=0) - c  # outgoing edge
            la, lb = np.hypot(a[:, 0], a[:, 1]), np.hypot(b[:, 0], b[:, 1])
            ok = (la > 1e-12) & (lb > 1e-12)
            a, b = a / np.maximum(la, 1e-300)[:, None], b / np.maximum(lb, 1e-300)[:, None]
            cross = (a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) * sign
            dot = a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]
            turn = np.arctan2(np.abs(cross), dot)  # exterior angle at the corner
            convex = ok & (cross > 0) & (turn >= _MITRE_TIP_MIN_TURN)
            if not convex.any():
                continue
            # right-hand normals point out of the solid for both ring orientations
            n1 = np.column_stack([a[:, 1], -a[:, 0]])[convex]
            n2 = np.column_stack([b[:, 1], -b[:, 0]])[convex]
            v, a, b = c[convex], a[convex], b[convex]
            # The two offset lines meet d * tan(turn / 2) past their feet; GEOS bevels a mitre
            # longer than 5 offsets, which that long a run along the offset lines still covers
            # (a point 6 d from a foot is at least 5 d from the corner).
            run = np.minimum(d * np.tan(turn[convex] / 2), 6.0 * d)[:, None]
            foot1, foot2 = v + n1 * d, v + n2 * d
            quads.append(np.stack([v, foot1, foot1 + a * run, foot2 - b * run, foot2, v], axis=1))
    if not quads:
        return np.array([], dtype=object)
    return shapely.polygons(np.concatenate(quads))


def _speck_tol(r: float) -> float:
    """The thin-part area (see `_thin_area`) below which tunnelling through parts thinner than `r`
    is ignored: about a speck of `r` across, the air could only tunnel into by about its own
    size - but never more than `_THIN_SPECK_NM2`, since a single substep as deep as the whole
    etch (see `_etch_steps`) would otherwise take a whole mask for a speck."""
    return min(r * r, _THIN_SPECK_NM2)


def _thin_area(blocked: BaseGeometry, r: float, window: tuple[float, float] | None = None) -> float:
    """The area of the parts of `blocked` thinner than `r` (nm^2): what a morphological opening
    by `r / 2` (mitre joins, so real corners survive it) fails to restore. An etch substep's air,
    dilated by its reach, tunnels through such a part of a slower-etching layer as if it weren't
    there - see `Geometry._reach_air`, the one caller.

    Measured without the overlay `blocked - opened` would cost on two polygons sharing most of
    their boundary: `blocked.area - opened.area`, plus the area of the opening outside `blocked`
    - which only the mitre tips of the dilation produce, so only their quads (`_mitre_tips`)
    that `blocked` doesn't cover are overlaid, a small geometry.

    `window`, if given, is the x-range the measurement is restricted to: the parts of a
    band-cropped mirror image near the crop cut are not real geometry. The restriction is a
    plain rectangle clip of the measured shapes (area needs no valid topology), never an input
    to the overlay.
    """
    if blocked.is_empty or r <= 0:
        return 0.0
    mitre = {"join_style": "mitre"}
    with np.errstate(divide="ignore", invalid="ignore"):  # GEOS mitre on a degenerate segment - harmless
        eroded = blocked.buffer(-r / 2, **mitre)
        opened = eroded.buffer(r / 2, **mitre) if not eroded.is_empty else eroded
    y0, y1 = blocked.bounds[1] - 1.0, blocked.bounds[3] + 1.0

    def measured(geom: BaseGeometry) -> BaseGeometry:
        if window is None or geom.is_empty:
            return geom
        return shapely.clip_by_rect(geom, window[0], y0, window[1], y1)

    blocked_measured = measured(blocked)
    if blocked_measured.is_empty:
        return 0.0
    if opened.is_empty:
        return blocked_measured.area
    area = blocked_measured.area - measured(opened).area
    tips = _mitre_tips(eroded, r / 2)
    if len(tips):
        # Only a tip region not entirely inside `blocked` can hold overshoot: a predicate per
        # quad against the prepared polygon (dilated by a hair, since a restored corner's tip
        # lies exactly on the outline), and an overlay only of the (usually few) that fail.
        within = blocked.buffer(_TOUCH_EPS, **mitre)
        shapely.prepare(within)
        poking = tips[~shapely.covers(within, tips)]
        if len(poking):
            # An exact mitre makes the two run ends coincide (to the last bit): buffer(0) turns
            # such a degenerate quad into a clean triangle before the union.
            try:
                hood = _robust_union(list(shapely.buffer(poking, 0)))
                # `_polygonal`: a quad edge lying along the opening leaves a line in the result
                under = _polygonal(opened.intersection(hood))
                overshoot = shapely.difference(under, blocked, grid_size=_COVER_GRID) if not under.is_empty else under
            except shapely.errors.GEOSException:
                overshoot = opened.buffer(0).difference(blocked.buffer(0))
            area += measured(overshoot).area
    return max(0.0, area)

_ALLOY_NAME = re.compile(r"^(In|Al)\d+(?:\.\d+)?Ga\d+(?:\.\d+)?N$")
_SEED_PROBE_DEPTH = 0.25  # nm - how far inside an exposed edge to look for the material it belongs to:
                          # deep enough to see past an etch's sub-angstrom residue film, far thinner
                          # than any real layer.
_RESIDUE_NM = 0.1  # nm - etched layers thinner than twice this are leftover residue, see Geometry.etch
_FRONT_SMOOTH = 0.2  # nm - bumps below this are trimmed off a growth front, see Geometry._deposit_faceted_once
_MIN_NUCLEATION_EDGE = 1.0  # nm - a shorter edge between two parallel ones is a step, not a facet, see _offset_named_facets
_MIN_SP_INV_FRACTION = 0.02  # slowest nonzero rate_sp_inv, relative to the fastest rate, see Geometry.deposit_faceted
_VICINAL_COS = math.cos(math.radians(10.0))
_VICINAL_SIN = math.sin(math.radians(10.0))  # an edge within 10 deg of a facet grows like it, see _offset_named_facets
_SHADED_VICINAL_COS = math.cos(math.radians(30.0))  # ... within 30 deg of the m-plane when the sidewalls are shaded
_UNION_GRID = 1e-4  # nm - snap grid for _robust_union's fallback
_UNION_COVER_TOL = 1e-3  # nm^2 - how much of an input a union may leave out before it's redone, see _covering_union
_COVER_GRID = 1e-6  # nm - snap grid that measures it
_SPLIT_TOL = 0.05  # nm - a layer vertex this close to a solid edge splits that edge (covers _SIMPLIFY_TOL drift)
_EDGE_SNAP = 1e-3  # nm - a vertex this close to a domain edge is put exactly on it, see Geometry._snap_edges:
                   # boolean ops leave them 1e-5..1e-4 nm off, and no real feature is a picometre wide
_GEODESIC_STEP = 0.25  # nm - an etch substep's reach is dilated in steps this long, see Geometry._reach_air
_ROUGH_CONTACT_TOL = 0.01  # nm - a rough film's face closer than this to the solid beneath rests on it (no grains there)
_ROUGH_MAX_GRAINS = 20000  # past this many grains along a rough film, they're spread out rather than drawn finer
_MASK_FACTOR_RATIO = 0.5  # a layer etched below this share of a factor's rate can't be tunnelled through by it, see Geometry.etch
_REACH_EPS = 0.03  # nm - the part of an etch substep's reach grown after cleaning, see Geometry._reach_air
_ETCH_SMOOTH_MAX = 0.25  # nm - cap on how far an etched layer's exposed edge is straightened, see Geometry._smooth_exposed
_DIRECTIONAL_COUPLING_NM = 0.5  # nm - how late a buried slower layer may start receding under a vertical beam, see Geometry._etch_steps
_PREVIEW_SUBSTEP_DIVISOR = 4  # an etch previewed runs this many times fewer substeps, see Geometry.etch
_GEODESIC_SIMPLIFY = 0.002  # nm - vertex-hygiene simplification between two geodesic steps, see Geometry._reach_air
_THIN_SPECK_NM2 = 16.0  # nm^2 - the most a thin part may total and still be a speck the air may tunnel into, see _speck_tol
_MITRE_TIP_MIN_TURN = math.radians(15.0)  # a corner turning less is too flat for its mitre tip to matter, see _mitre_tips


def seed_matches(material: str, seed_materials: list[str]) -> bool:
    """Whether a layer of `material` counts as one of `seed_materials`: an exact name, or the alloy
    family of a composition-named ternary nitride - `"InGaN"` matches `In0.10Ga0.90N`, `"AlGaN"`
    matches `Al0.20Ga0.80N` (see `structureforge.core.materials.indium_gan`), so a seed list
    doesn't have to spell out every composition a stack happens to use.
    """
    if material in seed_materials:
        return True
    m = _ALLOY_NAME.match(material)
    return bool(m) and f"{m.group(1)}GaN" in seed_materials


def _robust_union(geoms: list[BaseGeometry]) -> BaseGeometry:
    """`unary_union`, falling back to a snap-rounded union when GEOS's floating-point noding fails
    on many tiny, nearly coincident pieces (a "found two shells" / TopologyException)."""
    try:
        return unary_union(geoms)
    except shapely.errors.GEOSException:
        return shapely.union_all([g for g in geoms if not g.is_empty], grid_size=_UNION_GRID)


def _covering_union(geoms: list[BaseGeometry]) -> BaseGeometry:
    """`_robust_union`, checked: on nearly coincident edges, GEOS's floating-point overlay can
    take two pieces that only touch for overlapping ones and leave one of them out of the result,
    without raising - a whole substep's film missing from a growth, i.e. a slit through the
    crystal. A union covers each of its inputs; one that doesn't is redone snap-rounded, which
    never does that. Measured snap-rounded too: the floating-point overlay that would check it
    fails the same way (and `covers` is thrown by every last-bit disagreement).

    The pieces a growth merges don't overlap (each film is cut out of the solid before it), so
    their union has the sum of their areas: only a union short of it is checked - the check costs
    as much as the union itself."""
    geoms = [g for g in geoms if not g.is_empty]
    union = _robust_union(geoms)
    if (
        geoms
        and float(shapely.area(geoms).sum()) - union.area > _UNION_COVER_TOL
        and (shapely.area(shapely.difference(geoms, union, grid_size=_COVER_GRID)) > _UNION_COVER_TOL).any()
    ):
        union = shapely.union_all(geoms, grid_size=_UNION_GRID)
    return union


def sweep_union(geom: BaseGeometry, vector: tuple[float, float]) -> BaseGeometry:
    """The exact Minkowski sum of `geom` with the segment from (0,0) to `vector` - this is what
    "directional" deposition/etch actually mean: a uniform offset in one direction, rather than
    every direction like `.buffer()`.

    Computed directly rather than by sampling many intermediate translations: the union of `geom`
    and, for every edge of every ring of every part of `geom` whose outward normal has a positive
    component along `vector`, the parallelogram that edge sweeps out along `vector`, is exactly
    the swept region. A point of the sweep that isn't in `geom` is reached from `geom` along
    `vector`, so the straight path to it leaves `geom` somewhere - through an edge facing along
    `vector` (at a vertex, one of the two edges meeting there does) - and lies in that edge's
    parallelogram; edges facing away from, or parallel to, `vector` add nothing but cost: their
    (possibly degenerate, zero-area) quads are left out. Besides being exact rather than an
    approximation, this sidesteps a real GEOS robustness trap the earlier sampled version had:
    many translated copies at regular, exactly-aligned offsets is a textbook way to trigger a
    spurious "side location conflict" TopologyException in `unary_union` (axis-parallel edges
    landing exactly on top of each other at various samples) - a handful of edge-quads doesn't
    have that failure mode.
    """
    if geom.is_empty or (vector[0] == 0 and vector[1] == 0):
        return geom
    vx, vy = vector
    quads = []
    for part in (geom.geoms if isinstance(geom, MultiPolygon) else [geom]):
        if part.is_empty:
            continue
        # `orient` makes the exterior counter-clockwise and the holes clockwise, so the right-hand
        # normal (dy, -dx) of every edge points out of the solid for both.
        part = orient(part, sign=1.0)
        for ring in [part.exterior, *part.interiors]:
            c = np.asarray(ring.coords)
            p1, p2 = c[:-1], c[1:]
            d = p2 - p1
            facing = d[:, 1] * vx - d[:, 0] * vy > 1e-12
            if not facing.any():
                continue
            a, b = p1[facing], p2[facing]
            quads.append(np.stack([a, b, b + vector, a + vector, a], axis=1))
    if not quads:
        return geom
    pieces = [geom, *shapely.polygons(np.concatenate(quads))]
    return _clean(unary_union(pieces))


def _reach_factor(y: float, top: float, reach: float) -> float:
    """How much of its advance an m-plane keeps at height `y` when the growth reaches `reach`
    below `top` (see `Geometry.deposit_faceted`): all of it at the top, none that deep or deeper,
    linearly in between."""
    return min(1.0, max(0.0, 1.0 - (top - y) / reach))


def _split_at_level(coords: list[tuple[float, float]], level: float) -> list[tuple[float, float]]:
    """A closed ring's vertices with one more wherever an edge crosses height `level`."""
    out: list[tuple[float, float]] = []
    for i, (x1, y1) in enumerate(coords):
        x2, y2 = coords[(i + 1) % len(coords)]
        out.append((x1, y1))
        if (y1 - level) * (y2 - level) < 0:
            t = (level - y1) / (y2 - y1)
            out.append((x1 + t * (x2 - x1), level))
    return out


def _offset_named_facets(
    solid: BaseGeometry,
    named_directions: list[tuple[float, float, float, str]],
    edge_active: Callable[[tuple[float, float], tuple[float, float], tuple[float, float]], bool] | None = None,
    x_window: tuple[float, float] | None = None,
    lateral_reach: tuple[float, float] | None = None,
) -> dict[str, BaseGeometry]:
    """Advance each straight edge of `solid`'s rings (exterior and, for an enclosed void, interior -
    see `_fill_holes`) along its own outward normal by
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
    short chain of mitre points) are intersected pairwise. The grown area is the union of two kinds
    of convex piece: for every edge with a nonzero distance, the plain rectangle it sweeps along its
    normal; and at every corner, the fan from the vertex through its chain of mitre points (at a
    concave corner, only when the mitre lands close by - otherwise the two rectangles already
    meet). Every piece is clipped by the offset lines of the facets around it - its own edges, the
    facets nucleating at its corners, and up to two more edges each way across convex corners -
    so pieces built independently at neighbouring corners can't overshoot one another: a c-plane
    top consumed by its flanking SP facets closes to a single apex, and a nucleating SP facet
    trims the neighbouring c-plane strip exactly instead of leaving a sliver of it behind. Only
    true facets bound their neighbours: an off-axis, non-moving edge (a curved mask wall) does not.
    An edge within 10 degrees of a facet (a vicinal surface) grows like that facet, and bounds its
    neighbours with that facet's exact line.

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

    `edge_active(p1, p2, normal)`, if given, restricts growth to the edges it accepts (SAG: only
    edges made of a seed material) - every other edge is treated as rate 0 whatever its direction,
    and a new facet only nucleates at a convex corner whose two edges are both active.

    Overhangs: a named direction labelled "sp_inv" is an inverted semi-polar facet, facing down
    and out ({10-1-1}, normal (±sin θ, -cos θ)). Where a growing edge facing up or out (c, m or
    SP) sits right on top of a sidewall that doesn't move (a rate-0 or non-seed m-plane), that
    sidewall stops bounding it: the inverted facet nucleates at their corner, under the growing
    edge, and the crystal spreads past the sidewall's line with that facet as its underside. Its
    foot - where it meets the sidewall - then slides down that sidewall as it advances, at
    d / cos θ (mitred exactly: a steep inverted facet's foot outruns `mitre_or_bevel`'s limit),
    the way a shell nucleated on a nanowire's tip spreads down its bare sidewalls. Without an
    "sp_inv" direction nothing here changes.

    `x_window`, if given, skips every corner and edge lying entirely outside that x-range (the
    far parts of a mirror-padded solid, whose growth would be cropped away anyway).

    `lateral_reach` = (top, reach), if given, shades the m-planes (label "m", vicinal ones too):
    at height y they advance by their distance times `_reach_factor` - all of it at `top`, none
    from `reach` below it down. An edge crossing that depth is split there; each edge then sweeps
    a trapezoid instead of a rectangle, and bounds its neighbours with its tilted offset line. One
    shaded nowhere is pinned, like a rate-0 facet.
    """
    facet_normals = [(nx, ny) for nx, ny, _d, _label in named_directions]
    named_directions = [(nx, ny, d, label) for nx, ny, d, label in named_directions if d > 1e-9]
    named = [(nx, ny, d, label, math.atan2(ny, nx)) for nx, ny, d, label in named_directions]

    def outside(*xs: float) -> bool:
        return x_window is not None and (all(x < x_window[0] for x in xs) or all(x > x_window[1] for x in xs))

    def is_facet(nx: float, ny: float) -> bool:
        return any(abs(nx - fx) < _FACET_SNAP_TOL and abs(ny - fy) < _FACET_SNAP_TOL for fx, fy in facet_normals)

    def classify(nx: float, ny: float) -> tuple[float, str, tuple[float, float]]:
        """(distance, family label, the exact facet normal) for an edge normal - `(0, "", the
        normal itself)` when it isn't a facet."""
        for dx, dy, d, label, _ang in named:
            if abs(nx - dx) < _FACET_SNAP_TOL and abs(ny - dy) < _FACET_SNAP_TOL:
                return d, label, (dx, dy)
        # A vicinal surface - a few degrees off a facet, e.g. the floor of a sub-nm etch dip -
        # grows by step flow at that facet's rate rather than staying pinned: left pinned, it would
        # persist as a crevice carried up through every later layer.
        fx, fy, d, label = max(((dx, dy, d, label) for dx, dy, d, label, _ang in named), key=lambda e: e[0] * nx + e[1] * ny, default=(0.0, 0.0, 0.0, ""))
        # A shaded flank leans out as it thickens towards the top (see `lateral_reach`): still the
        # m-plane, roughened by the shade, further off it than 10 degrees on an already flared core.
        vicinal_cos = _SHADED_VICINAL_COS if lateral_reach is not None and label == "m" else _VICINAL_COS
        if fx * nx + fy * ny > vicinal_cos and not any(
            abs(nx - px) < _FACET_SNAP_TOL and abs(ny - py) < _FACET_SNAP_TOL for px, py in facet_normals
        ):
            return d, label, (fx, fy)
        return 0.0, "", (nx, ny)

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
            # Collinear edges advancing by different amounts (an active edge next to an inactive
            # one): keep both offsets, so the step between them is a straight cut at `v`.
            return [(p1, label2)] if abs(d1 - d2) < 1e-12 else [(p1, label1), (p2, label2)]
        px = (d1 * d - b * d2) / det
        py = (a * d2 - c * d1) / det
        point = (v[0] + px, v[1] + py)
        if math.hypot(point[0] - v[0], point[1] - v[1]) > 1.5 * (d1 + d2):
            return [(p1, label1), (p2, label2)]
        return [(point, label2)]

    def exact_mitre(
        v: tuple[float, float], n1: tuple[float, float], d1: float, n2: tuple[float, float], d2: float, label2: str
    ) -> tuple[tuple[float, float], str]:
        """Where offset lines n1@d1 and n2@d2 (both measured from `v`) cross, however far out."""
        (a, b), (c, d) = n1, n2
        det = a * d - b * c
        return (v[0] + (d1 * d - b * d2) / det, v[1] + (a * d2 - c * d1) / det), label2

    def between_ccw(angle: float, lo: float, hi: float) -> bool:
        # Same tolerance as `classify`: an edge whose normal is only a hair (float noise, e.g.
        # ~1e-9 rad left by `_merge_touching`'s tiny buffer) off a named direction is already
        # that facet, so the direction must not be inserted again as an "extra" - mitring a line
        # against a near-parallel copy of itself yields an arbitrary point (a spurious bump that
        # pokes outside the neighbouring facet's front, breaking the corner's symmetry).
        span = (hi - lo) % (2 * math.pi)
        rel = (angle - lo) % (2 * math.pi)
        return _FACET_SNAP_TOL < rel < span - _FACET_SNAP_TOL

    pieces_by_family: dict[str, list[BaseGeometry]] = {}

    def add_piece(label: str, geom: BaseGeometry) -> None:
        pieces_by_family.setdefault(label, []).append(geom)

    parts = list(solid.geoms) if isinstance(solid, MultiPolygon) else [solid]
    # An interior ring is an enclosed void (see `_fill_holes`): its walls grow exactly like the
    # outer surface. `orient` makes the exterior counter-clockwise and the holes clockwise, so
    # the same right-hand normal points out of the solid - into the void - for both.
    rings = [ring for part in parts for ring in (lambda q: [q.exterior, *q.interiors])(orient(part, sign=1.0))]
    for ring in rings:
        coords = list(ring.coords)[:-1]
        coords = [p for i, p in enumerate(coords) if p != coords[i - 1]]
        if lateral_reach is not None:
            coords = _split_at_level(coords, lateral_reach[0] - lateral_reach[1])
        n = len(coords)
        if n < 3:
            continue

        normals: list[tuple[float, float]] = []
        facet_dirs: list[tuple[float, float]] = []
        lengths: list[float] = []
        dists: list[float] = []
        labels: list[str] = []
        active: list[bool] = []
        for i in range(n):
            x1, y1 = coords[i]
            x2, y2 = coords[(i + 1) % n]
            length = math.hypot(x2 - x1, y2 - y1)
            nx, ny = (y2 - y1) / length, -(x2 - x1) / length
            normals.append((nx, ny))
            lengths.append(length)
            d, label, facet_dir = classify(nx, ny)
            is_active = not outside(x1, x2) and (edge_active is None or edge_active((x1, y1), (x2, y2), (nx, ny)))
            if not is_active:
                d, label = 0.0, ""
            active.append(is_active)
            dists.append(d)
            labels.append(label)
            facet_dirs.append(facet_dir)

        # How far each edge advances at its start and its end - the same unless shaded.
        d_from, d_to = list(dists), list(dists)
        shaded = [False] * n
        if lateral_reach is not None:
            for i in range(n):
                if labels[i] != "m":
                    continue
                d_from[i] = dists[i] * _reach_factor(coords[i][1], *lateral_reach)
                d_to[i] = dists[i] * _reach_factor(coords[(i + 1) % n][1], *lateral_reach)
                dists[i] = max(d_from[i], d_to[i])
                if dists[i] <= 1e-12:
                    # deep in the shade: pinned, like a rate-0 facet - but only by the shade, so not
                    # a bound for the rest of its own plane, higher up
                    dists[i], d_from[i], d_to[i], labels[i], shaded[i] = 0.0, 0.0, 0.0, "", True

        # A sub-nm step - a riser between two parallel terraces, possibly made of a few sub-nm
        # edges (an etch leaves the floor of an opening in terraces a few angstroms apart) - is
        # not a facet: it rises with its terraces (translated along their normal by their
        # distance, under their label), nucleates nothing and bounds nothing. Treated as a facet
        # of whatever orientation it happens to have, with a rate just under its critical value it
        # would widen at every step into a pit carried up through the whole crystal. A short edge
        # that isn't such a riser (a crystal just emerging above its mask) is left as it is.
        def terrace(i: int, step: int) -> int | None:
            for k in range(1, 5):
                j = (i + step * k) % n
                if lengths[j] >= _MIN_NUCLEATION_EDGE:
                    return j
            return None

        riser = [False] * n
        for i in range(n):
            if lengths[i] >= _MIN_NUCLEATION_EDGE or not active[i]:
                continue
            before, after = terrace(i, -1), terrace(i, 1)
            if before is None or after is None:
                continue
            (bx, by), (ax, ay) = normals[before], normals[after]
            if bx * ax + by * ay > _VICINAL_COS and active[before]:
                riser[i] = True
                nx, ny = normals[i]
                dists[i] = d_from[i] = d_to[i] = d_to[before] * max(0.0, nx * bx + ny * by)
                labels[i] = labels[before]
                facet_dirs[i] = facet_dirs[before]

        # Each edge line's offset half-plane {p : n.p <= n.v + d}: the bound every piece near that
        # edge must respect - including a rate-0 (pinned) line, which never moves. A vicinal edge
        # bounds with its facet's *exact* line, through its outermost end: its own slightly tilted
        # line, extended across a wide neighbour, would shave a wedge off that neighbour's front
        # (a 0.6 deg ramp at one end of an opening's floor cutting 2 nm off the far end of the
        # c-plane strip), while the exact line through the outer end lies outside everything the
        # edge itself grows.
        offsets: list[tuple[float, float, float]] = []
        for i in range(n):
            nx, ny = normals[i]
            (x1, y1), (x2, y2) = coords[i], coords[(i + 1) % n]
            if d_from[i] != d_to[i]:
                # a shaded edge: its offset line tilts, from one end's advance to the other's
                q1 = (x1 + nx * d_from[i], y1 + ny * d_from[i])
                q2 = (x2 + nx * d_to[i], y2 + ny * d_to[i])
                length = math.hypot(q2[0] - q1[0], q2[1] - q1[1])
                tx, ty = (q2[1] - q1[1]) / length, -(q2[0] - q1[0]) / length
                offsets.append((tx, ty, tx * q1[0] + ty * q1[1]))
            elif labels[i] and not is_facet(nx, ny):
                fx, fy = facet_dirs[i]
                offsets.append((fx, fy, max(fx * x1 + fy * y1, fx * x2 + fy * y2) + dists[i]))
            else:
                offsets.append((nx, ny, nx * x1 + ny * y1 + dists[i]))
        convex_at: list[bool] = []
        # Convex, or concave by only a few degrees (an angstrom-deep dip in a facet): a bound found
        # across such a corner still belongs to the same run of facets.
        passable: list[bool] = []
        for i in range(n):
            n_in, n_out = normals[i - 1], normals[i]
            turn = (-n_in[1]) * n_out[0] - n_in[0] * (-n_out[1])
            convex_at.append(turn > 1e-9)
            passable.append(turn > 1e-9 or (turn > -_VICINAL_SIN and n_in[0] * n_out[0] + n_in[1] * n_out[1] > 0))
        # ... and either corner of a riser, which is part of its terraces' facet, not a break in it.
        passable = [passable[i] or riser[i - 1] or riser[i] for i in range(n)]

        # Overhang corners (see the docstring): "start" where a growing edge sits on top of a static
        # sidewall - the inverted facet nucleates there - and "foot" where an inverted facet already
        # meets one. The sidewall is on the left of the crystal (normal (-1, 0), leaving the corner
        # downwards) or on its right (normal (1, 0), arriving at it from below); `grower` is the
        # other edge. Neither corner lets a bound through: the sidewall's line must not stop the
        # crystal spreading past it.
        inverted = {("left" if nx < 0 else "right"): (nx, ny, d, label, ang) for nx, ny, d, label, ang in named if label == "sp_inv"}
        overhang: list[str | None] = [None] * n
        overhang_side: list[str] = [""] * n
        grower = [0] * n
        for i in range(n):
            for side, edge, wall, sx in (("left", (i - 1) % n, i, -1.0), ("right", i, (i - 1) % n, 1.0)):
                wx, wy = normals[wall]
                if side not in inverted or dists[wall] > 0 or riser[wall] or abs(wx - sx) > _FACET_SNAP_TOL or abs(wy) > _FACET_SNAP_TOL:
                    continue
                if not active[edge] or dists[edge] <= 0 or riser[edge]:
                    continue
                (ex, ey), (ix, iy) = normals[edge], inverted[side][:2]
                if abs(ex - ix) < _FACET_SNAP_TOL and abs(ey - iy) < _FACET_SNAP_TOL:
                    overhang[i] = "foot"
                elif ex * sx >= -1e-9 and ey >= -1e-9:
                    overhang[i] = "start"
                else:
                    continue
                overhang_side[i], grower[i], passable[i] = side, edge, False
                break

        chains: list[list[tuple[float, float, float, str]]] = []
        for i in range(n):
            n_in, n_out = normals[i - 1], normals[i]
            chain = [(n_in[0], n_in[1], dists[i - 1], labels[i - 1])]
            if overhang[i] == "start":
                # Growing edge -> any facet nucleating on its way round -> inverted facet -> sidewall,
                # in traversal order: the sidewall comes last on the left, first on the right.
                inx, iny, ind, inl, ang_inv = inverted[overhang_side[i]]
                ex, ey = normals[grower[i]]
                ang_e = math.atan2(ey, ex)
                lo, hi = (ang_e, ang_inv) if overhang_side[i] == "left" else (ang_inv, ang_e)
                extras = sorted(
                    ((nx, ny, d, label) for nx, ny, d, label, ang in named if label != "sp_inv" and between_ccw(ang, lo, hi)),
                    key=lambda e: (math.atan2(e[1], e[0]) - lo) % (2 * math.pi),
                )
                inserted = [*extras, (inx, iny, ind, inl)] if overhang_side[i] == "left" else [(inx, iny, ind, inl), *extras]
                chain.extend(inserted)
                chain.append((n_out[0], n_out[1], dists[i], labels[i]))
                chains.append(chain)
                continue
            if convex_at[i] and active[i - 1] and active[i] and not (riser[i - 1] or riser[i]):
                ang_in = math.atan2(n_in[1], n_in[0])
                ang_out = math.atan2(n_out[1], n_out[0])
                extras = sorted(
                    ((nx, ny, d, label, ang) for nx, ny, d, label, ang in named if between_ccw(ang, ang_in, ang_out)),
                    key=lambda e: (e[4] - ang_in) % (2 * math.pi),
                )
                chain.extend((nx, ny, d, label) for nx, ny, d, label, _ang in extras)
            chain.append((n_out[0], n_out[1], dists[i], labels[i]))
            chains.append(chain)
        if lateral_reach is not None:
            # at a corner, each edge advances by its own advance at that corner, and an m-plane
            # nucleating there by the shade's
            for i, chain in enumerate(chains):
                factor = _reach_factor(coords[i][1], *lateral_reach)
                chain[0] = (*chain[0][:2], d_to[i - 1], chain[0][3])
                chain[-1] = (*chain[-1][:2], d_from[i], chain[-1][3])
                for k in range(1, len(chain) - 1):
                    if chain[k][3] == "m":
                        chain[k] = (*chain[k][:2], chain[k][2] * factor, "m")
        nucleated = [
            [(nx, ny, nx * coords[i][0] + ny * coords[i][1] + d) for nx, ny, d, _l in chains[i][1:-1]]
            for i in range(n)
        ]

        def bounds_around(first_edge: int, last_edge: int) -> list[tuple[float, float, float]]:
            """Offset half-planes of edges first_edge..last_edge plus up to two more each side,
            walking outwards only across convex corners: within a convex run the grown shape is the
            intersection of its facets' offset half-planes, so a piece must not cross any of them -
            this is what makes two fans closing a narrow facet meet at one apex instead of
            overshooting each other. Across a concave corner a neighbour's line is not a bound."""
            own = {k % n for k in range(first_edge, last_edge + 1)}
            idx = set(own)
            corners = {k % n for k in range(first_edge, last_edge + 2)}
            # Sub-nm edges (a step left by an etch, a sliver of tilt at a corner) don't count
            # towards the two: the real facet just past one must still bound this piece.
            k, counted = first_edge, 0
            while counted < 2 and first_edge - k < n - 1 and passable[k % n]:
                k -= 1
                idx.add(k % n)
                corners.add(k % n)
                counted += lengths[k % n] >= _MIN_NUCLEATION_EDGE
            k, counted = last_edge, 0
            while counted < 2 and k - last_edge < n - 1 and passable[(k + 1) % n]:
                k += 1
                idx.add(k % n)
                corners.add((k + 1) % n)
                counted += lengths[k % n] >= _MIN_NUCLEATION_EDGE
            # Only a real crystal facet bounds its neighbours (a rate-0 one included: its line is
            # pinned) - a vicinal one too, since it grows as that facet (see `classify`): a
            # sidewall a fraction of a degree off vertical, as the first strip up from an etched
            # floor leaves it, must still stop the semi-polar strip nucleating above it, or that
            # strip pokes several nm past the sidewall's own front and the sidewall, catching up
            # underneath over the following substeps, is left with a step. An off-axis edge - a
            # curved mask wall, a sub-nm etch dip - or a non-seed edge has no growth front of its
            # own to stop anything at.
            # A facet nucleating at a corner of the run bounds it too, or the neighbouring edge's
            # strip would poke past it and leave a sliver of edge for the next growth step to grow.
            # Not a vicinal neighbour of a vicinal piece of the same facet, though: on a flank
            # leaning out (a flared core, split where two layers meet), the exact line through the
            # lower edge's outer end would cut the upper edge's strip down to nothing.
            own_vicinal = {facet_dirs[j] for j in own if labels[j] and not is_facet(*normals[j])}
            own_dirs = {facet_dirs[j] for j in own}
            planes = [
                offsets[j]
                for j in idx
                if j in own or (
                    active[j] and not riser[j] and (is_facet(*normals[j]) or labels[j])
                    and not (labels[j] and facet_dirs[j] in own_vicinal)
                    and not (shaded[j] and facet_dirs[j] in own_dirs)
                )
            ]
            return planes + [plane for c in corners for plane in nucleated[c]]

        def clipped(points: list[tuple[float, float]], planes: list[tuple[float, float, float]]) -> BaseGeometry:
            # Sutherland-Hodgman against each half-plane, with a hair of slack so a piece lying
            # exactly on a bound isn't shaved to nothing by rounding.
            for a, b, h in planes:
                if len(points) < 3:
                    return Polygon()
                out = []
                for p, q in zip(points, points[1:] + points[:1]):
                    fp, fq = a * p[0] + b * p[1] - h - 1e-9, a * q[0] + b * q[1] - h - 1e-9
                    if fp <= 0:
                        out.append(p)
                    if (fp < 0 < fq) or (fq < 0 < fp):
                        s = fp / (fp - fq)
                        out.append((p[0] + s * (q[0] - p[0]), p[1] + s * (q[1] - p[1])))
                points = out
            # Every piece is a triangle or rectangle clipped by half-planes - convex, hence valid.
            return Polygon(points) if len(points) >= 3 else Polygon()

        for i in range(n):
            v = coords[i]
            n_in, d_in, lbl_in = normals[i - 1], d_to[i - 1], labels[i - 1]
            n_out, d_out, lbl_out = normals[i], d_from[i], labels[i]
            if (d_in <= 0 and d_out <= 0 and len(chains[i]) == 2) or outside(v[0]):
                continue
            p_in = (v[0] + n_in[0] * d_in, v[1] + n_in[1] * d_in)
            p_out = (v[0] + n_out[0] * d_out, v[1] + n_out[1] * d_out)

            chain = chains[i]
            pairs = list(zip(chain, chain[1:]))
            # At an overhang corner, the inverted facet meets the sidewall at its foot, exactly.
            wall_pair = (len(pairs) - 1 if overhang_side[i] == "left" else 0) if overhang[i] else None
            mitre_points = []
            for k, ((nx1, ny1, d1, l1), (nx2, ny2, d2, l2)) in enumerate(pairs):
                if k == wall_pair:
                    mitre_points.append(exact_mitre(v, (nx1, ny1), d1, (nx2, ny2), d2, l2))
                else:
                    mitre_points.extend(mitre_or_bevel(v, (nx1, ny1), d1, l1, (nx2, ny2), d2, l2))
            if not convex_at[i] and not overhang[i] and len(mitre_points) != 1:
                continue  # concave corner whose mitre lands too far out: the two edge strips suffice
            planes = bounds_around(grower[i], grower[i]) if overhang[i] else bounds_around(i - 1, i)
            fan = [(p_in, lbl_in), *mitre_points, (p_out, lbl_out)]
            for (p_a, label_a), (p_b, _label_b) in zip(fan, fan[1:]):
                piece = clipped([v, p_a, p_b], planes)
                if not piece.is_empty and piece.area > 1e-12:
                    add_piece(label_a, piece)

        for i in range(n):
            if dists[i] <= 0:
                continue
            v_i, v_next = coords[i], coords[(i + 1) % n]
            nx, ny = normals[i]
            strip = [v_i, v_next, (v_next[0] + nx * d_to[i], v_next[1] + ny * d_to[i]), (v_i[0] + nx * d_from[i], v_i[1] + ny * d_from[i])]
            if outside(v_i[0], v_next[0]):
                continue
            piece = clipped(strip, bounds_around(i, i))
            if not piece.is_empty and piece.area > 1e-12:
                add_piece(labels[i], piece)

    return {label: _clean(_robust_union(pcs)) for label, pcs in pieces_by_family.items()}


def beam_vector(angle_deg: float, length: float) -> tuple[float, float]:
    """Direction a beam travels into the structure (source above, travelling down), for a recipe
    tilted `angle_deg` from the surface normal (0 = straight down, positive tilts towards +x),
    scaled to `length`.
    """
    angle = math.radians(angle_deg)
    return (length * math.sin(angle), -length * math.cos(angle))


def _half_plane(nx: float, ny: float, h: float, extent: float = 1.0e6) -> Polygon:
    """The half-plane {p : n·p <= h} for a unit normal n, as a polygon `extent` nm across - far
    beyond any real domain, so intersecting with it is exact for every practical purpose.
    """
    px, py = nx * h, ny * h
    tx, ty = -ny * extent, nx * extent
    return Polygon([
        (px - tx, py - ty),
        (px + tx, py + ty),
        (px + tx - nx * extent, py + ty - ny * extent),
        (px - tx - nx * extent, py - ty - ny * extent),
    ])


def _line_parts(geom: BaseGeometry) -> list[BaseGeometry]:
    """Every non-degenerate LineString inside `geom`, flattening Multi*/GeometryCollections (a
    boundary/air intersection can mix lines with stray single points)."""
    if geom.is_empty:
        return []
    if geom.geom_type in ("LineString", "LinearRing"):
        return [geom] if geom.length > 0 else []
    if hasattr(geom, "geoms"):
        return [line for part in geom.geoms for line in _line_parts(part)]
    return []


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
    - **Selective etch runs in substeps.** Each substep finds, per distinct rate factor present,
      what that factor's reach of the current air can get to without passing through any
      markedly slower-etching layer (see `etch`), and removes from each layer only its own
      factor's share. A thinner, slower-etching layer (a resist mask, a stop layer) is never
      tunnelled through, the
      material under a mask is only reached round the mask's edge, and a slower layer being
      exposed (an oxide etch reaching the substrate) doesn't stop the faster material beside it
      from receding. This gets multi-material selectivity, masked undercut, and etch-through-
      once-a-mask-is-fully-consumed all correct without a full level-set solver, at the residual
      cost of a finite-step-size error bounded by roughly one substep's depth (see `etch`'s
      `steps` parameter) - the same kind of discretisation error any explicit time-stepping
      scheme has, not a structural limitation.
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
        return self._solid_of(unary_union(polys))

    def _solid_of(self, union: BaseGeometry) -> BaseGeometry:
        """`solid()` from the plain union of every layer, already computed - `deposit_faceted`
        keeps that union up to date substep by substep instead of re-merging every substep's film
        each time (see there)."""
        if union.is_empty:
            return Polygon()
        # Closing by `_SEAM_EPS`, not just merging parts that touch: two layers (or two substeps of
        # one growth) whose shared boundary disagrees in the last few bits leave a zero-width slit
        # *between* them, which every growth step reading this shape back would otherwise treat as
        # a real crevice in the surface - and grow spikes out of.
        merged = self._snap_edges(_clean(union))
        mitre = {"join_style": "mitre"}
        with np.errstate(divide="ignore", invalid="ignore"):  # GEOS mitre on a degenerate segment - harmless
            merged = _clean(merged.buffer(_SEAM_EPS, **mitre).buffer(-_SEAM_EPS, **mitre))
        return _fill_holes(merged) if _has_interior(merged) else merged  # only the spurious ones

    def bounds(self) -> tuple[float, float, float, float]:
        solid = self.solid()
        if solid.is_empty:
            return (0.0, 0.0, self.domain_width_nm, 0.0)
        return solid.bounds

    # -- domain-edge handling --------------------------------------------

    def _snap_edges(self, geom: BaseGeometry) -> BaseGeometry:
        """`geom` with every vertex within `_EDGE_SNAP` of a domain edge moved exactly onto it.

        Boolean ops leave a vertex that should sit on `x=0` at `x=1e-17` or so. `_mirror_pad`
        then unions the shape with a mirror copy whose matching vertex sits at `-1e-17`: no longer
        an exactly shared edge, so GEOS keeps the two as separate parts of a `MultiPolygon` and
        the domain edge reads back as an *exposed surface* - a conformal deposit coats it, an
        isotropic etch eats into it, one sub-step at a time, and `etch()` sees the substrate as
        "exposed" through it and guards the whole domain against the faster factors.
        """
        if geom.is_empty:
            return geom
        w = self.domain_width_nm

        def snap(coords: np.ndarray) -> np.ndarray:
            coords = coords.copy()
            x = coords[:, 0]
            x[np.abs(x) < _EDGE_SNAP] = 0.0
            x[np.abs(x - w) < _EDGE_SNAP] = w
            return coords

        return shapely.transform(geom, snap)

    def _mirror_pad(self, geom: BaseGeometry, band: float | None = None, bulk: bool = False) -> BaseGeometry:
        """`geom` unioned with its mirror images across `x=0` and `x=domain_width_nm` (see the
        class docstring: domain edges are symmetry boundaries). `band`, if given, keeps only the
        first `band` nm of each mirror image - enough for any operation whose reach from the domain
        is below it, at a fraction of the vertex count of the two full copies. `bulk` adds the
        wafer bulk under the whole padded width in the same union (see `_bulk_pad`)."""
        if geom.is_empty:
            return geom
        geom = self._snap_edges(geom)
        w = self.domain_width_nm
        if band is not None and band < w:
            y0, y1 = geom.bounds[1] - 1.0, geom.bounds[3] + 1.0
            src_left = _polygonal(geom.intersection(box(0.0, y0, band, y1)))
            src_right = _polygonal(geom.intersection(box(w - band, y0, w, y1)))
        else:
            band = w
            src_left = src_right = geom
        pieces = [
            geom,
            scale(src_left, xfact=-1, yfact=1, origin=(0.0, 0.0)),
            scale(src_right, xfact=-1, yfact=1, origin=(w, 0.0)),
        ]
        if bulk and self.floor_nm is not None:
            pieces.append(box(-band, self.floor_nm - _BULK_MARGIN, w + band, self.floor_nm))
        return _clean(_robust_union(pieces))

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

    def _reflect_into_domain(self, x: float) -> float:
        """Map an x from the mirror-padded copies (see `_mirror_pad`) back into [0, domain_width]."""
        w = self.domain_width_nm
        x = math.fmod(abs(x), 2 * w)
        return 2 * w - x if x > w else x

    def _material_at(self, x: float, y: float) -> str | None:
        """The material of the layer at (x, y), allowing for the mirror/bulk padding of `_pad`;
        falls back to the nearest layer (a probe can land a hair outside a simplified layer)."""
        return self._material_lookup()(x, y)

    def _material_lookup(self) -> Callable[[float, float], str | None]:
        """`_material_at` for many probes against the same stack: the live layers are gathered
        once, and each probe tests them all in one vectorised call - topmost (latest) layer first,
        then the nearest one - instead of one Python-level `contains` per layer. Only valid while
        the layers don't change."""
        live = [l for l in self.layers if not l.polygon.is_empty]
        polygons = np.array([l.polygon for l in live], dtype=object)
        floor = self.floor_nm
        bottom = self.layers[0].material if self.layers else None

        def material_at(x: float, y: float) -> str | None:
            if floor is not None and y < floor:
                return bottom
            if not live:
                return None
            px = self._reflect_into_domain(x)
            inside = np.flatnonzero(shapely.contains_xy(polygons, px, y))
            if inside.size:
                return live[inside[-1]].material
            return live[int(np.argmin(shapely.distance(polygons, Point(px, y))))].material

        return material_at

    def _split_at_layer_boundaries(self, padded: BaseGeometry, margin: float = math.inf) -> BaseGeometry:
        """`padded` (from `_pad(self.solid())`) with every layer vertex lying on one of its exterior
        edges inserted into that edge, so each edge of the result is made of a single material -
        `solid()` merges collinear edges of neighbouring layers (an InGaN cap's side flush with the
        GaN sidewall beneath it) into one. Edges further than `margin` outside the domain are left
        whole."""
        w = self.domain_width_nm
        # every vertex of every ring of every layer, in one vectorised call
        v = shapely.get_coordinates([l.polygon for l in self.layers if not l.polygon.is_empty])
        if not len(v):
            return padded
        v = np.vstack([v, np.column_stack([-v[:, 0], v[:, 1]]), np.column_stack([2 * w - v[:, 0], v[:, 1]])])

        def split_ring(coords: list[tuple[float, float]]) -> list[tuple[float, float]]:
            out: list[tuple[float, float]] = []
            for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
                out.append((x1, y1))
                if max(x1, x2) < -margin or min(x1, x2) > w + margin:
                    continue
                dx, dy = x2 - x1, y2 - y1
                length2 = dx * dx + dy * dy
                if length2 < 1e-12:
                    continue
                t = ((v[:, 0] - x1) * dx + (v[:, 1] - y1) * dy) / length2
                px, py = x1 + t * dx, y1 + t * dy
                dist = np.hypot(v[:, 0] - px, v[:, 1] - py)
                t_margin = _SPLIT_TOL / math.sqrt(length2)
                hit = (dist < _SPLIT_TOL) & (t > t_margin) & (t < 1 - t_margin)
                if hit.any():
                    for tt in np.unique(np.round(t[hit], 9)):
                        out.append((x1 + tt * dx, y1 + tt * dy))
            out.append(coords[-1])
            return out

        parts = list(padded.geoms) if isinstance(padded, MultiPolygon) else [padded]
        split = [
            Polygon(split_ring(list(p.exterior.coords)), [split_ring(list(r.coords)) for r in p.interiors])
            for p in parts
            if not p.is_empty
        ]
        return split[0] if len(split) == 1 else MultiPolygon(split)

    def _seed_edge_filter(
        self, seed_materials: list[str]
    ) -> Callable[[tuple[float, float], tuple[float, float], tuple[float, float]], bool]:
        """An `_offset_named_facets` `edge_active` predicate: an edge grows only if the material
        just inside its midpoint is a seed (see `seed_matches`). Edges belong to the exposed
        surface by construction, so a mask covering a seed hides it with no extra bookkeeping."""

        material_at = self._material_lookup()

        def active(p1: tuple[float, float], p2: tuple[float, float], normal: tuple[float, float]) -> bool:
            mx = (p1[0] + p2[0]) / 2 - normal[0] * _SEED_PROBE_DEPTH
            my = (p1[1] + p2[1]) / 2 - normal[1] * _SEED_PROBE_DEPTH
            material = material_at(mx, my)
            return material is not None and seed_matches(material, seed_materials)

        return active

    def _clamp_floor(self, y: float) -> float:
        return y if self.floor_nm is None else max(y, self.floor_nm)

    def _shadow(
        self, solid: BaseGeometry, angle_deg: float, y_min: float, y_max: float, mirrored: BaseGeometry | None = None
    ) -> BaseGeometry:
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
        `mirrored`: the mirror-padded `solid`, when the caller already has it (a band-cropped one
        is fine as long as its band covers the shadow's horizontal reach, see `_etch_band`; so is
        a bulk-padded one - the bulk only ever shadows what lies below the wafer).
        """
        if solid.is_empty:
            return solid
        padded = self._mirror_pad(solid) if mirrored is None else mirrored
        span = (y_max - y_min) + self.domain_width_nm + _MARGIN
        dx, dy = beam_vector(angle_deg, span)
        swept = sweep_union(padded, (dx, dy))
        return swept.difference(padded)

    @staticmethod
    def _outside_air(air: BaseGeometry, air_box: BaseGeometry) -> BaseGeometry:
        """The parts of `air` (= `air_box` minus the padded solid) connected to the outside, i.e.
        touching the box's own boundary. A part that doesn't is a void sealed inside the solid
        (see `_fill_holes`): no etchant and no precursor gets in, so an etch or a conformal
        deposit must not act on its walls."""
        if air.is_empty or not isinstance(air, MultiPolygon):
            return air
        edge = air_box.boundary
        kept = [part for part in air.geoms if part.intersects(edge)]
        if len(kept) == len(air.geoms):
            return air
        return _clean(unary_union(kept)) if kept else Polygon()

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
        film = grown.difference(padded)
        if _has_interior(padded):
            # Only the surface open to the outside is coated: a void sealed inside the solid
            # (see `_fill_holes`) stays empty.
            air_box = box(*grown.bounds).buffer(_MARGIN)
            film = film.intersection(self._outside_air(_clean(air_box.difference(padded)), air_box))
        film = self._crop(film, y_min - thickness_nm, y_max + thickness_nm)
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

    def deposit(
        self,
        material: str,
        thickness_nm: float,
        recipe: DepositionRecipe,
        roughness_nm: float | None = None,
        grain_nm: float | None = None,
        seed: int = 0,
        seed_materials: list[str] | None = None,
    ) -> None:
        """`roughness_nm`: the film's outer face is no longer smooth but made of grains, each
        standing up to `roughness_nm` above or sinking as far below the nominal thickness - see
        `_roughen`. `grain_nm` is their lateral size (4x `roughness_nm` if unset), `seed` the draw:
        the same seed on the same surface always gives the same grains.

        `seed_materials`: the film only forms on exposed surfaces made of one of these materials
        (as `deposit_epitaxial`'s SAG selectivity), spilling at most its thickness past their
        edges - see `_seed_surface`. None or empty: on every exposed surface."""
        before = len(self.layers)
        solid = self.solid() if roughness_nm or seed_materials else None
        if seed_materials:
            seed_surface = self._seed_surface(seed_materials)
            if seed_surface.is_empty:
                return
        if recipe.mode is DepositionMode.conformal:
            self.deposit_conformal(material, thickness_nm)
        else:
            self.deposit_directional(material, thickness_nm, recipe.angle_deg)
        if seed_materials and len(self.layers) > before:
            film = _drop_tiny(_clean(self.layers[-1].polygon.intersection(seed_surface.buffer(thickness_nm, quad_segs=12))))
            if film.is_empty:
                self.layers.pop()
            else:
                self.layers[-1].polygon = film
        if roughness_nm and len(self.layers) > before:
            film = self._roughen(self.layers[-1].polygon, solid, roughness_nm, grain_nm or 4 * roughness_nm, seed)
            if film.is_empty:
                self.layers.pop()
            else:
                self.layers[-1].polygon = film

    def _seed_surface(self, seed_materials: list[str]) -> BaseGeometry:
        """The exposed edges made of a seed material (see `_seed_edge_filter`), as lines - the only
        surface a selective deposit coats. Empty when no exposed edge is a seed."""
        if not any(seed_matches(l.material, seed_materials) for l in self.layers if not l.polygon.is_empty):
            return LineString()
        padded = self._split_at_layer_boundaries(self._pad(self.solid()), margin=1.0)
        edge_active = self._seed_edge_filter(seed_materials)
        lines = []
        for part in (padded.geoms if isinstance(padded, MultiPolygon) else [padded]):
            coords = list(orient(part, sign=1.0).exterior.coords)
            for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
                if max(x1, x2) < -1.0 or min(x1, x2) > self.domain_width_nm + 1.0:
                    continue
                length = math.hypot(x2 - x1, y2 - y1)
                if length < 1e-12:
                    continue
                normal = ((y2 - y1) / length, -(x2 - x1) / length)
                if edge_active((x1, y1), (x2, y2), normal):
                    lines.append(LineString([(x1, y1), (x2, y2)]))
        return unary_union(lines) if lines else LineString()

    def _roughen(
        self, film: BaseGeometry, solid: BaseGeometry, amplitude_nm: float, grain_nm: float, seed: int
    ) -> BaseGeometry:
        """`film`'s free face (the part not resting on `solid`) turned into grains: one every half
        grain along it, each a disk of half the grain size, its height drawn uniformly in
        [-amplitude, +amplitude]. A positive grain is a bump added on top of the face, a negative
        one a dimple cut into it - and one deeper than the film itself leaves a bare spot, so a
        roughness above the thickness gives islands rather than a closed film. The disks never
        reach into `solid`, and a face the deposit never coated (a shadowed flank) grows none.
        """
        radius = max(grain_nm / 2, amplitude_nm)  # a smaller disk would float off its own bump
        surface = film.boundary.difference(solid.buffer(_ROUGH_CONTACT_TOL))
        # the domain's own vertical edges are cuts, not a surface the film exposes
        x_min, x_max = _ROUGH_CONTACT_TOL, self.domain_width_nm - _ROUGH_CONTACT_TOL
        surface = surface.intersection(box(x_min, film.bounds[1] - 1.0, x_max, film.bounds[3] + 1.0))
        lines = [g for g in shapely.get_parts(shapely.line_merge(surface)) if isinstance(g, LineString) and g.length > 0]
        if not lines:
            return film
        total = sum(line.length for line in lines)
        spacing = max(radius, total / _ROUGH_MAX_GRAINS)
        points, normals = [], []
        for line in lines:
            n = max(1, round(line.length / spacing))
            at = (np.arange(n) + 0.5) * line.length / n
            delta = min(_ROUGH_CONTACT_TOL, line.length / (4 * n))
            p = shapely.get_coordinates(shapely.line_interpolate_point(line, at))
            ahead = shapely.get_coordinates(shapely.line_interpolate_point(line, np.minimum(at + delta, line.length)))
            behind = shapely.get_coordinates(shapely.line_interpolate_point(line, np.maximum(at - delta, 0.0)))
            tangent = ahead - behind
            normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
            normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
            points.append(p)
            normals.append(normal)
        p, normal = np.vstack(points), np.vstack(normals)
        # point every normal out of the matter, into the air the film faces
        whole = _robust_union([solid, film])
        probe = p + normal * _ROUGH_CONTACT_TOL
        normal[shapely.contains_xy(whole, probe[:, 0], probe[:, 1])] *= -1

        rng = np.random.default_rng(seed)
        height = rng.uniform(-amplitude_nm, amplitude_nm, len(p))
        bump, dimple = height > 0, height < 0
        # a bump's disk tops out `height` above the face, a dimple's bottoms out `-height` below it
        bumps = shapely.buffer(shapely.points(p[bump] + normal[bump] * (height[bump] - radius)[:, None]), radius, quad_segs=6)
        dimples = shapely.buffer(shapely.points(p[dimple] + normal[dimple] * (height[dimple] + radius)[:, None]), radius, quad_segs=6)
        rough = film
        if len(dimples):
            rough = rough.difference(_robust_union(list(dimples)))
        if len(bumps):
            rough = _robust_union([rough, *bumps])
        rough = rough.difference(solid)
        y_min, y_max = whole.bounds[1], whole.bounds[3]
        return self._crop(rough, y_min, y_max + 2 * radius)

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

        If `seed_materials` is non-empty the film nucleates *only* on exposed surfaces made of one
        of those materials (SAG selectivity) - a mask or a non-seed cap is just not a growth
        surface. An alloy family name matches every composition of it (`"InGaN"` matches
        `In0.10Ga0.90N`, see `seed_matches`). Pass an empty list / None to grow on all exposed
        surfaces.

        `provenance`, if given, is attached to the resulting `Layer` as-is (see
        `LayerProvenance`) — purely descriptive, this method never reads it back.
        """
        if thickness_nm <= 0 or not self.layers:
            return
        solid = self.solid()
        if solid.is_empty:
            return

        y_min, y_max = solid.bounds[1], solid.bounds[3]

        # --- growth direction(s) ---
        if orientation == "c_plane":
            vectors = [(0.0, thickness_nm)]
        elif orientation == "m_plane":
            # Symmetric lateral expansion on both sidewalls
            vectors = [(thickness_nm, 0.0), (-thickness_nm, 0.0)]
        elif orientation == "semi_polar":
            # Symmetric tilted facets: ±x tilt from vertical by angle_deg
            rad = math.radians(angle_deg)
            vectors = [(thickness_nm * math.sin(rad), thickness_nm * math.cos(rad)), (-thickness_nm * math.sin(rad), thickness_nm * math.cos(rad))]
        else:  # pragma: no cover
            raise ValueError(f"unknown epitaxial orientation {orientation!r}")

        padded = self._pad(solid)
        if seed_materials:
            # SAG: sweep only the exposed edges made of a seed material (see `_seed_edge_filter`) -
            # a mask, or a non-seed cap, simply isn't a growth surface, with no need to guess what
            # it shadows; the domain edges stay symmetry boundaries as for blanket growth.
            if not any(seed_matches(l.material, seed_materials) for l in self.layers if not l.polygon.is_empty):
                return
            margin = thickness_nm + 1.0
            padded = self._split_at_layer_boundaries(padded, margin=margin)
            edge_active = self._seed_edge_filter(seed_materials)
            quads = []
            for part in (padded.geoms if isinstance(padded, MultiPolygon) else [padded]):
                coords = list(orient(part, sign=1.0).exterior.coords)
                for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
                    if max(x1, x2) < -margin or min(x1, x2) > self.domain_width_nm + margin:
                        continue
                    length = math.hypot(x2 - x1, y2 - y1)
                    if length < 1e-12:
                        continue
                    normal = ((y2 - y1) / length, -(x2 - x1) / length)
                    facing = [(vx, vy) for vx, vy in vectors if vx * normal[0] + vy * normal[1] > 1e-9]
                    if not facing or not edge_active((x1, y1), (x2, y2), normal):
                        continue
                    for vx, vy in facing:
                        quads.append(Polygon([(x1, y1), (x2, y2), (x2 + vx, y2 + vy), (x1 + vx, y1 + vy)]))
            if not quads:
                return
            film_raw = _clean(_robust_union(quads))
        else:
            film_raw = _clean(unary_union([sweep_union(padded, v) for v in vectors]))

        film = self._crop(_clean(film_raw.difference(padded)), y_min - thickness_nm, y_max + thickness_nm)
        film = _drop_tiny(_close_seams(_clean(film.difference(solid)), solid))

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
        max_substep_nm: float = 10.0,
        rate_sp_inv: float = 0.0,
        material_sp_inv: str | None = None,
        lateral_reach_nm: float | None = None,
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
        faithfully follow the pencil/pyramid shape as it develops. A thick layer is grown the same
        way internally - in substeps of at most `max_substep_nm` of advance for the fastest facet -
        so facets can appear and vanish along the way (a crystal filling a mask opening, then
        overflowing it), and still lands as a single Layer per material.

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

        `rate_sp_inv` adds a fourth family, the inverted semi-polar facets ({10-1-1}: same angle
        from the c-axis as the SP ones, facing down and out), with `material_sp_inv` as its
        override. At 0 (the default) nothing changes; a nonzero rate must be at least 2% of the
        fastest one (ValueError otherwise - slower, its advance per substep is below the engine's
        resolution). Above 0, a crystal whose sidewalls don't grow
        (rate_m = 0, or not a seed material) is no longer held inside their lines: it spreads past
        them with an inverted facet as its underside, whose foot slides down the bare sidewall as
        it advances - a shell nucleating on a nanowire's tip and creeping down it, with no m-plane
        growth on the wire itself (see `_offset_named_facets`). A shell grown that way ends as a
        hexagon around the tip: c-plane top, SP facets, inverted SP facets, and an m-plane at its
        widest only if rate_m > 0.

        `lateral_reach_nm` shades the sidewalls, the way neighbouring wires keep the growth species
        from reaching the bottom of a forest of them: the m-planes advance at full `rate_m` at the
        top of the structure, less and less further down, and not at all `lateral_reach_nm` below
        it - a layer that widens from the flanks, thickest at the top and gone at that depth, its
        outer face straight. The top is the structure's highest point at each substep. Unset (the
        default), the sidewalls grow alike all the way down.

        `seed_materials` enables SAG selectivity (same semantics as `deposit_epitaxial`).

        `provenance`, if given, is attached as-is to every `Layer` this call creates (see
        `LayerProvenance`) - including each per-family split, when `material_c`/`material_m`/
        `material_sp`/`material_sp_inv` produce more than one. This method never reads it back.
        """
        max_rate = max(rate_c, rate_m, rate_sp, rate_sp_inv)
        if 0 < rate_sp_inv < _MIN_SP_INV_FRACTION * max_rate:
            # Substeps advance the fastest facet by up to `max_substep_nm`: slower than this, the
            # inverted facet moves a fraction of a nm per substep, and the sliver it adds at its
            # foot falls under the film's own noise filters - dropped on one side but not the
            # other, the overhang loses its symmetry and breaks into a staircase.
            raise ValueError(
                f"rate_sp_inv must be 0 or at least {_MIN_SP_INV_FRACTION:g} x the fastest rate "
                f"({_MIN_SP_INV_FRACTION * max_rate:.3g} here), got {rate_sp_inv:g}"
            )
        if lateral_reach_nm is not None and lateral_reach_nm <= 0:
            raise ValueError(f"lateral_reach_nm must be > 0, got {lateral_reach_nm:g}")
        if thickness_nm <= 0 or max_rate <= 0 or not self.layers:
            return
        if seed_materials and not any(
            seed_matches(l.material, seed_materials) for l in self.layers if not l.polygon.is_empty
        ):
            return
        # The union of every layer, kept up to date below as substeps add their films.
        union = unary_union([l.polygon for l in self.layers if not l.polygon.is_empty])
        solid_before = self._solid_of(union)
        if solid_before.is_empty:
            return

        # A single offset is only exact while no facet vanishes or appears mid-growth: grown in one
        # shot, a crystal filling a mask opening could never overflow it and facet, and a c-plane top
        # consumed by its flanking facets folds over itself. Substeps no larger than
        # `max_substep_nm` of advance let each topology change happen between two offsets. Each
        # substep's film is tagged so the next one keeps growing on it whatever `seed_materials`
        # says, then the substeps are merged back into one Layer per material.
        families = {"c": material_c or material, "m": material_m or material, "sp": material_sp or material}
        if rate_sp_inv > 0:
            families["sp_inv"] = material_sp_inv or material
        tags = {m: f"<growing>{m}" for m in set(families.values())}
        seeds = [*seed_materials, *tags.values()] if seed_materials else None
        n = max(1, math.ceil(max_rate * thickness_nm / max_substep_nm))
        start = len(self.layers)
        solid = solid_before
        for _ in range(n):
            count = len(self.layers)
            self._deposit_faceted_once(
                thickness_nm / n, rate_c, rate_m, rate_sp, semi_polar_angle_deg, seeds,
                {family: tags[m] for family, m in families.items()}, rate_sp_inv, solid, lateral_reach_nm,
            )
            # The solid the next substep grows on: last one's union plus this substep's films only -
            # `solid()` would re-merge every film grown so far, a cost growing with each substep
            # (quadratic over a thick layer).
            added = [l.polygon for l in self.layers[count:] if not l.polygon.is_empty]
            if added:
                union = _covering_union([union, *added])
                solid = self._solid_of(union)
        grown = self.layers[start:]
        del self.layers[start:]
        for layer_material, tag in tags.items():
            polys = [l.polygon for l in grown if l.material == tag]
            if not polys:
                continue
            film = _fill_holes(_clean(_covering_union(polys)))
            film = _close_seams(_clean(film.difference(solid_before)), solid_before)
            if _has_interior(film):
                # Subtracting the solid can leave the film wrapped round a speck its seam closing
                # made, with no layer in it: air, a construction artifact (see `_fill_holes`).
                film = _fill_holes(film, occupied=unary_union([l.polygon for l in self.layers if not l.polygon.is_empty]))
            film = _drop_tiny(film)
            if not film.is_empty:
                self.layers.append(Layer(material=layer_material, polygon=film, provenance=provenance))

    def _deposit_faceted_once(
        self,
        t: float,
        rate_c: float,
        rate_m: float,
        rate_sp: float,
        semi_polar_angle_deg: float,
        seed_materials: list[str] | None,
        materials_by_family: dict[str, str],
        rate_sp_inv: float = 0.0,
        solid: BaseGeometry | None = None,
        lateral_reach_nm: float | None = None,
    ) -> None:
        """One single-offset substep of `deposit_faceted`, appending one Layer per material in
        `materials_by_family` (keyed "c"/"m"/"sp", plus "sp_inv" when `rate_sp_inv` > 0).
        `solid`: `self.solid()`, when the caller already has it."""
        if solid is None:
            solid = self.solid()
        if solid.is_empty:
            return
        y_min, y_max = solid.bounds[1], solid.bounds[3]
        theta = math.radians(semi_polar_angle_deg)
        material = materials_by_family["c"]

        # --- SAG selectivity -----------------------------------------------
        # Always grow from the whole (padded) solid, so the domain edges stay symmetry boundaries;
        # SAG only decides which of its exposed edges are allowed to move - an edge made of a
        # non-seed material (a mask, a cap) is pinned like a rate-0 facet.
        reach = t * max(rate_c, rate_m, rate_sp, rate_sp_inv) * 4 + 1.0
        real = self._pad(solid)
        # The growth front, trimmed of sub-angstrom bumps (see `_offset_named_facets` for the
        # steps between terraces). Trim-only - intersected with the solid - so it never sits above
        # the real surface: the film still starts exactly on it, see the difference below.
        padded = _clean(real.intersection(real.simplify(_FRONT_SMOOTH, preserve_topology=True)))
        edge_active = None
        if seed_materials:
            if not any(seed_matches(l.material, seed_materials) for l in self.layers if not l.polygon.is_empty):
                return
            padded = self._split_at_layer_boundaries(padded, margin=reach)
            edge_active = self._seed_edge_filter(seed_materials)

        named_directions = [
            (0.0, 1.0, rate_c * t, "c"),
            (1.0, 0.0, rate_m * t, "m"),
            (-1.0, 0.0, rate_m * t, "m"),
            (math.sin(theta), math.cos(theta), rate_sp * t, "sp"),
            (-math.sin(theta), math.cos(theta), rate_sp * t, "sp"),
        ]
        if rate_sp_inv > 0:
            # Only named when growing: as a pinned facet it would bound its neighbours, changing
            # every growth that never asked for it.
            named_directions += [
                (math.sin(theta), -math.cos(theta), rate_sp_inv * t, "sp_inv"),
                (-math.sin(theta), -math.cos(theta), rate_sp_inv * t, "sp_inv"),
            ]
        pieces_by_family = _offset_named_facets(
            padded, named_directions, edge_active, x_window=(-reach, self.domain_width_nm + reach),
            lateral_reach=(y_max, lateral_reach_nm) if lateral_reach_nm is not None else None,
        )
        max_reach = t * (rate_c + rate_m + rate_sp) / max(math.cos(theta), 0.05) + 1.0

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

        # Neighbouring families' pieces overlap where an edge strip meets a corner fan: whatever an
        # earlier family (c, then m, then sp) already claimed isn't handed out a second time.
        claimed = solid
        for layer_material, new_areas in pieces_by_material.items():
            new_area = unary_union(new_areas)
            grown = _clean(unary_union([solid, new_area]))
            film = self._crop(
                _clean(grown.difference(real)),
                y_min - t,
                y_max + max_reach,
            )
            film = _fill_holes(_close_seams(_clean(film.difference(solid)), solid))
            film = _drop_tiny(_clean(film.difference(claimed)))
            if not film.is_empty:
                self.layers.append(Layer(material=layer_material, polygon=film))
                claimed = unary_union([claimed, film])

    def fill_facet_envelope(
        self,
        material: str,
        seed_materials: list[str] | None = None,
        c_plane: bool = True,
        m_plane: bool = False,
        semi_polar_angle_deg: float | None = None,
        top_level_nm: float | None = None,
        provenance: LayerProvenance | None = None,
    ) -> None:
        """"Catch up" the crystal planes: impose an ideal faceted shape instead of growing one
        rate by rate. Each chosen facet family - c-plane (0,1), m-plane (±1,0), semi-polar
        (±sin θ, cos θ) - is pushed outward until it just touches the outermost exposed point of
        the crystal, and everything below those planes that isn't already solid is filled with
        `material`. Typical uses: a sharp pyramid on a pedestal's flat top (semi-polar only - the
        two SP planes through the top's edges meet at the apex), or squaring up an irregular
        crystal into clean c/m/SP facets before growing a conformal MQW stack on it.

        The crystal is the exposed surface of `seed_materials` (default: `[material]`) - whatever
        is touching air, so a SAG mask covering the rest of the wafer keeps it out. Separate
        exposed regions (e.g. one per mask opening) each get their own envelope. Each envelope is
        bounded below by its region's lowest exposed point; with no lateral facet chosen (neither
        m-plane nor semi-polar) it's bounded by the region's own x-range, i.e. implicit m-planes.

        `top_level_nm`, if given, truncates every envelope with a c-plane at that absolute level
        (a truncated pyramid / flat-topped pencil of a chosen height). The envelope must be bounded
        above: give `c_plane`, `semi_polar_angle_deg` or `top_level_nm`.
        """
        if not c_plane and semi_polar_angle_deg is None and top_level_nm is None:
            raise ValueError("fill_facet_envelope needs c_plane, semi_polar_angle_deg or top_level_nm to bound the shape above")
        if not self.layers:
            return
        solid = self.solid()
        if solid.is_empty:
            return
        y_min, y_max = solid.bounds[1], solid.bounds[3]

        seeds = seed_materials or [material]
        seed_polys = [l.polygon for l in self.layers if seed_matches(l.material, seeds) and not l.polygon.is_empty]
        if not seed_polys:
            return
        # Air starts exactly at y_min so the wafer's backside never counts as exposed; the domain's
        # left/right edges don't either, since there is no air outside the domain box.
        air = self._domain_box(y_min, y_max + 1.0).difference(solid)
        exposed = unary_union(seed_polys).boundary.intersection(air.buffer(1e-5))
        lines = _line_parts(exposed)
        if not lines:
            return

        normals: list[tuple[float, float]] = []
        if c_plane:
            normals.append((0.0, 1.0))
        if m_plane:
            normals += [(1.0, 0.0), (-1.0, 0.0)]
        if semi_polar_angle_deg is not None:
            theta = math.radians(semi_polar_angle_deg)
            normals += [(math.sin(theta), math.cos(theta)), (-math.sin(theta), math.cos(theta))]
        if not m_plane and semi_polar_angle_deg is None:
            normals += [(1.0, 0.0), (-1.0, 0.0)]

        blobs = unary_union([line.buffer(1.0) for line in lines])
        envelopes = []
        for blob in (blobs.geoms if isinstance(blobs, MultiPolygon) else [blobs]):
            points = [p for line in lines if line.intersects(blob) for p in line.coords]
            base = min(y for _, y in points)
            top = top_level_nm if top_level_nm is not None else y_max + _BULK_MARGIN
            if top <= base:
                continue
            envelope = box(0.0, base, self.domain_width_nm, top)
            for nx, ny in normals:
                envelope = envelope.intersection(_half_plane(nx, ny, max(nx * x + ny * y for x, y in points)))
            envelopes.append(self._crop(envelope.difference(solid), base, top))

        film = _drop_tiny(_clean(unary_union(envelopes))) if envelopes else Polygon()
        if not film.is_empty:
            self.layers.append(Layer(material=material, polygon=film, provenance=provenance))

    def etch(
        self,
        recipe: EtchRecipe,
        depth_nm: float,
        materials: MaterialLibrary,
        steps: int | None = None,
        preview: bool = False,
    ) -> None:
        """Remove material along `recipe`'s direction, `depth_nm` deep for the recipe's reference
        (factor 1.0) material - other materials recede slower/faster per `recipe.factor_for`.
        Runs in substeps so a mixed-rate recipe advances each material's own front correctly,
        including etching through a thin masking layer into whatever sits below it.

        Each substep takes the current air (everything around the solid that isn't solid) and,
        per distinct rate factor present, finds what that factor's reach (`substep * factor`)
        of air can get to without passing through any *slower* layer - see `_reach_air`. A
        layer loses exactly its own intersection with its own factor's ring, so a mask is only
        ever consumed at its own rate, the material under it only from where the air actually
        reaches (round the mask's edge, never through it), and the material beside a slower
        layer keeps receding even once that layer is exposed.

        A substep works on the solid mirrored only within a band past each domain edge, as wide
        as anything the substep can reach there (see `_etch_band`), and on the plain dilation of
        the air whenever no slower layer is thin enough to be tunnelled through - a layer only
        ever loses its own intersection with the ring, so clipping the ring to what the air can
        pass through changes nothing unless the dilation has to be done in geodesic steps. Each
        factor's slower layers are probed for thin parts once, with enough margin to hold for as
        many substeps as their own recession allows (see `_tunnel_policy`), not every substep.

        Only a layer etched below `_MASK_FACTOR_RATIO` of a factor's rate counts as slower for it:
        GaN at 0.8x under an oxide etch is no mask. A thin part of such a layer recedes by at
        least half the reach from its own lit side each substep, so the air tunnelling through
        it gets at most `substep * (factor - its factor)` - under half a reach - ahead of the
        front, an error within the one-substep error the scheme has anyway; guarding it instead
        steps the dilation in `_GEODESIC_STEP`s over the whole domain whenever such a layer has a
        part thinner than the reach, which a shell thinned by shading always does.

        Afterwards, sub-angstrom films left by the final substep are removed, and the exposed
        edges of slowly-etched layers are straightened to within their own step height (see the
        end of this method and `_smooth_exposed`).

        `steps`, if not given, is chosen by `_etch_steps`; `preview` divides that by
        `_PREVIEW_SUBSTEP_DIVISOR` for a quick look (a few nm of error on an isotropic front).
        """
        if depth_nm <= 0 or not self.layers:
            return
        factors = [recipe.factor_for(materials.get(l.material)) for l in self.layers]
        live = [i for i, l in enumerate(self.layers) if not l.polygon.is_empty]
        if not live:
            return
        if steps is None:
            steps = self._etch_steps(recipe, depth_nm, factors, live)
            if preview:
                steps = max(1, math.ceil(steps / _PREVIEW_SUBSTEP_DIVISOR))
        substep = depth_nm / steps
        directional = recipe.mode is EtchMode.directional
        w = self.domain_width_nm
        # factor -> (substep index its verdict holds until, geodesic step of `_reach_air`, probe hint)
        tunnel: dict[float, tuple[int, float, int]] = {}

        solid = self.solid()
        for k in range(steps):
            if solid.is_empty:
                break
            live = [i for i, l in enumerate(self.layers) if not l.polygon.is_empty]
            distinct_factors = sorted({factors[i] for i in live})
            if not distinct_factors or distinct_factors[-1] <= 0:
                break
            y_min, y_max = solid.bounds[1], solid.bounds[3]
            reach_max = substep * distinct_factors[-1]
            # Padded (mirrored within a band + bulk-extended) once per substep: the wafer floor
            # and the domain's left/right edges are real edges of the raw `solid` polygon but not
            # real surface, so neither the air nor the shadow (directional mode only) may see them.
            band = self._etch_band(solid, reach_max, recipe.angle_deg if directional else None)
            x0, x1 = (-band, w + band) if band is not None else (-w, 2 * w)
            padded = self._mirror_pad(solid, band=band, bulk=True)
            shadow = self._shadow(solid, recipe.angle_deg, y_min, y_max, mirrored=padded) if directional else None

            # This substep's "air": everything in a box around the padded solid that isn't solid.
            # Each factor's erosion ring is the air dilated by that factor's reach - through air
            # and faster/equal material only, where a slower layer is thin: see _reach_air.
            air_box = box(x0 - reach_max - _MARGIN, y_min - reach_max - _MARGIN, x1 + reach_max + _MARGIN, y_max + reach_max + _MARGIN)
            air = self._outside_air(air_box.difference(padded), air_box)
            # Thin parts of a band-cropped mirror image near the crop cut are not real geometry.
            window = (x0 + reach_max, x1 - reach_max)

            ring_by_factor: dict[float, BaseGeometry] = {}
            for factor in distinct_factors:
                if factor <= 0:
                    continue
                reach = substep * factor
                slower = [i for i in live if factors[i] < factor * _MASK_FACTOR_RATIO]
                blocked: BaseGeometry = Polygon()
                step = math.inf
                if slower:
                    until, step, hint = tunnel.get(factor, (k, math.inf, steps))
                    if k >= until or step < reach:
                        # The slower layers are needed to probe them, and to clip the stepped
                        # dilation to what is passable while a probe's thin verdict holds.
                        blocked = self._mirror_pad(_robust_union([self.layers[i].polygon for i in slower]), band=band)
                    if k >= until:
                        tunnel[factor] = self._tunnel_policy(
                            blocked, reach, window, k, steps, substep * max(factors[i] for i in slower), hint
                        )
                        until, step, hint = tunnel[factor]
                ring = self._reach_air(air, air_box, blocked, reach, recipe, step)
                if shadow is not None:
                    ring = ring.difference(shadow)
                # Simplified once here, not per layer: a small dilation's arcs (48-gons, sagitta
                # far below the tolerance) would otherwise carry their vertices into every overlay.
                ring_by_factor[factor] = ring.simplify(_SIMPLIFY_TOL, preserve_topology=True)

            etched = [i for i in live if factors[i] > 0]
            removed = shapely.intersection(
                np.array([self.layers[i].polygon for i in etched], dtype=object),
                np.array([ring_by_factor[factors[i]] for i in etched], dtype=object),
            )
            removed_parts = [g for g in removed if not g.is_empty]
            if not removed_parts:
                continue
            # Validity fixes only between substeps, no simplification: Douglas-Peucker trims a
            # curved front by up to its tolerance every time it runs, and where it starts
            # differs from one side of an opening to the other - over forty substeps that
            # drift made a symmetric undercut visibly lopsided. Cleaned once, below.
            total_removed = unary_union(removed_parts).buffer(0)
            new_solid = solid.difference(total_removed).buffer(0)
            # Only a layer the removal touches changes; the others already lie inside
            # `new_solid`, to within the solid's own simplification tolerance.
            shapely.prepare(total_removed)
            for i in live:
                if shapely.intersects(total_removed, self.layers[i].polygon):
                    self.layers[i].polygon = _drop_tiny(self.layers[i].polygon.intersection(new_solid).buffer(0))
            # The next substep's solid is rebuilt from the layers rather than taken as
            # `new_solid`: the per-layer removals don't tile the ring to the last bit, and the
            # hairline slivers `solid - removed` keeps along the layer seams, owned by no layer,
            # would pile up substep after substep.
            solid = self.solid()
        for layer in self.layers:
            if not layer.polygon.is_empty:
                layer.polygon = _drop_tiny(_clean(layer.polygon))

        # The last substep can stop a hair short of a layer's far side, leaving a sub-angstrom film
        # (e.g. on the floor of a mask opening) that a real over-etch would clear - and that would
        # otherwise read as a mask covering the seed. Morphological opening removes only such films.
        for layer in self.layers:
            if layer.polygon.is_empty or recipe.factor_for(materials.get(layer.material)) <= 0:
                continue
            opened = layer.polygon.buffer(-_RESIDUE_NM, join_style="mitre").buffer(_RESIDUE_NM, join_style="mitre")
            layer.polygon = _drop_tiny(_clean(layer.polygon.intersection(opened)))

        # Time-stepping leaves its mark on a slowly-etched surface that a faster neighbour uncovers
        # bit by bit (the underside of a nitride mask as the oxide beneath it is undercut): each
        # substep bites one more `substep * factor` out of the part uncovered so far, so the front
        # that should be a smooth slope comes out as a staircase - or, with round erosion, a row
        # of scallops - of exactly that step height, drawn as a ragged, wavy edge. Straighten each
        # etched layer's *exposed* edges to within its own step height (never more than a couple
        # of monolayers); the edges it shares with other layers are left exactly as they are, so
        # neighbours still fit together without slits or overlaps.
        fastest = max((recipe.factor_for(materials.get(l.material)) for l in self.layers if not l.polygon.is_empty), default=0.0)
        for layer in self.layers:
            factor = recipe.factor_for(materials.get(layer.material))
            if layer.polygon.is_empty or factor <= 0 or factor >= fastest:
                continue  # the fastest layer is never uncovered by anything: its front is already smooth
            others = [l.polygon for l in self.layers if l is not layer and not l.polygon.is_empty]
            interface = unary_union(others) if others else Polygon()
            tol = min(1.5 * substep * factor, _ETCH_SMOOTH_MAX)
            layer.polygon = self._smooth_exposed(layer.polygon, tol, interface)

    def _etch_steps(self, recipe: EtchRecipe, depth_nm: float, factors: list[float], live: list[int]) -> int:
        """How many substeps `etch` runs (see its docstring for what they buy). The default,
        `max(10, depth / 2)` capped at 40, keeps every substep - hence every coupling error -
        near 2 nm. A beam straight down (directional, angle 0) couples far less: its sweep is
        exact for any substep (translations compose), it lights no sidewall, and its shadow is
        cast by overhangs only. A substep then only matters where (a) a slower layer is
        consumed and uncovers what lies below, (b) an overhang is consumed and uncovers what it
        shadowed, or (c) a slower layer not lit from the start is uncovered and begins to recede
        late - by at most a substep times its factor. Without (a) and (b) - no void, no
        downward-facing edge above the wafer floor, no part of a slower layer thinner than what
        it recedes over the whole etch - the substep is sized for (c) alone:
        `_DIRECTIONAL_COUPLING_NM` divided by the fastest factor among the slower layers with a
        top-facing edge that isn't on the exposed surface, and a single substep when there is
        none (a resist mask eroding at 10% is lit from the start: it couples to nothing).
        """
        default = min(40, max(10, round(depth_nm / 2)))
        if recipe.mode is not EtchMode.directional or abs(recipe.angle_deg) > 1e-9:
            return default
        f_max = max(factors[i] for i in live)
        if f_max <= 0:
            return default
        solid = self.solid()
        if solid.is_empty or _has_interior(solid) or self._has_overhang(solid):
            return default
        w = self.domain_width_nm
        slower = [i for i in live if 0 < factors[i] < f_max]
        if slower:
            f_slow = max(factors[i] for i in slower)
            r = 2 * depth_nm * f_slow
            blocked = self._mirror_pad(_robust_union([self.layers[i].polygon for i in slower]), band=r + 2 * _MARGIN)
            window = (-2 * _MARGIN, w + 2 * _MARGIN) if r + 2 * _MARGIN < w else (-w + r, 2 * w - r)
            if _thin_area(blocked, r, window) > _speck_tol(r):
                return default
        shapely.prepare(solid)
        buried = 0.0
        for i in slower:
            for part in (self.layers[i].polygon.geoms if isinstance(self.layers[i].polygon, MultiPolygon) else [self.layers[i].polygon]):
                part = orient(part, sign=1.0)
                for ring in [part.exterior, *part.interiors]:
                    c = np.asarray(ring.coords)
                    d = c[1:] - c[:-1]
                    up = d[:, 0] < -1e-12  # right-hand normal (dy, -dx) pointing up
                    if not up.any():
                        continue
                    mid = (c[1:][up] + c[:-1][up]) / 2
                    if shapely.contains_xy(solid, mid[:, 0], mid[:, 1] + _SPLIT_TOL).any():
                        buried = max(buried, factors[i])
        if buried <= 0:
            return 1
        return min(default, max(1, math.ceil(depth_nm * buried / _DIRECTIONAL_COUPLING_NM)))

    def _has_overhang(self, solid: BaseGeometry) -> bool:
        """Whether `solid` has a downward-facing edge above the wafer floor (an overhanging mask,
        a lifted film): the only thing a vertical beam's shadow is cast by. With no floor,
        the solid's own bottom counts, which is the safe answer."""
        for part in (solid.geoms if isinstance(solid, MultiPolygon) else [solid]):
            part = orient(part, sign=1.0)
            for ring in [part.exterior, *part.interiors]:
                c = np.asarray(ring.coords)
                d = c[1:] - c[:-1]
                down = d[:, 0] > 1e-12  # right-hand normal (dy, -dx) pointing down
                if self.floor_nm is not None:
                    down &= np.maximum(c[1:, 1], c[:-1, 1]) > self.floor_nm + _SPLIT_TOL
                if down.any():
                    return True
        return False

    def _etch_band(self, solid: BaseGeometry, reach_max: float, angle_deg: float | None) -> float | None:
        """How much of each mirror image an etch substep needs (see `_mirror_pad`): the substep's
        reach past the domain edge, plus - for a directional etch at `angle_deg` - how far
        sideways a feature's shadow travels across the structure's height, plus any sealed void
        (an interior ring of `solid`, or a pocket of air open to the edge and closed by its own
        mirror image) whose mirror image the crop would cut: cut open, the void would read as
        reachable air (see `_outside_air`). None when that comes to the whole domain anyway.
        """
        w = self.domain_width_nm
        y_min, y_max = solid.bounds[1], solid.bounds[3]
        band = reach_max + 2 * _MARGIN
        if angle_deg is not None:
            band += (y_max - y_min + reach_max + _MARGIN) * abs(math.tan(math.radians(angle_deg)))
        if band >= w:
            return None
        parts = solid.geoms if isinstance(solid, MultiPolygon) else [solid]
        spans = [(ring.bounds[0], ring.bounds[2]) for part in parts for ring in part.interiors]
        # A pocket of air open to a domain edge but not to the chamber - half a void centred on
        # the edge, an undercut reaching it - is sealed by its own mirror image, as an interior
        # ring of the full mirrored solid would be.
        top = y_max + _MARGIN
        pockets = _polygonal(box(0.0, y_min, w, top).difference(solid))
        for pocket in (pockets.geoms if isinstance(pockets, MultiPolygon) else [pockets]):
            if pocket.is_empty:
                continue
            x0, _y0, x1, y1 = pocket.bounds
            if y1 < top - 1e-9 and (x0 <= 1e-9 or x1 >= w - 1e-9):
                spans.append((x0, x1))
        grown = bool(spans)
        while grown and band < w:
            grown = False
            for a, b in spans:
                if a < band < b:
                    band, grown = b + _MARGIN, True
                if w - b < band < w - a:
                    band, grown = w - a + _MARGIN, True
        return None if band >= w else band

    @staticmethod
    def _tunnel_policy(
        blocked: BaseGeometry,
        reach: float,
        window: tuple[float, float],
        k: int,
        steps: int,
        recession: float,
        hint: int,
    ) -> tuple[int, float, int]:
        """Whether `_reach_air` must dilate the air in geodesic steps to keep it from tunnelling
        through a thin part of `blocked` (an etch factor's slower layers, mirror-padded) - and
        if so, how long a step is safe. Returns `(until, step, hint)`: the verdict holds for
        every substep index below `until`, `step` is the longest dilation that can't cross any
        part of `blocked` (`inf`: none is thin enough, a single dilation is fine), and `hint` is
        the widest probe worth trying next time.

        A part thinner than `reach` with total area above `_speck_tol(reach)` counts as thin
        (see `_thin_area`); anything smaller is a speck the air could only tunnel into by about
        its own size. The slower layers only recede, by at most `recession` from each exposed side
        per substep, so a part thicker than `reach + 2 * recession * n` now stays thicker than
        `reach` for the next `n` substeps: a probe that much wider gives a "thick" verdict good
        for those substeps - every remaining one for a mask that doesn't etch at all. The
        horizon starts at `hint` and is halved until a probe passes; once the plain probe is
        the only one that does, `hint` drops to 1 and wider probes are not tried again (a
        structure with parts between `reach` and a wider probe is the norm when the slower
        layers etch nearly as fast). A thin verdict's step comes from the thinnest part found
        by halving the probe width: a step shorter than it can't cross it, `_GEODESIC_STEP` is
        the floor the engine resolves, and a verdict at that floor can only improve, so it is
        kept for as long as the slower layers take to recede by the reach before probing again.
        """
        tol = _speck_tol(reach)
        remaining = steps - k
        h_max = remaining if recession <= 0 else max(1, min(remaining, math.ceil(reach / recession)))
        h = min(hint, h_max)
        while h > 1:
            if _thin_area(blocked, reach + 2 * h * recession, window) <= tol:
                return k + h, math.inf, min(2 * h, h_max)
            h //= 2
        if _thin_area(blocked, reach, window) <= tol:
            return k + 1, math.inf, 1
        r = reach / 2
        while r > _GEODESIC_STEP:
            if _thin_area(blocked, r, window) <= _speck_tol(r):
                return k + 1, max(_GEODESIC_STEP, 0.9 * r), 1
            r /= 2
        hold = remaining if recession <= 0 else max(1, min(remaining, math.ceil(reach / (2 * recession))))
        return k + hold, _GEODESIC_STEP, 1

    def _reach_air(
        self,
        air: BaseGeometry,
        air_box: BaseGeometry,
        blocked: BaseGeometry,
        reach: float,
        recipe: EtchRecipe,
        step: float = math.inf,
    ) -> BaseGeometry:
        """Everything within `reach` of `air` *without passing through `blocked`* - the region one
        substep of an etch at that reach can remove, for a factor whose slower-etching layers are
        `blocked` (mirror-padded). Isotropic: a round dilation of the air; directional: the air
        swept along the beam.

        A plain dilation would tunnel: a mask thinner than `reach` (a resist eroding away, a stop
        layer, a sub-nm residue) is no obstacle to it, and whatever sits beneath the mask would be
        removed as if the mask weren't there. Padding slower layers to "effectively infinite"
        thickness was the previous answer, but a layer extended upward within its own x-span
        swallows everything above it: the moment an oxide etch reaches the substrate, the whole
        domain is guarded and the undercut under the mask stops dead. Here, when `step` (from
        `_tunnel_policy`) says some part of `blocked` is thinner than `reach`, the dilation is
        done in steps no longer than it, each clipped to the passable region (everything but
        `blocked`): air creeps round a mask, never through it, and a slower layer anywhere -
        above, beneath, buried - bounds how far the faster erosion gets. Otherwise (`step` at
        least `reach`, the usual case) the plain dilation is returned as is: a thick blocker
        can't be tunnelled, and the caller only ever intersects the result with the layers of
        its own factor, which lie in the passable region anyway.

        The stepped case grows the last `_REACH_EPS` of the reach separately, *after* the result
        is cleaned. Where the dilated air is tangent to a layer interface (an undercut creeping
        along the underside of a mask), the arc meets that line in a run of sub-nm segments, and
        `_clean`'s simplification can drop the exact meeting point - leaving the ring a hair
        short of the interface and the layer with a sub-nm "roof" over what should be an open
        trench; a roof is a hole, the solid fills holes, and the etch stops. Overshooting the
        interface by that hair costs nothing (the ring is intersected with each layer separately)
        and makes the ring robust to it. Between two steps only a far finer simplification runs
        (`_GEODESIC_SIMPLIFY`, below the arcs' own sagitta): Douglas-Peucker always cuts a convex
        arc's corners inward, so a coarser one would bias the front, and none at all lets the
        vertex count explode step after step.
        """
        if air.is_empty or reach <= 0:
            return Polygon()
        isotropic = recipe.mode is EtchMode.isotropic

        def grow(geom: BaseGeometry, by: float) -> BaseGeometry:
            if isotropic:
                return geom.buffer(by, quad_segs=12)
            return sweep_union(geom, beam_vector(recipe.angle_deg, by))

        if blocked.is_empty or step >= reach:
            return grow(air, reach)
        eps = min(_REACH_EPS, reach / 2)
        inner = reach - eps
        passable = air_box.difference(blocked)
        n = max(1, math.ceil(inner / step))
        reached = air
        for _ in range(n):
            reached = grow(reached, inner / n).intersection(passable).simplify(_GEODESIC_SIMPLIFY, preserve_topology=True)
        return grow(_clean(reached), eps)

    def _smooth_exposed(self, geom: BaseGeometry, tol: float, interface: BaseGeometry) -> BaseGeometry:
        """`geom` with its *exposed* edges simplified to within `tol` (Douglas-Peucker), every
        vertex that touches `interface` (the other layers), a domain edge or the wafer floor kept
        exactly where it is. See the end of `etch` for why only the exposed edges: a layer's
        shared boundaries must stay bit-identical to its neighbours'. Any area the simplification
        adds is cut back to outside `interface` again, so it can't overlap a neighbour either."""
        if geom.is_empty or tol <= 0:
            return geom
        w = self.domain_width_nm
        floor = self.floor_nm

        def locked(pts: np.ndarray) -> np.ndarray:
            on_edge = (np.abs(pts[:, 0]) < _SPLIT_TOL) | (np.abs(pts[:, 0] - w) < _SPLIT_TOL)
            if floor is not None:
                on_edge |= np.abs(pts[:, 1] - floor) < _SPLIT_TOL
            if interface.is_empty:
                return on_edge
            # A vertex is free to move only if *both* edges it joins are fully exposed. Testing
            # the vertex alone isn't enough: where a neighbour ends partway along an edge (an
            # oxide undercut stopping under a resist's underside), that edge's far vertex touches
            # nothing itself, yet moving it would peel the edge off the neighbour - a hairline
            # slit of "air" the next substep would flood straight into.
            edges = shapely.linestrings(np.stack([pts, np.roll(pts, -1, axis=0)], axis=1))
            touching = shapely.distance(edges, interface) < _SPLIT_TOL
            return on_edge | touching | np.roll(touching, 1)

        def smooth_ring(ring) -> list[tuple[float, float]]:
            pts = np.asarray(ring.coords)[:-1]
            n = len(pts)
            if n < 4:
                return [tuple(p) for p in pts]
            lock = locked(pts)
            if not lock.any():
                simplified = Polygon(pts).simplify(tol, preserve_topology=True)
                return list(simplified.exterior.coords)[:-1] if not simplified.is_empty else [tuple(p) for p in pts]
            if lock.all():
                return [tuple(p) for p in pts]
            start = int(np.argmax(lock))
            order = [(start + k) % n for k in range(n)] + [start]
            out: list[tuple[float, float]] = []
            run: list[int] = [order[0]]
            for idx in order[1:]:
                run.append(idx)
                if lock[idx]:
                    chain = [tuple(pts[i]) for i in run]
                    if len(chain) > 2:
                        chain = list(LineString(chain).simplify(tol, preserve_topology=False).coords)
                    out.extend(chain[:-1])
                    run = [idx]
            return out

        parts = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
        rebuilt = []
        for part in parts:
            shell = smooth_ring(part.exterior)
            holes = [smooth_ring(h) for h in part.interiors]
            if len(shell) < 3:
                continue
            rebuilt.append(Polygon(shell, [h for h in holes if len(h) >= 3]))
        if not rebuilt:
            return geom
        result = _clean(unary_union(rebuilt))
        if not interface.is_empty:
            result = result.difference(interface)
        result = _drop_tiny(_clean(result))
        return result if not result.is_empty else geom

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
