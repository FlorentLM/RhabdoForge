from typing import Tuple, Optional, Sequence, Callable, List, Union
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import ConvexHull, Delaunay, cKDTree, Voronoi


class Polygon2D:
    """
    A 2D region with conveniences:

    - Delaunay triangulation for fast inside tests
    - cKDTree of the raw cloud for buffered tests
    - ConvexHull
    - Boundary polyline (for plotting)
    - Area
    """

    def __init__(self,
            boundary: ArrayLike,
            raw_points2d: ArrayLike,
            hull: 'ConvexHull'
        ):

        self.boundary: np.ndarray = np.asarray(boundary, dtype=float)   # ordered polyline (M, 2)
        self.hull: 'ConvexHull' = hull
        self.equations: np.ndarray = np.asarray(hull.equations, dtype=float)
        self.area: float = polygon_area(self.boundary)

        self._delaunay: 'Delaunay' = Delaunay(self.boundary)
        self._tree: 'cKDTree' = cKDTree(np.asarray(raw_points2d, dtype=float))

    @property
    def tree(self) -> 'cKDTree':
        return self._tree

    @classmethod
    def from_points(cls, points2d: ArrayLike, smooth: bool = False, n_boundary: int = 300) -> 'Polygon2D':

        points2d = np.asarray(points2d, dtype=float)
        hull = ConvexHull(points2d)

        if smooth:
            boundary = smooth_hull(points2d, hull, n=n_boundary)
            hull = ConvexHull(boundary)  # hull of the smoothed ring -> ghosts/equations
        else:
            boundary = points2d[hull.vertices]

        return cls(boundary, points2d, hull)

    def inside(self, points2d: ArrayLike, buffer: float = 0.0) -> np.ndarray:

        points2d = np.atleast_2d(np.asarray(points2d, dtype=float))
        inside = self._delaunay.find_simplex(points2d) >= 0
        if buffer > 0:
            out = np.where(~inside)[0]
            if len(out):
                d, _ = self._tree.query(points2d[out])
                inside[out] = d < buffer
        return inside

    def signed_distance(self, points2d: ArrayLike) -> np.ndarray:
        """Signed distance to the hull faces (<0 inside)."""
        return signed_distance(points2d=points2d, domain=self)


# Some indexing helpers

def pack_edge_keys(i: ArrayLike, j: ArrayLike, big: int) -> np.ndarray:
    """
    Encode undirected edges (i, j) as int64 keys: lo*big + hi, with lo=min, hi=max.
    'big' must exceed the largest node id so the mapping is bijective.
    """
    i = np.asarray(i, dtype=np.int64)
    j = np.asarray(j, dtype=np.int64)
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    return lo * np.int64(big) + hi


def unpack_edge_keys(keys: ArrayLike, big: int) -> Tuple[np.ndarray, np.ndarray]:
    """Inverse of pack_edge_keys -> (lo, hi)."""
    keys = np.asarray(keys, dtype=np.int64)
    big = np.int64(big)
    return keys // big, keys % big


def polygon_area(points2d: ArrayLike, signed: bool = False) -> float:
    """
    Shoelace area of a 2D polygon whose vertices are given in order.

    Absolute area unless 'signed' is True (positive for CCW winding).
    Degenerate polygons (< 3 vertices) have zero area.
    """
    p = np.asarray(points2d, dtype=np.float64)
    if p.shape[0] < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    a = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(a) if signed else float(abs(a))


def fan_decompose(points2d: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """
    Triangle-fan decomposition of an ordered simple polygon.

    Fans from vertex 0, returns the (centroid, area) of each fan triangle.
    Degenerate polygons (< 3 vertices) give empty arrays.

    Returns:
        centroids: (T, 2), triangle centroids, T = len(poly) - 2
        areas: (T,), triangle areas
    """
    p = np.asarray(points2d, dtype=np.float64)
    if p.shape[0] < 3:
        return np.zeros((0, 2)), np.zeros(0)

    v0 = p[0]
    e1, e2 = p[1:-1] - v0, p[2:] - v0
    areas = 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])
    centroids = (v0 + p[1:-1] + p[2:]) / 3.0
    return centroids, areas


def triangle_circumradii(points: np.ndarray, simplices: np.ndarray) -> np.ndarray:
    """Calculation of triangle circumradii in 2D or 3D."""
    pts = np.asarray(points)
    a, b, c = pts[simplices[:, 0]], pts[simplices[:, 1]], pts[simplices[:, 2]]

    # Side lengths
    ab = np.linalg.norm(a - b, axis=-1)
    bc = np.linalg.norm(b - c, axis=-1)
    ca = np.linalg.norm(c - a, axis=-1)

    areas = triangle_areas(a, b, c)
    return (ab * bc * ca) / (4.0 * areas + 1e-300)


def find_boundary_indices(simplices: np.ndarray, n_points: int) -> np.ndarray:
    """Finds indices of points on the boundary of a triangle mesh (edges used once)."""

    # Sort edges so (i, j) and (j, i) are identical
    edges = np.sort(np.concatenate([
        simplices[:, [0, 1]],
        simplices[:, [1, 2]],
        simplices[:, [2, 0]]
    ]), axis=1)

    big = int(n_points + 1)
    keys = pack_edge_keys(edges[:, 0], edges[:, 1], big)
    unique_keys, counts = np.unique(keys, return_counts=True)
    boundary_keys = unique_keys[counts == 1]
    lo, hi = unpack_edge_keys(boundary_keys, big)

    return np.unique(np.stack([lo, hi]))


def triangle_areas(a: ArrayLike, b: ArrayLike, c: ArrayLike) -> np.ndarray:
    """
    Areas of triangles from their three vertex arrays.
    2D uses the scalar cross (shoelace), 3D uses half the cross-product norm.

    Args:
        - a, b, c: each (..., D) with D in {2, 3}
    """
    a, b, c = (np.asarray(a, dtype=np.float64),
               np.asarray(b, dtype=np.float64),
               np.asarray(c, dtype=np.float64))
    e1, e2 = b - a, c - a

    if a.shape[-1] == 2:
        return 0.5 * np.abs(e1[..., 0] * e2[..., 1] - e1[..., 1] * e2[..., 0])
    if a.shape[-1] == 3:
        return 0.5 * np.linalg.norm(np.cross(e1, e2), axis=-1)

    raise ValueError(f'vertices must be 2D or 3D, got last-axis size {a.shape[-1]}')


def polygon_centroid(points2d: ArrayLike) -> np.ndarray:
    """
    Area-weighted centroid of an ordered simple polygon
    (true centre of mass, not the vertex mean).

    Falls back to the vertex mean for a zero-area or degenerate polygon.
    """
    centroids, areas = fan_decompose(points2d)
    total = areas.sum()
    if total <= 1e-14:
        p = np.asarray(points2d, dtype=np.float64)
        return p.mean(axis=0) if p.shape[0] else np.full(2, np.nan)
    return (centroids * areas[:, None]).sum(axis=0) / total


def weighted_polygon_centroids(
        cells: Sequence[Optional[np.ndarray]],
        fallback: ArrayLike,
        weight_fn: Optional[Callable] = None,
        clip_equations: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Weighted centroids of a set of (optionally clipped) ordered polygons.

    Each cell is fan-decomposed, each fan triangle is weighted by its area times
    'weight_fn' (evaluated at the triangle centroid), and the cell's result is the
    weighted mean of those centroids.

    Fallback per cell i: weighted centroid -> (if zero weight) polygon vertex mean -> (if no usable polygon) fallback[i]

    Args:
        cells: length-N sequence of (n_i, 2) ordered vertex arrays
        fallback: (N, 2) default position per cell (e.g. the current points)
        weight_fn: (M, 2) -> (M,) weights at query points
            None -> area only (= geometric centroid)
        clip_equations: Optional, half-plane rows, each cell is clipped against
            them (Sutherland-Hodgman) before decomposition.
    """
    fallback = np.asarray(fallback, dtype=np.float64)
    n = len(cells)
    if fallback.shape[0] != n:
        raise ValueError(f"fallback has {fallback.shape[0]} rows but there are {n} cells")

    out = fallback.copy()
    poly_mean = np.full((n, 2), np.nan)  # per valid-but-zero-weight cell

    centroids_all, areas_all, owner_all = [], [], []
    for i, cell in enumerate(cells):
        if cell is None or len(cell) < 3:
            continue

        poly = cell if clip_equations is None else clip_polygon(cell, clip_equations)
        if len(poly) < 3:
            continue

        poly_mean[i] = polygon_centroid(poly)

        cents, areas = fan_decompose(poly)
        keep = areas > 1e-14
        if keep.any():
            centroids_all.append(cents[keep])
            areas_all.append(areas[keep])
            owner_all.append(np.full(int(keep.sum()), i))

    has_cell = np.isfinite(poly_mean[:, 0])
    out[has_cell] = poly_mean[has_cell]  # default = polygon vertex mean
    if not centroids_all:
        return out

    C = np.vstack(centroids_all)
    owner = np.concatenate(owner_all)
    w = np.concatenate(areas_all)
    if weight_fn is not None:
        w = w * np.asarray(weight_fn(C)).ravel()

    den = np.bincount(owner, weights=w, minlength=n)
    num = np.stack([np.bincount(owner, weights=w * C[:, 0], minlength=n),
                    np.bincount(owner, weights=w * C[:, 1], minlength=n)], axis=1)
    good = den > 1e-16
    out[good] = num[good] / den[good, None]  # override with weighted centroid
    return out


def order_polygon_ccw(points2d: np.ndarray) -> np.ndarray:
    """
    Order polygon vertices counter-clockwise about their centroid.

    For already-ordered input (e.g. a scipy Voronoi region) this only changes the
    starting vertex, use it to make a raw vertex set safe for the shoelace / fan
    routines, which assume ordered vertices.
    """
    p = np.asarray(points2d, dtype=np.float64)
    if p.shape[0] < 3:
        return p
    c = p.mean(axis=0)
    order = np.argsort(np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0]))
    return p[order]


def clip_polygon(points2d: Sequence[ArrayLike], hull_equations: ArrayLike) -> np.ndarray:
    """
    Clip a convex polygon against a convex region (Sutherland-Hodgman).

    'hull_equations' are scipy ConvexHull half-plane rows [nx, ny, offset],
    a point is interior where (normal . x + offset) <= 0.
    https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm#Pseudocode
    """
    out = [np.asarray(p, dtype=np.float64) for p in points2d]
    hull_equations = np.asarray(hull_equations, dtype=np.float64)

    for eq in hull_equations:
        normal, offset = eq[:2], eq[2]

        if len(out) < 3:
            return np.zeros((0, 2))

        inp = out
        out = []
        n_verts = len(inp)

        for i in range(n_verts):
            cur = inp[i]
            nxt = inp[(i + 1) % n_verts]
            d_cur = normal @ cur + offset
            d_nxt = normal @ nxt + offset

            if d_cur <= 0:
                out.append(cur)
                if d_nxt > 0:
                    t = d_cur / (d_cur - d_nxt)
                    out.append(cur + t * (nxt - cur))
            elif d_nxt <= 0:
                t = d_cur / (d_cur - d_nxt)
                out.append(cur + t * (nxt - cur))

    return np.array(out) if out else np.zeros((0, 2))


# Voronoi / hull routines

def voronoi_cells(vor: 'Voronoi', n_cells: Optional[int] = None) -> List[np.ndarray]:
    """
    Ordered vertex arrays for the first 'n_cells' input points of a scipy Voronoi
    (all of them if None). Unbounded or empty cells become None.
    """
    n = vor.point_region.shape[0] if n_cells is None else int(n_cells)
    cells = []
    for i in range(n):
        region = vor.regions[vor.point_region[i]]
        if not region or -1 in region:
            cells.append(None)
        else:
            cells.append(vor.vertices[region])

    return cells


def smooth_hull(
        points2d: ArrayLike,
        hull: 'ConvexHull' = None,
        n: int = 300,
        smoothing: float = 0.0,
) -> np.ndarray:
    """
    Smooth a convex hull into a closed periodic cubic spline.
    Returns (n, 2) points along the smoothed boundary.
    'smoothing' is splprep's 's' parameter (so 0 = pass by all vertices, >0 makes it miss some).
    """
    from scipy.interpolate import splprep, splev

    points2d = np.asarray(points2d, dtype=np.float32)
    if hull is None:
        hull = ConvexHull(points2d)

    coords = points2d[hull.vertices]
    coords = np.vstack([coords, coords[0]])   # Close the ring
    tck, _ = splprep([coords[:, 0], coords[:, 1]], s=smoothing, per=True)
    u = np.linspace(0.0, 1.0, n)
    x, y = splev(u, tck)
    return np.column_stack([x, y])


def signed_distance(points2d: ArrayLike, domain: Union['Polygon2D', 'ConvexHull']) -> np.ndarray:
    """
    Signed distance to the nearest convex-hull face (<0 inside, >0 outside).
    """
    points2d = np.atleast_2d(np.asarray(points2d, dtype=np.float64))
    eq = domain.equations
    return (points2d @ eq[:, :2].T + eq[:, 2]).max(axis=1)


def resample_contour(boundary: ArrayLike, spacing_fn: Callable) -> np.ndarray:
    """
    Walk a closed contour and drop a point every local target spacing.

    Greedy arc-length walk: spacing_fn is the step taken from point, so the ring
    is dense where the target spacing is small and sparse where it is large.

    Returns:
        ordered ring of boundary points (R, 2)
    """
    b = np.asarray(boundary, dtype=np.float64)

    closed = np.vstack([b, b[:1]])
    seg = np.diff(closed, axis=0)
    seglen = np.linalg.norm(seg, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seglen)])

    total = float(s[-1])
    if total <= 0:
        return b.copy()

    def pos_at(t):
        t = t % total
        k = int(np.searchsorted(s, t, side='right') - 1)
        k = min(max(k, 0), len(seglen) - 1)
        f = (t - s[k]) / max(seglen[k], 1e-12)
        return closed[k] + f * seg[k]

    out, t, guard = [], 0.0, 0
    guard_max = int(50 * total / max(np.median(seglen), 1e-9)) + 1000
    while t < total and guard < guard_max:
        p = pos_at(t)
        out.append(p)
        step = float(np.ravel(spacing_fn(p[None]))[0])
        t += max(step, total * 1e-4)
        guard += 1

    ring = np.asarray(out)
    # Drop last point if the walk wrapped back to the first
    if len(ring) > 2:
        first_step = float(np.ravel(spacing_fn(ring[:1]))[0])
        if np.linalg.norm(ring[-1] - ring[0]) < 0.5 * first_step:
            ring = ring[:-1]
    return ring