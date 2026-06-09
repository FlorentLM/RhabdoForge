import numpy as np
from scipy.spatial import ConvexHull, Delaunay, cKDTree


def polygon_area(polygon, signed: bool = False) -> float:
    """
    Shoelace area of a 2D polygon whose vertices are given in order.

    Absolute area unless 'signed' is True (positive for CCW winding).
    Degenerate polygons (< 3 vertices) have zero area.
    """
    p = np.asarray(polygon, dtype=np.float64)
    if p.shape[0] < 3:
        return 0.0
    x, y = p[:, 0], p[:, 1]
    a = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return float(a) if signed else float(abs(a))


def clip_polygon(vertices: np.ndarray, hull_equations: np.ndarray) -> np.ndarray:
    """
    Clip a convex polygon against a convex region (Sutherland-Hodgman).

    'hull_equations' are scipy ConvexHull half-plane rows [nx, ny, offset],
    a point is interior where (normal . x + offset) <= 0.
    https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm#Pseudocode
    """
    out = list(vertices)

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


def smooth_hull(
        pts_2d: np.ndarray,
        hull: ConvexHull = None,
        n: int = 300,
        smoothing: float = 0.0,
) -> np.ndarray:
    """
    Smooth a convex hull into a closed periodic cubic spline.
    Returns (n, 2) points along the smoothed boundary.
    'smoothing' is splprep's 's' parameter (so 0 = pass by all vertices, >0 makes it miss some).
    """
    from scipy.interpolate import splprep, splev

    pts_2d = np.asarray(pts_2d, dtype=float)
    if hull is None:
        hull = ConvexHull(pts_2d)

    coords = pts_2d[hull.vertices]
    coords = np.vstack([coords, coords[0]])   # Close the ring
    tck, _ = splprep([coords[:, 0], coords[:, 1]], s=smoothing, per=True)
    u = np.linspace(0.0, 1.0, n)
    x, y = splev(u, tck)
    return np.column_stack([x, y])


class Polygon2D:
    """
    A 2D region with conveniences:

    - Delaunay triangulation for fast inside tests
    - cKDTree of the raw cloud for buffered tests
    - ConvexHull (Lloyd needs its .points / .vertices / .equations)
    - Boundary polyline (for plotting)
    - Area
    """

    def __init__(self, boundary_pts: np.ndarray, raw_pts: np.ndarray, hull: ConvexHull):
        self.boundary = np.asarray(boundary_pts, dtype=float)   # (M, 2) ordered polyline
        self.hull = hull   # scipy ConvexHull (for Lloyd)
        self.area = polygon_area(self.boundary)
        self._delaunay = Delaunay(self.boundary)
        self._tree = cKDTree(np.asarray(raw_pts, dtype=float))

    @classmethod
    def from_points(cls, pts_2d: np.ndarray, smooth: bool = False,
                    n_boundary: int = 300) -> 'Polygon2D':

        pts_2d = np.asarray(pts_2d, dtype=float)
        hull = ConvexHull(pts_2d)

        if smooth:
            boundary = smooth_hull(pts_2d, hull, n=n_boundary)
            hull = ConvexHull(boundary)  # hull of the smoothed ring -> Lloyd ghosts/equations
        else:
            boundary = pts_2d[hull.vertices]

        return cls(boundary, pts_2d, hull)

    def inside(self, points: np.ndarray, buffer: float = 0.0) -> np.ndarray:

        points = np.atleast_2d(np.asarray(points, dtype=float))
        inside = self._delaunay.find_simplex(points) >= 0
        if buffer > 0:
            out = np.where(~inside)[0]
            if len(out):
                d, _ = self._tree.query(points[out])
                inside[out] = d < buffer
        return inside