import warnings
from typing import Callable, Tuple
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, Voronoi, cKDTree

from insectvision.geometry.linalg import tangent_frames
from insectvision.geometry.hexatic import hexatic_rest_vectors
from insectvision.geometry.neighbours import mean_neighbour_distance, delaunay_edges
from insectvision.geometry.polygons import (
    Polygon2D, polygon_area, weighted_polygon_centroids, voronoi_cells, mirror_across_hull, order_polygon_ccw
)


class OmmatidiaSpacing2D:
    """
    Local inter-ommatidial spacing field for a 2D cloud

    One thin-plate-spline RBF is fit over normalised local spacing

    spacing(points2d) -> target local spacing (spring_relaxation, density_warp)
    density(points2d) -> point number density (lloyd_relaxation)
    density_scale: rescales the target lattice (>1 packs more lenses)

    In 2D density ~1/spacing**2, so spacing shrinks by sqrt(density_scale)
    """

    def __init__(
            self,
            points2d: np.ndarray,
            smoothing: float = 0.1,
            k: int = 6,
            density_scale: float = 1.0,
            clip_norm: Tuple[float, float] = (0.1, 5.0),
            density_exponent: float = 2.0,
    ):

        points2d = np.asarray(points2d, dtype=float)

        self.density_scale = float(density_scale)
        self.density_exponent = float(density_exponent)
        self._clip_lo, self._clip_hi = clip_norm

        tree = cKDTree(points2d)
        spacing = mean_neighbour_distance(tree, points2d, k=k)

        # Reference scale: mean over the inner 80% so that boundary points (inflated spacing, incomplete rings)
        # don't pull the value up
        # (property of the raw cloud so *not* scaled by density_scale)

        p10, p90 = np.percentile(spacing, [10, 90])
        core = (spacing >= p10) & (spacing <= p90)
        self.mean_spacing = float(spacing[core].mean()) if core.any() else float(spacing.mean())

        # TPS is linear in the fitted values so density_scale can be applied at query time
        self._rbf = RBFInterpolator(
            points2d, spacing / self.mean_spacing,
            kernel='thin_plate_spline', smoothing=smoothing,
        )

    def _norm_spacing(self, points2d: np.ndarray) -> np.ndarray:
        """Normalised (mean_spacing = 1), density-scaled, clamped spacing."""

        points2d = np.atleast_2d(np.asarray(points2d, dtype=float))
        s = self._rbf(points2d).ravel() / np.sqrt(self.density_scale)
        # Clamp bc TPS extrapolates wildly outside the hull
        return np.clip(s, self._clip_lo, self._clip_hi)

    def spacing(self, points2d: np.ndarray) -> np.ndarray:
        """Target local spacing at 'points2d', in input units. Pass as spacing_fn."""
        return self._norm_spacing(points2d) * self.mean_spacing

    def density(self, points2d: np.ndarray) -> np.ndarray:
        """Point number density at 'points2d' (~ 1 / spacing**exponent). Pass as density_fn."""
        return 1.0 / self._norm_spacing(points2d) ** self.density_exponent



def hexagonal_grid(spacing: float, angle: float, extent: float) -> np.ndarray:
    """
    Make a regular hexagonal grid.
    """
    b1 = np.array([1.0, 0.0])
    b2 = np.array([np.cos(angle), np.sin(angle)])

    n = int(np.ceil(extent / spacing)) + 1
    ij = np.mgrid[-n:n + 1, -n:n + 1].reshape(2, -1).T
    grid = (ij[:, 0:1] * b1 + ij[:, 1:2] * b2) * spacing

    return grid


def align_grid(grid: np.ndarray, points2d: np.ndarray) -> np.ndarray:
    """
    Align a hex grid (rigid transform) to match a point cloud.
    """

    tree_raw = cKDTree(points2d)
    domain = Polygon2D.from_points(points2d)

    def loss(p):
        cos, sin = np.cos(p[2]), np.sin(p[2])
        mat = np.array([[cos, -sin], [sin, cos]])
        t = grid @ mat.T + p[:2]

        t_tree = cKDTree(t)
        d_fwd, _ = t_tree.query(points2d)
        mask = domain.inside(t)

        if np.any(mask):
            d_bwd, _ = tree_raw.query(t[mask])
        else:
            d_bwd = np.array([1e3])

        return 2.0 * np.mean(d_fwd ** 2) + np.mean(d_bwd ** 2)

    res = minimize(
        loss,
        x0=[0, 0, 0],
        bounds=[
            (-0.1, 0.1), (-0.1, 0.1),
            (-np.pi / 4, np.pi / 4),
        ],
    )

    pos, rot = res.x[:2], res.x[2]
    c, s = np.cos(rot), np.sin(rot)
    R_mat = np.array([[c, -s], [s, c]])
    aligned = grid @ R_mat.T + pos

    return aligned


def lloyd_relaxation(
        points2d: np.ndarray,
        density_fn: Callable,
        boundary: ConvexHull,
        max_iter: int = 20,
        convergence_tol: float = 1e-6,
        relaxation_factor: float = 0.85,
        verbose: bool = False,
) -> np.ndarray:
    """
    Voronoi relaxation (Lloyd's algorithm, 10.1109/TIT.1982.1056489).

    Args:
        points2d: (N, 2)
        density_fn: (M, 2) -> (M,)
        boundary: ConvexHull of the target domain
        max_iter: int
        convergence_tol: Stop when mean displacement < this
        relaxation_factor: Under-relaxation factor in (0, 1]
            1.0 = full step to centroid (that can oscillate)
            0.5-0.8 = smoother convergence
        verbose: Print info
    """

    points2d = points2d.copy()
    n_real = len(points2d)

    # get data domain (to place the ghost ring fallback)
    domain_center = np.mean(boundary.points[boundary.vertices], axis=0)
    domain_radius = np.max(np.linalg.norm(boundary.points[boundary.vertices] - domain_center, axis=1))

    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    r = domain_radius * 3.0
    ghosts_points = domain_center + np.column_stack([np.cos(angles), np.sin(angles)]) * r

    # estim initial spacing to define mirror depth and boundary expansion
    tree = cKDTree(points2d)
    d, _ = tree.query(points2d, k=2)
    mean_nn = float(np.mean(d[:, 1]))

    mirror_depth = mean_nn * 3.0

    # Expand the hard boundary constraints by 1.5x mean spacing
    # (pure safety net, the mirrored points do the actual bounding)
    expanded_equations = boundary.equations.copy()
    expanded_equations[:, 2] -= mean_nn * 1.5

    for it in range(max_iter):
        mirrored_ghosts = mirror_across_hull(points2d, boundary.equations, mirror_depth)
        ghosts = mirrored_ghosts if len(mirrored_ghosts) > 0 else ghosts_points

        all_pts = np.vstack([points2d, ghosts])
        try:
            vor = Voronoi(all_pts)
        except Exception as exc:
            if verbose:
                print(f"  Lloyd iter {it}: Voronoi failed ({exc}), stopping.")
            break

        # Voronoi edges between a point and its reflection lie exactly on the true
        # boundary, so we don't want the hard clipper to interfere.
        new_pts = weighted_polygon_centroids(
            voronoi_cells(vor, n_real),
            fallback=points2d,
            weight_fn=density_fn,
            clip_equations=expanded_equations,
        )
        # TODO: Could use the polygon_centroid function instead

        step = new_pts - points2d
        points_next = points2d + relaxation_factor * step

        disp = np.linalg.norm(relaxation_factor * step, axis=1)

        if verbose:
            print(f"  Lloyd iter {it:3d}:  mean d = {disp.mean():.6f}, max d = {disp.max():.6f}")

        points2d = points_next

        if disp.mean() < convergence_tol:
            if verbose:
                print(f"  Converged at iteration {it}.")
            break

    return points2d


def density_warp(
        points2d: np.ndarray,
        spacing_fn: Callable,
        reference_spacing: float,
        exponent: float = 1.0,
) -> np.ndarray:
    """
    Warp a point set so that local spacing matches a target density field.

    Args:
        points2d: (N, 2)
        spacing_fn: (M, 2) -> (M,) Target local spacing at each point
        reference_spacing (float): The spacing of the uniform input grid
        exponent (float): Warp strength. Lower values keep more points at the boundary
    """
    pts = points2d.copy()

    s = spacing_fn(pts)     # centre of compression
    weights = 1.0 / np.maximum(s, 1e-12)
    centroid = np.average(pts, axis=0, weights=weights)

    disp = pts - centroid

    scale = (s / reference_spacing) ** exponent
    pts = centroid + disp * scale[:, None]

    return pts


def spring_relaxation(
        points2d,
        spacing_fn,
        theta_fn=None,
        confidence_fn=None,
        max_iter=120,
        retriangulate_every=0,
        max_length_factor=1.8,
        force_cap=2.0,
        dt=0.05,
        convergence_tol=1e-3,
        verbose=False,
    ):
    """
    Density adaptation via spring relaxation.

    Each Delaunay edge acts as a spring. If theta_fn is given, the rest state
    is a vector whose direction is the edge's bearing snapped to the nearest
    local hexatic axis and whose length is the local target spacing

    -> edges are pulled toward both the right length and the right orientation.

    Without theta_fn the rest state is a scalar length only.

    With an orientation field, topology must follow it (T1 transitions), so set retriangulate_every nedds to be >0.

    Args:
        points2d: (N, 2) initial positions (a roughly hex grid)
        spacing_fn: (M, 2) -> (M,) target local spacing
        theta_fn: (M, 2) -> (M,) local hexatic-axis angle [rad], or None
        retriangulate_every: recompute Delaunay edges every k iters (0 = never)
        dt: step size (fraction of mean spacing)
        convergence_tol: relative early-stop (fraction of local spacing)
        verbose: print progress
    """
    pts = points2d.copy()
    edges = delaunay_edges(pts, max_length_factor=max_length_factor)

    for it in range(max_iter):
        if retriangulate_every and it > 0 and it % retriangulate_every == 0:
            edges = delaunay_edges(pts, max_length_factor=max_length_factor)

        mid = 0.5 * (pts[edges[:, 0]] + pts[edges[:, 1]])
        target_lengths = spacing_fn(mid).ravel()
        current_diff = pts[edges[:, 1]] - pts[edges[:, 0]]
        cur_len = np.maximum(np.linalg.norm(current_diff, axis=1, keepdims=True), 1e-12)
        iso_rest = target_lengths[:, None] * current_diff / cur_len  # length-only rest

        if theta_fn is None:
            rest = iso_rest
        else:
            ax_rest = hexatic_rest_vectors(current_diff, theta_fn(mid), target_lengths)

            if confidence_fn is not None:
                w = np.clip(confidence_fn(mid), 0.0, 1.0)[:, None]  # taper to isotropic
                rest = w * ax_rest + (1.0 - w) * iso_rest
            else:
                rest = ax_rest

        fpe = (current_diff - rest) / target_lengths[:, None]
        fmag = np.linalg.norm(fpe, axis=1, keepdims=True)
        force_per_edge = np.where(fmag > force_cap, fpe * force_cap / fmag, fpe)

        forces = np.zeros_like(pts)
        np.add.at(forces, edges[:, 0], force_per_edge)
        np.add.at(forces, edges[:, 1], -force_per_edge)

        deg = np.zeros(len(pts))
        np.add.at(deg, edges[:, 0], 1)
        np.add.at(deg, edges[:, 1], 1)
        forces /= np.maximum(deg, 1)[:, None]

        node_scale = spacing_fn(pts).ravel()
        disp = dt * node_scale[:, None] * forces
        norms = np.linalg.norm(disp, axis=1, keepdims=True)
        cap = 0.5 * node_scale[:, None]
        applied = np.where(norms > cap, disp * cap / norms, disp)
        pts += applied

        mean_disp = float(np.linalg.norm(applied, axis=1).mean())
        ref = float(np.median(node_scale))

        if verbose and (it % 10 == 0 or it == max_iter - 1):
            print(f"  spring iter {it:3d}:  mean |disp|/spacing = {mean_disp / max(ref, 1e-12):.5f}")

        if mean_disp < convergence_tol * ref:
            if verbose:
                print(f"  spring converged at iter {it}.")
            break

    return pts


def voronoi_estimation(
        positions,
        directions,
        k=18,
        packing=1.0,
        n_iter=1,
        shell_factor=1.5,
        min_ring=4,
        fill_sweeps=6
    ):
    """
    Per-ommatidium facet diameter from the local Voronoi cell area on the eye surface.

    First-ring spacing is taken over a *shell* (neighbours within shell_factor x the
    nearest) rather than the kNN-6, so boundary lenses aren't biased high by second-ring neighbours.

    Boundary lenses are filled from their interior neighbours.

    Args:
        positions: (N, 3) lens world positions.
        directions: (N, 3) lens optical axes (unit, the local surface normal).
        k: neighbours used to bound each local cell (>= ~12 so the first ring is enclosed).
        packing: scale on the hex flat-to-flat diameter (1.0 = facets tile edge-to-edge).
        n_iter: neighbour-median smoothing passes applied to the result.
    """
    positions = np.asarray(positions, dtype=float)
    directions = np.asarray(directions, dtype=float)
    N = len(positions)

    tree = cKDTree(positions)
    kq = max(0, min(k + 1, N))
    dist, idx = tree.query(positions, k=kq)
    nn = dist[:, 1:]
    ring_idx = idx[:, 1:min(7, kq)]

    shell = np.where(nn <= shell_factor * dist[:, 1:2], nn, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        ref = np.nanmedian(shell, axis=1)

    ref = np.where(np.isfinite(ref), ref, dist[:, 1])
    n_shell = np.sum(np.isfinite(shell), axis=1)
    boundary = n_shell < min_ring

    D = np.full(N, np.nan, dtype=float)

    # Per-lens tangent basis
    right_b, up_b = tangent_frames(directions)

    for i in range(N):
        nb = idx[i, 1:]
        u, v = right_b[i], up_b[i]
        rel = positions[nb] - positions[i]
        pts = np.vstack([[0.0, 0.0], np.column_stack([rel @ u, rel @ v])])
        fb = packing * ref[i]

        try:
            vor = Voronoi(pts)
            region = vor.regions[vor.point_region[0]]
            if region and (-1 not in region):
                poly = order_polygon_ccw(vor.vertices[region])
                area = polygon_area(poly)
                dv = packing * np.sqrt(2.0 * area / np.sqrt(3.0))
                if 0.6 * fb < dv < 1.5 * fb:
                    D[i] = dv
        except Exception:
            pass

        if np.isnan(D[i]) and not boundary[i]:
            D[i] = fb  # interior fallback

    D[boundary] = np.nan  # Never trust boundary fallback

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for _ in range(fill_sweeps):  # propagate interior values outward
            med = np.nanmedian(np.concatenate([D[:, None], D[ring_idx]], axis=1), axis=1)
            D[boundary] = med[boundary]

    D = np.where(np.isfinite(D), D, packing * ref)  # last resort

    for _ in range(int(n_iter)):
        D = np.median(np.concatenate([D[:, None], D[ring_idx]], axis=1), axis=1)

    return D.astype(np.float32)
