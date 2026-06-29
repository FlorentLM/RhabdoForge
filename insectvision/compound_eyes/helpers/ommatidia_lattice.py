import warnings
from typing import Callable, Sequence, Optional
import numpy as np
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, Voronoi, cKDTree

from insectvision.geometry.linalg import tangent_frames
from insectvision.geometry.hexatic import hexatic_rest_vectors
from insectvision.geometry.neighbours import delaunay_edges
from insectvision.geometry.polygons import Polygon2D, weighted_polygon_centroids, voronoi_cells, mirror_across_hull


def hexagonal_grid(
        spacing: float = 1.0,
        angles: float | Sequence[float] = np.pi / 3,
        extent: float = 10.0
    ) -> np.ndarray:
    """
    Make a hexagonal grid with exact coverage for any geometry.

    Args:
        spacing: The length of the first basis vector (b1).
        angles:
            - If float: The angle between b1 and b2 (standard hex = pi/3).
            - If 3-item list: The three internal angles (alpha, beta, gamma) of the
              lattice triangle. Sum is normalised to 180 degrees.
        extent: The radius of the circular domain to cover.
    """

    if np.isscalar(angles):
        # Standard case: equilateral triangle sides if angle is 60 deg
        b1 = np.array([spacing, 0.0])
        b2 = np.array([spacing * np.cos(angles), spacing * np.sin(angles)])

    else:
        # Squished case: triangle with angles a, b, c
        angles = np.atleast_1d(angles).astype(float)
        if len(angles) != 3:
            raise ValueError('Provide 3 angles for a squished lattice.')

        angles = angles * (np.pi / np.sum(angles))
        a, b, c = angles

        s2 = spacing
        s1 = s2 * np.sin(a) / np.sin(b)

        b1 = np.array([s2, 0.0])
        b2 = np.array([s1 * np.cos(c), s1 * np.sin(c)])

    B = np.column_stack([b1, b2])

    try:
        inv_B = np.linalg.inv(B)
    except np.linalg.LinAlgError:
        raise ValueError('Lattice angles result in a degenerate (collinear) basis.')

    n1 = int(np.ceil(extent * np.linalg.norm(inv_B[0, :])))
    n2 = int(np.ceil(extent * np.linalg.norm(inv_B[1, :])))

    i_range = np.arange(-n1, n1 + 1)
    j_range = np.arange(-n2, n2 + 1)
    ii, jj = np.meshgrid(i_range, j_range)

    indices = np.stack([ii.ravel(), jj.ravel()], axis=0)
    grid = (B @ indices).T

    dist_sq = np.sum(grid ** 2, axis=1)
    mask = dist_sq <= (extent ** 2)

    return grid[mask]


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
        fixed_mask: Optional[np.ndarray] = None,
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
        max_iter: int
        convergence_tol: Stop when mean displacement < this
        relaxation_factor: Under-relaxation factor in (0, 1]
            1.0 = full step to centroid (that can oscillate)
            0.5-0.8 = smoother convergence
        fixed_mask: (N,) bool, optional. Points where True are never moved and act
            as fixed walls bounding their neighbours' Voronoi cells.
        verbose: Print info
    """

    points2d = points2d.copy()
    move = ~np.asarray(fixed_mask, dtype=bool)

    for it in range(max_iter):

        vor = Voronoi(points2d)
        cells = voronoi_cells(vor, len(points2d))
        new_pts = weighted_polygon_centroids(cells, fallback=points2d, weight_fn=density_fn)

        step = new_pts - points2d
        step[~move] = 0.0

        points2d = points2d + relaxation_factor * step
        if np.linalg.norm(relaxation_factor * step[move], axis=1).mean() < convergence_tol:
            if verbose:
                print(f"  Llyod converged at iteration {it}.")
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
        k: int=18,
        packing: float= 1.0,
        n_iter: int=1,
        shell_factor: float= 1.5,
        min_ring: int=4,
        fill_sweeps: int=6,
        n_jobs: int = -1
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

    def _worker_voronoi_dv(pts_2d, packing, ref_i):
        """
        pts_2d: (k, 2) array of neighbour projections
        packing: float
        ref_i: local spacing reference for clamping
        """
        try:
            vor = Voronoi(pts_2d)

            # Get region for the first point (which is our center 0,0)
            region_idx = vor.point_region[0]
            region = vor.regions[region_idx]

            # Check if cell is bounded
            if region and (-1 not in region):
                # Scipy Voronoi regions are already ordered, so shoelace logic from polygons.py is inlined
                v = vor.vertices[region]

                # Shoelace area (inlined)
                x = v[:, 0]
                y = v[:, 1]
                area = 0.5 * np.abs(
                    np.dot(x[:-1], y[1:]) + x[-1] * y[0] -
                    (np.dot(y[:-1], x[1:]) + y[-1] * x[0])
                )

                dv = packing * np.sqrt(2.0 * area / np.sqrt(3.0))

                fb = packing * ref_i
                if 0.6 * fb < dv < 1.5 * fb:
                    return dv
        except Exception:
            pass
        return np.nan

    # Spacing reference
    nn = dist[:, 1:]

    shell = np.where(nn <= shell_factor * dist[:, 1:2], nn, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        ref = np.nanmedian(shell, axis=1)

    ref = np.where(np.isfinite(ref), ref, dist[:, 1])

    # Boundary detection
    n_shell = np.sum(np.isfinite(shell), axis=1)
    boundary = n_shell < min_ring

    # Generate basis for every point
    right_b, up_b = tangent_frames(directions)

    rel_pos = positions[idx] - positions[:, np.newaxis, :]

    u_coords = np.einsum('nij,nj->ni', rel_pos, right_b)
    v_coords = np.einsum('nij,nj->ni', rel_pos, up_b)

    all_pts_2d = np.stack([u_coords, v_coords], axis=-1)

    # Only process non-boundary points
    active_indices = np.where(~boundary)[0]

    results = Parallel(n_jobs=n_jobs)(
        delayed(_worker_voronoi_dv)(
            all_pts_2d[i],
            packing,
            ref[i]
        ) for i in active_indices
    )

    D = np.full(N, np.nan, dtype=float)
    D[active_indices] = results

    # Propagate interior values to boundary
    ring_idx = idx[:, 1:min(7, kq)]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        for _ in range(fill_sweeps):
            # Look at neighbour values (including the ones just filled)
            neighbor_vals = D[ring_idx]
            med = np.nanmedian(neighbor_vals, axis=1)   # Median of neighbours for boundary points
            mask = boundary & np.isnan(D)     # only update boundary points that are still NaN
            D[mask] = med[mask]

    D = np.where(np.isfinite(D), D, packing * ref)  # last resort

    # Smoothing
    for _ in range(int(n_iter)):
        D = np.median(np.concatenate([D[:, None], D[ring_idx]], axis=1), axis=1)

    return D.astype(np.float32)
