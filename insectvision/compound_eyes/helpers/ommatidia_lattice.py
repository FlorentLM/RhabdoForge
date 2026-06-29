from typing import Callable, Sequence, Optional
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import Voronoi, cKDTree

from insectvision.geometry.polygons import Polygon2D, weighted_polygon_centroids, voronoi_cells


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