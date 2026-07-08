from typing import Callable, Optional, Tuple
import numpy as np
from numpy.typing import ArrayLike
from numpy.fft import fft2, ifft2, fftfreq
from scipy.optimize import minimize
from scipy.spatial import cKDTree, ConvexHull
from scipy.interpolate import RegularGridInterpolator, griddata

from insectvision.geometry.linalg import rotate2d
from insectvision.geometry.polygons import Polygon2D
from insectvision.geometry.ghosting import ghosts_from_mirror, ghosts_from_growth_2d
from insectvision.geometry.neighbours import delaunay_edges, metric_spacing


def align_grid(grid: ArrayLike, points2d: ArrayLike) -> np.ndarray:
    """
    Align a hex grid (rigid transform) to match a point cloud.
    """
    grid = np.asarray(grid, float)
    points2d = np.asarray(points2d, float)

    source_domain = Polygon2D.from_points(points2d)

    def loss(p):
        cos, sin = np.cos(p[2]), np.sin(p[2])
        mat = np.array([[cos, -sin], [sin, cos]])
        t = grid @ mat.T + p[:2]

        t_tree = cKDTree(t)
        d_fwd, _ = t_tree.query(points2d)
        mask = source_domain.inside(t)

        if np.any(mask):
            d_bwd, _ = source_domain.tree.query(t[mask])
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


# Relaxation

def density_warp(
        points2d: ArrayLike,
        spacing_fn: Callable,
        reference_spacing: float,
        exponent: float = 1.0,
    ) -> np.ndarray:
    """
    Warp a point set so that local spacing matches a target density field.

    Args:
        - points2d: (N, 2)
        - spacing_fn: Target local spacing function at each point, (M, 2) -> (M,)
        - reference_spacing: float, The spacing of the uniform input grid
        - exponent: float, Warp strength. Lower value = keeps more points at the boundary
    """

    pts = np.copy(points2d).astype(np.float64)

    s = spacing_fn(pts)     # centre of compression
    weights = 1.0 / np.maximum(s, 1e-12)
    centroid = np.average(pts, axis=0, weights=weights)

    disp = pts - centroid

    scale = (s / reference_spacing) ** exponent
    pts = centroid + disp * scale[:, None]

    return pts


def spring_relaxation(
        points2d: ArrayLike,
        spacing_fn: Callable,
        theta_fn: Callable,
        bond_dirs: ArrayLike,
        max_iter: int = 120,
        retriangulate_every: int = 5,
        dt: float = 0.1,
        force_cap: float = 2.0,
        convergence_tol: float = 1e-3,
        verbose: bool = False,
        domain: Optional['Polygon2D'] = None,
        ghost_depth_factor: float = 0.0,
        ghost_source: str = 'lattice'
    ) -> np.ndarray:
    """
    Local spring relaxation with orientation-following bonds.

    For each edge the ideal bond is chosen in the local lattice frame R(-theta) and
    rotated back by theta(midpoint).
    Thus rows curve to follow theta, and periodic retriangulation lets
    dislocations nucleate wherever the lattice can't stay both straight and regular.

    Ghost points (ghost_depth_factor > 0): points within 'ghost_depth_factor * local
    spacing' of a reference boundary are mirrored to the outside and added to the
    cloud before triangulating, giving the edge points symmetric spring pressure to
    prevent the boundary from collapsing inward.

    ghost_source:
        - 'hull': reflect across domain.hull (the smooth contour), requires 'domain'.
        - 'edge': reflect across ConvexHull(pts), the lattice's own current outer edge, so the
                    mirror plane moves with the lattice.
        - 'none': no ghosts (free boundary, will contract during relaxation).
    """

    ghost_source = 'none' if not ghost_source else str(ghost_source).lower()

    if ghost_source not in ('lattice', 'hull', 'edge', 'none'):
        raise ValueError(f"ghost_source must be 'lattice', 'hull', 'edge' or 'none', got {ghost_source!r}")

    pts = np.copy(points2d).astype(np.float64)

    n_real = len(pts)
    use_ghosts = (ghost_depth_factor > 0.0 and ghost_source != 'none'
                  and (ghost_source == 'edge' or domain is not None))

    def build_cloud(p):
        if not use_ghosts:
            return p, np.zeros((0, 2))

        local_spacing = spacing_fn(p).ravel()

        if ghost_source == 'lattice':
            g = ghosts_from_growth_2d(
                p, theta_fn, spacing_fn, bond_dirs, domain,
                avg_spacing=np.median(local_spacing),
                boundary_band=ghost_depth_factor,       # band in units of local spacing
            )
        else:
            depth = ghost_depth_factor * local_spacing  # per-point mirror depth
            if ghost_source == 'edge':
                # Mirror across the lattice's own current outer edge (moves with the cloud)
                try:
                    g = ghosts_from_mirror(p, depth, domain=ConvexHull(p))
                except Exception:
                    return p, np.zeros((0, 2))   # degenerate cloud? skip ghosts this pass
            else:  # 'hull'
                g = ghosts_from_mirror(p, depth, domain=domain)

        if len(g):
            # Drop ghosts sitting on top of a real point (self-images for 'edge'),
            # thresholded by the local spacing at each ghost
            d, _ = cKDTree(p).query(g)
            g = g[d > 0.3 * spacing_fn(g).ravel()]

        return (np.vstack([p, g]), g) if len(g) else (p, np.zeros((0, 2)))

    cloud, ghosts = build_cloud(pts)

    edges = delaunay_edges(cloud, max_length_factor=1.8)

    bond_dirs = np.asarray(bond_dirs, dtype=np.float64)

    for it in range(max_iter):
        if retriangulate_every and it > 0 and it % retriangulate_every == 0:
            cloud, ghosts = build_cloud(pts)
            edges = delaunay_edges(cloud, max_length_factor=1.8)
            if verbose:
                print(f'  spring relaxation [it {it}]: retriangulating')
        else:
            # refresh the (moved) real block, ghosts stay frozen between retriangulations
            cloud = np.vstack([pts, ghosts]) if use_ghosts else pts

        e0, e1 = edges[:, 0], edges[:, 1]

        mid = 0.5 * (cloud[e0] + cloud[e1])
        cur = cloud[e1] - cloud[e0]
        th = theta_fn(mid).ravel()

        # Into lattice frame
        loc = rotate2d(cur, -th)
        u = loc / np.maximum(np.linalg.norm(loc, axis=1, keepdims=True), 1e-12)

        bond_dirs_unit = bond_dirs / np.maximum(np.linalg.norm(bond_dirs, axis=1, keepdims=True), 1e-12)
        best_idx = np.argmax(u @ bond_dirs_unit.T, axis=1)

        ideal = bond_dirs[best_idx]

        L = spacing_fn(mid).ravel()

        # Rest vector back in world frame
        rest = L[:, None] * rotate2d(ideal, th)

        fpe = (cur - rest) / np.maximum(np.linalg.norm(rest, axis=1, keepdims=True), 1e-12)
        fmag = np.linalg.norm(fpe, axis=1, keepdims=True)
        fpe = np.where(fmag > force_cap, fpe * force_cap / fmag, fpe)

        n_all = len(cloud)

        forces, deg = np.zeros((n_all, 2)), np.zeros(n_all)
        np.add.at(forces, e0, fpe)
        np.add.at(forces, e1, -fpe)
        np.add.at(deg, e0, 1)
        np.add.at(deg, e1, 1)

        # Ghosts points are slaved to the real cloud (only the real block moves)
        forces, deg = forces[:n_real], deg[:n_real]

        node_scale = spacing_fn(pts).ravel()
        disp = dt * node_scale[:, None] * forces / np.maximum(deg, 1)[:, None]
        norms = np.linalg.norm(disp, axis=1, keepdims=True)
        cap = 0.5 * node_scale[:, None]
        disp = np.where(norms > cap, disp * cap / np.maximum(norms, 1e-12), disp)
        pts = pts + disp

        if np.linalg.norm(disp, axis=1).mean() < convergence_tol * np.median(node_scale):
            if verbose:
                print(f'  spring relaxation [it {it}]: converged')
            return pts

    if verbose:
        print(f'  spring relaxation [it {it}]: hit max_iter={max_iter}')
    return pts


def density_correct(
        points2d: ArrayLike,
        target_spacing_fn: Callable,
        domain: Optional['Polygon2D'],
        n_iter: int = 3,
        relax: float = 0.6,
        grid_n: int = 192,
        pad: float = 0.2,
        k: int = 6,
        verbose: bool = False
    ) -> np.ndarray:
    """
    Smooth, area-correcting transport (linearised optimal transport).

    div(u) = 1 - (s_achieved / s_target)^2
        -> positive where the lattice is too dense (-> expand), negative where too sparse (-> contract)
    Then u = grad(phi) turns this into one Poisson solve per pass, which is solved on a grid by FFT.
    """

    pts = np.copy(points2d).astype(np.float64)

    lo = domain.boundary.min(0) - pad
    hi = domain.boundary.max(0) + pad

    xs = np.linspace(lo[0], hi[0], grid_n)
    ys = np.linspace(lo[1], hi[1], grid_n)
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    gx, gy = np.meshgrid(xs, ys, indexing='ij')
    grid_pts = np.column_stack([gx.ravel(), gy.ravel()])
    inside = domain.inside(grid_pts).reshape(gx.shape)

    KX, KY = np.meshgrid(2 * np.pi * fftfreq(grid_n, dx),
                         2 * np.pi * fftfreq(grid_n, dy), indexing='ij')
    K2 = KX ** 2 + KY ** 2
    K2[0, 0] = 1.0   # 1.0 to avoid div by zero on the DC mode

    for it in range(n_iter):

        s_ach = metric_spacing(query_points=pts, k=k)
        s_tgt = target_spacing_fn(pts).ravel()

        # Positive where the lattice is too dense, negative where too sparse
        src = 1.0 - (s_ach / np.maximum(s_tgt, 1e-12)) ** 2

        G = griddata(pts, src, (gx, gy), method='linear', fill_value=0.0)
        G = np.where(inside, np.nan_to_num(G), 0.0)
        G = G - G.mean()                                     # FFT solvability

        phi = np.real(ifft2(-fft2(G) / K2))                  # solves laplacian(phi) = G
        ux, uy = np.gradient(phi, dx, dy)

        fx = RegularGridInterpolator((xs, ys), ux, bounds_error=False, fill_value=0.0)
        fy = RegularGridInterpolator((xs, ys), uy, bounds_error=False, fill_value=0.0)
        u = np.column_stack([fx(pts), fy(pts)])

        # Per-point capping at half the local spacing for safety on steep gradients
        cap = 0.5 * s_tgt
        un = np.linalg.norm(u, axis=1)
        scale = np.where(un > cap, cap / np.maximum(un, 1e-12), 1.0)
        pts = pts + relax * u * scale[:, None]

        if verbose:
            print(f'  density_correct iter {it}: mean |u| = {un.mean():.4f}')

    return pts


# TODO: This one should move
def mirror_bilateral(
        positions: ArrayLike,
        directions: ArrayLike,
        shift: float = 0.0,
        source_side: str = 'right'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Translate one eye's positions along X by 'shift', then mirror across X=0 to build the other side.
    """

    positions = np.asarray(positions, dtype=np.float32)
    directions = np.asarray(directions, dtype=np.float32)

    src_pos = positions.copy()
    src_dir = directions.copy()

    src_pos[:, 0] += shift

    mir_pos = src_pos.copy()
    mir_pos[:, 0] *= -1

    mir_dir = src_dir.copy()
    mir_dir[:, 0] *= -1

    if str(source_side).lower()[:1] == 'r':
        R_pos, L_pos, R_dir, L_dir = src_pos, mir_pos, src_dir, mir_dir
    elif str(source_side).lower()[:1] == 'l':
        L_pos, R_pos, L_dir, R_dir = src_pos, mir_pos, src_dir, mir_dir
    else:
        raise ValueError(f"Unknown source_side '{source_side!r}'")

    both_eyes_pos = np.vstack([L_pos, R_pos])
    both_eyes_dirs = np.vstack([L_dir, R_dir])
    both_eyes_ids = np.concatenate([np.zeros(len(L_pos)), np.ones(len(R_pos))])

    return both_eyes_pos, both_eyes_dirs, both_eyes_ids