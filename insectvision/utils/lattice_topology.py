from typing import Callable, Tuple
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, Voronoi, cKDTree, Delaunay

from insectvision.utils.math import project_to_stereo, stereo_to_sphere
# from species_models.plots import plot_fitted_comparison

# TODO: Might benefit from a bit of cleaning here too
# TODO: Fix plot imports


def estimate_density(
    pts_2d: np.ndarray,
    smoothing: float = 0.1,
    k_neighbours: int = 7,
) -> Tuple[Callable, float]:
    """
    Estimate a density field from a 2D point cloud.

    Takes local inter-point spacing, smoothes with RBF, and inverts to give density.

    Note: density is proportional to 1/spacing, not 1/spacing**2 (to keep gradients cool for the Lloyd relaxation).
        The quadratic relationship emerges from the Voronoi cell areas.

    Returns:
        density_fn (callable): (M, 2) -> (M,)
        mean_spacing (float): mean nearest-neighbour distance in the cloud
    """
    tree = cKDTree(pts_2d)
    dist, _ = tree.query(pts_2d, k=k_neighbours)
    spacing = dist[:, 1:].mean(axis=1)
    mean_spacing = float(spacing.mean())

    # RBF of normalised spacing
    rbf = RBFInterpolator(
        pts_2d, spacing / mean_spacing,
        kernel='thin_plate_spline', smoothing=smoothing,
    )

    def density_fn(pts: np.ndarray) -> np.ndarray:
        pts = np.atleast_2d(pts)
        s = rbf(pts).ravel()

        # Clamp bc RBF can go wild outside the data...
        s = np.clip(s, 0.3, 5.0)

        return 1.0 / s ** 2

    return density_fn, mean_spacing


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


def align_grid(grid: np.ndarray, pts_2d: np.ndarray) -> np.ndarray:
    """
    Align a hex grid (rigid transform) to match a point cloud.
    """

    tree_raw = cKDTree(pts_2d)
    hull = Delaunay(pts_2d)

    def loss(p):
        pos, rot = p[:2], p[2]
        c, s = np.cos(rot), np.sin(rot)
        mat = np.array([[c, -s], [s, c]])
        t = grid @ mat.T + pos

        t_tree = cKDTree(t)
        d_fwd, _ = t_tree.query(pts_2d)

        mask = _is_inside(t, hull, tree_raw, buffer=0)
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
    mat = np.array([[c, -s], [s, c]])
    aligned = grid @ mat.T + pos

    return aligned


def clip_polygon(vertices: np.ndarray, hull_equations: np.ndarray) -> np.ndarray:
    """
    Clip a (convex) polygon against a (convex) region
    https://en.wikipedia.org/wiki/Sutherland%E2%80%93Hodgman_algorithm#Pseudocode
    """

    out = list(vertices)

    for eq in hull_equations:
        normal, offset = eq[:2], eq[2]

        if len(out) < 3:
            return np.zeros([0, 2])

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

    return np.array(out) if out else np.zeros([0, 2])


def weighted_centroid(polygon: np.ndarray, density_fn: Callable) -> np.ndarray:
    """
    Density weighted centroid of a (convex) polygon.
    (fan triangulation -> triangles centroids -> done)
    """
    n = len(polygon)
    if n < 3:
        return polygon.mean(axis=0)

    v0 = polygon[0]
    total_w = 0.0
    wc = np.zeros(2)

    for i in range(1, n - 1):
        tri = np.array([v0, polygon[i], polygon[i + 1]])
        centre = tri.mean(axis=0)
        area = 0.5 * abs(np.cross(tri[1] - tri[0], tri[2] - tri[0]))

        if area < 1e-14:
            continue

        rho = density_fn(centre.reshape(1, -1))[0]
        w = area * rho
        wc += w * centre
        total_w += w

    return wc / total_w if total_w > 1e-16 else polygon.mean(axis=0)


def weighted_centroids_batch(
        points: np.ndarray,
        voronoi: Voronoi,
        n_real: int,
        boundary_equations: np.ndarray,
        density_fn: Callable,
) -> np.ndarray:
    """
    Compute density-weighted centroids for many points.
    """
    new_pts = points[:n_real].copy()

    for i in range(n_real):
        region_idx = voronoi.point_region[i]
        region = voronoi.regions[region_idx]

        if not region or -1 in region:
            # should not happen with proper bounding ghosts
            continue

        cell = voronoi.vertices[region]
        if len(cell) < 3:
            continue

        clipped = clip_polygon(cell, boundary_equations)
        if len(clipped) < 3:
            continue

        new_pts[i] = weighted_centroid(clipped, density_fn)

    return new_pts


def psi6_from_adjacency(pts_2d: np.ndarray, adj) -> np.ndarray:
    """
    Hexatic order parameter using topological (Delaunay) neighbours.
    This handles varying density correctly (unlike kNN) because the adjacency follows connectivity, not metric distance.
    i.e. points with 5 or 7 neighbours are handled just fine by the phasor average.
    """
    N = len(pts_2d)
    psi6 = np.zeros(N)

    for i in range(N):
        nb = adj[i]

        if len(nb) < 2:
            continue

        delta = pts_2d[nb] - pts_2d[i]
        angles = np.arctan2(delta[:, 1], delta[:, 0])
        psi6[i] = abs(np.mean(np.exp(6j * angles)))

    return psi6


def spacing_from_adjacency(pts_2d: np.ndarray, adj) -> np.ndarray:
    """
    Mean distance to Delaunay neighbours (more robust than kNN at boundaries).
    """
    N = len(pts_2d)
    spacing = np.zeros(N)

    for i in range(N):
        nb = adj[i]
        if not nb:
            continue
        spacing[i] = np.linalg.norm(pts_2d[nb] - pts_2d[i], axis=1).mean()

    return spacing


def delaunay_edges(pts_2d: np.ndarray, max_length_factor: float = 0.0):
    """
    Compute Delaunay edges in the stereo plane.
    Optionally prunes edges longer than max_length_factor * local spacing (disabled if <= 0)
    """

    tri = Delaunay(pts_2d)
    edges = set()
    for simplex in tri.simplices:
        for i in range(3):
            a, b = simplex[i], simplex[(i + 1) % 3]
            edges.add((min(a, b), max(a, b)))
    edges = np.array(list(edges))

    if max_length_factor > 0:
        # Prune long edges (convex hull boundary, not real neighbours)

        tree = cKDTree(pts_2d)
        d, _ = tree.query(pts_2d, k=7)
        local_spacing = d[:, 1:].mean(axis=1)

        lengths = np.linalg.norm(pts_2d[edges[:, 0]] - pts_2d[edges[:, 1]], axis=1)
        mean_local = 0.5 * (local_spacing[edges[:, 0]] + local_spacing[edges[:, 1]])
        to_keep = lengths < mean_local * max_length_factor

        return edges[to_keep]

    return edges


##

def _place_ghost_ring(
        domain_center: np.ndarray,
        domain_radius: float,
        n_ghosts: int = 64,
        radius_factor: float = 3.0,
) -> np.ndarray:
    """
    Place ghost points on a circle far outside the data so that every real point's Voronoi cell is finite
    """
    angles = np.linspace(0, 2 * np.pi, n_ghosts, endpoint=False)
    r = domain_radius * radius_factor
    ghosts = domain_center + np.column_stack([np.cos(angles), np.sin(angles)]) * r
    return ghosts


def _is_inside(points: np.ndarray, delaunay: Delaunay, tree: cKDTree, buffer: float = 0.0) -> np.ndarray:
    inside = delaunay.find_simplex(points) >= 0
    if buffer > 0:
        outside = np.where(~inside)[0]
        if len(outside):
            d, _ = tree.query(points[outside])
            inside[outside] = d < buffer
    return inside


def lloyd_relaxation(
        points: np.ndarray,
        density_fn: Callable,
        boundary: ConvexHull,
        max_iter: int = 20,
        convergence_tol: float = 1e-6,
        relaxation_factor: float = 0.8,
        verbose: bool = False,
) -> np.ndarray:
    """
    Voronoi relaxation (Lloyd's algorithm, 10.1109/TIT.1982.1056489).

    Args:
        points: (N, 2)
        density_fn: (M, 2) -> (M,)
        boundary: ConvexHull of the target domain
        max_iter: int
        convergence_tol: Stop when mean displacement < this
        relaxation_factor: Under-relaxation factor in (0, 1]
            1.0 = full step to centroid (that can oscillate)
            0.5-0.8 = smoother convergence
        verbose: Print info
    """

    points = points.copy()
    n_real = len(points)

    # get data domain (to place the ghost ring)
    domain_center = np.mean(boundary.points[boundary.vertices], axis=0)
    domain_radius = np.max(np.linalg.norm(boundary.points[boundary.vertices] - domain_center, axis=1))

    ghosts = _place_ghost_ring(domain_center, domain_radius, n_ghosts=128)

    for it in range(max_iter):
        all_pts = np.vstack([points, ghosts])

        try:
            vor = Voronoi(all_pts)
        except Exception as exc:
            if verbose:
                print(f"  Lloyd iter {it}: Voronoi failed ({exc}), stopping.")
            break

        new_pts = weighted_centroids_batch(
            all_pts, vor, n_real, boundary.equations, density_fn
        )

        step = new_pts - points
        points_next = points + relaxation_factor * step

        disp = np.linalg.norm(relaxation_factor * step, axis=1)

        if verbose:
            print(f"  Lloyd iter {it:3d}:  mean d = {disp.mean():.6f}, max d = {disp.max():.6f}")

        points = points_next

        if disp.mean() < convergence_tol:
            if verbose:
                print(f"  Converged at iteration {it}.")
            break

    return points


def density_warp(
        points: np.ndarray,
        spacing_fn: Callable,
        reference_spacing: float,
        exponent: float = 1.0,
) -> np.ndarray:
    """
    Warp a point set so that local spacing matches a target density field.

    Args:
        points: (N, 2)
        spacing_fn: (M, 2) -> (M,) Target local spacing at each point
        reference_spacing (float): The spacing of the uniform input grid
        exponent (float): Warp strength. Lower values keep more points at the boundary
    """
    pts = points.copy()

    s = spacing_fn(pts)     # centre of compression
    weights = 1.0 / np.maximum(s, 1e-12)
    centroid = np.average(pts, axis=0, weights=weights)

    disp = pts - centroid

    scale = (s / reference_spacing) ** exponent
    pts = centroid + disp * scale[:, None]

    return pts


def spring_relaxation(
        points: np.ndarray,
        spacing_fn: Callable,
        n_iterations: int = 120,
        dt: float = 0.05,
        verbose: bool = False,
) -> np.ndarray:
    """
    Density adaptation via spring relaxation.

    Each Delaunay edge acts as a spring, their rest length are set by the local spacing.
    Forces are integrated with gradient descent (no velocity accumulation) and clamped for stability.

    Delaunay topology is computed once from the initial grid and then fixed to preserves hex row structure.

    Args:
        points: (N, 2) Initial grid positions (should be a regular hex grid)
        spacing_fn: (M, 2) -> (M,) Returns the target local spacing at each query point
        n_iterations (int):
        dt (float): Step size (in fraction of mean spacing)
        verbose (bool):
    """
    pts = points.copy()
    edges = delaunay_edges(pts)

    # initial mean edge length (for normalisation)
    diff0 = pts[edges[:, 0]] - pts[edges[:, 1]]
    mean_edge_len = np.linalg.norm(diff0, axis=1).mean()

    for it in range(n_iterations):

        # Target spacing at each edge midpoint
        midpoints = 0.5 * (pts[edges[:, 0]] + pts[edges[:, 1]])
        target_lengths = spacing_fn(midpoints).ravel()

        current_diff = pts[edges[:, 1]] - pts[edges[:, 0]]
        current_lengths = np.linalg.norm(current_diff, axis=1)
        current_lengths = np.maximum(current_lengths, 1e-12)
        unit = current_diff / current_lengths[:, None]

        # Spring force: strain = (actual - target) / target
        strain = (current_lengths - target_lengths) / target_lengths
        force_per_edge = strain[:, None] * unit

        # Accumulate forces per point
        forces = np.zeros_like(pts)
        np.add.at(forces, edges[:, 0], force_per_edge)
        np.add.at(forces, edges[:, 1], -force_per_edge)

        displacement = dt * mean_edge_len * forces
        max_disp = 0.5 * mean_edge_len
        norms = np.linalg.norm(displacement, axis=1, keepdims=True)
        displacement = np.where(norms > max_disp, displacement * max_disp / norms, displacement)

        pts += displacement

        if verbose and (it % 20 == 0 or it == n_iterations - 1):
            rms_strain = np.sqrt(np.mean(strain ** 2))
            max_strain = np.max(np.abs(strain))
            mean_disp = np.linalg.norm(displacement, axis=1).mean()
            print(f"  Spring iter {it:3d}:  "
                  f"RMS strain = {rms_strain:.4f}  "
                  f"max |strain| = {max_strain:.4f}  "
                  f"mean disp = {mean_disp:.6f}")

    return pts


##


def fit_lattice(
        raw_directions: np.ndarray,
        density_scale: float = 1.0,
        lattice_angle: float = np.pi / 3,
        relaxation_max_iter: int = 20,
        relaxation_factor: float = 0.8,
        smoothing: float = 0.1,
        boundary_buffer_factor: float = 0.6,
        verbose: bool = False,
        show_plots: bool = False,
) -> np.ndarray:
    """
    Generate a true hexagonal lattice on the unit sphere with local
    density matching that of an observed point set.

    Pipeline is: uniform grid -> global density warp -> spring cleaning (haha) -> Lloyd's

    Args:
        raw_directions: (N, 3) Measured / digitised unit directions on the sphere
        density_scale (float): Multiplier on target density
        lattice_angle (float): Angle between hex basis vectors (pi/3 = regular hexagonal)
        relaxation_max_iter (int): Upper limit for Lloyd's relaxation
        relaxation_factor (float): Under-relaxation (0, 1]. 0.8 is a good default
        smoothing (float): RBF smoothing for the density estimator
        boundary_buffer_factor (float): Fraction of mean spacing used as boundary buffer during pruning.
        verbose (bool): Whether to print progress
        show_plots (bool): Whether to display plots
    """

    pts_2d, fwd, rgt, up = project_to_stereo(raw_directions)

    density_fn, mean_spacing = estimate_density(pts_2d, smoothing)

    hull = ConvexHull(pts_2d)
    delaunay = Delaunay(pts_2d)
    tree_raw = cKDTree(pts_2d)

    prune_buffer = mean_spacing * boundary_buffer_factor

    domain_area = hull.volume  # ConvexHull.volume = area in 2D
    n_target = int(len(pts_2d) * density_scale)
    target_spacing = np.sqrt(domain_area / (n_target * np.sin(lattice_angle)))

    if verbose:
        print(f"Reference: {len(pts_2d)} points, mean spacing {mean_spacing:.5f}")
        print(f"Domain area: {domain_area:.5f}, target count: ~{n_target}")
        print(f"Target spacing: {target_spacing:.5f} (density_scale={density_scale:.2f}x)")

    extent = np.max(np.abs(pts_2d)) + 4 * mean_spacing
    light_buffer = prune_buffer + 3.0 * mean_spacing

    # Spacing function (target inter-ommatidial distance at each point)
    tree_ref = cKDTree(pts_2d)
    d_ref, _ = tree_ref.query(pts_2d, k=7)
    local_spacing = d_ref[:, 1:].mean(axis=1) / np.sqrt(density_scale)

    spacing_rbf = RBFInterpolator(
        pts_2d, local_spacing,
        kernel='thin_plate_spline', smoothing=smoothing,
    )

    def spacing_fn(pts):
        pts = np.atleast_2d(pts)
        s = spacing_rbf(pts).ravel()
        return np.clip(s, mean_spacing * 0.2, mean_spacing * 5.0)

    # Uniform hex grid aligned to ref data -> warped globally to match density
    # (scales displacement from centroid by local spacing ratio)
    current_spacing = target_spacing

    for trial in range(5):
        grid = hexagonal_grid(current_spacing, lattice_angle, extent)
        aligned = align_grid(grid, pts_2d)

        mask = _is_inside(aligned, delaunay, tree_raw, buffer=light_buffer)
        lattice = aligned[mask]

        lattice = density_warp(lattice, spacing_fn, current_spacing)

        mask = _is_inside(lattice, delaunay, tree_raw, buffer=prune_buffer)
        lattice = lattice[mask]
        n_survived = len(lattice)

        if verbose:
            print(f"  Trial {trial}: spacing={current_spacing:.5f}, "
                  f"survived={n_survived} (target {n_target})")

        if n_survived == 0:
            current_spacing *= 0.7
            continue

        # Adjust spacing
        ratio = n_target / n_survived
        current_spacing *= 1.0 / np.sqrt(ratio)  # in 2D, count scales as 1/spacing**2

        if abs(n_survived - n_target) / max(n_target, 1) < 0.03:
            break  # close enough

    if verbose:
        print(f"After global warp + prune: {len(lattice)} points")

    # Spring relaxation (local cleanup, preserves hex topology and rows)
    lattice = spring_relaxation(
        lattice, spacing_fn,
        n_iterations=40,
        dt=0.05,
        verbose=verbose,
    )

    # Prune
    mask = _is_inside(lattice, delaunay, tree_raw, buffer=prune_buffer)
    lattice = lattice[mask]

    if verbose:
        print(f"After spring relaxation: {len(lattice)} points")

    # Lloyd's relaxation (final hex regularity polish)
    lattice = lloyd_relaxation(
        lattice, density_fn, hull,
        max_iter=relaxation_max_iter,
        relaxation_factor=relaxation_factor,
        verbose=verbose,
    )

    # and prune again (Lloyd's might have pushed points slightly outside)
    mask = _is_inside(lattice, delaunay, tree_raw, buffer=prune_buffer)
    lattice = lattice[mask]

    if verbose:
        print(f"Final lattice: {len(lattice)} points")

    # if show_plots:
    #     plot_fitted_comparison(pts_2d, lattice, density_fn)

    return stereo_to_sphere(lattice, fwd, rgt, up)


def fit_lattice_from_density(
        density_fn: Callable,
        boundary_pts: np.ndarray,
        n_target: int,
        stereo_frame: Tuple[np.ndarray, np.ndarray, np.ndarray],
        lattice_angle: float = np.pi / 3,
        relaxation_max_iter: int = 20,
        relaxation_factor: float = 0.8,
        verbose: bool = False,
        show_plots: bool = False,
) -> np.ndarray:
    """
    Generate a true lattice from an explicit density function and boundary.

    Args:
        density_fn: (M, 2) -> (M,) Density (in the stereographic plane)
        boundary_pts: (K, 2) Points (in the stereographic plane) defining the boundary
        n_target (int): Approximate number of ommatidia to generate
        stereo_frame: Orthonormal frame (fwd, rgt, up) for back-projection to the sphere
        lattice_angle (float): Angle between hex basis vectors (pi/3 = regular hexagonal)
        relaxation_max_iter (int): Upper limit for Lloyd's relaxation
        relaxation_factor (float): Under-relaxation (0, 1]. 0.8 is a good default
        verbose (bool): Whether to print progress
        show_plots (bool): Whether to display plots
    """

    hull = ConvexHull(boundary_pts)
    delaunay = Delaunay(boundary_pts)
    tree = cKDTree(boundary_pts)

    fwd, rgt, up = stereo_frame

    domain_area = hull.volume  # 2D ConvexHull.volume = area
    target_spacing = np.sqrt(domain_area / (n_target * np.sin(lattice_angle)))

    if verbose:
        print(f"Domain area: {domain_area:.5f},  target spacing: {target_spacing:.5f}")

    # Seed grid
    extent = np.max(np.abs(boundary_pts)) + 2 * target_spacing
    grid = hexagonal_grid(target_spacing, lattice_angle, extent)

    # Centre on data domain
    centre = boundary_pts.mean(axis=0)
    grid = grid + centre

    # Prune
    prune_buffer = target_spacing * 0.5
    mask = _is_inside(grid, delaunay, tree, buffer=prune_buffer)
    lattice = grid[mask]

    if verbose:
        print(f"Initial grid: {len(lattice)} points")

    # Lloyd's relaxation (final hex regularity polish)
    lattice = lloyd_relaxation(
        lattice, density_fn, hull,
        max_iter=relaxation_max_iter,
        relaxation_factor=relaxation_factor,
        verbose=verbose,
    )

    # and prune again (Lloyd's might have pushed points slightly outside)
    mask = _is_inside(lattice, delaunay, tree, buffer=prune_buffer)
    lattice = lattice[mask]

    if verbose:
        print(f"Final lattice: {len(lattice)} points")

    # if show_plots:
    #     plot_fitted_comparison(boundary_pts, lattice, density_fn)

    return stereo_to_sphere(lattice, fwd, rgt, up)