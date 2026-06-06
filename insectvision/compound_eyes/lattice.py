import warnings
from typing import Callable, Tuple
import numpy as np
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, Voronoi, cKDTree, Delaunay

from insectvision.utils.hexatic import phasor_from_points, hexatic_rest_vectors, hexatic_order
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

    # reference mean spacing using inner 80% (avoid boundary from polluting target density)
    p10, p90 = np.percentile(spacing, [10, 90])
    core_mask = (spacing >= p10) & (spacing <= p90)
    mean_spacing = float(spacing[core_mask].mean()) if core_mask.any() else float(spacing.mean())

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


def delaunay_adjacency(pts_2d: np.ndarray, max_length_factor: float = 0.0):
    """One-ring neighbour lists from a 2D Delaunay triangulation.
    If max_length_factor > 0, drop edges longer than that times local spacing
    (removes boundary bearings corruptions)."""
    tri = Delaunay(pts_2d)
    adj = [set() for _ in range(len(pts_2d))]
    for sx in tri.simplices:
        for i in range(3):
            a, b = int(sx[i]), int(sx[(i + 1) % 3])
            adj[a].add(b); adj[b].add(a)
    adj = [np.fromiter(s, dtype=np.intp, count=len(s)) for s in adj]
    if max_length_factor > 0:
        tree = cKDTree(pts_2d)
        d, _ = tree.query(pts_2d, k=min(7, len(pts_2d)))
        loc = d[:, 1:].mean(axis=1)
        adj = [nb[np.linalg.norm(pts_2d[nb] - pts_2d[i], axis=1) < max_length_factor * loc[i]]
               for i, nb in enumerate(adj)]
    return adj


def hexatic_axis_field(pts_2d, adj=None, smoothing=0.5, min_order=0.5,
                       max_length_factor=1.8, return_confidence=False):
    """Continuous local hexatic-axis interpolant.

    Points with |Psi6| < min_order (incomplete rings at the boundary) are excluded from the fit,
    so the boundary inherits the smooth interior field instead of defining its own noisy one.
    """

    pts_2d = np.asarray(pts_2d, dtype=np.float64)

    if adj is None:
        adj = delaunay_adjacency(pts_2d, max_length_factor=max_length_factor)

    z6 = phasor_from_points(pts_2d, adj)
    order = hexatic_order(z6)

    keep = order >= min_order
    if keep.sum() < 4:  # safety: nothing trusted
        keep = np.ones(len(pts_2d), dtype=bool)

    rbf_re = RBFInterpolator(pts_2d[keep], z6[keep].real, kernel='thin_plate_spline', smoothing=smoothing)
    rbf_im = RBFInterpolator(pts_2d[keep], z6[keep].imag, kernel='thin_plate_spline', smoothing=smoothing)

    def theta_fn(q):
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        return (np.angle(rbf_re(q) + 1j * rbf_im(q)) / 6.0).astype(np.float64)

    if not return_confidence:
        return theta_fn
    rbf_c = RBFInterpolator(pts_2d[keep], order[keep], kernel='thin_plate_spline', smoothing=smoothing)

    def conf_fn(q):
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        return np.clip(rbf_c(q), 0.0, 1.0)

    return theta_fn, conf_fn


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


def _mirror_across_hull(points: np.ndarray, equations: np.ndarray, depth: float) -> np.ndarray:
    """
    Mirror points close to the hull edges to the outside.
    This creates a symmetric Voronoi pressure that prevents edge points
    from squashing against the boundary.
    """
    mirrored = []
    for eq in equations:
        normal, offset = eq[:2], eq[2]

        # ConvexHull convention: normals point outward
        dist = points @ normal + offset

        mask = (dist > -depth) & (dist <= 0)    # points inside the hull and within 'depth' of the edge
        if not np.any(mask):
            continue

        close_pts = points[mask]
        dist_close = dist[mask]

        # Reflect across edge
        mirrored_pts = close_pts - 2.0 * dist_close[:, None] * normal[None, :]
        mirrored.append(mirrored_pts)

    if mirrored:
        return np.vstack(mirrored)
    return np.zeros((0, 2))


def lloyd_relaxation(
        points: np.ndarray,
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

    # get data domain (to place the ghost ring fallback)
    domain_center = np.mean(boundary.points[boundary.vertices], axis=0)
    domain_radius = np.max(np.linalg.norm(boundary.points[boundary.vertices] - domain_center, axis=1))
    fallback_ghosts = _place_ghost_ring(domain_center, domain_radius, n_ghosts=128)

    # estim initial spacing to define mirror depth and boundary expansion
    tree = cKDTree(points)
    d, _ = tree.query(points, k=2)
    mean_nn = float(np.mean(d[:, 1]))

    mirror_depth = mean_nn * 3.0

    # Expand the hard boundary constraints by 1.5x mean spacing
    # (pure safety net, the mirrored points do the actual bounding)
    expanded_equations = boundary.equations.copy()
    expanded_equations[:, 2] -= mean_nn * 1.5

    for it in range(max_iter):
        mirrored_ghosts = _mirror_across_hull(points, boundary.equations, mirror_depth)
        ghosts = mirrored_ghosts if len(mirrored_ghosts) > 0 else fallback_ghosts

        all_pts = np.vstack([points, ghosts])
        try:
            vor = Voronoi(all_pts)
        except Exception as exc:
            if verbose:
                print(f"  Lloyd iter {it}: Voronoi failed ({exc}), stopping.")
            break

        # Voronoi edges between a point and its reflection lie exactly on the true boundary
        # so we don't want the hard clipper to interfere
        new_pts = weighted_centroids_batch(
            all_pts, vor, n_real, expanded_equations, density_fn
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
        points,
        spacing_fn,
        theta_fn=None,
        confidence_fn=None,
        n_iterations=120,
        retriangulate_every=0,
        max_length_factor=1.8,
        force_cap=2.0,
        dt=0.05,
        verbose=False
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
        points: (N, 2) initial positions (a roughly hex grid)
        spacing_fn: (M, 2) -> (M,) target local spacing
        theta_fn: (M, 2) -> (M,) local hexatic-axis angle [rad], or None
        retriangulate_every: recompute Delaunay edges every k iters (0 = never)
        dt: step size (fraction of mean spacing)
        verbose: print progress
    """
    pts = points.copy()
    edges = delaunay_edges(pts, max_length_factor=max_length_factor)
    mean_edge_len = np.linalg.norm(pts[edges[:, 0]] - pts[edges[:, 1]], axis=1).mean()

    for it in range(n_iterations):
        if retriangulate_every and it > 0 and it % retriangulate_every == 0:
            edges = delaunay_edges(pts, max_length_factor=max_length_factor)
            mean_edge_len = np.linalg.norm(pts[edges[:, 0]] - pts[edges[:, 1]], axis=1).mean()

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

        force_per_edge = np.clip((current_diff - rest) / target_lengths[:, None], -force_cap, force_cap)
        forces = np.zeros_like(pts)
        np.add.at(forces, edges[:, 0], force_per_edge)
        np.add.at(forces, edges[:, 1], -force_per_edge)

        disp = dt * mean_edge_len * forces
        norms = np.linalg.norm(disp, axis=1, keepdims=True)
        cap = 0.5 * mean_edge_len
        pts += np.where(norms > cap, disp * cap / norms, disp)

    return pts


def facet_diameters(
        positions,
        directions,
        k=18,
        packing=1.0,
        smooth_iter=1,
        shell_factor=1.5,
        min_ring=4,
        fill_sweeps=6
    ):
    """
    Per-ommatidium facet diameter from the local Voronoi cell area on the eye surface.

    First-ring spacing is taken over a *shell* (neighbours within shell_factor x the
    nearest) rather than the kNN-6, so boundary/spur lenses aren't biased high by
    second-ring points padding the count. Boundary lenses (< min_ring genuine
    first-ring neighbours, or no bounded Voronoi cell) are filled from their interior
    neighbours instead of trusting their own inflated fallback.

    Args:
        positions: (N, 3) lens world positions.
        directions: (N, 3) lens optical axes (unit; the local surface normal).
        k: neighbours used to bound each local cell (>= ~12 so the first ring is enclosed).
        packing: scale on the hex flat-to-flat diameter (1.0 = facets tile edge-to-edge).
        smooth_iter: neighbour-median smoothing passes applied to the result.
    """
    positions = np.asarray(positions, dtype=float)
    directions = np.asarray(directions, dtype=float)
    N = len(positions)

    tree = cKDTree(positions)
    kq = min(k + 1, N)
    dist, idx = tree.query(positions, k=kq)  # col 0 is self
    nn = dist[:, 1:]
    ring_idx = idx[:, 1:min(7, kq)]

    shell = np.where(nn <= shell_factor * dist[:, 1:2], nn, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        ref = np.nanmedian(shell, axis=1)

    ref = np.where(np.isfinite(ref), ref, dist[:, 1])
    n_shell = np.sum(np.isfinite(shell), axis=1)
    boundary = n_shell < min_ring

    D = np.full(N, np.nan, dtype=float)

    for i in range(N):
        nb = idx[i, 1:]
        n = directions[i] / max(np.linalg.norm(directions[i]), 1e-12)
        a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        u = np.cross(n, a)
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        rel = positions[nb] - positions[i]
        pts2d = np.vstack([[0.0, 0.0], np.column_stack([rel @ u, rel @ v])])
        fb = packing * ref[i]

        try:
            vor = Voronoi(pts2d)
            region = vor.regions[vor.point_region[0]]
            if region and (-1 not in region):
                poly = vor.vertices[region]
                c = poly.mean(axis=0)
                poly = poly[np.argsort(np.arctan2(poly[:, 1] - c[1], poly[:, 0] - c[0]))]
                area = 0.5 * abs(np.dot(poly[:, 0], np.roll(poly[:, 1], -1))
                                 - np.dot(poly[:, 1], np.roll(poly[:, 0], -1)))
                dv = packing * np.sqrt(2.0 * area / np.sqrt(3.0))
                if 0.6 * fb < dv < 1.5 * fb:
                    D[i] = dv
        except Exception:
            pass

        if np.isnan(D[i]) and not boundary[i]:
            D[i] = fb  # interior fallback

    D[boundary] = np.nan  # Never trust boundary fallback

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for _ in range(fill_sweeps):  # propagate interior values outward
            med = np.nanmedian(np.concatenate([D[:, None], D[ring_idx]], axis=1), axis=1)
            D[boundary] = med[boundary]

    D = np.where(np.isfinite(D), D, packing * ref)  # last resort

    for _ in range(int(smooth_iter)):
        D = np.median(np.concatenate([D[:, None], D[ring_idx]], axis=1), axis=1)

    return D.astype(np.float32)


def fit_lattice(
        raw_directions: np.ndarray,
        density_scale: float = 1.0,
        lattice_angle: float = np.pi / 3,
        relaxation_max_iter: int = 20,
        relaxation_factor: float = 0.8,
        smoothing: float = 0.1,
        axis_smoothing: float = 0.5,
        boundary_buffer_factor: float = 1.1,
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

    # Local hexatic-axis field from the reference data
    ref_adj = delaunay_adjacency(pts_2d, max_length_factor=1.8)
    axis_fn, conf_fn = hexatic_axis_field(pts_2d, ref_adj, smoothing=axis_smoothing,
                                          min_order=0.5, return_confidence=True)
    theta0 = float(axis_fn(pts_2d.mean(axis=0, keepdims=True))[0])

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

        c0, s0 = np.cos(theta0), np.sin(theta0)
        grid = grid @ np.array([[c0, -s0], [s0, c0]]).T  # pre-orient seed to the field
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
        theta_fn=axis_fn,
        n_iterations=80,
        retriangulate_every=10,
        dt=0.05,
        verbose=verbose,
        confidence_fn=conf_fn
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

    # Re-assert the hexatic axes after the isotropic Lloyd polish
    lattice = spring_relaxation(
        lattice, spacing_fn, theta_fn=axis_fn,
        n_iterations=15, retriangulate_every=5, dt=0.03,
        confidence_fn=conf_fn,
        verbose=verbose,
    )
    mask = _is_inside(lattice, delaunay, tree_raw, buffer=prune_buffer)
    lattice = lattice[mask]

    # TODO: All these steps might be a bit too... cautious, but it's pretty quick anyway

    if verbose:
        print(f"Final lattice: {len(lattice)} points")

    # if show_plots:
    #     plot_fitted_comparison(pts_2d, lattice, density_fn)

    return stereo_to_sphere(lattice, fwd, rgt, up)

#
# def fit_lattice_from_density(
#         density_fn: Callable,
#         boundary_pts: np.ndarray,
#         n_target: int,
#         stereo_frame: Tuple[np.ndarray, np.ndarray, np.ndarray],
#         lattice_angle: float = np.pi / 3,
#         relaxation_max_iter: int = 20,
#         relaxation_factor: float = 0.8,
#         verbose: bool = False,
#         show_plots: bool = False,
# ) -> np.ndarray:
#     """
#     Generate a true lattice from an explicit density function and boundary.
#
#     Args:
#         density_fn: (M, 2) -> (M,) Density (in the stereographic plane)
#         boundary_pts: (K, 2) Points (in the stereographic plane) defining the boundary
#         n_target (int): Approximate number of ommatidia to generate
#         stereo_frame: Orthonormal frame (fwd, rgt, up) for back-projection to the sphere
#         lattice_angle (float): Angle between hex basis vectors (pi/3 = regular hexagonal)
#         relaxation_max_iter (int): Upper limit for Lloyd's relaxation
#         relaxation_factor (float): Under-relaxation (0, 1]. 0.8 is a good default
#         verbose (bool): Whether to print progress
#         show_plots (bool): Whether to display plots
#     """
#
#     hull = ConvexHull(boundary_pts)
#     delaunay = Delaunay(boundary_pts)
#     tree = cKDTree(boundary_pts)
#
#     fwd, rgt, up = stereo_frame
#
#     domain_area = hull.volume  # 2D ConvexHull.volume = area
#     target_spacing = np.sqrt(domain_area / (n_target * np.sin(lattice_angle)))
#
#     if verbose:
#         print(f"Domain area: {domain_area:.5f},  target spacing: {target_spacing:.5f}")
#
#     # Seed grid
#     extent = np.max(np.abs(boundary_pts)) + 2 * target_spacing
#     grid = hexagonal_grid(target_spacing, lattice_angle, extent)
#
#     # Centre on data domain
#     centre = boundary_pts.mean(axis=0)
#     grid = grid + centre
#
#     # Prune
#     prune_buffer = target_spacing * 0.5
#     mask = _is_inside(grid, delaunay, tree, buffer=prune_buffer)
#     lattice = grid[mask]
#
#     if verbose:
#         print(f"Initial grid: {len(lattice)} points")
#
#     # Lloyd's relaxation (final hex regularity polish)
#     lattice = lloyd_relaxation(
#         lattice, density_fn, hull,
#         max_iter=relaxation_max_iter,
#         relaxation_factor=relaxation_factor,
#         verbose=verbose,
#     )
#
#     # and prune again (Lloyd's might have pushed points slightly outside)
#     mask = _is_inside(lattice, delaunay, tree, buffer=prune_buffer)
#     lattice = lattice[mask]
#
#     if verbose:
#         print(f"Final lattice: {len(lattice)} points")
#
#     # if show_plots:
#     #     plot_fitted_comparison(boundary_pts, lattice, density_fn)
#
#     return stereo_to_sphere(lattice, fwd, rgt, up)