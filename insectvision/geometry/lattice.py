from typing import TYPE_CHECKING, Sequence, List, Callable, Tuple, Optional
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

from insectvision.geometry.linalg import principal_axis_angle
from insectvision.geometry.neighbours import ragged_neighbours

if TYPE_CHECKING:
    from insectvision.geometry.polygons import Polygon2D
    from insectvision.geometry.neighbours import NeighbourGraph


# Lattice generation

def create_hexagonal_grid(
        spacing: float = 1.0,
        angles: float | Sequence[float] = np.pi / 3,
        extent: float = 10.0,
        degrees: bool = False,
) -> np.ndarray:
    """
    Generate a perfectly regular (but potentially skewed) hexagonal grid.

    Args:
        spacing: The length of the first basis vector (b1).
        angles:
            - if float: The angle between b1 and b2 (standard hex = pi/3).
            - if 3-item list: The three internal angles (alpha, beta, gamma) of the lattice triangle.
                Sum is normalised to pi (or 180° if working in degrees).
        extent: The radius of the circular domain to cover.
    """
    B = compute_lattice_basis(spacing, angles, degrees=degrees)
    inv_B = np.linalg.inv(B)

    # Determine required integer range to cover the circular extent
    n1 = int(np.ceil(extent * np.linalg.norm(inv_B[:, 0])))
    n2 = int(np.ceil(extent * np.linalg.norm(inv_B[:, 1])))

    ii, jj = np.meshgrid(np.arange(-n1, n1 + 1), np.arange(-n2, n2 + 1))
    indices = np.stack([ii.ravel(), jj.ravel()], axis=1)

    grid = indices @ B
    mask = np.sum(grid ** 2, axis=1) <= (extent ** 2)
    return grid[mask]


def create_ghosts_ring(
        points2d: ArrayLike,
        theta_fn: Callable,
        spacing_fn: Callable,
        bond_dirs: ArrayLike,
        domain: 'Polygon2D',
        avg_spacing: float,
        boundary_band: float = 1.5,
        missing_tol: float = 0.5,
        merge_tol: float = 0.4,
    ) -> np.ndarray:
    """
    One outward ring of ghosts grown from the lattice.

    For each near-boundary site, take the 6 ideal bond directions rotated into the world
    frame by the local axis angle, and keep a ghost at p + L*b when (a) that bond has no
    existing neighbour and (b) it points further outside the domain than the site itself.
    Ghosts proposed by adjacent sites are then merged.

    Args:
        - points2d: (N, 2) current lattice
        - theta_fn: (M,2) -> (M,) local hexatic axis angle (rad)
        - spacing_fn: (M,2) -> (M,) local target spacing
        - bond_dirs: (6, 2) ideal unit bond directions of the base cell (e.g. base_bond_dirs(lattice_angles))
        - domain: Polygon2D with .signed_distance (<0 inside)
        - avg_spacing: scalar reference spacing (sets the boundary band width)
        - boundary_band: consider sites within this * avg_spacing of the boundary
        - missing_tol: a bond counts as satisfied if a real point is within this * L
        - merge_tol: merge ghosts closer than this * median spacing

    Returns:
        (G, 2) ghost points (possibly empty)
    """
    pts = np.asarray(points2d, dtype=np.float64)
    bond_dirs = np.asarray(bond_dirs, dtype=np.float64)

    d_hull = domain.signed_distance(pts)                      # <0 inside
    near = np.where(d_hull > -boundary_band * avg_spacing)[0]
    if len(near) == 0:
        return np.zeros((0, 2))

    P = pts[near]
    L = spacing_fn(P).ravel()
    th = theta_fn(P).ravel()
    c, s = np.cos(th), np.sin(th)

    # Rotate the 6 base bond dirs by theta at each site -> (m, 6, 2)
    bx, by = bond_dirs[:, 0], bond_dirs[:, 1]
    wx = c[:, None] * bx[None, :] - s[:, None] * by[None, :]
    wy = s[:, None] * bx[None, :] + c[:, None] * by[None, :]
    targets = P[:, None, :] + L[:, None, None] * np.stack([wx, wy], axis=-1)
    targets = targets.reshape(-1, 2)
    L_rep = np.repeat(L, 6)
    d_site_rep = np.repeat(d_hull[near], 6)

    tree = cKDTree(pts)
    d_near, _ = tree.query(targets)
    d_target = domain.signed_distance(targets)

    keep = (d_near > missing_tol * L_rep) & (d_target > d_site_rep)   # missing AND outward
    cand = targets[keep]
    if len(cand) == 0:
        return np.zeros((0, 2))

    # Merge ghosts proposed by adjacent boundary sites
    ct = cKDTree(cand)
    done = np.zeros(len(cand), dtype=bool)
    L_med = float(np.median(spacing_fn(cand).ravel()))
    out = []
    for i in range(len(cand)):
        if done[i]:
            continue
        grp = ct.query_ball_point(cand[i], r=merge_tol * L_med)
        out.append(cand[grp].mean(axis=0))
        done[grp] = True

    return np.asarray(out)


# Basis & bond geometry

def compute_lattice_basis(
        spacing: float = 1.0,
        angles: float | Sequence[float] = np.pi / 3,
        degrees: bool = False
) -> np.ndarray:
    """
    Compute the 2x2 basis matrix B = [b1, b2] for a hexagonal lattice.
    """
    angles = np.asarray(angles, dtype=np.float64)
    if degrees:
        angles = np.deg2rad(angles)

    if angles.ndim == 0:
        # Standard: spacing is length of both vectors
        b1 = np.array([spacing, 0.0])
        b2 = np.array([spacing * np.cos(angles), spacing * np.sin(angles)])
    else:
        # Squished: triangle with internal angles a, b, c
        if angles.size != 3:
            raise ValueError('Provide 3 angles for a squished lattice.')

        angles = angles * (np.pi / np.sum(angles))
        a, b, c = angles

        # Law of sines: s1 / sin(a) = s2 / sin(b)
        # 'spacing' treated as the length of the primary horizontal bond (s2)
        s2 = spacing
        s1 = s2 * np.sin(a) / np.sin(b)

        b1 = np.array([s2, 0.0])
        b2 = np.array([s1 * np.cos(c), s1 * np.sin(c)])

    return np.vstack([b1, b2])


def base_bond_dirs(lattice_angles: float | Sequence[float], degrees: bool = False) -> np.ndarray:
    """
    Computes the 6 nearest-neighbour bond directions from lattice geometry.
    """
    basis = compute_lattice_basis(spacing=1.0, angles=lattice_angles, degrees=degrees)
    b1, b2 = basis[0], basis[1]

    # The 6 neighbours are b1, b2, and the closing side
    dirs = np.vstack([
        b1,         # Right
        b2,         # Top-right-ish
        b2 - b1,    # Top-left-ish
        -b1,        # Left
        -b2,        # Bottom-left-ish
        b1 - b2     # Bottom-right-ish
    ])
    return dirs / np.linalg.norm(dirs, axis=1, keepdims=True)


def local_bearings(
        points2d: ArrayLike,
        theta_fn: Callable,
        offsets_deg: Sequence[float] = (0.0, 60.0, 120.0),
        degrees: bool = False
) -> np.ndarray:
    """
    Evaluate the principal lattice bearings at arbitrary coordinates.

    Args:
        - points2d: (N, 2) spatial coordinates to query
        - theta_fn: Continuous hexatic axis interpolant
        - offsets: The relative angles of the lattice axes
        - degrees: If True, inputs/outputs are in degrees

    Returns:
        (N, 3) array of absolute row bearings
    """
    pts = np.atleast_2d(np.asarray(points2d, dtype=np.float64))
    base_angles_rad = theta_fn(pts).ravel()[:, None]

    offs = np.deg2rad(np.asarray(offsets_deg, dtype=np.float64))

    bearings_rad = (base_angles_rad + offs) % np.pi
    return np.rad2deg(bearings_rad) if degrees else bearings_rad


def bearings_to_angles(bearings: ArrayLike, degrees: bool = False) -> np.ndarray:
    """
    Convert three absolute row bearings into the three internal
    angles of the lattice unit-triangle.
    """
    bearings = np.asarray(bearings, dtype=np.float64)
    if degrees:
        bearings = np.deg2rad(bearings)

    if len(bearings) < 3:
        print('Could not resolve 3 rows, using regular 60° hex')
        # Fallback to perfect hex if the lattice is too disordered to trace
        angles = np.array([np.pi / 3, np.pi / 3, np.pi / 3])
    else:
        b = np.sort(bearings % np.pi)[:3]

        # Internal angles are the gaps between the three lines
        a1 = b[1] - b[0]
        a2 = b[2] - b[1]
        a3 = np.pi - (a1 + a2)  # wrap-around angle
        angles = np.array([a1, a2, a3])

    return np.rad2deg(angles) if degrees else angles


# Lattice graph measurements

def first_ring_gap(
        points2d: ArrayLike,
        neighbours: 'NeighbourGraph',
        degrees: bool = False
    ) -> np.ndarray:
    """
    Largest empty angular sector between consecutive first-ring neighbour bearings

    Complete ring (incl 5- or 7-fold disclinations) should have ~ 2*pi/degree
    Any cell missing a sector should be >= ~2*pi/3
    """

    points2d = np.asarray(points2d, dtype=np.float64)
    neighbours_list = ragged_neighbours(neighbours)
    gap = np.full(len(points2d), 2.0 * np.pi)

    for i, neighb_indices in enumerate(neighbours_list):
        if neighb_indices.size < 2:
            continue

        d = points2d[neighb_indices] - points2d[i]
        ang = np.sort(np.arctan2(d[:, 1], d[:, 0]))
        diffs = np.diff(np.concatenate([ang, ang[:1] + 2.0 * np.pi]))
        gap[i] = diffs.max()

    return np.rad2deg(gap) if degrees else gap


# Row tracing

def _walk(
        points2d: np.ndarray,
        neighbours: List[np.ndarray],
        start: int,
        heading: np.ndarray,
        max_steps: int,
        cos_step: float,
        cos_global: float,
) -> np.ndarray:
    """
    Greedy single-direction row walk.
    """
    path, cur = [], start
    h0 = heading / (np.linalg.norm(heading) + 1e-12)
    h, visited = h0, {start}

    for _ in range(max_steps):
        neighb = neighbours[cur]
        if neighb.size == 0:
            break

        vec = points2d[neighb] - points2d[cur]
        u = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
        dots = u @ h
        j = int(np.argmax(dots))

        if dots[j] < cos_step or (u[j] @ h0) < cos_global:
            break

        nxt = int(neighb[j])
        if nxt in visited:
            break

        path.append(nxt)
        visited.add(nxt)
        h, cur = u[j], nxt

    return np.array(path, dtype=int)


def trace_lattice_rows(
        points2d: ArrayLike,
        neighbours: 'NeighbourGraph',
        seed: Optional[Sequence[float]] = None,
        bearings: Optional[ArrayLike] = None,
        theta_fn: Optional[Callable] = None,
        offsets_deg: Sequence[float] = (0.0, 60.0, 120.0),
        max_steps: int = 500,
        step_tol_deg: float = 30.0,
        global_tol_deg: float = 70.0,
        degrees: bool = False
) -> Tuple[np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Trace lattice rows outward from the centre seed along the local axis directions.

    Provide either explicit 'bearings' or a hexatic interpolant function 'theta_fn'.

    Returns:
        seed_idx: Index of the central point.
        rows: List of (N, 2) arrays, each representing a sorted row of points.
        bearings: (R,) array of fitted angles for each row.
    """

    points2d = np.asarray(points2d, dtype=np.float64)

    # If no seed point, take closest to centroid
    seed = points2d.mean(axis=0) if seed is None else np.array(seed, dtype=np.float64)
    seed_idx = int(np.argmin(np.linalg.norm(points2d - seed, axis=1)))
    seed_pos = points2d[seed_idx]

    cos_step = np.cos(np.deg2rad(step_tol_deg))
    cos_global = np.cos(np.deg2rad(global_tol_deg))

    if bearings is None:
        if theta_fn is None:
            raise ValueError("Must provide either 'bearings' or 'theta_fn'")
        bearings_rad = local_bearings(seed_pos, theta_fn, offsets_deg, degrees=False)[0]
    else:
        bearings = np.asarray(bearings, dtype=np.float64)
        bearings_rad = np.deg2rad(bearings) if degrees else bearings

    neighb_list = ragged_neighbours(neighbours)

    out_rows, out_bearings = [], []

    for brg in bearings_rad[:3]:
        h0 = np.array([np.cos(brg), np.sin(brg)])

        # Walk both directions
        fwd = _walk(points2d, neighb_list, seed_idx, h0, max_steps, cos_step, cos_global)
        bwd = _walk(points2d, neighb_list, seed_idx, -h0, max_steps, cos_step, cos_global)

        # Combine into a single ordered row: [reversed backward] + [seed_idx] + [forward]
        row_pts = np.vstack([points2d[bwd[::-1]], points2d[[seed_idx]], points2d[fwd]])

        # Fit bearing
        b = principal_axis_angle(row_pts)
        if not np.isnan(b):
            out_rows.append(row_pts)
            out_bearings.append(b % np.pi)

    out_bearings = np.array(out_bearings)
    if degrees:
        out_bearings = np.rad2deg(out_bearings)
    return seed_pos, out_rows, out_bearings