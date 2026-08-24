from typing import TYPE_CHECKING, Sequence, List, Callable, Tuple, Optional
import numpy as np
from numpy.typing import ArrayLike

from rhabdoforge.geometry.linalg import principal_axes
from rhabdoforge.geometry.neighbours import ragged_neighbours
from rhabdoforge.utils import norm_l2

if TYPE_CHECKING:
    from rhabdoforge.geometry.neighbours import NeighbourGraph


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
    # return norm_l2(dirs)
    return dirs


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


def bond_ioa(
        bearings: ArrayLike,
        minor: ArrayLike,
        major: ArrayLike,
        axis: ArrayLike = 0.0,
    ) -> np.ndarray:
    """
    Interommatidial angle for bonds at the given bearings (elliptical anisotropy model).
    IOA is 'major' along 'axis' and 'minor' perpendicular to it, smoothly interpolated between.

    Reduces to isotropic exactly when minor == major.

    TODO / Note: (minor, major) is lossy (it's 2-value summary of a possibly 3-length squished hex),
        and it does not encode which hex direction the major axis goes along... Might need to improve that.
        For exact per-bond IOAs, should pass true per-bond lengths instead (length ~ IOA at small angles).
        For mesh ghosts the ellipse is fine (because ghosts are discarded after closing boundary cells).

    Args:
        - bearings: bond bearings in radians (e.g. base_angle + k*60deg)
        - minor: tightest angular separation (broadcastable)
        - major: widest angular separation (broadcastable)
        - axis: bearing of the major axis (defaults to 0 == bond 0)
    """
    phi = np.asarray(bearings, dtype=np.float64) - np.asarray(axis, dtype=np.float64)
    major = np.asarray(major, dtype=np.float64)
    minor = np.asarray(minor, dtype=np.float64)

    c, s = np.cos(phi), np.sin(phi)
    return np.hypot(major * c, minor * s)


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

def lattice_confidence(
        hex_order: ArrayLike,
        is_boundary: ArrayLike,
        hex_order_lo: float = 0.25,
        hex_order_hi: float = 0.9,
        edge_weight: float = 0.25,
    ) -> np.ndarray:
    """
    Per-lens confidence in [0, 1] that the *local* lattice metric can be trusted.
    Used for the fitting and during model construction (spacing, IOA, whitening covariance).

    Args:
        hex_order: (N,) |Psi6| in [0, 1]
        is_boundary: (N,) bool mask (e.g. from identify_boundary_points)
        hex_order_lo/hi: linear ramp for order contribution
        edge_weight: multiplier for boundary points (distrust cue)
    """
    hex_order = np.asarray(hex_order, dtype=np.float64)
    is_boundary = np.asarray(is_boundary, dtype=bool)

    conf = np.clip((hex_order - hex_order_lo) / max(hex_order_hi - hex_order_lo, 1e-9), 0.0, 1.0)

    return (conf * np.where(is_boundary, float(edge_weight), 1.0)).astype(np.float32)


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

        # Fit bearing by PCA
        major = principal_axes(row_pts)[0][:, 0]
        b = np.arctan2(major[1], major[0])

        if not np.isnan(b):
            out_rows.append(row_pts)
            out_bearings.append(b % np.pi)

    out_bearings = np.array(out_bearings)
    if degrees:
        out_bearings = np.rad2deg(out_bearings)
    return seed_pos, out_rows, out_bearings
