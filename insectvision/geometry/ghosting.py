from typing import List, Optional, Tuple, Union, Callable
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree, ConvexHull

from insectvision.geometry.fields import interpolate_hexatic_field
from insectvision.geometry.lattice import bond_ioa
from insectvision.geometry.linalg import rotate2d, tangent_frames
from insectvision.geometry.neighbours import merge_close_points
from insectvision.geometry.spherical import sphere_to_stereo, radius_of_curvature
from insectvision.geometry.polygons import Polygon2D
from insectvision.utils import norm_l2


def combine_clouds(
        real_pos: ArrayLike,
        ghost_pos: Optional[ArrayLike] = None,
        real_dirs: Optional[ArrayLike] = None,
        ghost_dirs: Optional[ArrayLike] = None,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    Combine real and ghost point clouds, returning the joint arrays and a boolean mask
    of which points are real. Useful for cleanly building a unified tree for querying closed neighbourhoods.

    Returns:
        cloud_pos: (N + G, D)
        cloud_dirs: (N + G, D) if directions provided, else None
        is_real: (N + G,) bool
    """
    real_pos = np.asarray(real_pos, dtype=np.float64)
    N = len(real_pos)

    if ghost_pos is None or len(ghost_pos) == 0:
        cloud_pos = real_pos.copy()
    else:
        cloud_pos = np.vstack([real_pos, np.asarray(ghost_pos, dtype=np.float64)])

    is_real = np.arange(len(cloud_pos)) < N

    cloud_dirs = None
    if real_dirs is not None:
        real_dirs = np.asarray(real_dirs, dtype=np.float64)
        if ghost_dirs is None or len(ghost_dirs) == 0:
            cloud_dirs = real_dirs.copy()
        else:
            cloud_dirs = np.vstack([real_dirs, np.asarray(ghost_dirs, dtype=np.float64)])

    return cloud_pos, cloud_dirs, is_real


# Reject + merge kernel (shared)

def ghost_growth_kernel(
        existing: ArrayLike,
        candidates: ArrayLike,
        step_len: ArrayLike,
        outward_ok: ArrayLike,
        satisfied_factor: float = 0.5,
        merge_factor: float = 0.5,
        payload: Optional[ArrayLike] = None,
        normalize_payload: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Reject + merge kernel shared by the growth-based ghost producers.
    Works in 2D or 3D: 'existing' and 'candidates' just need matching last axis.

    Given per-boundary-site candidate points (already stepped out along the ideal
    bond directions by a coordinate-specific stepper), drop candidates that
        (a) already have a real neighbour ('bond satisfied'), or
        (b) do not point outward,
    then merge the survivors proposed by adjacent sites.

    Args:
        - existing: (M, D) cloud to test against (real + ghosts already added)
        - candidates: (C, D) proposed ghost points
        - step_len: (C,) ideal bond length at each candidate (rejection/merge scale)
        - outward_ok: (C,) bool, True where the candidate is 'more outside' than its site
        - satisfied_factor: reject if an existing point is within this * step_len
        - merge_factor: merge survivors closer than this * median(kept step_len)
        - payload: (C, P) optional per-candidate vector to co-merge (e.g. normals)
        - normalize_payload: renormalise merged payloads (for unit vectors)

    Returns:
        (G, D) ghosts, or ((G, D), (G, P)) if 'payload' is given. Possibly empty.
    """
    candidates = np.asarray(candidates, dtype=np.float64)
    step_len = np.asarray(step_len, dtype=np.float64).ravel()
    outward_ok = np.asarray(outward_ok, dtype=bool).ravel()

    if len(candidates) == 0:
        return (np.zeros((0, candidates.shape[1])),
                np.zeros((0, payload.shape[1]))) if payload is not None else np.zeros((0, candidates.shape[1]))

    # Reject candidates that are too close to existing points or pointing inward
    d_near, _ = cKDTree(existing).query(candidates)

    # Scale rejection by the local step length (spacing)
    reject_r = satisfied_factor * step_len
    keep = (d_near > reject_r) & outward_ok

    if not keep.any():
        return (np.zeros((0, candidates.shape[1])),
                np.zeros((0, payload.shape[1]))) if payload is not None else np.zeros((0, candidates.shape[1]))

    cand = candidates[keep]
    pay = payload[keep] if payload is not None else None

    # Determine merge radius based on the median spacing of surviving candidates
    merge_r = merge_factor * float(np.median(step_len[keep]))

    # Greedy merge (prevents chaining/collapsing the ring)
    tree = cKDTree(cand)
    done = np.zeros(len(cand), dtype=bool)

    out_pts: List[np.ndarray] = []
    out_pay: List[np.ndarray] = []

    for i in range(len(cand)):
        if done[i]:
            continue

        # Find all proposal siblings within merge_r
        indices = tree.query_ball_point(cand[i], r=merge_r)

        # Only consider those not already consumed by a previous merge
        valid = [idx for idx in indices if not done[idx]]
        if not valid:
            continue

        # Merge the cluster
        cluster_idx = np.array(valid)
        out_pts.append(cand[cluster_idx].mean(axis=0))

        if pay is not None:
            v = pay[cluster_idx].mean(axis=0)
            if normalize_payload:
                v /= np.linalg.norm(v) + 1e-12
            out_pay.append(v)

        # Mark all points in this cluster as consumed
        done[cluster_idx] = True

    merged_pts = np.array(out_pts)
    if pay is not None:
        return merged_pts, np.array(out_pay)

    return merged_pts


# Ghost producers

# Planar version (2D fitting frame)

def ghosts_from_growth_2d(
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

    For each near-boundary location, step the ideal bonds out (rotated into the world
    frame by the local axis angle) and keep a ghost at p + L*b when (a) that bond has
    no existing neighbour and (b) it points further outside the domain than the site
    itself. Ghosts proposed by adjacent sites are then merged.

    Args:
        - points2d: (N, 2) current lattice
        - theta_fn: (M,2) -> (M,) local hexatic axis angle (rad)
        - spacing_fn: (M,2) -> (M,) local target spacing
        - bond_dirs: (b, 2) ideal unit bond directions of the base cell (e.g. base_bond_dirs(lattice_angles))
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

    d_hull = domain.signed_distance(pts)  # < 0 inside
    near = np.where(d_hull > -boundary_band * avg_spacing)[0]
    if near.size == 0:
        return np.zeros((0, 2))

    P = pts[near]
    L = spacing_fn(P).ravel()
    th = theta_fn(P).ravel()
    b = len(bond_dirs)

    cand_dirs = rotate2d(bond_dirs[None], th[:, None])
    cand = (P[:, None, :] + L[:, None, None] * cand_dirs).reshape(-1, 2)

    step_len = np.repeat(L, b)
    d_site = np.repeat(d_hull[near], b)   # signed dist of the site that proposed each candidate

    # Outward == candidate sits further outside the hull than its proposing site
    outward_ok = domain.signed_distance(cand) > d_site

    return ghost_growth_kernel(
        existing=pts,
        candidates=cand,
        step_len=step_len,
        outward_ok=outward_ok,
        satisfied_factor=missing_tol,
        merge_factor=merge_tol,
    )


def _one_sphere_ring(
        curr_pos: np.ndarray,
        curr_norms: np.ndarray,
        seeds: np.ndarray,
        theta_fn: Callable,
        frame: Tuple[np.ndarray, np.ndarray, np.ndarray],
        stereo_domain: 'Polygon2D',
        real_tree: cKDTree,
        real_ioa: np.ndarray,
        fallback_R: float,
        collision_factor: float,
        merge_factor: float,
        aniso_axis: str,
) -> Tuple[np.ndarray, np.ndarray]:

    P = curr_pos[seeds]
    D = curr_norms[seeds]
    M = len(seeds)

    _, j = real_tree.query(P)
    minor, major = real_ioa[j, 0], real_ioa[j, 1]

    R = radius_of_curvature(P, D, cKDTree(curr_pos), curr_norms)
    R = np.where(np.isfinite(R), R, fallback_R)

    T_right, T_up = tangent_frames(D)

    # Get local lattice orientation
    q_seed, *_ = sphere_to_stereo(D, frame)
    th = np.asarray(theta_fn(q_seed), dtype=np.float64).ravel()

    # Construct 6-fold bond fan
    phi = np.deg2rad(np.arange(0, 360, 60))
    bearings = th[:, None] + phi[None, :]  # (M, 6)

    # Calculate IOA for each bond in the fan
    aniso = th if aniso_axis == 'hexatic' else 0.0
    step_ioa = bond_ioa(bearings, minor[:, None], major[:, None], axis=aniso[:, None])

    # Construct unit direction vectors for each bond
    tan_vec = (T_right[:, None, :] * np.cos(bearings)[..., None] +
               T_up[:, None, :] * np.sin(bearings)[..., None])

    v_dir = D[:, None, :] * np.cos(step_ioa)[..., None] + tan_vec * np.sin(step_ioa)[..., None]
    v_dir = norm_l2(v_dir.reshape(-1, 3))

    # Calculate step positions
    v_pos = P[:, None, :] + R[:, None, None] * (v_dir.reshape(M, 6, 3) - D[:, None, :])
    step_len = R[:, None] * step_ioa

    # Outward test
    q_cand, *_ = sphere_to_stereo(v_dir, frame)
    cand_sd = stereo_domain.signed_distance(q_cand).reshape(M, 6)
    seed_sd = stereo_domain.signed_distance(q_seed)
    outward_ok = (cand_sd > seed_sd[:, None]).ravel()

    return ghost_growth_kernel(
        existing=curr_pos,
        candidates=v_pos.reshape(-1, 3),
        step_len=step_len.ravel(),
        outward_ok=outward_ok,
        satisfied_factor=collision_factor,
        merge_factor=merge_factor,
        payload=v_dir,
        normalize_payload=True
    )


# 3D version: mirrors the planar shape one ring at a time

def ghosts_from_growth_3d(
        positions: ArrayLike,
        directions: ArrayLike,
        is_edge: ArrayLike,
        ioa_angles: ArrayLike,
        n_rows: int = 3,
        curvature_radius: Optional[float] = None,
        collision_factor: float = 0.5,
        merge_factor: float = 0.7,
        field_smoothing: float = 0.2,
        aniso_axis: str = 'hexatic',
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Grow 'n_rows' outward rings of virtual ommatidia on the sphere, following the
    smoothed hexatic axis field and the local (measured) interommatidial angles.

    Args:
        - positions: (N, 3) real lens world positions
        - directions: (N, 3) real optical axes
        - is_edge: (N,) bool, boundary ommatidia to seed the first ring from
        - ioa_angles: (N, 2) measured (minor, major) IOA per real ommatidium
        - n_rows: number of outward rings to grow
        - curvature_radius: fallback radius when the local estimate fails (None -> global median)
        - collision_factor / merge_factor: passed to propose_ghost_ring
        - field_smoothing: smoothing for the hexatic interpolant
        - aniso_axis: 'hexatic' -> major-IOA axis follows the local hexatic bearing

    Returns:
        (virtual_positions (G, 3), virtual_directions (G, 3))
    """
    positions = np.asarray(positions, dtype=np.float64)
    directions = norm_l2(np.asarray(directions, dtype=np.float64))
    ioa_angles = np.asarray(ioa_angles, dtype=np.float64).reshape(-1, 2)
    is_edge = np.asarray(is_edge, dtype=bool)

    if len(positions) < 3:
        return np.zeros((0, 3)), np.zeros((0, 3))

    # Fixed stereo frame + orientation field + outward-test domain, from the real eye
    pts2d, fwd, right, up = sphere_to_stereo(directions)
    frame = (fwd, right, up)
    theta_fn = interpolate_hexatic_field(pts2d, smoothing=field_smoothing)
    stereo_domain = Polygon2D.from_points(pts2d)

    real_tree = cKDTree(positions)

    # Global fallback radius of curvature (same estimator, one shot) if none supplied
    if curvature_radius is None:
        radii = radius_of_curvature(
            query_positions=positions,
            query_normals=directions,
            tree=real_tree,
            tree_normals=directions,
        )
        # Reduce to a single scalar for the global fallback
        fallback_R = np.nanmedian(radii)
        if not np.isfinite(fallback_R):
            fallback_R = 1.0
    else:
        fallback_R = float(curvature_radius)

    curr_pos = positions.copy()
    curr_dirs = directions.copy()
    seeds = np.where(is_edge)[0]
    n0 = len(positions)

    for _ in range(max(0, n_rows)):
        if seeds.size == 0:
            break

        new_pos, new_dirs = _one_sphere_ring(
            curr_pos, curr_dirs, seeds, theta_fn, frame, stereo_domain,
            real_tree, ioa_angles, fallback_R, collision_factor, merge_factor, aniso_axis,
        )
        if len(new_pos) == 0:
            break

        start = len(curr_pos)
        curr_pos = np.vstack([curr_pos, new_pos])
        curr_dirs = np.vstack([curr_dirs, new_dirs])
        seeds = np.arange(start, len(curr_pos))

    return curr_pos[n0:], curr_dirs[n0:]


# Planar mirror mode (standalone, does not use the kernel)

def ghosts_from_mirror(
        points2d: ArrayLike,
        depth: Union[float, ArrayLike],
        domain: Union['Polygon2D', 'ConvexHull'],
        merge_factor: float = 0.5,
) -> np.ndarray:
    """
    Mirror points lying within 'depth' of the boundary edges to the outside.
    Creates a symmetric Voronoi pressure that stops edge points from squashing
    against the boundary.

    Args:
        - points2d: (N, 2) array of real ommatidia positions.
        - depth: float or (N,) array, local search distance for mirroring.
        - domain: Polygon2D or ConvexHull containing the .equations [nx, ny, offset].
        - merge_factor: factor of local depth used for merging overlapping ghosts.
    """
    pts = np.asarray(points2d, dtype=np.float64)
    d_limit = np.broadcast_to(np.asarray(depth, dtype=np.float64), pts.shape[0])

    mirrored = []

    for eq in domain.equations:
        normal, offset = eq[:2], eq[2]

        d_face = pts @ normal + offset

        # points within 'depth' of this specific edge
        mask = (d_face > -d_limit) & (d_face <= 0)

        if np.any(mask):
            # Reflect: since dist is negative (inside), this moves the point outside
            mirrored.append(pts[mask] - 2.0 * d_face[mask, None] * normal[None, :])

    if not mirrored:
        return np.zeros((0, 2))

    all_ghosts = np.vstack(mirrored)
    m_rad = merge_factor * np.median(d_limit)

    return merge_close_points(all_ghosts, radius=m_rad, reduce=np.median)