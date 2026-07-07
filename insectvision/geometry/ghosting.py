from typing import List, Optional, Tuple, Union, Callable
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

from insectvision.geometry.fields import interpolate_hexatic_field
from insectvision.geometry.lattice import bond_ioa
from insectvision.geometry.linalg import rotate2d
from insectvision.geometry.neighbours import merge_close_points
from insectvision.geometry.spherical import sphere_to_stereo, stereo_to_sphere, radius_of_curvature
from insectvision.geometry.polygons import Polygon2D
from insectvision.utils import norm_l2


# Reject + merge kernel (shared)

def _ghost_growth_kernel(
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
        - payload: (C, P) optional per-candidate vector to co-merge (e.g. directions)
        - normalize_payload: renormalise merged payloads (for unit vectors)

    Returns:
        (G, D) ghosts, or ((G, D), (G, P)) if 'payload' is given. Possibly empty.
    """
    existing = np.asarray(existing, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    step_len = np.asarray(step_len, dtype=np.float64).ravel()
    outward_ok = np.asarray(outward_ok, dtype=bool).ravel()

    D = candidates.shape[1] if candidates.ndim == 2 else 2
    pay = None if payload is None else np.asarray(payload, dtype=np.float64)
    P = D if pay is None else (pay.shape[1] if pay.ndim == 2 else D)

    def _empty():
        return (np.zeros((0, D)), np.zeros((0, P))) if pay is not None else np.zeros((0, D))

    if len(candidates) == 0:
        return _empty()

    # Reject: bonds already satisfied by an existing point, and inward candidates.
    d_near, _ = cKDTree(existing).query(candidates)
    keep = (d_near > satisfied_factor * step_len) & outward_ok
    if not keep.any():
        return _empty()

    cand = candidates[keep]
    pay = None if pay is None else pay[keep]
    merge_r = merge_factor * float(np.median(step_len[keep]))

    # Merge siblings proposed by adjacent sites: greedy, index-order
    # (deterministic: the lowest surviving index claims its ball). Payload averaged in lockstep.
    tree = cKDTree(cand)
    done = np.zeros(len(cand), dtype=bool)
    out_pts: List[np.ndarray] = []
    out_pay: List[np.ndarray] = []

    for i in range(len(cand)):
        if done[i]:
            continue

        grp = tree.query_ball_point(cand[i], r=merge_r)
        out_pts.append(cand[grp].mean(axis=0))

        if pay is not None:
            v = pay[grp].mean(axis=0)
            out_pay.append(v / max(np.linalg.norm(v), 1e-12) if normalize_payload else v)
        done[grp] = True

    merged = np.asarray(out_pts)
    return (merged, np.asarray(out_pay)) if pay is not None else merged


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

    return _ghost_growth_kernel(
        existing=pts,
        candidates=cand,
        step_len=step_len,
        outward_ok=outward_ok,
        satisfied_factor=missing_tol,
        merge_factor=merge_tol,
    )


def _one_sphere_ring(
        curr_pos: np.ndarray,
        curr_dirs: np.ndarray,
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
    """
    One outward ring on the sphere. Returns (pos, dirs) of accepted ghosts.
    """

    fwd, right, up = frame
    P = curr_pos[seeds]        # (M, 3)
    D = curr_dirs[seeds]       # (M, 3)
    M = len(seeds)

    # Measured IOA at the nearest *real* ommatidium (seeds may be ghosts in later rings)
    _, j = real_tree.query(P)
    minor, major = real_ioa[j, 0], real_ioa[j, 1]

    # Local radius of curvature from the current cloud
    R = radius_of_curvature(
        query_pos=P,
        query_dirs=D,
        tree=cKDTree(curr_pos),
        cloud_normals=curr_dirs
    )
    R = np.where(np.isfinite(R), R, fallback_R)  # (M,)

    # Base tangent from the local hexatic bearing
    q_seed, *_ = sphere_to_stereo(D, frame)   # (M, 2)
    base_angle = np.asarray(theta_fn(q_seed), dtype=np.float64).ravel()

    uv = np.stack([np.cos(base_angle), np.sin(base_angle)], axis=1)
    d1 = stereo_to_sphere(q_seed + 1e-6 * uv, fwd, right, up)
    t = d1 - D
    t = t - np.einsum('md,md->m', t, D)[:, None] * D      # project onto tangent plane of D
    base_tan = norm_l2(t)
    cross_db = np.cross(D, base_tan)

    # 6-fold bond fan, anisotropic IOA as the angular step
    phi = np.arange(6) * (np.pi / 3.0)
    cph, sph = np.cos(phi), np.sin(phi)
    axis = base_angle if aniso_axis == 'hexatic' else np.zeros(M)
    bearings = base_angle[:, None] + phi[None, :]         # (M, 6)
    step_ioa = bond_ioa(bearings, minor[:, None], major[:, None], axis[:, None])  # (M, 6)

    step_vec = (base_tan[:, None, :] * cph[None, :, None]
                + cross_db[:, None, :] * sph[None, :, None])                       # (M, 6, 3)
    v_dir = norm_l2(D[:, None, :] * np.cos(step_ioa)[..., None]
                    + step_vec * np.sin(step_ioa)[..., None])                      # (M, 6, 3)
    v_pos = P[:, None, :] + R[:, None, None] * (v_dir - D[:, None, :])             # (M, 6, 3)
    step_len = R[:, None] * step_ioa                                               # (M, 6) arc len

    # Outward test
    v_dir_flat = v_dir.reshape(-1, 3)
    q_cand, *_ = sphere_to_stereo(v_dir_flat, frame)
    cand_sd = stereo_domain.signed_distance(q_cand).reshape(M, 6)
    seed_sd = stereo_domain.signed_distance(q_seed)

    outward_ok = (cand_sd > seed_sd[:, None]).ravel()

    return _ghost_growth_kernel(
        existing=curr_pos,
        candidates=v_pos.reshape(-1, 3),
        step_len=step_len.ravel(),
        outward_ok=outward_ok,
        satisfied_factor=collision_factor,
        merge_factor=merge_factor,
        payload=v_dir_flat,
        normalize_payload=True,
    )


# 3D version: mirrors the planar shape one ring at a time

def ghosts_from_growth_3d(
        positions: ArrayLike,
        directions: ArrayLike,
        is_edge: ArrayLike,
        ioa_angles: ArrayLike,
        n_rows: int = 2,
        curvature_radius: Optional[float] = None,
        collision_factor: float = 0.7,
        merge_factor: float = 1.1,
        field_smoothing: float = 0.1,
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
        - n_rows: number of outward rings
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
            query_pos=positions,
            query_dirs=directions,
            tree=real_tree,
            cloud_normals=directions,
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

# TODO: This should accept 'domain' and have a signature closer to ghosts_from_growth
#    ...or just get rid of it?

def ghosts_from_mirror(points2d: ArrayLike, equations: ArrayLike, depth: float) -> np.ndarray:
    """
    Mirror points lying within 'depth' of the hull edges to the outside.

    Creates a symmetric Voronoi pressure that stops edge points from squashing
    against the boundary. 'equations' are scipy ConvexHull half-plane rows
    [nx, ny, offset] with outward-pointing normals.
    """
    points2d = np.asarray(points2d, dtype=np.float64)
    equations = np.asarray(equations, dtype=np.float64)

    mirrored = []
    for eq in equations:
        normal, offset = eq[:2], eq[2]

        # scipy.ConvexHull convention: normals point outward
        dist = points2d @ normal + offset

        mask = (dist > -depth) & (dist <= 0)  # inside the hull and within 'depth' of the edge
        if not np.any(mask):
            continue

        close_pts = points2d[mask]
        dist_close = dist[mask]

        # Reflect across the edge
        mirrored_pts = close_pts - 2.0 * dist_close[:, None] * normal[None, :]
        mirrored.append(mirrored_pts)

    if not mirrored:
        return np.zeros((0, 2))

    return merge_close_points(np.vstack(mirrored), radius=0.5 * depth, reduce=np.mean)