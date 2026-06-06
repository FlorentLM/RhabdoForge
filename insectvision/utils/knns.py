from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
from scipy.spatial import cKDTree
from insectvision.utils.math import norm_l2


def knn(
        tree: cKDTree,
        query_pts: np.ndarray,
        k: int,
        drop_self: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    k-nearest-neighbour query against a prebuilt tree.

    Args:
        tree: a prebuilt cKDTree (owned/cached by the caller)
        query_pts: (Q, D) query points
        k: number of neighbours requested. Clamped to the tree size.
        drop_self: if True, query k+1 and drop the first (self) column. Use for
            self-queries (query points are tree members). Set False when querying
            external points, where the nearest match is not the point itself.

    Returns:
        (distances, indices), both (Q, k_eff) with k_eff = min(k, n - 1) when drop_self else min(k, n)
        When k_eff == 0, returns (Q, 0) arrays
    """
    query_pts = np.atleast_2d(query_pts)
    n = tree.n
    max_k = (n - 1) if drop_self else n
    k_eff = min(int(k), max_k)

    if k_eff <= 0:
        q = query_pts.shape[0]
        return np.empty((q, 0), dtype=float), np.empty((q, 0), dtype=np.intp)

    kq = k_eff + 1 if drop_self else k_eff
    dist, idx = tree.query(query_pts, k=kq)

    if idx.ndim == 1:  # kq == 1
        idx = idx.reshape(-1, 1)
        dist = dist.reshape(-1, 1)
    if drop_self:
        idx = idx[:, 1:]
        dist = dist[:, 1:]

    return dist, idx.astype(np.intp)


def local_spacing(
        tree: cKDTree,
        query_pts: Optional[np.ndarray] = None,
        k: int = 6,
) -> np.ndarray:
    """
    Mean distance from each query point to its k nearest *other* points.

    'k' counts neighbours excluding the point itself. If 'query_pts' is None the
    tree's own data is used (the common "mean spacing of this cloud" case).
    Returns NaN for points with no available neighbours.
    """
    pts = tree.data if query_pts is None else query_pts
    dist, _ = knn(tree, pts, k, drop_self=True)
    if dist.shape[1] == 0:
        return np.full(dist.shape[0], np.nan)
    return dist.mean(axis=1)


def angle_to_chord(angle_rad) -> np.ndarray:
    """Great-circle angle (rad) -> Euclidean chord length on the unit sphere."""
    return 2.0 * np.sin(0.5 * np.asarray(angle_rad, dtype=float))


def chord_to_angle(chord) -> np.ndarray:
    """Euclidean chord length on the unit sphere -> great-circle angle (rad)."""
    return 2.0 * np.arcsin(np.clip(0.5 * np.asarray(chord, dtype=float), -1.0, 1.0))


def within_angle(tree: cKDTree, query_dirs: np.ndarray, angle_rad: float) -> List[np.ndarray]:
    """
    Neighbours within a great-circle 'angle_rad' of each query direction.

    'tree' must be built over unit direction vectors. Returns scipy
    query_ball_point output (a list of index arrays, one per query direction).
    """
    return tree.query_ball_point(query_dirs, r=angle_to_chord(angle_rad))


def lookat_topk(
        pos: np.ndarray,
        dirs: np.ndarray,
        targets: np.ndarray,
        k: int,
        conflict_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Local indices (Q, k_eff) of the lenses best looking at each target.

    Score is the dot product of each lens optical axis with the unit vector from
    that lens to the target. Columns are ordered best-first. 'conflict_mask' (a
    boolean over the lenses) excludes conflicted lenses from selection.
    """
    desired = targets[:, None, :] - pos[None, :, :]
    desired = norm_l2(desired, axis=-1)
    dots = np.einsum('jk,ijk->ij', dirs, desired)

    if conflict_mask is not None:
        dots[:, conflict_mask] = -np.inf

    k_eff = min(k, dots.shape[1])
    part = np.argpartition(dots, -k_eff, axis=1)[:, -k_eff:]
    top = np.take_along_axis(dots, part, axis=1)
    order = np.argsort(top, axis=1)[:, ::-1]
    return np.take_along_axis(part, order, axis=1)
