from typing import Tuple
import numpy as np
from scipy.spatial import KDTree

from insectvision.engine.utils import WORLD_UP, WORLD_RIGHT


# TODO: rotate_vectors and tangent_frames are pure math utils, not eye utils

def rotate_vectors(vectors: np.ndarray, axes: np.ndarray, angles: np.ndarray, degrees: bool = True) -> np.ndarray:
    """Rotate vectors around axes (Rodrigues formula)."""

    angles_arr = np.asarray(angles)
    angles_rad = np.deg2rad(angles_arr) if degrees else angles_arr

    if angles_rad.ndim == 0:
        cos_a = np.cos(angles_rad)
        sin_a = np.sin(angles_rad)
    else:
        cos_a = np.cos(angles_rad)[:, np.newaxis]
        sin_a = np.sin(angles_rad)[:, np.newaxis]

    rotated = (
            vectors * cos_a
            + np.cross(axes, vectors) * sin_a
            + axes * np.sum(axes * vectors, axis=1, keepdims=True) * (1 - cos_a)
    )
    return rotated


def tangent_frames(direction: np.ndarray):
    """
    Compute orthonormal frame for one or more directions.
    """
    # TODO: This can gimbal lock
    # TODO: Use this function in more places

    dots = np.abs(direction @ WORLD_UP)
    ref_ups = np.where(dots[:, np.newaxis] > 0.999, WORLD_RIGHT, WORLD_UP)

    local_right = np.cross(direction, ref_ups)
    local_right /= np.linalg.norm(local_right, axis=1, keepdims=True)
    local_up = np.cross(local_right, direction)

    return local_right, local_up


##

def compute_lattice_properties(
        directions: np.ndarray,
        positions: np.ndarray,
        k: int = 8,
        neighbour_dist_factor: float = 1.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate local lattice properties from lens positions (used by both receptor array construction paths).
    """

    N = len(directions)
    if N <= k:
        z = np.zeros(N, dtype=np.float32)
        return z, z, z, np.zeros(N, dtype=np.uint32)

    # Physical direction vectors from common centre
    eye_center = np.mean(positions, axis=0)
    phys_dirs = positions - eye_center
    norms = np.linalg.norm(phys_dirs, axis=1, keepdims=True)
    np.divide(phys_dirs, norms, out=phys_dirs, where=norms != 0)

    phys_kdtree = KDTree(phys_dirs)
    distances, indices = phys_kdtree.query(phys_dirs, k=k + 1)
    nb_indices = indices[:, 1:]
    nb_distances = distances[:, 1:]

    if nb_indices.size == 0:
        z = np.zeros(N, dtype=np.float32)
        return z, z, z, np.zeros(N, dtype=np.uint32)

    angular_sep = 2.0 * np.arcsin(np.clip(nb_distances / 2.0, -1.0, 1.0))
    dist_to_closest = angular_sep[:, 0]
    is_immediate = angular_sep <= dist_to_closest[:, np.newaxis] * neighbour_dist_factor
    nb_counts = np.sum(is_immediate, axis=1)

    # Local tangent planes
    dot_up = np.abs(phys_dirs @ WORLD_UP)
    is_polar = dot_up > 0.999
    ref_ups = np.where(is_polar[:, np.newaxis], WORLD_RIGHT, WORLD_UP)

    local_y = ref_ups - phys_dirs * np.sum(phys_dirs * ref_ups, axis=1, keepdims=True)
    local_y /= np.linalg.norm(local_y, axis=1, keepdims=True)
    local_x = np.cross(local_y, phys_dirs)

    nb_phys = phys_dirs[nb_indices]
    delta = nb_phys - phys_dirs[:, np.newaxis, :]

    proj_x = np.sum(delta * local_x[:, np.newaxis, :], axis=2)
    proj_y = np.sum(delta * local_y[:, np.newaxis, :], axis=2)

    tilts = np.zeros(N, dtype=np.float32)
    ioa_major = np.zeros(N, dtype=np.float32)
    ioa_minor = np.zeros(N, dtype=np.float32)

    for i in range(N):

        mask = is_immediate[i]
        pts = np.column_stack([proj_x[i, mask], proj_y[i, mask]])

        if pts.shape[0] < 2:
            avg = np.mean(angular_sep[i, mask]) if np.any(mask) else 0.0
            ioa_major[i], ioa_minor[i], tilts[i] = avg, avg, 0.0
            continue

        cov = np.cov(pts, rowvar=False)
        evals, evecs = np.linalg.eigh(cov)
        primary = evecs[:, np.argmax(evals)]
        tilts[i] = np.arctan2(primary[1], primary[0])

        ct, st = np.cos(-tilts[i]), np.sin(-tilts[i])
        ax = proj_x[i, mask] * ct - proj_y[i, mask] * st
        ay = proj_x[i, mask] * st + proj_y[i, mask] * ct
        angles_aligned = np.arctan2(ay, ax)

        sep = angular_sep[i, mask]

        maj_idx = np.argsort(np.abs(np.sin(angles_aligned)))[:2]
        min_idx = np.argsort(np.abs(np.cos(angles_aligned)))[:2]

        ioa_major[i] = np.mean(sep[maj_idx])
        ioa_minor[i] = np.mean(sep[min_idx])

    final_minor = np.minimum(ioa_minor, ioa_major)
    final_major = np.maximum(ioa_minor, ioa_major)

    return final_minor, final_major, tilts, nb_counts.astype(np.uint32)