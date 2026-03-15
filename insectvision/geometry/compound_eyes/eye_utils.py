import numpy as np

from insectvision.engine.utils import WORLD_UP, WORLD_RIGHT


# TODO: these are pure math utils, not eye utils, they could be moved elsewhere

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