from typing import Optional, Union
import numpy as np
from numpy.typing import ArrayLike

from insectvision.utils.shared import norm_l2


def tangent_frames(
        directions: ArrayLike,
        world_up: Optional[ArrayLike] = None,
        world_right: Optional[ArrayLike] = None,
):
    """
    Orthonormal basis (right, up) for each direction vector.

    Handles poles / gimbal lock by blending the reference vector near the
    singularity. Accepts a single (3,) vector or an (N, 3) batch.
    """
    from insectvision.engine.world_utils import WORLD_UP, WORLD_RIGHT

    if world_up is None:
        world_up = WORLD_UP
    if world_right is None:
        world_right = WORLD_RIGHT

    dirs = np.asarray(directions, dtype=np.float32)
    is_1d = (dirs.ndim == 1)
    if is_1d:
        dirs = dirs[np.newaxis, :]

    forward = norm_l2(dirs)

    dots = np.abs(forward @ world_up)

    blend = np.clip((dots - 0.98) / (1.0 - 0.98), 0.0, 1.0)[:, np.newaxis]
    ref_ups = (1.0 - blend) * world_up + blend * world_right

    right = norm_l2(np.cross(forward, ref_ups))
    up = norm_l2(np.cross(right, forward))

    if is_1d:
        return right[0], up[0]
    return right, up


def tangent_bearing(
        target_dirs: ArrayLike,
        home_dirs: ArrayLike,
        right: ArrayLike,
        up: ArrayLike,
        degrees: bool = False,
) -> np.ndarray:
    """
    Project (target - home) onto a (right, up) tangent plane and return the bearing.
    """

    delta = np.asarray(target_dirs) - np.asarray(home_dirs)
    u = np.sum(delta * np.asarray(right), axis=-1)
    v = np.sum(delta * np.asarray(up), axis=-1)
    bearing = np.arctan2(v, u)

    return np.rad2deg(bearing) if degrees else bearing


def rotate_in_tangent_plane(
        vectors: ArrayLike,
        normals: ArrayLike,
        angles: ArrayLike,
        normalize: bool = True,
) -> np.ndarray:
    """
    Rotate tangent vectors about their local surface normal by 'angles'.

    Args:
        vectors: (..., 3) vectors to rotate (don't need to be exactly perpendicular to n)
        normals: (..., 3) unit surface normals (e.g. lens optical axes)
        angles:  (...,)   rotation angle per row (rad)
        normalize: renormalise the result
    """
    v = np.asarray(vectors, dtype=np.float64)
    n = np.asarray(normals, dtype=np.float64)
    a = np.asarray(angles, dtype=np.float64)[..., None]

    rotated = v * np.cos(a) + np.cross(n, v) * np.sin(a)
    if normalize:
        rotated = norm_l2(rotated).astype(np.float32)
    return rotated.astype(np.float32)


def rotate_vectors(
        vectors: Union[np.ndarray, list],
        axes: Union[np.ndarray, list],
        angles: Union[np.ndarray, float],
        degrees: bool = True,
        normalize_axes: bool = False,
) -> np.ndarray:
    """Rotate vectors around arbitrary axes (Rodrigues' formula)."""

    v = np.asarray(vectors)
    k = np.asarray(axes)
    theta = np.asarray(angles)

    if normalize_axes:
        k = k / np.linalg.norm(k, axis=-1, keepdims=True)

    if degrees:
        theta = np.deg2rad(theta)

    theta = theta[..., np.newaxis]

    c = np.cos(theta)
    s = np.sin(theta)
    cross = np.cross(k, v, axis=-1)
    dot = np.sum(k * v, axis=-1, keepdims=True)

    return v * c + cross * s + k * dot * (1.0 - c)


def local_to_world(coords: ArrayLike, *basis: ArrayLike) -> np.ndarray:
    """
    Express local coordinates in world space against a set of basis vectors.
    The result is *not* renormalised.
    """
    coords = np.asarray(coords)
    if coords.shape[-1] != len(basis):
        raise ValueError(
            f"local_to_world: coords last axis ({coords.shape[-1]}) must equal "
            f"the number of basis vectors ({len(basis)})"
        )
    out = coords[..., 0, None] * np.asarray(basis[0])
    for i in range(1, len(basis)):
        out = out + coords[..., i, None] * np.asarray(basis[i])
    return out
