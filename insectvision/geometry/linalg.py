from typing import Optional, Tuple
import numpy as np
from numpy.typing import ArrayLike

from insectvision.utils import _match_batch, norm_l2


def tangent_frames(
        directions: ArrayLike,
        world_up: Optional[ArrayLike] = None,
        world_right: Optional[ArrayLike] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
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

    dirs = np.asarray(directions, dtype=np.float64)
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
        target_directions: ArrayLike,
        home_directions: ArrayLike,
        right: ArrayLike,
        up: ArrayLike,
        degrees: bool = False,
) -> np.ndarray:
    """
    Project (target - home) onto a (right, up) tangent plane and return the bearing.
    """

    target, home = _match_batch(target_directions, home_directions)
    delta = target - home

    r_vec = _match_batch(delta, right)[1]
    u_vec = _match_batch(delta, up)[1]

    u = np.einsum('...k,...k->...', delta, r_vec)
    v = np.einsum('...k,...k->...', delta, u_vec)

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


# TODO: This one is unused, might get rid of
def rotate_vectors(
        vectors: ArrayLike,
        axes: ArrayLike,
        angles: ArrayLike,
        degrees: bool = True,
        normalize_axes: bool = False,
) -> np.ndarray:
    """Rotate vectors around arbitrary axes (Rodrigues' formula)."""

    v = np.asarray(vectors, dtype=np.float64)
    k = np.asarray(axes, dtype=np.float64)
    theta = np.asarray(angles, dtype=np.float64)

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
    out_coords = np.asarray(coords)
    if coords.shape[-1] != len(basis):
        raise ValueError(f'local_to_world: coords last axis must equal number of basis vectors')

    b0 = _match_batch(out_coords, basis[0])[1]
    out = out_coords[..., 0, None] * b0

    for i in range(1, len(basis)):
        bi = _match_batch(out_coords, basis[i])[1]
        out = out + out_coords[..., i, None] * bi
    return out


def rot2d(theta: float | ArrayLike, degrees: bool = False) -> np.ndarray:
    """2x2 rotation matrix for a single angle or array of."""
    theta = np.asarray(theta, dtype=np.float64)
    if degrees:
        theta = np.deg2rad(theta)
    c, s = np.cos(theta), np.sin(theta)
    out = np.stack([[c, -s], [s, c]]).squeeze()
    return out if out.ndim == 2 else out.T


def principal_axis_angle(points2d: ArrayLike, degrees: bool = False) -> float:
    """
    Orientation of the major principal axis of a 2D point set (PCA).
    Sign-ambiguous (a line, not a ray).
    """
    points2d = np.asarray(points2d, dtype=float)

    if len(points2d) < 2:
        return np.nan

    c = points2d - points2d.mean(axis=0)
    evals, evecs = np.linalg.eigh(c.T @ c)
    vec = evecs[:, int(np.argmax(evals))]

    angle = float(np.arctan2(vec[1], vec[0]))

    return np.rad2deg(angle) if degrees else angle

