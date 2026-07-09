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
    from insectvision.utils import WORLD_UP, WORLD_RIGHT

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


def projected_bearing(
        vectors: ArrayLike,
        right: ArrayLike,
        up: ArrayLike,
        degrees: bool = False,
) -> np.ndarray:
    """
    Angular bearing of vectors projected onto a (right, up) plane.

    Args:
        vectors: (..., 3) array of vectors to project.
        right: (..., 3) basis vector representing the 0-radian axis.
        up: (..., 3) basis vector representing the pi/2-radian axis.
        degrees: If True, returns result in degrees, otherwise radians.
    """
    v = np.asarray(vectors, dtype=np.float64)
    r = np.asarray(right, dtype=np.float64)
    u = np.asarray(up, dtype=np.float64)

    # Project vectors onto the basis axes
    x = np.einsum('...k,...k->...', v, r)
    y = np.einsum('...k,...k->...', v, u)

    bearing = np.arctan2(y, x)
    return np.rad2deg(bearing) if degrees else bearing


def tangent_bearing(
        target_directions: ArrayLike,
        ref_directions: ArrayLike,
        right: ArrayLike,
        up: ArrayLike,
        degrees: bool = False,
) -> np.ndarray:
    """
    Project (target - reference) onto a (right, up) tangent plane and return the bearing.
    """
    target, ref = _match_batch(target_directions, ref_directions)
    delta = target - ref
    r_vec = _match_batch(delta, right)[1]
    u_vec = _match_batch(delta, up)[1]

    return projected_bearing(delta, r_vec, u_vec, degrees=degrees)


def project_to_tangent(vectors: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """
    Project vectors onto the tangent planes defined by the given normals.
    Assumes normals are unit vectors.
    """
    dot = np.einsum('...i,...i->...', vectors, normals)
    return vectors - dot[..., None] * normals


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


def rotation_matrix2d(theta: float | ArrayLike, degrees: bool = False) -> np.ndarray:
    """
    Return a 2x2 rotation matrix (or stack of) for angle theta.
    """
    th = np.deg2rad(theta) if degrees else np.asarray(theta, dtype=np.float64)
    c, s = np.cos(th), np.sin(th)

    out = np.stack([
        np.stack([c, -s], axis=-1),
        np.stack([s, c], axis=-1)
    ], axis=-2)
    return out.squeeze()


def rotate2d(vecs: ArrayLike, theta: ArrayLike, degrees: bool = False) -> np.ndarray:
    """
    Rotate 2D vectors by angle theta.

    Args:
        vecs: (..., 2) array of vectors.
        theta: (...) rotation angle(s).
        degrees: If True, theta is in degrees.
    """
    v = np.asarray(vecs, dtype=np.float64)
    th = np.deg2rad(theta) if degrees else np.asarray(theta, dtype=np.float64)

    c, s = np.cos(th), np.sin(th)
    x, y = v[..., 0], v[..., 1]

    return np.stack([c * x - s * y, s * x + c * y], axis=-1)


def principal_axes(points: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """
    Principal axes of a point set (PCA), any dimensionality.

    Centres the cloud and eigendecomposes its covariance, returning the axes as
    the columns of 'axes' ordered major -> minor (descending variance) with the
    matching variances alongside.

    Args:
        points: (N, D) point cloud (N points in D dimensions).

    Returns:
        axes: (D, D) unit eigenvectors as columns, sorted by descending variance,
            so axes[:, 0] is the major axis. Sign is arbitrary (PCA is sign-ambiguous).
        variances: (D,) eigenvalues (variances) in descending order.
        Both are NaN-filled when fewer than 2 points are supplied.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2:
        raise ValueError(f'points must be (N, D), got shape {pts.shape}')

    d = pts.shape[1]
    if pts.shape[0] < 2:
        return np.full((d, d), np.nan), np.full(d, np.nan)

    c = pts - pts.mean(axis=0)
    evals, evecs = np.linalg.eigh(c.T @ c)

    order = np.argsort(evals)[::-1]     # major -> minor
    return evecs[:, order], evals[order]