from typing import Tuple, Union, Any, Optional
import numpy as np
from numpy.typing import ArrayLike


def norm_minmax(
        array: ArrayLike,
        axis: Any = None,
        inplace: bool = False,
        eps: float = 1e-9
) -> np.ndarray:

    if inplace:
        arr = np.asarray(array)
        if not np.issubdtype(arr.dtype, np.floating):
            raise ValueError("In-place normalisation requires a float array dtype.")
    else:
        arr = np.array(array, dtype=float, copy=True)

    keep = (axis is not None)
    vmin = np.nanmin(arr, axis=axis, keepdims=keep)
    vmax = np.nanmax(arr, axis=axis, keepdims=keep)

    np.subtract(arr, vmin, out=arr)
    np.divide(arr, (vmax - vmin) + eps, out=arr)

    return arr


def norm_l2(
        vectors: ArrayLike,
        axis: int = -1,
        inplace: bool = False,
        eps: float = 1e-9
) -> np.ndarray:

    if inplace:
        arr = np.asarray(vectors)
        if not np.issubdtype(arr.dtype, np.floating):
            raise ValueError("In-place normalisation requires a float array dtype.")
    else:
        arr = np.array(vectors, dtype=float, copy=True)

    norms = np.linalg.norm(arr, axis=axis, keepdims=True)

    np.divide(arr, norms, out=arr, where=norms > eps)

    return arr


def tangent_frames(
        directions: ArrayLike,
        world_up: Optional[ArrayLike] = None,
        world_right: Optional[ArrayLike] = None
    ):
    """
    Computes orthonormal basis vectors (right, up) for given direction vectors.
    Handles poles/gimbal lock by switching reference vectors.
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

    fwd = norm_l2(dirs)

    dots = np.abs(fwd @ world_up)

    blend = np.clip((dots - 0.98) / (1.0 - 0.98), 0.0, 1.0)[:, np.newaxis]
    ref_ups = (1.0 - blend) * world_up + blend * world_right

    right = np.cross(fwd, ref_ups)
    right = norm_l2(right)

    up = np.cross(right, fwd)
    up = norm_l2(up)

    if is_1d:
        return right[0], up[0]

    return right, up


##
# TODO: frame transforms below should be generalised to accept a reference / handedness.
# TODO: And the ones in the eye generations sripts should be moved here


def project_to_stereo(
        dirs_3d: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stereographic projection of unit directions onto a
    tangent plane centred on the mean viewing direction.

    Returns:
        pts_2d: (N, 2)
        fwd, rgt, up: (3,) orthonormal frame of projection plane
    """

    centre_dir = np.mean(dirs_3d, axis=0)
    fwd = centre_dir / np.linalg.norm(centre_dir)
    rgt, up = tangent_frames(fwd)

    denom = 1.0 + np.dot(dirs_3d, fwd)
    pts_2d = np.column_stack([
        np.dot(dirs_3d, rgt) / denom,
        np.dot(dirs_3d, up) / denom,
    ])
    return pts_2d, fwd, rgt, up


def stereo_to_sphere(
        pts_2d: np.ndarray,
        fwd: np.ndarray,
        rgt: np.ndarray,
        up: np.ndarray
) -> np.ndarray:
    """
    Inverse stereographic projection (2D plane -> unit sphere)
    """
    x, y = pts_2d[:, 0], pts_2d[:, 1]
    r2 = x ** 2 + y ** 2
    denom = 1.0 + r2

    dirs = (
        (2.0 / denom)[:, None] * (x[:, None] * rgt + y[:, None] * up)
        + ((1.0 - r2) / denom)[:, None] * fwd
    )
    dirs = norm_l2(dirs)

    return dirs


def rotate_vectors(
        vectors: Union[np.ndarray, list],
        axes: Union[np.ndarray, list],
        angles: Union[np.ndarray, float],
        degrees: bool = True,
        normalize_axes: bool = False
) -> np.ndarray:
    """Rotate vectors around axes (Rodrigues formula)."""

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


def wrap_angle(angles: ArrayLike) -> np.ndarray:
    """Wrap angle(s) to (-pi, pi]."""
    a = np.asarray(angles)
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def weighted_circ_mean(angles: ArrayLike, weights: Optional[ArrayLike] = None,
                       axis: Optional[int] = None) -> np.ndarray:
    """Weighted circular mean of angles."""
    a = np.asarray(angles)
    w = np.ones_like(a) if weights is None else np.asarray(weights)
    s = np.sum(w * np.sin(a), axis=axis)
    c = np.sum(w * np.cos(a), axis=axis)
    return np.arctan2(s, c)


def weighted_circ_std(angles: ArrayLike, weights: Optional[ArrayLike] = None, axis: Optional[int] = None) -> np.ndarray:
    """Circular standard deviation (rad). 0 = perfectly rigid, grows with shear."""
    a = np.asarray(angles)
    w = np.ones_like(a) if weights is None else np.asarray(weights)
    s = np.sum(w * np.sin(a), axis=axis)
    c = np.sum(w * np.cos(a), axis=axis)
    R = np.hypot(s, c) / np.clip(np.sum(w, axis=axis), 1e-9, None)
    R = np.clip(R, 1e-9, 1.0)
    return np.sqrt(-2.0 * np.log(R))


def cartesian_to_spherical(directions: ArrayLike, degrees: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Converts (N, 3) Cartesian vectors to Azimuth and Elevation following the project convention."""
    d = np.asarray(directions)
    az = np.arctan2(d[..., 0], -d[..., 2])
    el = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
    if degrees:
        return np.degrees(az), np.degrees(el)
    return az, el


def spherical_to_cartesian(azimuth: ArrayLike, elevation: ArrayLike, radius: float = 1.0, degrees: bool = False) -> np.ndarray:
    """Converts spherical coordinates to cartesian coordinates in the internal reference frame."""
    az = np.radians(azimuth) if degrees else np.asarray(azimuth)
    el = np.radians(elevation) if degrees else np.asarray(elevation)
    x = radius * np.sin(az) * np.cos(el)
    y = radius * np.sin(el)
    z = -radius * np.cos(az) * np.cos(el)
    return np.stack([x, y, z])


def spherical_gradients(azimuth: ArrayLike, elevation: ArrayLike, degrees: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Compute spatial gradients with respect to Azimuth and Elevation on the unit sphere."""
    az = np.radians(azimuth) if degrees else np.asarray(azimuth)
    el = np.radians(elevation) if degrees else np.asarray(elevation)

    cos_az, sin_az = np.cos(az), np.sin(az)
    cos_el, sin_el = np.cos(el), np.sin(el)

    az_grad = np.column_stack([cos_az * cos_el, np.zeros_like(az), sin_az * cos_el])
    el_grad = np.column_stack([-sin_az * sin_el, cos_el, cos_az * sin_el])
    return az_grad, el_grad


def tangent_bearing(target_dirs: ArrayLike, home_dirs: ArrayLike, right: ArrayLike, up: ArrayLike, degrees: bool = False) -> np.ndarray:
    """Projects difference vectors onto a tangent plane and returns the bearing angle."""
    delta = np.asarray(target_dirs) - np.asarray(home_dirs)
    u = np.sum(delta * np.asarray(right), axis=-1)
    v = np.sum(delta * np.asarray(up), axis=-1)
    bearing = np.arctan2(v, u)
    return np.degrees(bearing) if degrees else bearing


def neighbour_smooth(
        values: np.ndarray,
        neighbours: np.ndarray,
        mask: Optional[np.ndarray] = None,
        n_iter: int = 2,
        method: str = 'mean'
) -> np.ndarray:
    """
    Smooth a 1D field over a precomputed neighbour graph.

    Args:
        values: (N,) array of values to smooth.
        neighbours: (N, k) integer array of neighbour indices. -1 padding is ignored.
        mask: (N,) bool array. If provided, only True items are updated.
        n_iter: Smoothing passes.
        method: 'mean' or 'median'.
    """
    out = np.asarray(values, dtype=np.float64).copy()
    if n_iter <= 0:
        return out.astype(values.dtype)

    update_mask = mask if mask is not None else np.ones(out.shape[0], dtype=bool)
    valid_nb = neighbours >= 0
    safe_nb = np.where(valid_nb, neighbours, 0)

    for _ in range(n_iter):
        nb_vals = out[safe_nb]

        if method == 'mean':
            sum_nb = np.sum(np.where(valid_nb, nb_vals, 0.0), axis=1)
            count_nb = np.maximum(np.sum(valid_nb, axis=1), 1)
            cur = 0.5 * out + 0.5 * (sum_nb / count_nb)
        elif method == 'median':
            nb_vals = np.where(valid_nb, nb_vals, np.nan)
            stacked = np.concatenate([out[:, None], nb_vals], axis=1)
            with np.errstate(all='ignore'):
                cur = np.nanmedian(stacked, axis=1)
        else:
            raise ValueError(f"Unknown method {method}")

        out = np.where(update_mask, cur, out)

    return out.astype(values.dtype)


##

def icosahedron_faces() -> np.ndarray:
    """
    Base z-axis-aligned icosahedron.
    Returns (20, 3, 3) face vertices.
    """

    G = (1 + np.sqrt(5)) / 2.0

    p = np.array([
        [G, -G, -G, G, 1, 1, -1, -1, 0, 0, 0, 0],
        [0, 0, 0, 0, G, -G, -G, G, 1, 1, -1, -1],
        [1, 1, -1, -1, 0, 0, 0, 0, G, -G, -G, G]
    ], dtype=np.float32).T

    p /= np.linalg.norm(p[0])
    ang = np.arctan(p[0, 0] / p[0, 2])

    ca, sa = np.cos(ang), np.sin(ang)
    rot = np.array([[ca, 0, -sa], [0, 1, 0], [sa, 0, ca]])
    p = np.inner(rot, p).T
    p = p[[0, 3, 4, 8, -1, 5, -2, -3, 7, 1, 6, 2]]

    tri = np.array([
        [1, 2, 3, 4, 5, 6, 2, 7, 2, 8, 3, 9, 10, 10, 6, 6, 7, 8, 9, 10],
        [2, 3, 4, 5, 1, 7, 1, 8, 8, 9, 9, 10, 5, 6, 1, 11, 11, 11, 11, 11],
        [0, 0, 0, 0, 0, 1, 7, 2, 3, 3, 4, 4, 4, 5, 5, 7, 8, 9, 10, 6]
    ]).T
    return p[tri]


def barycentric_coords(n_subdiv: int) -> np.ndarray:
    """
    Barycentric coordinates for subdivided reference triangle.
    """

    vals = np.linspace(0, 1, n_subdiv + 1)
    num = int((n_subdiv + 1) * (n_subdiv + 2) / 2)
    bc = np.zeros((num, 3))

    shifts = np.arange(n_subdiv + 1, 0, -1)
    starts = np.zeros(n_subdiv + 1, dtype=int)
    starts[1:] = np.cumsum(shifts[:-1])
    stops = starts + shifts

    for i, (s, e, sh) in enumerate(zip(starts, stops, shifts)):
        bc[s:e, 0] = vals[sh - 1::-1]
        bc[s:e, 1] = vals[:sh]
        bc[s:e, 2] = vals[i]
    return bc


def subdivide_icosahedron(n_subdiv: int) -> np.ndarray:
    """
    Subdivide icosahedron via barycentric interpolation onto unit sphere.
    """

    verts = icosahedron_faces()
    bary = barycentric_coords(n_subdiv)

    all_v = np.einsum('ij,kjl->kil', bary, verts).reshape(-1, 3)
    all_v /= np.linalg.norm(all_v, axis=1)[:, np.newaxis]
    _, iu = np.unique(np.round(all_v, 6), axis=0, return_index=True)

    return all_v[iu].astype(np.float32)


def icosphere(points: int) -> np.ndarray:
    """
    Uniform icosphere (icosahedron subdivision method).
    """
    lod = int(np.round(np.sqrt((max(points, 12) - 2) / 10.0)))
    dirs = subdivide_icosahedron(lod).astype(np.float32)

    if np.abs(points - len(dirs)) > 1:
        print(f"Note: {len(dirs)} ommatidia for subdivision level {lod}.")

    return dirs


def fibonacci_sphere(points: int) -> np.ndarray:
    """
    Uniform points on unit sphere (Fibonacci method).
    """
    phi = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(points)
    y = 1 - (i / float(points - 1)) * 2
    r = np.sqrt(1 - y * y)
    theta = phi * i

    return np.column_stack([np.cos(theta) * r, y, np.sin(theta) * r])