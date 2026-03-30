from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike


def normalise_vectors(vectors: ArrayLike):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return np.divide(vectors, norms, out=vectors, where=norms != 0)


def tangent_frames(directions: ArrayLike, world_up=None, world_right=None):
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

    fwd = normalise_vectors(dirs)

    dots = np.abs(fwd @ world_up)
    is_polar = dots > 0.999
    ref_ups = np.where(is_polar[:, np.newaxis], world_right, world_up)

    # Right-handed orthonormal basis
    right = np.cross(fwd, ref_ups)
    right = normalise_vectors(right)

    up = np.cross(right, fwd)
    up = normalise_vectors(up)

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
    dirs = normalise_vectors(dirs)

    return dirs


def spherical_to_cartesian(azimuth, elevation, radius=1.0, degrees=False):
    """
    Converts spherical coordinates to cartesian coordinates in internal reference frame.
    """
    az_rad = np.radians(azimuth) if degrees else np.array(azimuth)
    el_rad = np.radians(elevation) if degrees else np.array(elevation)

    x = radius * np.cos(el_rad) * np.cos(az_rad)
    y = radius * np.cos(el_rad) * np.sin(az_rad)
    z = radius * np.sin(el_rad)

    return np.stack([x, y, z])


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


