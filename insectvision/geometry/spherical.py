from typing import TYPE_CHECKING, Tuple, Optional, Sequence
import numpy as np
from numpy.typing import ArrayLike

from insectvision.utils import norm_l2
from insectvision.geometry.linalg import tangent_frames, local_to_world
from insectvision.geometry.neighbours import knn

if TYPE_CHECKING:
    from scipy.spatial import cKDTree


# Spherical <-> Cartesian

def cartesian_to_spherical(directions: ArrayLike, degrees: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    (N, 3) Cartesian -> (azimuth, elevation).
    """
    d = np.asarray(directions)
    az = np.arctan2(d[..., 0], -d[..., 2])
    el = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
    if degrees:
        return np.rad2deg(az), np.rad2deg(el)
    return az, el


def spherical_to_cartesian(azimuth: ArrayLike, elevation: ArrayLike, radius: float = 1.0, degrees: bool = False) -> np.ndarray:
    """
    (azimuth, elevation) -> (N, 3) Cartesian.
    """
    az = np.deg2rad(azimuth) if degrees else np.asarray(azimuth, dtype=np.float32)
    el = np.deg2rad(elevation) if degrees else np.asarray(elevation, dtype=np.float32)
    x = radius * np.sin(az) * np.cos(el)
    y = radius * np.sin(el)
    z = -radius * np.cos(az) * np.cos(el)
    return np.stack([x, y, z], axis=-1)


def spherical_gradients(azimuth: ArrayLike, elevation: ArrayLike, degrees: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Spatial gradient directions w.r.t. azimuth and elevation on the unit sphere.
    """

    az = np.deg2rad(azimuth) if degrees else np.asarray(azimuth, dtype=np.float32)
    el = np.deg2rad(elevation) if degrees else np.asarray(elevation, dtype=np.float32)

    cos_az, sin_az = np.cos(az), np.sin(az)
    cos_el, sin_el = np.cos(el), np.sin(el)

    az_grad = np.column_stack([cos_az * cos_el, np.zeros_like(az), sin_az * cos_el])
    el_grad = np.column_stack([-sin_az * sin_el, cos_el, cos_az * sin_el])

    if degrees:
        return np.rad2deg(az_grad), np.rad2deg(el_grad)
    return az_grad, el_grad


# Stereographic projection

def sphere_to_stereo(
        directions: ArrayLike,
        basis: Optional[Sequence[ArrayLike]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stereographic projection of unit directions onto a tangent plane.

    Args:
        directions: (N, 3) array of unit vectors.
        basis: Optional (forward, right, up) sequence of basis vectors. If None, the orthonormal frame is
               centered on the mean of the input directions.

    Returns:
        points2d: (N, 2)
        forward, right, up: (3,) orthonormal frame of the projection plane
    """
    directions = np.atleast_2d(np.asarray(directions, dtype=np.float64))

    if basis is None:
        centre_dir = np.mean(directions, axis=0)
        forward = norm_l2(centre_dir)
        right, up = tangent_frames(forward)
    else:
        forward, right, up = basis
        forward = np.asarray(forward, dtype=np.float64)
        right = np.asarray(right, dtype=np.float64)
        up = np.asarray(up, dtype=np.float64)

    denom = 1.0 + (directions @ forward)

    points2d = np.column_stack([
        (directions @ right) / denom,
        (directions @ up) / denom,
    ])

    return points2d, forward, right, up


def stereo_to_sphere(
        points2d: ArrayLike,
        forward: ArrayLike,
        right: ArrayLike,
        up: ArrayLike,
) -> np.ndarray:
    """
    Inverse stereographic projection (2D plane -> unit sphere).
    """
    points2d = np.asarray(points2d, dtype=np.float64)

    forward = np.asarray(forward, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    x, y = points2d[:, 0], points2d[:, 1]
    r2 = x ** 2 + y ** 2
    denom = 1.0 + r2

    dirs = local_to_world(
        np.column_stack([2.0 * x / denom, 2.0 * y / denom, (1.0 - r2) / denom]),
        right, up, forward,
    )
    return norm_l2(dirs)


def angle_to_chord(angle_rad: ArrayLike) -> np.ndarray:
    """Great-circle angle (rad) -> Euclidean chord length on the unit sphere."""
    return 2.0 * np.sin(0.5 * np.asarray(angle_rad, dtype=np.float64))


def chord_to_angle(chord: ArrayLike) -> np.ndarray:
    """Euclidean chord length on the unit sphere -> great-circle angle (rad)."""
    return 2.0 * np.arcsin(np.clip(0.5 * np.asarray(chord, dtype=np.float64), -1.0, 1.0))


def normals_to_ellipsoid(directions: ArrayLike, rx: float, ry: float, rz: float) -> np.ndarray:
    """
    Map viewing directions to positions on an ellipsoid such that the directions are the surface normals.
    """
    directions = np.asarray(directions, dtype=np.float64)

    nx = directions[:, 0]
    ny = directions[:, 1]
    nz = directions[:, 2]

    # Calculate the scale factor for the normal mapping
    K = np.sqrt((nx * rx) ** 2 + (ny * ry) ** 2 + (nz * rz) ** 2)

    x = (rx ** 2 * nx) / K
    y = (ry ** 2 * ny) / K
    z = (rz ** 2 * nz) / K

    return np.column_stack([x, y, z])


def radius_of_curvature(
        query_positions: ArrayLike,
        query_normals: ArrayLike,
        tree: 'cKDTree',
        tree_normals: ArrayLike,
        k: int = 7,
) -> np.ndarray:
    """
    Estimate local radius of curvature per query point: R ~= (spatial distance) / (great-circle angle).

    Args:
        - query_positions: (N, 3) positions to evaluate
        - query_normals: (N, 3) normals at those positions
        - tree: cKDTree containing the cloud positions
        - tree_normals: (M, 3) normals of the points corresponding to the tree data
        - k: Number of neighbours to average over

    Returns:
        (N,) array of radii. Points with no angular variation return NaN.
    """
    q_pos = np.atleast_2d(np.asarray(query_positions, dtype=np.float64))
    q_nrm = np.atleast_2d(np.asarray(query_normals, dtype=np.float64))

    dist, idx = knn(tree, q_pos, k=k, drop_self=True)
    if idx.size == 0:
        return np.full(len(q_pos), np.nan)

    # Calculate angles between query normals and neighbour normals
    # chord length on unit sphere -> angle in radians
    chords = np.linalg.norm(tree_normals[idx] - q_nrm[:, None, :], axis=-1)
    angles = chord_to_angle(chords)

    # R = arc_length / angle
    valid = (angles > 1e-5) & (dist > 0)

    with np.errstate(all='ignore'):
        ratios = np.where(valid, dist / np.where(valid, angles, 1.0), np.nan)
        return np.nanmedian(ratios, axis=1)