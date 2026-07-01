from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike
from insectvision.geometry.linalg import tangent_frames, local_to_world
from insectvision.utils.shared import norm_l2


# Spherical <-> Cartesian

def cartesian_to_spherical(directions: ArrayLike, degrees: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    (N, 3) Cartesian -> (azimuth, elevation).
    """
    d = np.asarray(directions)
    az = np.arctan2(d[..., 0], -d[..., 2])
    el = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
    if degrees:
        return np.degrees(az), np.degrees(el)
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
        return np.degrees(az_grad), np.degrees(el_grad)
    return az_grad, el_grad


# Stereographic projection

def sphere_to_stereo(directions: ArrayLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stereographic projection of unit directions onto a tangent plane centred
    on the mean viewing direction.

    Returns:
        points2d: (N, 2)
        forward, right, up: (3,) orthonormal frame of the projection plane
    """

    directions = np.asarray(directions, dtype=np.float64)

    centre_dir = np.mean(directions, axis=0)
    forward = norm_l2(centre_dir)
    right, up = tangent_frames(forward)

    denom = 1.0 + np.dot(directions, forward)
    points2d = np.column_stack([
        np.dot(directions, right) / denom,
        np.dot(directions, up) / denom,
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
