from __future__ import annotations
from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike
from insectvision.utils.math import norm_l2, tangent_frames


# Spherical <-> Cartesian

def cartesian_to_spherical(directions: ArrayLike, degrees: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """(N, 3) Cartesian -> (azimuth, elevation)."""
    d = np.asarray(directions)
    az = np.arctan2(d[..., 0], -d[..., 2])
    el = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
    if degrees:
        return np.degrees(az), np.degrees(el)
    return az, el


def spherical_to_cartesian(azimuth: ArrayLike, elevation: ArrayLike, radius: float = 1.0, degrees: bool = False) -> np.ndarray:
    """
    (azimuth, elevation) -> (3, N) Cartesian, in the internal reference frame.
    """

    az = np.radians(azimuth) if degrees else np.asarray(azimuth)
    el = np.radians(elevation) if degrees else np.asarray(elevation)
    x = radius * np.sin(az) * np.cos(el)
    y = radius * np.sin(el)
    z = -radius * np.cos(az) * np.cos(el)
    return np.stack([x, y, z])


def spherical_gradients(azimuth: ArrayLike, elevation: ArrayLike, degrees: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """
    Spatial gradient directions w.r.t. azimuth and elevation on the unit sphere.
    """

    az = np.radians(azimuth) if degrees else np.asarray(azimuth)
    el = np.radians(elevation) if degrees else np.asarray(elevation)

    cos_az, sin_az = np.cos(az), np.sin(az)
    cos_el, sin_el = np.cos(el), np.sin(el)

    az_grad = np.column_stack([cos_az * cos_el, np.zeros_like(az), sin_az * cos_el])
    el_grad = np.column_stack([-sin_az * sin_el, cos_el, cos_az * sin_el])

    if degrees:
        return np.degrees(az_grad), np.degrees(el_grad)
    return az_grad, el_grad


# Stereographic projection

def project_to_stereo(dirs_3d: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Stereographic projection of unit directions onto a tangent plane centred
    on the mean viewing direction.

    Returns:
        pts_2d: (N, 2)
        fwd, rgt, up: (3,) orthonormal frame of the projection plane
    """
    centre_dir = np.mean(dirs_3d, axis=0)
    fwd = norm_l2(centre_dir)
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
        up: np.ndarray,
) -> np.ndarray:
    """
    Inverse stereographic projection (2D plane -> unit sphere).
    """

    x, y = pts_2d[:, 0], pts_2d[:, 1]
    r2 = x ** 2 + y ** 2
    denom = 1.0 + r2

    dirs = (
        (2.0 / denom)[:, None] * (x[:, None] * rgt + y[:, None] * up)
        + ((1.0 - r2) / denom)[:, None] * fwd
    )
    return norm_l2(dirs)
