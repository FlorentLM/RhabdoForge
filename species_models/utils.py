from typing import Tuple

import numpy as np

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


def position_eyes(
        sphere_positions: np.ndarray,
        sphere_directions: np.ndarray,
        HW_mm: float,
        FW_mm: float,
        EL_mm: float,
        ED_mm: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale a single spherical eye to realistic dimensions (and create both eyes).

    Args:
        sphere_positions: (N, 3) positions on unit sphere
        sphere_directions: (N, 3) unit direction vectors
        HW_mm: Head width in mm
        FW_mm: Frons width (distance between eyes) in mm
        EL_mm: Eye length (vertical extent) in mm
        ED_mm: Eye depth (anterior-posterior extent) in mm

    Returns:
        origins: (2N, 3) positions for both eyes
        directions: (2N, 3) direction vectors for both eyes
        eye_id: (2N,) array with 0 for left eye, 1 for right eye
    """
    eye_width_mm = (HW_mm - FW_mm) / 2.0
    eye_height_mm = EL_mm

    # Scale to ellipsoid
    scale_factors = np.array([eye_width_mm / 2.0, eye_height_mm / 2.0, ED_mm / 2.0])
    left_eye_origins = sphere_positions * scale_factors

    # Position left eye: medial edge at x = -FW_mm/2
    current_medial_x = left_eye_origins[:, 0].max()
    target_medial_x = -FW_mm / 2.0
    translation_x = target_medial_x - current_medial_x
    left_eye_origins[:, 0] += translation_x

    # Create right eye by mirroring
    right_eye_origins = left_eye_origins.copy()
    right_eye_origins[:, 0] *= -1

    # Directions: mirror X component for right eye
    left_eye_dirs = sphere_directions.copy()
    right_eye_dirs = sphere_directions.copy()
    right_eye_dirs[:, 0] *= -1

    origins = np.vstack([left_eye_origins, right_eye_origins])
    directions = np.vstack([left_eye_dirs, right_eye_dirs])
    eye_id = np.concatenate([
        np.zeros(len(sphere_positions), dtype=int),
        np.ones(len(sphere_positions), dtype=int)
    ])

    return origins, directions, eye_id