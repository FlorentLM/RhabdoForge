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