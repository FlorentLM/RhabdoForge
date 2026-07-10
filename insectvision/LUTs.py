from typing import Callable

import numpy as np
from numpy._typing import ArrayLike
from scipy.interpolate import Akima1DInterpolator
from scipy.special import j1


def akima_interp_fn(x: ArrayLike, y: ArrayLike, fill_value: float) -> 'Callable':
    """
    Akima interpolator that returns 'fill_value' for queries outside range.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    akima_fn = Akima1DInterpolator(x, y)

    def wrapper(query_x):
        query_x = np.asarray(query_x)
        mask_oob = (query_x < x.min()) | (query_x > x.max())
        vals = akima_fn(query_x)
        return np.where(mask_oob, fill_value, vals)

    return wrapper


def airy_sensitivity_lut(size: int = 256, range: float = 6.0) -> np.ndarray:
    """
    Map x from 0.0 (centre) to 4.0 (deep in the tails)
    where x is normalised such that x=0.5 is the half max
    """

    x_vals = np.linspace(0, range, size)
    lut_data = []

    for x_norm in x_vals:
        # Scale for Airy FWHM
        x = 3.232 * x_norm
        if x < 1e-6:
            val = 1.0
        else:
            val = (2.0 * j1(x) / x) ** 2
        lut_data.append(float(val))

    return np.array(lut_data, dtype=np.float32)


def lorentzian_sensitivity_lut(size: int = 256) -> np.ndarray:
    """
    Lorentzian (Cauchy) profile.
    Heavy tails: stays brighter further from the centre compared to a Gaussian.
    """
    x_vals = np.linspace(0, 4.0, size)
    # At x=0.5, val = 1 / (1 + (0.5/0.5)^2) = 0.5
    lut_data = 1.0 / (1.0 + (x_vals / 0.5) ** 2)
    return np.array(lut_data, dtype=np.float32)


def leakage_sensitivity_lut(size: int = 256, pedestal_height: float = 0.05, pedestal_width: float = 3.0) -> np.ndarray:
    """
    Sum-of-Gaussians.
    Simulates a narrow optical core with a wide 'pedestal' caused by
    light leakage between ommatidia (common in insect eye measurements).
    """
    x_vals = np.linspace(0, 4.0, size)
    core = np.exp(-2.77258872224 * x_vals ** 2) # Core Gaussian (standard GAUSS_K)
    wide = np.exp(-2.77258872224 * (x_vals / pedestal_width) ** 2)   # wide Gaussian pedestal
    # re-normalised so peak is 1.0
    combined = (core + pedestal_height * wide) / (1.0 + pedestal_height)
    return np.array(combined, dtype=np.float32)
