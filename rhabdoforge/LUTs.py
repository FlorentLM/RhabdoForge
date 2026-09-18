from typing import Callable, Optional
import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import Akima1DInterpolator
from scipy.special import j1

from rhabdoforge.types import METADATA_BIT_LAYOUT


# Angular range for sensitivity LUTs, in units of the acceptance angle (FWHM)
LUT_RANGE = 4.0
LUT_SIZE = 256
LUT_MODE_SLOTS = 1 << METADATA_BIT_LAYOUT['rhab_R'][1]   # rhabdomere types a per-type LUT can index


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


def gaussian_sensitivity_lut() -> np.ndarray:
    """Pure Gaussian target profile, baked onto the same grid as the other LUT-based targets."""
    x_vals = np.linspace(0, LUT_RANGE, LUT_SIZE)
    return np.exp(-2.77258872224 * x_vals ** 2).astype(np.float32)


def airy_sensitivity_lut() -> np.ndarray:
    """
    Map x from 0.0 (centre) to LUT_RANGE (deep in the tails)
    where x is normalised such that x=0.5 is the half max

    Note: 'range' defaulted to 6.0 while the shader has always indexed this table as if it
    spanned 0->4, stretching the profile by 1.5x and making its outer third unreachable.
    Both now derive from LUT_RANGE.
    """

    x_vals = np.linspace(0, LUT_RANGE, LUT_SIZE)
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


def lorentzian_sensitivity_lut() -> np.ndarray:
    """
    Lorentzian (Cauchy) profile.
    Heavy tails: stays brighter further from the centre compared to a Gaussian.
    """
    x_vals = np.linspace(0, LUT_RANGE, LUT_SIZE)
    # At x=0.5, val = 1 / (1 + (0.5/0.5)^2) = 0.5
    lut_data = 1.0 / (1.0 + (x_vals / 0.5) ** 2)
    return np.array(lut_data, dtype=np.float32)


def leakage_sensitivity_lut(pedestal_height: float = 0.05, pedestal_width: float = 3.0) -> np.ndarray:
    """
    Sum-of-Gaussians.
    Simulates a narrow optical core with a wide 'pedestal' caused by
    light leakage between ommatidia (common in insect eye measurements).
    """
    x_vals = np.linspace(0, LUT_RANGE, LUT_SIZE)
    core = np.exp(-2.77258872224 * x_vals ** 2) # Core Gaussian (standard GAUSS_K)
    wide = np.exp(-2.77258872224 * (x_vals / pedestal_width) ** 2)   # wide Gaussian pedestal
    # re-normalised so peak is 1.0
    combined = (core + pedestal_height * wide) / (1.0 + pedestal_height)
    return np.array(combined, dtype=np.float32)


def invert_lut_cdf(lut: ArrayLike) -> np.ndarray:
    """
    Inverse CDF (area-weighted, isotropic 2D) of one or more LUT_SIZE-long profiles
    (flat-packed)
    """
    lut = np.atleast_1d(np.asarray(lut, dtype=np.float64))
    slots = lut.size // LUT_SIZE
    lut = lut.reshape(slots, LUT_SIZE)

    r = np.linspace(0.0, LUT_RANGE, LUT_SIZE)
    u = np.linspace(0.0, 1.0, LUT_SIZE)
    icdf = np.empty_like(lut)

    for i in range(slots):
        w = lut[i] * r         # 2D polar area element
        cdf = np.concatenate(([0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(r))))
        total = cdf[-1]
        icdf[i] = r if total <= 0.0 else np.interp(u, cdf / total, r)

    return icdf.reshape(-1).astype(np.float32)


def waveguide_sensitivity_lut(
        diameters_um: ArrayLike,
        wavelengths_um: ArrayLike,
        f_number: float,
        n_rhabdomere: ArrayLike = None,
        n_surround: ArrayLike = None,
        defocus_um: float = 0.0,
        focal_um: Optional[float] = None,
        slots: Optional[int] = None,
    ) -> np.ndarray:
    """
    One angular sensitivity profile per rhabdomere (Stavenga's waveguide mode sum)
    (flat-packed fixed-size blocks -> SSBO -> shader indexes by rhab_R)

    Slots past the bundle's rhabdomere count fall back to a Gaussian.
    """
    from rhabdoforge.compound_eyes.helpers.acceptance import RhabdomereOptics
    from rhabdoforge.compound_eyes.helpers.waveguide import solve_modes

    optics = RhabdomereOptics(
        diameter_um=np.atleast_1d(np.asarray(diameters_um, dtype=np.float32)),
        wavelength_um=np.atleast_1d(np.asarray(wavelengths_um, dtype=np.float32)),
        n_rhabdomere=None if n_rhabdomere is None else np.atleast_1d(np.asarray(n_rhabdomere, dtype=np.float32)),
        n_surround=None if n_surround is None else np.atleast_1d(np.asarray(n_surround, dtype=np.float32)),
    )

    x_vals = np.linspace(0, LUT_RANGE, LUT_SIZE)
    gaussian = np.exp(-2.77258872224 * x_vals ** 2)     # for unused slots

    if slots is None:
        slots = LUT_MODE_SLOTS

    lut = np.tile(gaussian, slots).astype(np.float32)

    for r in range(min(optics.nb_rhabdomeres, slots)):
        modes = solve_modes(
            v_number=float(optics.v_number[r]),
            f_number=f_number,
            diameter_um=float(optics.diameter_um[r]),
            wavelength_um=float(optics.wavelength_um[r]),
            defocus_um=defocus_um,
            focal_um=focal_um,
        )

        # d_sweep (D = d/b units) -> theta/Drho, where 0.5 is half max
        profile_x = 0.5 * modes.d_sweep / modes.d_half
        lut[r * LUT_SIZE:(r + 1) * LUT_SIZE] = np.interp(x_vals, profile_x, modes.sensitivity)

    return lut
