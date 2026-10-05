"""
Spectral helpers: visual pigment sensitivity from the Govardovskii et al. 2000 A1 template [1].

    [1] Govardovskii et al. 2000: "In search of the visual pigment template", 10.1017/S0952523800009663
"""
import numpy as np
from numpy.typing import ArrayLike

RGB_NM = (610.0, 550.0, 465.0)     # Approximate dominant wavelengths of the display primaries


def _template(lam: np.ndarray, lam_max: np.ndarray) -> np.ndarray:
    """A1 pigment template (alpha + beta bands), un-normalised."""
    x = lam_max / lam
    a = 0.8795 + 0.0459 * np.exp(-(lam_max - 300.0) ** 2 / 11940.0)
    alpha = 1.0 / (np.exp(69.7 * (a - x)) + np.exp(28.0 * (0.922 - x)) + np.exp(-14.9 * (1.104 - x)) + 0.674)
    beta = 0.26 * np.exp(-((lam - (189.0 + 0.315 * lam_max)) / (-40.5 + 0.195 * lam_max)) ** 2)
    return alpha + beta


def opsin_sensitivity(lam_max_nm: ArrayLike, channel_nm: ArrayLike = RGB_NM) -> np.ndarray:
    """
    Relative sensitivity of each receptor to each channel, normalised to 1 at its own peak.

    - lam_max_nm: (R,) peak wavelengths (nm)
    - channel_nm: (C,) channel wavelengths (nm)

    Returns (R, C) float32. Single wavelength per channel (no primary spectrum integration).
    """
    lm = np.atleast_1d(np.asarray(lam_max_nm, dtype=np.float64))[:, None]
    ch = np.atleast_1d(np.asarray(channel_nm, dtype=np.float64))[None, :]
    return (_template(ch, lm) / _template(lm, lm)).astype(np.float32)
