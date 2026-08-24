from typing import Optional
import numpy as np
from numpy.typing import ArrayLike


def resultant(
        angles: ArrayLike,
        weights: Optional[ArrayLike] = None,
        axis: Optional[int] = None,
        fold: int = 1,
        degrees: bool = False
    ) -> np.ndarray:
    """
    Mean resultant phasor z = <w * exp(i*kfold*theta)> (weights normalised), reduced along 'axis'.
    Rows whose weights sum to <= 0 return 0+0j.

    fold=1 -> circular mean/variance
    fold=2 -> nematic
    fold=6 -> hexatic
    """
    angles = np.asarray(angles, dtype=np.float64)

    a = np.deg2rad(angles) if degrees else angles
    z = np.exp(1j * fold * a)

    if weights is None:
        return z.mean(axis=axis)

    w = np.asarray(weights, dtype=np.float64)
    num = (w * z).sum(axis=axis)
    den = w.sum(axis=axis)

    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def fold_angle(angles: ArrayLike, degrees: bool = False) -> np.ndarray:
    """
    Wrap angle(s) to [0, pi/2.0] (or [0, 90] if degrees is True).
    (Nematic distance to the nearest axis)
    """
    period = 180.0 if degrees else np.pi
    a = np.asarray(angles)
    wrapped = a % period
    return np.minimum(wrapped, period - wrapped)


def wrap_angle(angles: ArrayLike, degrees: bool = False) -> np.ndarray:
    """
    Wrap angle(s) to (-pi, pi] (or (-180, 180] if degrees is True).
    """
    a = np.deg2rad(angles) if degrees else np.asarray(angles)
    wrapped = (a + np.pi) % (2.0 * np.pi) - np.pi
    return np.rad2deg(wrapped) if degrees else wrapped