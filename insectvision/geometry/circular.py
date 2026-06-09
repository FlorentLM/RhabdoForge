from typing import Optional
import numpy as np
from numpy.typing import ArrayLike


def resultant(angles, weights=None, axis=None, fold=1, degrees: bool = False):
    """
    Mean resultant phasor z = <w * exp(i*kfold*theta)> (weights normalised), reduced along 'axis'.
    Rows whose weights sum to <= 0 return 0+0j.

    fold=1 -> circular mean/variance
    fold=2 -> nematic
    fold=6 -> hexatic
    """
    a = np.deg2rad(angles) if degrees else np.asarray(angles)
    z = np.exp(1j * fold * a)

    if weights is None:
        return z.mean(axis=axis)

    w = np.asarray(weights, dtype=np.float64)
    num = (w * z).sum(axis=axis)
    den = w.sum(axis=axis)

    res = np.divide(num, den, out=np.zeros_like(num), where=den > 0)

    return np.rad2deg(res) if degrees else res


def wrap_angle(angles: ArrayLike, degrees: bool = False) -> np.ndarray:
    """
    Wrap angle(s) to (-pi, pi]
    """
    a = np.deg2rad(angles) if degrees else np.asarray(angles)
    wrapped = (a + np.pi) % (2.0 * np.pi) - np.pi
    return np.rad2deg(wrapped) if degrees else wrapped


def circ_mean(
        angles: ArrayLike,
        weights: Optional[ArrayLike] = None,
        axis: Optional[int] = None,
        degrees: bool = False
    ) -> np.ndarray:
    """
    Circular mean of angles (rad), optionally weighted
    """
    return np.angle(resultant(angles=angles, weights=weights, axis=axis, degrees=degrees))


def circ_std(
        angles: ArrayLike,
        weights: Optional[ArrayLike] = None,
        axis: Optional[int] = None,
        degrees: bool = False
    ) -> np.ndarray:
    """
    Circular standard deviation (rad), optionally weighted
    0 = perfectly concentrated, grows with spread
    """
    a = np.deg2rad(angles) if degrees else angles
    R = np.clip(np.abs(resultant(angles=a, weights=weights, axis=axis, degrees=False)), 1e-9, 1.0)
    return np.sqrt(-2.0 * np.log(R))