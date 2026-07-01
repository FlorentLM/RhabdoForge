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
    Circular mean of angles, optionally weighted
    """
    a = np.deg2rad(angles) if degrees else np.asarray(angles)
    m = np.angle(resultant(angles=a, weights=weights, axis=axis, degrees=False))
    return np.rad2deg(m) if degrees else m


def circ_std(
        angles: ArrayLike,
        weights: Optional[ArrayLike] = None,
        axis: Optional[int] = None,
        degrees: bool = False
    ) -> np.ndarray:
    """
    Circular standard deviation, optionally weighted
    0 = perfectly concentrated, grows with spread
    """
    a = np.deg2rad(angles) if degrees else angles
    r = np.clip(np.abs(resultant(angles=a, weights=weights, axis=axis, degrees=False)), 1e-9, 1.0)
    return np.sqrt(-2.0 * np.log(np.deg2rad(r) if degrees else r))