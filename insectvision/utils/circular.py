from __future__ import annotations
from typing import Optional
import numpy as np
from numpy.typing import ArrayLike


def wrap_angle(angles: ArrayLike, degrees: bool = False) -> np.ndarray:
    """
    Wrap angle(s) to (-pi, pi].
    """
    a = np.radians(angles) if degrees else np.asarray(angles)
    wrapped = (a + np.pi) % (2.0 * np.pi) - np.pi
    return np.degrees(wrapped) if degrees else wrapped


def weighted_circ_mean(
        angles: ArrayLike,
        weights: Optional[ArrayLike] = None,
        axis: Optional[int] = None,
        degrees: bool = False
    ) -> np.ndarray:
    """
    Weighted circular mean of angles (rad).
    """
    a = np.radians(angles) if degrees else np.asarray(angles)
    w = np.ones_like(a) if weights is None else np.asarray(weights)
    s = np.sum(w * np.sin(a), axis=axis)
    c = np.sum(w * np.cos(a), axis=axis)

    circmean =  np.arctan2(s, c)

    return np.degrees(circmean) if degrees else circmean


def weighted_circ_std(
        angles: ArrayLike,
        weights: Optional[ArrayLike] = None,
        axis: Optional[int] = None,
        degrees: bool = False
    ) -> np.ndarray:
    """
    Circular standard deviation (rad). 0 = perfectly concentrated, grows with spread.
    """
    a = np.radians(angles) if degrees else np.asarray(angles)
    w = np.ones_like(a) if weights is None else np.asarray(weights)
    s = np.sum(w * np.sin(a), axis=axis)
    c = np.sum(w * np.cos(a), axis=axis)

    R = np.hypot(s, c) / np.clip(np.sum(w, axis=axis), 1e-9, None)
    R = np.clip(R, 1e-9, 1.0)

    return np.sqrt(-2.0 * np.log(R))
