from typing import Optional, Callable, Tuple, Union
import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import RBFInterpolator

from insectvision.geometry.circular import resultant
from insectvision.geometry.neighbours import beta_skeleton_neighbours, NeighbourGraph, ragged_neighbours


def hexatic_axis_angle(z6: ArrayLike) -> np.ndarray:
    """Local axis angle theta = arg(z6)/6, in (-pi/6, pi/6]."""
    return (np.angle(z6) / 6.0).astype(np.float32)


def hexatic_order(z6: ArrayLike) -> np.ndarray:
    """|Psi6| in [0, 1]."""
    return np.abs(z6).astype(np.float32)


def compute_hexatic_phasor(points2d: ArrayLike, neighbours: 'NeighbourGraph') -> np.ndarray:
    """
    Per-point 6-fold order parameter (psi_6) from a 2D cloud + neighbour graph.
    Points with < 2 neighbours return 0+0j.
    """
    points2d = np.asarray(points2d, dtype=np.float64)
    nbr = ragged_neighbours(neighbours)
    z6 = np.zeros(len(points2d), dtype=np.complex128)

    for i, nb in enumerate(nbr):
        if nb.size < 2:
            continue
        # Bearings of neighbours relative to self
        d = points2d[nb] - points2d[i]
        angles = np.arctan2(d[:, 1], d[:, 0])
        z6[i] = resultant(angles=angles, fold=6)

    return z6


def hexatic_interpolant_fn(
        points2d: ArrayLike,
        neighbours: Optional[ArrayLike] = None,
        smoothing: float = 0.5,
        min_order: float = 0.5,
        return_confidence: bool = False
    ) -> Union['Callable', Tuple['Callable', 'Callable']]:
    """
    Continuous local hexatic-axis interpolant.

    Points with |Psi6| < min_order (incomplete rings at the boundary) are excluded from the fit,
    so the boundary inherits the smooth interior field instead of defining its own noisy one.
    """
    points2d = np.asarray(points2d, dtype=np.float64)

    if neighbours is None:
        neighbours = beta_skeleton_neighbours(points2d)

    z6 = compute_hexatic_phasor(points2d, neighbours)
    order = hexatic_order(z6)

    keep = order >= min_order
    if keep.sum() < 4:  # safety: nothing trusted
        keep = np.ones(len(points2d), dtype=bool)

    rbf_re = RBFInterpolator(points2d[keep], z6[keep].real, kernel='thin_plate_spline', smoothing=smoothing)
    rbf_im = RBFInterpolator(points2d[keep], z6[keep].imag, kernel='thin_plate_spline', smoothing=smoothing)

    def theta_fn(q):
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        return (np.angle(rbf_re(q) + 1j * rbf_im(q)) / 6.0).astype(np.float64)

    if not return_confidence:
        return theta_fn

    rbf_c = RBFInterpolator(points2d[keep], order[keep], kernel='thin_plate_spline', smoothing=smoothing)

    def conf_fn(q):
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        return np.clip(rbf_c(q), 0.0, 1.0)

    return theta_fn, conf_fn
