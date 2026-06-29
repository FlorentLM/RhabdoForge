from typing import Sequence, Optional
import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import RBFInterpolator

from insectvision.geometry.circular import resultant
from insectvision.geometry.neighbours import beta_skeleton_neighbours


def hexatic_axis_angle(z6) -> np.ndarray:
    """Local axis angle theta = arg(z6)/6, in (-pi/6, pi/6]."""
    return (np.angle(z6) / 6.0).astype(np.float32)


def hexatic_order(z6) -> np.ndarray:
    """|Psi6| in [0, 1]."""
    return np.abs(z6).astype(np.float32)


def phasor_from_points(points2d: np.ndarray, neighbours: Sequence[int] | Sequence[Sequence[int]]) -> np.ndarray:
    """
    Per-point 6-fold phasor from a 2D cloud + neighbour lists.

    'adj[i]' is an array/list of neighbour indices of point i (e.g. Delaunay one-ring).
    Points with < 2 neighbours get 0+0j.
    """

    points2d = np.asarray(points2d, dtype=np.float64)
    z6 = np.zeros(len(points2d), dtype=np.complex128)

    for i, nb in enumerate(neighbours):
        nb = np.asarray(nb, dtype=np.intp)
        if nb.size < 2:
            continue
        d = points2d[nb] - points2d[i]
        z6[i] = resultant(angles=np.arctan2(d[:, 1], d[:, 0]), axis=-1, fold=6)

    return z6


def hexatic_axis_field(
        points2d: ArrayLike,
        neighbours: Optional[ArrayLike] = None,
        smoothing: float = 0.5,
        min_order: float = 0.5,
        return_confidence: bool = False
    ):
    """
    Continuous local hexatic-axis interpolant.

    Points with |Psi6| < min_order (incomplete rings at the boundary) are excluded from the fit,
    so the boundary inherits the smooth interior field instead of defining its own noisy one.
    """
    points2d = np.asarray(points2d, dtype=np.float64)

    if neighbours is None:
        neighbours = beta_skeleton_neighbours(points2d)

    z6 = phasor_from_points(points2d, neighbours)
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
