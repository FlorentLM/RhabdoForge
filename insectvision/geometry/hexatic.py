import numpy as np
from numpy.typing import ArrayLike

from insectvision.geometry.circular import resultant
from insectvision.geometry.neighbours import NeighbourGraph, neighbour_bearings


def hexatic_axis_angle(z6: ArrayLike) -> np.ndarray:
    """Local axis angle theta = arg(z6)/6, in (-pi/6, pi/6]."""
    return (np.angle(z6) / 6.0).astype(np.float32)


def hexatic_order(z6: ArrayLike) -> np.ndarray:
    """|Psi6| in [0, 1]."""
    return np.abs(z6).astype(np.float32)


def compute_psi6(points2d: ArrayLike, neighbours: 'NeighbourGraph') -> np.ndarray:
    """
    Per-point 6-fold order parameter (psi_6) from a 2D cloud + neighbour graph.
    Points with < 2 neighbours return 0+0j.
    """
    bearings = neighbour_bearings(points2d, neighbours)

    z6 = np.zeros(len(points2d), dtype=np.complex128)

    for i, angles in enumerate(bearings):
        if angles.size < 2:
            continue
        z6[i] = resultant(angles=angles, fold=6)

    return z6