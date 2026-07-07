from dataclasses import dataclass
from typing import Callable, Optional, Tuple
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree
from scipy.interpolate import RBFInterpolator

from insectvision.geometry.polygons import Polygon2D
from insectvision.geometry.hexatic import hexatic_interpolant_fn
from insectvision.geometry.neighbours import (
    delaunay_neighbours, graph_spacing, NeighbourGraph
)
from insectvision.geometry.lattice import get_lattice_angles
from insectvision.geometry.smoothing import smooth_scalars


def _get_spacing_field(
        points2d: ArrayLike,
        neighbours: 'NeighbourGraph',
        smoothing: float = 0.1,
        clip_norm: Tuple[float, float] = (0.1, 2.0)
    ) -> Tuple[float, 'Callable']:
    """
    Continuous local-spacing field from the raw cloud (mean neighbour distance).
    Returns (mean_spacing, spacing_fn), where spacing_fn maps (M, 2) -> (M,).
    """

    points2d = np.asarray(points2d, dtype=np.float64)
    spacing = graph_spacing(points2d, neighbours)

    valid = np.isfinite(spacing)
    if not np.all(valid) and np.any(valid):
        _, nn = cKDTree(points2d[valid]).query(points2d[~valid])
        spacing[~valid] = spacing[valid][nn]
    elif not np.any(valid):
        spacing[:] = 1.0

    # no NaNs anymore so mean smoothing over the (padded) first-ring graph
    spacing = smooth_scalars(spacing, neighbours, n_iter=3, method='mean')

    # Ref scale: mean over inner 90% so the boundary doesn't inflate it
    lo, hi = np.percentile(spacing, [5, 95])
    core = (spacing >= lo) & (spacing <= hi)
    mean_spacing = float(spacing[core].mean()) if core.any() else float(spacing.mean())

    rbf = RBFInterpolator(points2d, spacing / mean_spacing, kernel='thin_plate_spline', smoothing=smoothing)

    def spacing_fn(q):
        spacing = rbf(np.atleast_2d(np.asarray(q, dtype=np.float64))).ravel()
        return np.clip(spacing, clip_norm[0], clip_norm[1]) * mean_spacing

    return mean_spacing, spacing_fn


@dataclass
class EyeMeasurements:
    """
    Measurements from a source 2D cloud (used to feed generation).
    """
    domain: 'Polygon2D'
    spacing_fn: 'Callable'
    theta_fn: 'Callable'
    lattice_angles: np.ndarray
    mean_spacing: float
    n_source: int
    source_points: Optional[np.ndarray] = None

    @classmethod
    def from_points(cls,
            points2d: ArrayLike,
            density_smoothing: float = 0.1,
            axes_smoothing: float = 0.4,
            min_hex_order: float = 0.2,
            max_length_factor:float = 1.8,
            smooth_domain: bool = True
        ) -> 'EyeMeasurements':

        points2d = np.asarray(points2d, dtype=float)

        neighbours = delaunay_neighbours(points2d, max_length_factor=max_length_factor)

        mean_spacing, spacing_fn = _get_spacing_field(points2d, neighbours, smoothing=density_smoothing)
        theta_fn = hexatic_interpolant_fn(
            points2d, neighbours, smoothing=axes_smoothing, min_order=min_hex_order
        )
        lattice_angles = get_lattice_angles(points2d, neighbours, theta_fn)

        domain = Polygon2D.from_points(points2d, smooth=smooth_domain)

        return cls(domain, spacing_fn, theta_fn, lattice_angles, mean_spacing, len(points2d), points2d)
