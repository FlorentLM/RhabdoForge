from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Sequence, List
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree
from scipy.interpolate import RBFInterpolator

from insectvision.geometry.linalg import principal_axis_angle
from insectvision.geometry.polygons import Polygon2D
from insectvision.geometry.hexatic import hexatic_axis_field
from insectvision.geometry.neighbours import (
    delaunay_neighbours, smooth_scalars, graph_spacing, walk_rows, NeighbourGraph
)


def get_spacing_field(
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
        s = rbf(np.atleast_2d(np.asarray(q, dtype=np.float64))).ravel()
        return np.clip(s, clip_norm[0], clip_norm[1]) * mean_spacing

    return mean_spacing, spacing_fn


def trace_lattice_rows(
        points2d: ArrayLike,
        neighbours: 'NeighbourGraph',
        axis_fn: 'Callable',
        offsets: Sequence[float] = (0.0, 60.0, 120.0),
        degrees: bool = True
    ) -> Tuple[int, List[np.ndarray], np.ndarray]:
    """
    Trace lattice rows outward from the centre seed along the local axis directions.

    Returns (seed, rows, bearings): the centre-seed index, the ordered (k, 2) row
    point-arrays, and their fitted bearings (rad, folded to [0, pi)). Rows whose
    bearing could not be fitted are dropped, so rows and bearings stay aligned.
    """

    points2d = np.asarray(points2d, dtype=np.float64)
    centre = points2d.mean(axis=0)
    seed = int(np.argmin(np.linalg.norm(points2d - centre, axis=1)))
    theta_seed = float(axis_fn(points2d[seed][None])[0])

    offsets_rad = np.deg2rad(offsets) if degrees else np.asarray(offsets, dtype=np.float64)
    rows = walk_rows(points2d, neighbours, seed, theta_seed + offsets_rad)

    kept_rows, kept_bearings = [], []
    for r in rows:
        b = principal_axis_angle(r)
        if not np.isnan(b):
            kept_rows.append(r)
            kept_bearings.append(b % np.pi)

    return seed, kept_rows, np.array(kept_bearings)


def get_lattice_angles(
        points2d: ArrayLike,
        neighbours: 'NeighbourGraph',
        axis_fn: 'Callable'
    ) -> np.ndarray:
    """Three internal angles of the mean unit cell (sums to pi), defaults to 60/60/60 if unresolved."""

    _, _, bearings = trace_lattice_rows(points2d, neighbours, axis_fn)

    if len(bearings) < 3:
        print('  Could not resolve 3 rows, using regular 60° hex')
        return np.array([np.pi / 3, np.pi / 3, np.pi / 3])

    bearings = np.sort(bearings)
    a1, a2 = bearings[1] - bearings[0], bearings[2] - bearings[1]
    return np.array([a1, a2, np.pi - (a1 + a2)])


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

        mean_spacing, spacing_fn = get_spacing_field(points2d, neighbours, smoothing=density_smoothing)
        theta_fn = hexatic_axis_field(
            points2d, neighbours, smoothing=axes_smoothing, min_order=min_hex_order
        )
        lattice_angles = get_lattice_angles(points2d, neighbours, theta_fn)

        domain = Polygon2D.from_points(points2d, smooth=smooth_domain)

        return cls(domain, spacing_fn, theta_fn, lattice_angles, mean_spacing, len(points2d), points2d)
