"""
Hexatic (6-fold) orientation order utilities.

A *hexatic axis* is a direction defined modulo 60 deg: the three lattice
axes {theta, theta+60, theta+120} are equivalent, and each axis is a line
(mod 180 deg). We represent the local axis by the 6-fold phasor

    z6 = < exp(6i * bearing) >      (mean over a point's neighbours)

whose magnitude |z6| in [0, 1] is the hexatic order parameter Psi6
(1 = perfect hex, 0 = isotropic disorder) and whose phase gives the axis angle

    theta = arg(z6) / 6         in (-pi/6, pi/6]

Everything here is 2D and frame-agnostic: callers pass bearings already measured
in whatever tangent frame they care about.
"""
from __future__ import annotations
from typing import Sequence
import numpy as np

_AXES_3 = np.array([0.0, np.pi / 3.0, 2.0 * np.pi / 3.0])


def hexatic_axis_angle(z6) -> np.ndarray:
    """Local axis angle theta = arg(z6)/6, in (-pi/6, pi/6]."""
    return np.angle(z6) / 6.0


def hexatic_order(z6) -> np.ndarray:
    """|Psi6| in [0, 1]."""
    return np.abs(z6)


def hexatic_phasor(bearings, weights=None, axis=-1) -> np.ndarray:
    """Mean 6-fold phasor < exp(6i*bearing) > along 'axis' (optionally weighted).

    Rows whose weights sum to <= 0 return 0+0j (undefined axis, |Psi6| = 0).
    """

    z = np.exp(6j * np.asarray(bearings, dtype=np.float64))
    if weights is None:
        return z.mean(axis=axis)

    w = np.asarray(weights, dtype=np.float64)
    num = (z * w).sum(axis=axis)
    den = w.sum(axis=axis)

    return np.divide(num, den, out=np.zeros_like(num), where=den > 0)


def phasor_from_points(pts_2d: np.ndarray, adj: Sequence) -> np.ndarray:
    """Per-point 6-fold phasor from a 2D cloud + neighbour lists.

    'adj[i]' is an array/list of neighbour indices of point i (e.g. Delaunay one-ring).
    Points with < 2 neighbours get 0+0j.
    """

    pts_2d = np.asarray(pts_2d, dtype=np.float64)
    z6 = np.zeros(len(pts_2d), dtype=np.complex128)

    for i, nb in enumerate(adj):
        nb = np.asarray(nb, dtype=np.intp)
        if nb.size < 2:
            continue
        d = pts_2d[nb] - pts_2d[i]
        z6[i] = hexatic_phasor(np.arctan2(d[:, 1], d[:, 0]))

    return z6


def snap_bearing_to_axes(bearing, theta) -> np.ndarray:
    """Angle of the nearest hexatic axis *line* to each bearing.

    Axes are {theta, theta+60, theta+120} (each a line, mod 180 deg). Returns the
    absolute angle of the nearest line (a line orientation, sign not meaningful).
    'bearing' and 'theta' broadcast against each other.
    """

    bearing = np.asarray(bearing, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64)
    axes = theta[..., None] + _AXES_3

    # signed distance to each axis line, folded into (-pi/2, pi/2]
    d = np.angle(np.exp(2j * (bearing[..., None] - axes))) / 2.0
    k = np.argmin(np.abs(d), axis=-1)

    return np.take_along_axis(axes, k[..., None], axis=-1)[..., 0]


def hexatic_rest_vectors(edge_vec: np.ndarray, theta, length) -> np.ndarray:
    """Rest vectors for oriented springs.

    For each edge (row of 'edge_vec'), snap its bearing to the nearest local
    hexatic axis line, and return a vector of magnitude 'length' along that line,
    signed to point the same way as 'edge_vec'
    (so the spring never tries to flip an edge by 180 deg)

    """
    edge_vec = np.asarray(edge_vec, dtype=np.float64)
    length = np.asarray(length, dtype=np.float64)
    bearing = np.arctan2(edge_vec[:, 1], edge_vec[:, 0])
    snap = snap_bearing_to_axes(bearing, np.asarray(theta, dtype=np.float64))
    rest = length[:, None] * np.stack([np.cos(snap), np.sin(snap)], axis=1)
    flip = np.einsum('ij,ij->i', rest, edge_vec) < 0.0
    rest[flip] *= -1.0
    return rest


def smooth_phasor(z6, neighbours, weights=None, iterations=3, include_self=True) -> np.ndarray:
    """Smooth a per-point 6-fold phasor over a precomputed neighbour graph.

    Operates on the complex phasor so the 60 deg ambiguity is respected (no domain
    walls, unlike nematic/vector smoothing of the resulting axis). Renormalises to
    a unit phasor each pass, confidence lives in 'weights'.

    Args:
        z6: (N,) complex per-point phasors (need not be unit)
        neighbours: (N, k) int neighbour indices; entries < 0 are ignored (padding)
        weights: (N,) per-point confidence (e.g. |Psi6|); None -> uniform
        iterations: smoothing passes
        include_self: keep each point's own phasor in its average
    """

    z6 = np.asarray(z6, dtype=np.complex128).copy()
    nb = np.asarray(neighbours, dtype=np.intp)
    N = z6.shape[0]
    w = np.ones(N) if weights is None else np.asarray(weights, dtype=np.float64)

    valid = nb >= 0
    safe = np.where(valid, nb, 0)

    for _ in range(int(iterations)):
        zw = z6 * w
        num = np.where(valid, zw[safe], 0.0 + 0.0j).sum(axis=1)
        den = np.where(valid, w[safe], 0.0).sum(axis=1)
        if include_self:
            num = num + zw
            den = den + w
        z6 = np.divide(num, np.maximum(den, 1e-12))
        z6 = np.divide(z6, np.maximum(np.abs(z6), 1e-12))
    return z6


def smooth_tilt(tilt, neighbours, weights=None, iterations=3) -> np.ndarray:
    """Smooth per-point hexatic axis angles (mod 60 deg) via 6-fold phasor averaging.
    Returns angles in (-pi/6, pi/6]."""
    z6 = np.exp(6j * np.asarray(tilt, dtype=np.float64))
    z6 = smooth_phasor(z6, neighbours, weights=weights, iterations=iterations)
    return hexatic_axis_angle(z6)