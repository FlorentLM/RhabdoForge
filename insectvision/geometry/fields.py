from typing import Optional, Sequence, Union, Callable, Tuple
import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

from insectvision.geometry.hexatic import compute_psi6, hexatic_order
from insectvision.geometry.neighbours import (
    NeighbourGraph, padded_neighbours, knn, beta_skeleton_neighbours, graph_spacing
)


# Discrete graph smoothing

def smooth_scalars(
        values: ArrayLike,
        neighbours: NeighbourGraph,
        mask: Optional[ArrayLike] = None,
        method: str = 'mean',
        n_iter: int = 2,
        include_self: bool = True
    ) -> np.ndarray:
    """
    Smooth a 1D scalar field over a precomputed neighbour graph.

    Args:
        - values: values to smooth, (N,)
        - neighbours: NeighbourGraph (ragged or dense, dense entries < 0 are padding)
        - n_iter: smoothing passes
        - mask: optional bool array, only True entries are updated, (N,)
        - method: 'mean' (half-self half-neighbours) or 'median' (self + neighbours)
    """

    method = 'mean' if not method else str(method).lower()

    out = np.copy(values).astype(np.float64)
    if n_iter <= 0:
        return out.astype(values.dtype)

    update_mask = np.asarray(mask) if mask is not None else np.ones(out.shape[0], dtype=bool)

    neighb_indices, valid_neighb_mask = padded_neighbours(neighbours, masked=True)

    for _ in range(n_iter):
        nb_vals = out[neighb_indices]

        if method == 'mean':
            num = np.sum(np.where(valid_neighb_mask, nb_vals, 0.0), axis=1)
            den = np.sum(valid_neighb_mask, axis=1)

            if include_self:
                num += out
                den += 1

            cur = np.divide(num, np.maximum(den, 1))  # avoid div by zero

        elif method == 'median':
            nb_vals_masked = np.where(valid_neighb_mask, nb_vals, np.nan)
            if include_self:
                # stack self with neighbours
                data = np.hstack([out[:, None], nb_vals_masked])
            else:
                data = nb_vals_masked

            with np.errstate(all='ignore'):
                cur = np.nanmedian(data, axis=1)
                # if a point has no neighbours and no self, fallback to original
                cur = np.where(np.isnan(cur), out, cur)
        else:
            raise ValueError(f'Unknown method {method!r}')

        out[update_mask] = cur[update_mask]

    return out.astype(values.dtype)


def smooth_phasors(
        values: ArrayLike,
        neighbours: NeighbourGraph,
        weights: Optional[ArrayLike] = None,
        n_iter: int = 3,
        include_self: bool = True,
) -> np.ndarray:
    """
    Smooth a per-point complex phasor over a precomputed neighbour graph.

    Args:
        values: (N,) complex phasors (need not be unit). For a hexatic field these
            are exp(6i*theta), for a nematic field exp(2i*theta), etc.
        neighbours: NeighbourGraph (ragged or dense, dense entries < 0 are padding).
        n_iter: smoothing passes.
        weights: (N,) per-point confidence (e.g. |Psi|), None -> uniform.
        include_self: keep each point's own phasor in its average.
    """

    z = np.copy(values).astype(np.complex128)
    w = np.ones(z.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64)

    neighb_indices, valid_neighb_mask = padded_neighbours(neighbours)

    for _ in range(n_iter):
        zw = z * w
        num = np.where(valid_neighb_mask, zw[neighb_indices], 0.0 + 0.0j).sum(axis=1)
        den = np.where(valid_neighb_mask, w[neighb_indices], 0.0).sum(axis=1)
        if include_self:
            num += zw
            den += w

        z = np.divide(num, np.maximum(den, 1e-12))
        z = np.divide(z, np.maximum(np.abs(z), 1e-12))

    return z


def smooth_nematic_vectors(
        values: ArrayLike,
        neighbours: NeighbourGraph,
        n_iter: int = 10,
        include_self: bool = True,
) -> np.ndarray:
    """
    Smooth sign-ambiguous (nematic) unit vectors over a precomputed neighbour graph.

    Each neighbour is flipped into the same hemisphere as the centre vector
    before averaging, so 180-degree-equivalent directors don't cancel. Used for
    director-like fields (e.g. bundle / saccade axes) where orientation matters
    but sign does not.

    Args:
        values: (M, D) unit vectors
        neighbours: NeighbourGraph (ragged or dense, dense entries < 0 are padding).
            A dense knn() output (self dropped) is the typical input.
        n_iter: smoothing passes
        include_self: keep each vector's own value in its average
    """

    out = np.copy(values).astype(np.float64)
    if n_iter <= 0:
        return out

    neighb_indices, valid_neighb_mask = padded_neighbours(neighbours, masked=True)

    for _ in range(n_iter):
        base = out
        neigh = out[neighb_indices]

        # Align sign per neighbour
        dots = np.einsum('id,ikd->ik', base, neigh)
        neigh = np.where(dots[..., None] < 0, -neigh, neigh)

        # Zero invalid padded neighbours
        neigh = np.where(valid_neighb_mask[..., None], neigh, 0.0)

        sum_vecs = neigh.sum(axis=1)
        counts = valid_neighb_mask.sum(axis=1, keepdims=True)

        if include_self:
            sum_vecs += base
            counts += 1

        avg = np.divide(sum_vecs, np.maximum(counts, 1))

        # Re-normalise, falling back to previous vector if magnitude is destroyed
        norms = np.linalg.norm(avg, axis=1, keepdims=True)
        out = np.where(norms > 1e-8, avg / np.maximum(norms, 1e-8), base)

    return out


def smooth_field_partitioned(
        values: ArrayLike,
        neighbours: Optional[Sequence[np.ndarray]] = None,  # TODO: should accept NeighbourGraph and do the pad / rag
        weights: Optional[np.ndarray] = None,
        kind: str = 'scalar',
        partition: Optional[np.ndarray] = None,
        positions: Optional[np.ndarray] = None,
        groups: Optional[Sequence[np.ndarray]] = None,
        k: int = 8,
        min_group: int = 2,
        mask: Optional[np.ndarray] = None,
        method: str = 'mean',
        n_iter: int = 2,
        include_self: bool = True,
    ) -> np.ndarray:
    """
    Smooth a per-element field independently within disjoint groups.
    Groups smaller than 'min_group' pass through untouched.

    Two ways to supply the groups and their neighbour graphs:

      1. Label mode (transient trees built here):
            partition: (N,) group label per element (np.unique-able)
            positions: (N, D) coordinates, a cKDTree is built per group and
                       queried for k neighbours (self dropped).
         Use when you only have positions and a labelling.

      2. Precomputed mode (caller owns the graphs, e.g. cached KD-trees or a non-Euclidean metric):
            groups:     sequence of global-index arrays, one per group
            neighbours: matching sequence of (n_g, k) *group-local* neighbour
                        index arrays (e.g. straight from knn(), -1 padding ok).

    Args:
        kind: which kernel to run, 'scalar' | 'phasor' | 'nematic'.
        k: neighbours per element (label mode only, clamped to group size).
        n_iter: smoothing passes per group.
        weights: (N,) per-element confidence, sliced per group ('phasor' only).
        mask: (N,) bool update mask, sliced per group ('scalar' only).
        method: 'mean' | 'median' ('scalar' only).
        include_self: keep each element's own value in its average
            ('phasor' / 'nematic' only).

    Returns:
        Smoothed field, same shape and dtype as 'values'.
    """

    kind = 'scalar' if not kind else str(kind).lower()
    method = 'mean' if not method else str(method).lower()

    if kind not in ('scalar', 'phasor', 'nematic'):
        raise ValueError(f"Unknown kind {kind!r}, expected 'scalar', 'phasor' or 'nematic'")

    label_mode = partition is not None
    precomp_mode = groups is not None

    if label_mode == precomp_mode:
        raise ValueError("Provide exactly one of 'partition' or 'groups'")
    if label_mode and positions is None:
        raise ValueError("'partition' mode requires 'positions'")
    if precomp_mode and neighbours is None:
        raise ValueError("'groups' mode requires a matching 'neighbours' sequence")

    out = np.asarray(values).copy()
    if n_iter <= 0:
        return out

    floor = max(int(min_group), 2)  # a group needs >= 2 members to have a neighbour

    # Build work list of (group_global_idx, group_local_neighbours).
    if label_mode:
        positions = np.asarray(positions, dtype=float)
        work = []
        for label in np.unique(partition):
            gi = np.flatnonzero(partition == label)
            if gi.size < floor:
                continue
            _, nb_local = knn(cKDTree(positions[gi]), None, k)
            work.append((gi, nb_local))
    else:
        if len(groups) != len(neighbours):
            raise ValueError("'groups' and 'neighbours' must have the same length")
        work = [(np.asarray(gi, dtype=np.intp), nb)
                for gi, nb in zip(groups, neighbours)
                if np.asarray(gi).size >= floor]

    for gi, nb_local in work:
        vals = out[gi]
        if kind == 'scalar':
            sm = smooth_scalars(
                values=vals, neighbours=nb_local, mask=None if mask is None else mask[gi], method=method, n_iter=n_iter,
            )
        elif kind == 'phasor':
            sm = smooth_phasors(
                values=vals,
                neighbours=nb_local,
                weights=None if weights is None else weights[gi],
                n_iter=n_iter,
                include_self=include_self
            )
        else:  # nematic
            sm = smooth_nematic_vectors(values=vals, neighbours=nb_local, n_iter=n_iter, include_self=include_self)

        out[gi] = sm.astype(out.dtype, copy=False)

    return out


# Continuous field interpolation

def interpolate_hexatic_field(
        points2d: ArrayLike,
        neighbours: Optional[NeighbourGraph] = None,
        smoothing: float = 0.5,
        min_order: float = 0.5,
        return_confidence: bool = False
    ) -> Union[Callable, Tuple[Callable, Callable]]:
    """
    Creates a continuous function Theta(x, y) for local lattice orientation.

    Points with |Psi6| < min_order (incomplete rings at the boundary) are excluded from the fit,
    so the boundary inherits the smooth interior field instead of defining its own noisy one.
    
    Args:
        - points2d: (N, 2) lattice coordinates
        - neighbours: Optional graph. If None, a Beta-skeleton is computed
        - smoothing: Regularisation parameter for the RBF interpolator
        - min_order: Threshold for Hexatic Order (|Psi6|). Points below 
            this are excluded from the fit, e.g. for allowing the boundary to inherit
            the orientation of the (better ordered) interior
    """
    points2d = np.asarray(points2d, dtype=np.float64)

    if neighbours is None:
        neighbours = beta_skeleton_neighbours(points2d)

    z6 = compute_psi6(points2d=points2d, neighbours=neighbours)
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


def interpolate_spacing_field(
        points2d: ArrayLike,
        neighbours: NeighbourGraph,
        smoothing: float = 0.1,
        clip_norm: Tuple[float, float] = (0.1, 2.0),
        return_mean_spacing: bool = False
    ) -> Union[Callable, Tuple[Callable, float]]:
    """
    Creates a continuous function S(x, y) for local lattice spacing

    Args:
        - points2d: (N, 2) lattice coordinates
        - neighbours: The first-ring neighbour graph
        - smoothing: Regularisation parameter for the RBF interpolator.
            Higher = stiffer, smoother surface
            0.0 forces the surface to pass exactly through every point.
        - clip_norm: (min, max) bounds for the field relative to the global mean
                (prevents the field from exploding at boundaries)


    Returns (spacing_fn, mean_spacing), where spacing_fn maps (M, 2) -> (M,).
    """
    points2d = np.asarray(points2d, dtype=np.float64)
    spacing = graph_spacing(points2d, neighbours)

    valid = np.isfinite(spacing)
    if not np.all(valid) and np.any(valid):
        _, nn = cKDTree(points2d[valid]).query(points2d[~valid])
        spacing[~valid] = spacing[valid][nn]
    elif not np.any(valid):
        spacing[:] = 1.0

    # no NaNs anymore so mean smoothing over the first-ring graph
    spacing = smooth_scalars(values=spacing, neighbours=neighbours, method='mean', n_iter=3)

    # Ref scale: mean over inner 90% so the boundary doesn't inflate it
    lo, hi = np.percentile(spacing, [5, 95])
    core = (spacing >= lo) & (spacing <= hi)
    mean_spacing = float(spacing[core].mean()) if core.any() else float(spacing.mean())

    rbf = RBFInterpolator(points2d, spacing / mean_spacing, kernel='thin_plate_spline', smoothing=smoothing)

    def spacing_fn(q):
        spacing = rbf(np.atleast_2d(np.asarray(q, dtype=np.float64))).ravel()
        return np.clip(spacing, clip_norm[0], clip_norm[1]) * mean_spacing

    if return_mean_spacing:
        return spacing_fn, mean_spacing
    return spacing_fn


def interpolate_scalar_field(
        points2d: ArrayLike,
        values: ArrayLike,
        neighbours: Optional[NeighbourGraph] = None,
        smoothing: float = 0.1
    ) -> Callable:
    """
    Creates a generic continuous interpolator F(x, y) for any scalar quantity (e.g., sensitivity, aperture).

    Args:
        - points2d: (N, 2) array of spatial coordinates
        - values: (N,) property measured or assigned at each point
        - neighbours: Optional connectivity graph. If provided, the discrete
            values are pre-smoothed over the graph.
        - smoothing: Regularisation parameter for the RBF interpolator.
            Higher = stiffer, smoother surface
            0.0 forces the surface to pass exactly through every point.
    """
    # TODO: Use this

    if neighbours is not None:
        values = smooth_scalars(values=values, neighbours=neighbours, n_iter=2)

    rbf = RBFInterpolator(points2d, values, kernel='thin_plate_spline', smoothing=smoothing)
    return lambda q: rbf(np.atleast_2d(q)).ravel()