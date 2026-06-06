from __future__ import annotations
from typing import Optional, Tuple
import numpy as np


def _masked_neighbours(neighbours: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(valid, safe) for an (N, k) neighbour array using -1 (or any < 0) as padding.

    'valid' is the boolean keep-mask, 'safe' is 'neighbours' with padding clamped
    to 0 so it can be used for fancy-indexing without going out of bounds.
    """
    nb = np.asarray(neighbours, dtype=np.intp)
    valid = nb >= 0
    safe = np.where(valid, nb, 0)
    return valid, safe


def neighbour_smooth(
        values: np.ndarray,
        neighbours: np.ndarray,
        mask: Optional[np.ndarray] = None,
        n_iter: int = 2,
        method: str = 'mean',
) -> np.ndarray:
    """
    Smooth a 1-D scalar field over a precomputed neighbour graph.

    Args:
        values: (N,) values to smooth
        neighbours: (N, k) neighbour indices, entries < 0 are ignored (padding)
        mask: (N,) bool. If given, only True entries are updated
        n_iter: smoothing passes
        method: 'mean' (half-self half-neighbours) or 'median' (self + neighbours)
    """
    out = np.asarray(values, dtype=np.float64).copy()
    if n_iter <= 0:
        return out.astype(values.dtype)

    update_mask = mask if mask is not None else np.ones(out.shape[0], dtype=bool)
    valid_nb, safe_nb = _masked_neighbours(neighbours)

    for _ in range(n_iter):
        nb_vals = out[safe_nb]

        if method == 'mean':
            sum_nb = np.sum(np.where(valid_nb, nb_vals, 0.0), axis=1)
            count_nb = np.maximum(np.sum(valid_nb, axis=1), 1)
            cur = 0.5 * out + 0.5 * (sum_nb / count_nb)
        elif method == 'median':
            nb_vals = np.where(valid_nb, nb_vals, np.nan)
            stacked = np.concatenate([out[:, None], nb_vals], axis=1)
            with np.errstate(all='ignore'):
                cur = np.nanmedian(stacked, axis=1)
        else:
            raise ValueError(f"Unknown method {method!r}")

        out = np.where(update_mask, cur, out)

    return out.astype(values.dtype)


def smooth_phasor(
        z: np.ndarray,
        neighbours: np.ndarray,
        weights: Optional[np.ndarray] = None,
        iterations: int = 3,
        include_self: bool = True,
) -> np.ndarray:
    """
    Smooth a per-point complex phasor over a precomputed neighbour graph.

    Operating on the phasor (rather than the recovered angle) respects whatever
    n-fold ambiguity is baked into it, so there are no domain walls. The result
    is renormalised to unit modulus each pass, confidence lives in 'weights'.

    Args:
        z: (N,) complex phasors (need not be unit). For a hexatic field these are
            exp(6i*theta), for a nematic field, exp(2i*theta), etc.
        neighbours: (N, k) int neighbour indices, entries < 0 are ignored (padding).
        weights: (N,) per-point confidence (e.g. |Psi|), None -> uniform.
        iterations: smoothing passes.
        include_self: keep each point's own phasor in its average.
    """
    z = np.asarray(z, dtype=np.complex128).copy()
    N = z.shape[0]
    w = np.ones(N) if weights is None else np.asarray(weights, dtype=np.float64)

    valid, safe = _masked_neighbours(neighbours)

    for _ in range(int(iterations)):
        zw = z * w
        num = np.where(valid, zw[safe], 0.0 + 0.0j).sum(axis=1)
        den = np.where(valid, w[safe], 0.0).sum(axis=1)
        if include_self:
            num = num + zw
            den = den + w
        z = np.divide(num, np.maximum(den, 1e-12))
        z = np.divide(z, np.maximum(np.abs(z), 1e-12))
    return z


def smooth_nematic_vectors(
        vectors: np.ndarray,
        neighbours: np.ndarray,
        iterations: int = 10,
) -> np.ndarray:
    """
    Smooth sign-ambiguous (nematic) unit vectors over a precomputed graph.

    Each neighbour is flipped into the same hemisphere as the centre vector
    before averaging, so 180-degree-equivalent directors don't cancel. Used for
    director-like fields (e.g. bundle / saccade axes) where orientation matters
    but sign does not.

    Unlike 'smooth_phasor' / 'neighbour_smooth', this expects a *dense* (M, k)
    neighbour array with no padding (every entry a valid index), e.g. straight
    from a cKDTree query with self dropped.

    Args:
        vectors: (M, D) unit vectors.
        neighbours: (M, k) neighbour indices into 'vectors'.
        iterations: smoothing passes.
    """
    out = np.asarray(vectors, dtype=np.float64).copy()
    if iterations <= 0:
        return out
    nb = np.asarray(neighbours, dtype=np.intp)

    for _ in range(int(iterations)):
        base = out
        neigh = out[nb]
        dots = np.einsum('id,ikd->ik', base, neigh)  # align sign per neighbour
        neigh = np.where(dots[..., None] < 0, -neigh, neigh)

        stacked = np.concatenate([base[:, None, :], neigh], axis=1)
        avg = stacked.mean(axis=1)
        norms = np.linalg.norm(avg, axis=1, keepdims=True)
        out = np.where(norms > 1e-8, avg / np.clip(norms, 1e-8, None), base)

    return out
