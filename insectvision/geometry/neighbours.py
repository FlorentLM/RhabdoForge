from typing import List, Optional, Tuple, Sequence
import numpy as np
from scipy.spatial import cKDTree, Delaunay


# 2D neighbour graphs

def _delaunay_pairs(points2d: np.ndarray) -> set:
    """Unique undirected (i, j) Delaunay edges (i < j)."""
    tri = Delaunay(points2d)
    pairs = set()
    for sx in tri.simplices:
        for i in range(3):
            a, b = int(sx[i]), int(sx[(i + 1) % 3])
            pairs.add((a, b) if a < b else (b, a))
    return pairs


def _prune_long_edges(points2d: np.ndarray, edges: np.ndarray, max_length_factor: float) -> np.ndarray:
    """
    Boolean keep-mask for edges shorter than max_length_factor x local spacing.

    Local spacing is the kNN mean distance at each endpoint; an edge survives iff
    its length is below max_length_factor times the *mean* of its two endpoints'
    spacing. The criterion is symmetric, so the surviving graph is symmetric too
    (edge i-j is kept for both i and j, or for neither).
    """
    spacing = mean_neighbour_distance(
        cKDTree(points2d), points2d, k=min(6, max(1, len(points2d) - 1))
    )
    lengths = np.linalg.norm(points2d[edges[:, 0]] - points2d[edges[:, 1]], axis=1)
    mean_local = 0.5 * (spacing[edges[:, 0]] + spacing[edges[:, 1]])
    return lengths < mean_local * max_length_factor


def delaunay_edges(points2d: np.ndarray, max_length_factor: float = 0.0) -> np.ndarray:
    """
    Unique undirected Delaunay edges in the plane, as a sorted (E, 2) int array.

    If max_length_factor > 0, prunes edges longer than max_length_factor x local
    spacing (convex-hull boundary edges, not real neighbours). Disabled if <= 0.
    """
    points2d = np.asarray(points2d)
    pairs = _delaunay_pairs(points2d)
    edges = np.array(sorted(pairs)) if pairs else np.zeros((0, 2), dtype=int)

    if max_length_factor > 0 and len(edges):
        edges = edges[_prune_long_edges(points2d, edges, max_length_factor)]

    return edges


def delaunay_neighbours(points2d: np.ndarray, max_length_factor: float = 0.0) -> List[np.ndarray]:
    """
    One-ring neighbour lists from a 2D Delaunay triangulation.

    This is the adjacency-list view of delaunay_edges, so the optional
    max_length_factor pruning uses the exact same (symmetric) criterion: an edge
    is kept for both endpoints or for neither.
    """
    points2d = np.asarray(points2d)
    neighbour_lists: List[list] = [[] for _ in range(len(points2d))]

    for a, b in delaunay_edges(points2d, max_length_factor=max_length_factor):
        a, b = int(a), int(b)
        neighbour_lists[a].append(b)
        neighbour_lists[b].append(a)

    return [np.array(s, dtype=np.intp) for s in neighbour_lists]


# knn queries against a prebuilt KD-tree

def knn(
        tree: cKDTree,
        query_pts: np.ndarray,
        k: int,
        drop_self: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    k-nearest-neighbour query against a prebuilt tree.

    Args:
        tree: a prebuilt cKDTree (owned/cached by the caller)
        query_pts: (Q, D) query points
        k: number of neighbours requested. Clamped to the tree size.
        drop_self: if True, query k+1 and drop the first (self) column. Use for
            self-queries (query points are tree members). Set False when querying
            external points, where the nearest match is not the point itself.

    Returns:
        (distances, indices), both (Q, k_eff) with k_eff = min(k, n - 1) when drop_self else min(k, n)
        When k_eff == 0, returns (Q, 0) arrays
    """
    query_pts = np.atleast_2d(query_pts)
    n = tree.n
    max_k = (n - 1) if drop_self else n
    k_eff = min(int(k), max_k)

    if k_eff <= 0:
        q = query_pts.shape[0]
        return np.empty((q, 0), dtype=float), np.empty((q, 0), dtype=np.intp)

    kq = k_eff + 1 if drop_self else k_eff
    dist, idx = tree.query(query_pts, k=kq)

    if idx.ndim == 1:  # kq == 1
        idx = idx.reshape(-1, 1)
        dist = dist.reshape(-1, 1)
    if drop_self:
        idx = idx[:, 1:]
        dist = dist[:, 1:]

    return dist, idx.astype(np.intp)


def mean_neighbour_distance(
        tree: cKDTree,
        query_pts: Optional[np.ndarray] = None,
        k: int = 6,
) -> np.ndarray:
    """
    Per-point mean distance to its k nearest neighbours (the local point-spacing
    scale).

    If 'query_pts' is None the tree's own data is used (i.e. the mean spacing of
    this cloud).
    Returns NaN for points with no available neighbours.
    """
    pts = tree.data if query_pts is None else query_pts
    dist, _ = knn(tree, pts, k, drop_self=True)
    if dist.shape[1] == 0:
        return np.full(dist.shape[0], np.nan)
    return dist.mean(axis=1)


# Neighbourhood-based smoothing

def _masked_neighbours(neighbours: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(valid, safe) for an (N, k) neighbour array using -1 (or any < 0) as padding.

    'valid' is the boolean keep-mask, 'safe' is 'neighbours' with padding clamped
    to 0 so it can be used for fancy-indexing without going out of bounds.
    """
    nb = np.asarray(neighbours, dtype=np.intp)
    valid = nb >= 0
    safe = np.where(valid, nb, 0)
    return valid, safe


def smooth_scalars(
        values: np.ndarray,
        neighbours: np.ndarray,
        n_iter: int = 2,
        mask: Optional[np.ndarray] = None,
        method: str = 'mean',
) -> np.ndarray:
    """
    Smooth a 1D scalar field over a precomputed neighbour graph.

    Args:
        values: (N,) values to smooth
        neighbours: (N, k) neighbour indices, entries < 0 are ignored (padding)
        n_iter: smoothing passes
        mask: (N,) bool. If given, only True entries are updated
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


def smooth_phasors(
        values: np.ndarray,
        neighbours: np.ndarray,
        n_iter: int = 3,
        weights: Optional[np.ndarray] = None,
        include_self: bool = True,
) -> np.ndarray:
    """
    Smooth a per-point complex phasor over a precomputed neighbour graph.

    Args:
        values: (N,) complex phasors (need not be unit). For a hexatic field these
            are exp(6i*theta), for a nematic field exp(2i*theta), etc.
        neighbours: (N, k) int neighbour indices, entries < 0 are ignored (padding).
        n_iter: smoothing passes.
        weights: (N,) per-point confidence (e.g. |Psi|), None -> uniform.
        include_self: keep each point's own phasor in its average.
    """
    z = np.asarray(values, dtype=np.complex128).copy()
    w = np.ones(z.shape[0]) if weights is None else np.asarray(weights, dtype=np.float64)

    valid_nb, safe_nb = _masked_neighbours(neighbours)

    for _ in range(n_iter):
        zw = z * w
        num = np.where(valid_nb, zw[safe_nb], 0.0 + 0.0j).sum(axis=1)
        den = np.where(valid_nb, w[safe_nb], 0.0).sum(axis=1)
        if include_self:
            num += zw
            den += w

        z = np.divide(num, np.maximum(den, 1e-12))
        z = np.divide(z, np.maximum(np.abs(z), 1e-12))

    return z


def smooth_nematic_vectors(
        values: np.ndarray,
        neighbours: np.ndarray,
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
        neighbours: (M, k) neighbour indices, entries < 0 are ignored (padding).
            A dense knn() output (self dropped) is the typical input.
        n_iter: smoothing passes
        include_self: keep each vector's own value in its average
    """
    out = np.asarray(values, dtype=np.float64).copy()
    if n_iter <= 0:
        return out

    valid_nb, safe_nb = _masked_neighbours(neighbours)

    for _ in range(n_iter):
        base = out
        neigh = out[safe_nb]

        # Align sign per neighbour
        dots = np.einsum('id,ikd->ik', base, neigh)
        neigh = np.where(dots[..., None] < 0, -neigh, neigh)

        # Zero invalid padded neighbours
        neigh = np.where(valid_nb[..., None], neigh, 0.0)

        sum_vecs = neigh.sum(axis=1)
        counts = valid_nb.sum(axis=1, keepdims=True)

        if include_self:
            sum_vecs += base
            counts += 1

        avg = np.divide(sum_vecs, np.maximum(counts, 1))
        norms = np.linalg.norm(avg, axis=1, keepdims=True)

        # Re-normalise, falling back to previous vector if magnitude is destroyed
        out = np.where(norms > 1e-8, avg / np.clip(norms, 1e-8, None), base)

    return out


def smooth_field_partitioned(
        values: np.ndarray,
        *,
        kind: str = 'scalar',
        partition: Optional[np.ndarray] = None,
        positions: Optional[np.ndarray] = None,
        groups: Optional[Sequence[np.ndarray]] = None,
        neighbours: Optional[Sequence[np.ndarray]] = None,
        k: int = 8,
        n_iter: int = 2,
        min_group: int = 2,
        weights: Optional[np.ndarray] = None,
        mask: Optional[np.ndarray] = None,
        method: str = 'mean',
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
            _, nb_local = knn(cKDTree(positions[gi]), positions[gi], k)
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
                vals, nb_local, n_iter=n_iter,
                mask=None if mask is None else mask[gi], method=method,
            )
        elif kind == 'phasor':
            sm = smooth_phasors(
                vals, nb_local, n_iter=n_iter,
                weights=None if weights is None else weights[gi], include_self=include_self,
            )
        else:  # nematic
            sm = smooth_nematic_vectors(vals, nb_local, n_iter=n_iter, include_self=include_self)

        out[gi] = sm.astype(out.dtype, copy=False)

    return out