from typing import List, Optional, Tuple, Sequence, Callable
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree, Delaunay

from insectvision.utils import norm_l2

# A neighbour graph, can be:
#   ragged: a sequence (list / object-array) of per-point neighbour-index arrays
#   dense: an (N, k) integer array with entries < 0 used as padding
NeighbourGraph = np.ndarray | Sequence[ArrayLike]


def _is_dense(neighbours: NeighbourGraph) -> bool:
    """True for dense (N, k) array representation, False for ragged."""
    return isinstance(neighbours, np.ndarray) and neighbours.ndim == 2


def ragged_neighbours(neighbours: NeighbourGraph) -> List[np.ndarray]:
    """
    Normalise a NeighbourGraph to a list of per-point neighbour-index arrays.
    Padding and any negative index are stripped.
    """
    if _is_dense(neighbours):
        dense = np.asarray(neighbours, dtype=np.intp)
        return [row[row >= 0] for row in dense]

    out: List[np.ndarray] = []
    for nb in neighbours:
        nb = np.asarray(nb, dtype=np.intp)
        out.append(nb[nb >= 0])
    return out


def padded_neighbours(neighbours: NeighbourGraph, pad_value: int = -1) -> np.ndarray:
    """
    Densify a NeighbourGraph to a (N, kmax) int array padded with 'pad_value'.

    Ragged input is padded to the widest row, an already-dense array is returned
    unchanged (cast to intp).
    """
    if _is_dense(neighbours):
        return np.asarray(neighbours, dtype=np.intp)

    n = len(neighbours)
    kmax = max((len(nb) for nb in neighbours), default=0)
    out = np.full((n, kmax), pad_value, dtype=np.intp)
    for i, nb in enumerate(neighbours):
        nb = np.asarray(nb, dtype=np.intp)
        out[i, :nb.size] = nb
    return out


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


def delaunay_edges(points2d: np.ndarray, max_length_factor: float = 0.0) -> np.ndarray:
    """
    Unique undirected Delaunay edges in the plane, as a sorted (E, 2) int array.
    max_length_factor > 0 prunes edges longer than max_length_factor x local spacing. Disabled if <= 0.
    """

    points2d = np.asarray(points2d, dtype=np.float64)
    pairs = _delaunay_pairs(points2d)
    edges = np.array(sorted(pairs)) if pairs else np.zeros((0, 2), dtype=int)

    if max_length_factor > 0 and len(edges):
        # Local spacing is the kNN mean distance at each endpoint, an edge survives iff
        # its length is below max_length_factor x the mean of its two endpoints' spacing.

        # The criterion is symmetric, so the surviving graph is symmetric too
        # (edge i-j is kept for both i and j, or for neither)
        spacing = ball_spacing(cKDTree(points2d), None, k=min(6, max(1, len(points2d) - 1)))

        lengths = np.linalg.norm(points2d[edges[:, 0]] - points2d[edges[:, 1]], axis=1)
        mean_local = 0.5 * (spacing[edges[:, 0]] + spacing[edges[:, 1]])
        keep = lengths < mean_local * max_length_factor
        edges = edges[keep]

    return edges


def delaunay_neighbours(points2d: ArrayLike, max_length_factor: float = 0.0) -> List[np.ndarray]:
    """
    One-ring neighbour lists from a 2D Delaunay triangulation.

    This is the adjacency-list view of delaunay_edges, so the optional
    max_length_factor pruning uses the exact same (symmetric) criterion: an edge
    is kept for both endpoints or for neither.
    """

    points2d = np.asarray(points2d, dtype=np.float64)
    neighbour_lists: List[list] = [[] for _ in range(len(points2d))]

    for a, b in delaunay_edges(points2d, max_length_factor=max_length_factor):
        a, b = int(a), int(b)
        neighbour_lists[a].append(b)
        neighbour_lists[b].append(a)

    return [np.array(s, dtype=np.intp) for s in neighbour_lists]


def beta_skeleton_edges(points2d: ArrayLike, beta: float = 1.0) -> np.ndarray:
    """
    β-skeleton restricted to Delaunay edges: a Delaunay subgraph for beta <= 1
        - beta = 1.0 is the Gabriel graph
        - beta slightly <1 re-admits edges that shear pushes past 90° (noop on well-ordered lattices)

    Reject a Delaunay edge iff an opposite vertex subtends an angle > theta c where:
        theta c = pi - arcsin(beta)        # 90° at beta=1, ~116° at beta=0.9

    Note: Testing only the (<=2) Delaunay-opposite vertices is exact for beta <= 1, since
    the forbidden region is a subset of the diametral disk
    """

    p = np.asarray(points2d, dtype=np.float64)
    n = len(p)
    if n < 3:
        i, j = np.triu_indices(n, k=1)
        return np.stack([i, j], axis=1).astype(int)

    cos_tc = np.cos(np.pi - np.arcsin(beta))  # beta <= 1

    s = Delaunay(p).simplices

    i = s[:, [0, 1, 2]].ravel()
    j = s[:, [1, 2, 0]].ravel()
    k = s[:, [2, 0, 1]].ravel()

    a, b = p[i] - p[k], p[j] - p[k]
    cos_ang = np.einsum('ab,ab->a', a, b) / (
            np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-300)

    reject = cos_ang < cos_tc  # angle at k is > theta_c

    lo, hi = np.minimum(i, j), np.maximum(i, j)
    key = lo.astype(np.int64) * n + hi
    order = np.argsort(key, kind='stable')

    uniq, start = np.unique(key[order], return_index=True)

    bad = np.maximum.reduceat(reject[order].astype(np.int8), start).astype(bool)
    good = uniq[~bad]

    return np.stack([good // n, good % n], axis=1).astype(int)


def beta_skeleton_neighbours(points2d: ArrayLike, beta: float = 1.0) -> List[np.ndarray]:
    """
    First-ring neighbour lists.
    """

    points2d = np.asarray(points2d, dtype=np.float64)

    nb: List[list] = [[] for _ in range(len(points2d))]

    for a, b in beta_skeleton_edges(points2d, beta):
        a, b = int(a), int(b)
        nb[a].append(b)
        nb[b].append(a)

    return [np.array(s, dtype=np.intp) for s in nb]


# kNN queries

def knn(
        tree: Optional[cKDTree] = None,
        query_points: Optional[ArrayLike] = None,
        k: int = 6,
        drop_self: bool = True,
        self_indices: Optional[ArrayLike] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    k-nearest-neighbour query against a prebuilt tree. Or not. Whatever.

    Args:
        tree: Ideally, a prebuilt cKDTree (owned/cached by the caller).
            If none passed, it warns and builds one over the query points.
        query_points: Points to query (Q, D). If None, the tree's own data is used (i.e. mean spacing of this cloud).
        k: number of neighbours requested. Clamped to the tree size.
        drop_self: if True, query k+1 and drop the first (self) column. Use for
            self-queries (query points are tree members). Set False when querying
            external points, where the nearest match is not the point itself.

    Returns:
        (distances, indices), both (Q, k_eff) with k_eff = min(k, n - 1) when drop_self, or min(k, n) otherwise.
        When k_eff == 0, returns (Q, 0) arrays.
    """

    if tree is None and query_points is None:
        raise ValueError('You must pass either a tree or query_points.')

    owns_data = False

    if query_points is None:
        query_points = tree.data
        owns_data = True

    query_points = np.atleast_2d(np.asarray(query_points, dtype=np.float64))

    if tree is None:
        tree = cKDTree(query_points)
        owns_data = True

    max_k = (tree.n - 1) if drop_self else tree.n
    k_clipped = max(0, min(int(k), max_k))

    if k_clipped <= 0:
        q = query_points.shape[0]
        return np.empty((q, 0), dtype=float), np.empty((q, 0), dtype=np.intp)

    kq = k_clipped + 1 if drop_self else k_clipped
    dist, idx = tree.query(query_points, k=kq)

    if idx.ndim == 1:
        idx = idx.reshape(-1, 1)
        dist = dist.reshape(-1, 1)

    if drop_self:
        if self_indices is None and owns_data:
            self_indices = np.arange(query_points.shape[0], dtype=np.intp)

        if self_indices is None:  # External tree + explicit points
            idx, dist = idx[:, 1:], dist[:, 1:]

        else:
            self_indices = np.asarray(self_indices, dtype=np.intp)

            is_self = idx == self_indices[:, None]  # <=1 True per row
            drop = is_self.copy()
            drop[~is_self.any(axis=1), -1] = True  # no self found: drop farthest
            keep = ~drop
            idx = idx[keep].reshape(idx.shape[0], -1)
            dist = dist[keep].reshape(dist.shape[0], -1)

    return dist, idx.astype(np.intp)


# Two ways to measure local spacing:
#
#   ball_spacing: metric (kNN ball), robust to topology/defects
#   graph_spacing: topological (explicit edges), reflects actual bond lengths

def ball_spacing(
        tree: Optional[cKDTree] = None,
        query_points: Optional[np.ndarray] = None,
        k: int = 6,
        reduce: 'Callable' = np.mean,
) -> np.ndarray:
    """
    Local spacing as the reduce (mean by default) of distances to the k nearest
    neighbours (metric kNN ball, robust to mesh topology/defects)

    'reduce' must accept an 'axis' argument (np.mean, np.median, np.min, np.max...)
    NaN for points with no available neighbours.
    """
    dist, _ = knn(tree, query_points, k, drop_self=True)
    if dist.shape[1] == 0:
        return np.full(dist.shape[0], np.nan)
    return reduce(dist, axis=1)


def graph_spacing(
        points2d: ArrayLike,
        neighbours: 'NeighbourGraph',
        reduce: 'Callable' = np.mean
    ) -> np.ndarray:
    """
    Local point spacing over a neighbour graph with choice of reduce function (mean by default).

    'reduce' must accept an 'axis' argument (np.mean, np.median, np.min, np.max...)
    NaN for isolated points.
    """
    points2d = np.asarray(points2d, dtype=float)
    nbr = ragged_neighbours(neighbours)
    out = np.full(len(points2d), np.nan)
    for i, nb in enumerate(nbr):
        if nb.size:
            out[i] = reduce(np.linalg.norm(points2d[nb] - points2d[i], axis=1), axis=-1)
    return out


def k_lookat(
        positions: ArrayLike,
        directions: ArrayLike,
        targets: ArrayLike,
        k: int,
        exclude_mask: Optional[ArrayLike] = None
    ) -> np.ndarray:
    """
    For each target, the indices of the k directed points that best face it.

    Each of the M sources has a position and a unit direction. A source is scored
    against a target by the dot product of its direction with the unit vector from
    the source to the target: +1 means it points straight at the target, -1 away.

    Args:
        - positions: source positions, (M, 3)
        - directions: source unit directions, (M, 3)
        - targets: target positions, (Q, 3)
        - k: number of sources to return per target
        - exclude_mask: optional bool array, True entries are never selected, (M,)

    Returns:
        (Q, k_eff) int array of source indices, best-first per row,
        with k_eff = min(k, number of selectable sources).
    """

    positions = np.asarray(positions, dtype=np.float32)
    directions = np.asarray(directions, dtype=np.float32)
    targets = np.asarray(targets, dtype=np.float32)

    desired = targets[:, None, :] - positions[None, :, :]
    desired = norm_l2(desired, axis=-1)
    dots = np.einsum('jk,ijk->ij', directions, desired)

    if exclude_mask is not None:
        exclude_mask = np.asarray(exclude_mask, dtype=bool)
        dots[:, exclude_mask] = -np.inf

    k_clipped = max(0, min(k, dots.shape[1]))
    part = np.argpartition(dots, -k_clipped, axis=1)[:, -k_clipped:]
    top = np.take_along_axis(dots, part, axis=1)
    order = np.argsort(top, axis=1)[:, ::-1]
    return np.take_along_axis(part, order, axis=1)


# Neighbourhood-based smoothing

def _masked_neighbours(neighbours: NeighbourGraph) -> Tuple[np.ndarray, np.ndarray]:
    """(valid, safe) for a neighbour graph.

    Ragged input is densified.
    'valid' is the boolean keep-mask, 'safe' is the dense indices with padding
    clamped to 0 so it can be used for fancy-indexing without going out of bounds.
    """
    nb = padded_neighbours(neighbours) if not _is_dense(neighbours) else np.asarray(neighbours, dtype=np.intp)
    valid = nb >= 0
    safe = np.where(valid, nb, 0)
    return valid, safe


def smooth_scalars(
        values: ArrayLike,
        neighbours: NeighbourGraph,
        n_iter: int = 2,
        mask: Optional[ArrayLike] = None,
        method: str = 'mean',
) -> np.ndarray:
    """
    Smooth a 1D scalar field over a precomputed neighbour graph.

    Args:
        values: values to smooth, (N,)
        neighbours: NeighbourGraph (ragged or dense, dense entries < 0 are padding)
        n_iter: smoothing passes
        mask: optional bool array, only True entries are updated, (N,)
        method: 'mean' (half-self half-neighbours) or 'median' (self + neighbours)
    """

    method = 'mean' if not method else str(method).lower()

    out = np.copy(values).astype(np.float64)
    if n_iter <= 0:
        return out.astype(values.dtype)

    update_mask = np.asarray(mask, dtype=bool) if mask is not None else np.ones(out.shape[0], dtype=bool)
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
        values: ArrayLike,
        neighbours: NeighbourGraph,
        n_iter: int = 3,
        weights: Optional[ArrayLike] = None,
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
        values: ArrayLike,
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


# Lattice-row tracing and graph measures


def first_ring_gap(points2d: ArrayLike, neighbours: NeighbourGraph, degrees: bool = False) -> np.ndarray:
    """
    Largest empty angular sector between consecutive first-ring neighbour bearings

    Complete ring (incl 5- or 7-fold disclinations) should have ~ 2*pi/degree
    Any lens missing a sector should be >= ~2*pi/3
    """

    points2d = np.asarray(points2d, dtype=np.float64)
    nbr = ragged_neighbours(neighbours)
    gap = np.full(len(points2d), 2.0 * np.pi)

    for i, nb in enumerate(nbr):
        if nb.size < 2:
            continue
        d = points2d[nb] - points2d[i]
        ang = np.sort(np.arctan2(d[:, 1], d[:, 0]))
        diffs = np.diff(np.concatenate([ang, ang[:1] + 2.0 * np.pi]))
        gap[i] = float(diffs.max())

    return np.rad2deg(gap) if degrees else gap


def _walk(
        points2d: np.ndarray,
        neighbours: Sequence[np.ndarray],
        start: int,
        heading: np.ndarray,
        max_steps: int,
        cos_step: float,
        cos_global: float,
    ) -> np.ndarray:
    """
    Greedy single-direction walk: step to the neighbour best aligned with the heading.
    'neighbours' is the list-of-arrays form (ragged)
    """
    path, cur = [], start
    h0 = heading / (np.linalg.norm(heading) + 1e-12)
    h, visited = h0, {start}

    for _ in range(max_steps):
        nb = neighbours[cur]
        if len(nb) == 0:
            break

        vec = points2d[nb] - points2d[cur]
        u = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
        dots = u @ h
        j = int(np.argmax(dots))
        if dots[j] < cos_step or (u[j] @ h0) < cos_global:
            break
        nxt = int(nb[j])
        if nxt in visited:
            break
        path.append(nxt)
        visited.add(nxt)
        h, cur = u[j], nxt
    return np.array(path, dtype=int)


def walk_rows(
        points2d: ArrayLike,
        neighbours: NeighbourGraph,
        seed: int,
        bearings: ArrayLike,
        max_steps: int = 500,
        step_tol: float = 30.0,
        global_tol: float = 70.0,
    ) -> List[np.ndarray]:
    """
    Trace lattice rows out of 'seed' along each heading in 'bearings' (rad).
    Each row is walked both ways and returned as an ordered (k, 2) array of the points.

    Args:
        - step_tol: float, caps the per-step turn (degrees)
        - global_tol: float, cumulative drift from the initial heading (degrees)
    """

    fams: List[np.ndarray] = []
    cos_step = np.cos(np.deg2rad(step_tol))
    cos_global = np.cos(np.deg2rad(global_tol))

    points2d = np.asarray(points2d, dtype=np.float64)
    nbr = ragged_neighbours(neighbours)

    for brg in bearings:
        h0 = np.array([np.cos(brg), np.sin(brg)])
        fwd = _walk(points2d, nbr, seed, h0, max_steps, cos_step, cos_global)
        bwd = _walk(points2d, nbr, seed, -h0, max_steps, cos_step, cos_global)
        fams.append(np.vstack([points2d[bwd[::-1]], points2d[[seed]], points2d[fwd]]))

    return fams
