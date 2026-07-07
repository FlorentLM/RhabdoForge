from typing import List, Optional, Tuple, Sequence, Callable, Union
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree, Delaunay
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from insectvision.utils import norm_l2

# A neighbour graph, can be:
#   ragged: a sequence (list / object-array) of per-point neighbour-index arrays
#   dense: an (N, k) integer array with entries < 0 used as padding
NeighbourGraph = np.ndarray | Sequence[ArrayLike]


def ragged_neighbours(neighbours: NeighbourGraph) -> List[np.ndarray]:
    """
    Normalise a NeighbourGraph to a list of per-point neighbour-index arrays.
    Padding and any negative index are stripped.
    """
    if isinstance(neighbours, np.ndarray) and neighbours.ndim == 2:
        dense = np.asarray(neighbours, dtype=np.intp)
        return [row[row >= 0] for row in dense]

    out: List[np.ndarray] = []
    for nb in neighbours:
        nb = np.asarray(nb, dtype=np.intp)
        out.append(nb[nb >= 0])
    return out


def padded_neighbours(neighbours: NeighbourGraph, pad_value: int = -1, masked: bool = True) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Densify a NeighbourGraph to a (N, kmax) int array padded with 'pad_value'.

    Ragged input is padded to the widest row, an already-dense array is returned
    unchanged (cast to intp).

    If 'masked' is True, this returns the dense indices with padding clamped to 0
    (so it can be used for fancy-indexing without going out of bounds) and the validity boolean mask.
    """
    if isinstance(neighbours, np.ndarray) and neighbours.ndim == 2:
        neighb = np.asarray(neighbours, dtype=np.intp)
    else:
        n = len(neighbours)
        kmax = max((len(nb) for nb in neighbours), default=0)
        neighb = np.full((n, kmax), pad_value, dtype=np.intp)
        for i, nb in enumerate(neighbours):
            nb = np.asarray(nb, dtype=np.intp)
            neighb[i, :nb.size] = nb

    if masked:
        valid_mask = neighb >= 0
        safe_neighb = np.where(valid_mask, neighb, 0)
        return safe_neighb, valid_mask

    return neighb


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


# Two ways to measure local spacing:
#
#   ball_spacing: metric (kNN ball), robust to topology/defects
#   graph_spacing: topological (explicit edges), reflects actual bond lengths

def ball_spacing(
        tree: Optional[cKDTree] = None,
        query_points: Optional[np.ndarray] = None,
        k: int = 6,
        reduce: Callable = np.mean,
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
        reduce: Callable = np.mean
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


def merge_close_points(points: ArrayLike, radius: float, reduce: Optional[Callable] = np.mean) -> np.ndarray:
    """
    Merge points (2D or 3D) within 'radius' (each group becomes one output point).

    'reduce' must accept an 'axis' argument (np.mean, np.median, np.min, np.max...),
    or if None, the lowest-index point of each component is preserved.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n < 2:
        return points

    pairs = cKDTree(points).query_pairs(r=float(radius), output_type='ndarray')
    if len(pairs) == 0:
        return points

    graph = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    n_comp, labels = connected_components(graph, directed=False)

    # No reduction (keep first point of each group)
    if reduce is None:
        first = np.full(n_comp, n, dtype=np.intp)
        np.minimum.at(first, labels, np.arange(n))
        return points[first]

    # Fast path for mean
    if reduce is np.mean:
        counts = np.bincount(labels, minlength=n_comp)
        sums = np.stack([np.bincount(labels, weights=points[:, d], minlength=n_comp)
                         for d in range(points.shape[1])], axis=1)
        return sums / counts[:, None]

    # Generic reduction (np.median, np.max, etc.)
    idx = np.argsort(labels)
    sorted_labels = labels[idx]
    sorted_points = points[idx]

    diff = np.where(np.diff(sorted_labels))[0] + 1
    groups = np.split(sorted_points, diff)

    return np.array([reduce(g, axis=0) for g in groups])