"""
User-facing views of Model data buffers.

BaseView      : Abstract base defining the ViewField descriptor properties
SpatialQueries: Mixin, shared spatial-queries (trees, graphs, knn)
OmmatidiumView: Indexes the Ommatidia axis
CartridgeView : Like OmmatidiumView, but rhabdomeres follow neural superposition
RhabdomereView: Indexes the Rhabdomeres axis
EyeView       : An OmmatidiumView over one eye
"""
import logging
from typing import TYPE_CHECKING, Optional, Tuple, Union, Generator, Dict
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree, Delaunay

from insectvision.utils import norm_l2

from insectvision.compound_eyes.buffers import _BIT_LAYOUT
from insectvision.geometry.ghosting import ghosts_from_growth_3d
from insectvision.geometry.linalg import tangent_frames, local_to_world
from insectvision.geometry.neighbours import knn, k_lookat, beta_skeleton_edges, identify_boundary_points, first_ring_gap
from insectvision.geometry.spherical import (
    cartesian_to_spherical, spherical_gradients, angle_to_chord, chord_to_angle, sphere_to_stereo, radius_of_curvature
)

if TYPE_CHECKING:
    from insectvision.compound_eyes.model import Model
    from insectvision.compound_eyes.buffers import Buffer
    from insectvision.compound_eyes import RhabdomereBundle

logger = logging.getLogger(__name__)


class ViewField:
    """
    Descriptor that routes property access to the underlying Buffer slice.
    """

    def __init__(self, field_name: str, level: str, doc: Optional[str] = None):
        self.field_name = field_name
        self.level = level      # 'ommatidia' or 'rhabdomere'
        self.is_metadata = field_name in _BIT_LAYOUT    # TODO: can be removed when the per-ommatidium metadata is moved
        self.is_flag = self.is_metadata and _BIT_LAYOUT[field_name][1] == 1  # single-bit metadata = booleans:
        self.__doc__ = doc

    def _index(self, obj):
        if self.level == 'rhabdomere':
            return obj.rhab_indices
        if self.is_metadata:  # per-ommatidia but stored per-rhab
            return obj.omm_indices * obj.R  # slot 0 of each ommatidium
        return obj.omm_indices

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        out = obj._buffer[self.field_name, self._index(obj)]
        return out.astype(bool) if self.is_flag else out

    def __set__(self, obj, value):
        obj._buffer[(self.field_name, self._index(obj))] = value


class BaseView:
    """
    Abstract base class for all views. Provides access to GPU-backed data.
    """
    __slots__ = ()

    @property
    def model(self) -> 'Model':
        raise NotImplementedError

    @property
    def _buffer(self) -> 'Buffer':
        return self.model._buf

    @property
    def bundle(self) -> 'RhabdomereBundle':
        """The rhabdomere bundle model."""
        return self.model._bundle

    @property
    def omm_indices(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def rhab_indices(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def N(self) -> int:
        idx = self.omm_indices
        return int(idx.shape[0]) if idx.ndim else 1

    @property
    def R(self) -> int:
        return self.model._R

    @property
    def shape(self) -> Tuple[int, int]:
        return self.N, self.R

    @property
    def size(self) -> int:
        return self.N * self.R

    def __len__(self) -> int:
        return self.N

    # Geometry

    @property
    def positions(self) -> np.ndarray:
        return self._buffer['position', self.omm_indices]

    @positions.setter
    def positions(self, value: ArrayLike):
        self._buffer['position', self.omm_indices] = np.asarray(value, dtype=np.float32).reshape(-1, 3)
        self.model._bump_spatial_ver()

    @property
    def directions(self) -> np.ndarray:
        return self._buffer['forward', self.omm_indices]

    @directions.setter
    def directions(self, value: ArrayLike):
        fwd = norm_l2(np.asarray(value).reshape(-1, 3)).astype(np.float32)
        self._buffer['forward', self.omm_indices] = fwd
        self._buffer['right', self.omm_indices], self._buffer['up', self.omm_indices] = tangent_frames(fwd)

        self.model._bump_spatial_ver()

    forward = directions

    right = ViewField('right', 'ommatidia')
    up = ViewField('up', 'ommatidia')

    @property
    def frame(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Subview-level (e.g. eye-level) reference frame (forward, right, up), world-anchored.
        forward = mean viewing direction, right/up via tangent frames

        (i.e. the same OpenGL convention (+X right, +Y up, -Z fwd) as the per-ommatidium frames).
        """
        if self.N == 0:
            raise ValueError('frame undefined for an empty view')
        _, f, r, u = sphere_to_stereo(self.directions)
        return f, r, u

    @property
    def curvature_radius(self) -> np.ndarray:
        """
        The local radius of curvature (μm or world units) for each ommatidium in this view.
        Returns an array of shape (N,).
        """
        if self.N == 0:
            return np.array([], dtype=np.float32)

        tree = self._get_tree('positions')
        if tree is None:
            return np.full(self.N, np.nan, dtype=np.float32)

        return radius_of_curvature(
            query_positions=self.positions,
            query_normals=self.directions,
            tree=tree,
            tree_normals=self.directions,
        ).astype(np.float32)

    # Transformation methods

    def translate(self, offset: ArrayLike) -> 'Model':
        """
        Translate all ommatidia (and rhabdomeres) positions by 'offset'.
        """
        self.positions = self.positions + np.asarray(offset, dtype=np.float32).reshape(3)
        return self

    def scale(self, factor: float) -> 'Model':
        """
        Scale all positions about the origin by 'factor'.
        Note: only world-scale geometry scales. Ommatidum and rhabdomere
        diameters (μm) and the focal length (μm) are at ommatidium
        scale and are unchanged.
        """
        self.positions = self.positions * factor
        return self

    def rotate(self, R_mat: ArrayLike) -> 'Model':
        """
        Rotate all positions, directions, and tangent frames by the 3x3
        rotation matrix 'R_mat'.
        """
        R_mat = np.asarray(R_mat, dtype=np.float64)

        if R_mat.shape != (3, 3):
            raise ValueError(f'R_mat must be 3x3, got {R_mat.shape}')

        if not np.allclose(R_mat @ R_mat.T, np.eye(3), atol=1e-3):
            logger.warning('rotate() called with non-orthonormal matrix, results may be off...')

        self.positions = self.positions @ R_mat.T
        self.directions = self.directions @ R_mat.T

        return self

    @property
    def azimuth(self):
        az, _ = cartesian_to_spherical(self.directions)
        return az

    @property
    def elevation(self):
        _, el = cartesian_to_spherical(self.directions)
        return el

    @property
    def rhabdomere_azimuth(self) -> np.ndarray:
        """The azimuth of every rhabdomere in this view, shaped (N, R)."""
        az, _ = cartesian_to_spherical(self.rhabdomeres.directions)
        return az.reshape(self.N, self.R)

    @property
    def rhabdomere_elevation(self) -> np.ndarray:
        """The elevation of every rhabdomere in this view, shaped (N, R)."""
        _, el = cartesian_to_spherical(self.rhabdomeres.directions)
        return el.reshape(self.N, self.R)

    @property
    def rhabdomere_positions(self) -> np.ndarray:
        """
        World-space positions of all rhabdomere tips in this view (N, R, 3).
        """
        return self.rhabdomeres.positions.reshape(self.N, self.R, 3)

    @property
    def rhabdomere_directions(self) -> np.ndarray:
        """
        World-space viewing directions of all rhabdomeres in this view (N, R, 3).
        """
        return self.rhabdomeres.directions.reshape(self.N, self.R, 3)

    # Lattice properties

    interommatidial_angles = ViewField('ioa_angles', 'ommatidia')
    interommatidial_tilt = ViewField('ioa_tilt', 'ommatidia')
    acceptance_angles_rest = ViewField('rest_acc_angles', 'rhabdomere')
    acceptance_angles_current = ViewField('curr_acc_angles', 'rhabdomere')

    @property
    def hexatic_order(self) -> np.ndarray:
        """Per-ommatidium hexatic order |Ψ6| of the lens lattice (1 = perfect hex, 0 = isotropic)."""
        return self.model._hexatic_order[self.omm_indices]

    @property
    def trust(self) -> np.ndarray:
        """Per-ommatidium metric confidence in [0, 1] (see Model._compute_trust_field)."""
        return self.model._trust[self.omm_indices]

    # Per-ommatidium neighbourhood-related properties

    eye_index = ViewField('eye_id', 'ommatidia')

    is_edge = ViewField('is_edge', 'ommatidia',
        doc="Whether these ommatidia are at the boundary of the eye.")

    is_binocular = ViewField('is_binocular', 'ommatidia',
        doc="Whether these ommatidia are part of the binocular area.")

    @property
    def is_interior(self) -> np.ndarray:
        """Whether these ommatidia are at *not* at the boundary of the eye."""
        return ~self.is_edge

    @property
    def is_monocular(self) -> np.ndarray:
        """Whether these ommatidia are *not* part of the binocular area."""
        return ~self.is_binocular

    @property
    def binocular_fraction(self) -> float:
        mask = self.is_binocular
        return float(np.mean(mask)) if mask.size > 0 else 0.0

    # Per-ommatidium specifics (optics)

    focal_length = ViewField('focal_um', 'ommatidia',
        doc="Per-ommatidium focal legnth (μm)")

    aperture = ViewField('aperture_um', 'ommatidia',
        doc="Per-ommatidium aperture (lens diameter) (μm)")

    # Per-ommatidium rhabdomeres bundles properties

    chi = ViewField('chi', 'ommatidia',
        doc="Per-ommatidium bundle orientation (chi)")
    bundle_orientation = chi

    @property
    def chirality(self) -> np.ndarray:
        """Returns +1 or -1 for each ommatidium."""
        return self._omm_chirality(self.omm_indices)

    def _focal_axis_local(self, theta) -> np.ndarray:
        """
        A focal-plane unit axis at angle 'theta' (rad, in the bundle's own frame),
        expressed in each ommatidium's (right, up) tangent coords, shape (N, 2).
        """
        chi = self._buffer['chi', self.omm_indices]
        vx = np.cos(theta) * self.chirality  # mirror x by chirality
        vy = np.broadcast_to(np.float32(np.sin(theta)), chi.shape)
        c, s = np.cos(chi), np.sin(chi)  # rotate by chi
        lx = c * vx - s * vy
        ly = s * vx + c * vy
        return np.stack([lx, ly], axis=-1).astype(np.float32)

    # Local (tangent-plane, 2D) axes

    @property
    def orientation_field_local(self) -> np.ndarray:
        """Bundle local X-axis (chi) in (right, up) tangent coords. Shape (N, 2)."""
        return self._focal_axis_local(self.bundle.flow_axis_rad)

    @property
    def main_axis_field_local(self) -> np.ndarray:
        """Bundle main axis (e.g. R3-R6) in (right, up) tangent coords. Shape (N, 2)."""
        return self._focal_axis_local(self.bundle.main_axis_rad)

    @property
    def saccade_field_local(self) -> np.ndarray:
        """Microsaccade actuation axis in (right, up) tangent coords. Shape (N, 2)."""
        return np.asarray(self._buffer['saccade_dxdy', self.omm_indices], dtype=np.float32)

    # World-space axes

    @property
    def orientation_field(self) -> np.ndarray:
        """Bundle local X-axis (chi) projected into world space. Shape (N, 3)."""
        return local_to_world(self.orientation_field_local, self.right, self.up)

    @property
    def main_axis_field(self) -> np.ndarray:
        """Bundle main axis (e.g. R3-R6) projected into world space. Shape (N, 3)."""
        return local_to_world(self.main_axis_field_local, self.right, self.up)

    @property
    def saccade_field(self) -> np.ndarray:
        """Microsaccade actuation axis in world coordinates. Shape (N, 3)."""
        return local_to_world(self.saccade_field_local, self.right, self.up)

    # retina_field = ViewField('retina_dxdy', 'ommatidia')      # TODO: disabled for now

    tau_rise = ViewField('tau_rise', 'ommatidia',
        doc="Rhabdomeres mechanical contraction rise time (same for all rhabdomeres in a given ommatidium)")

    tau_relax = ViewField('tau_relax', 'ommatidia',
        doc="Rhabdomeres mechanical relaxation time (same for all rhabdomeres in a given ommatidium)")

    tau_adapt_fast = ViewField('tau_adapt_fast', 'ommatidia',
        doc="Rhabdomeres fast adaptation EMA  (same for all rhabdomeres in a given ommatidium)")

    tau_adapt_slow = ViewField('tau_adapt_slow', 'ommatidia',
        doc="Rhabdomeres slow adaptation EMA  (same for all rhabdomeres in a given ommatidium)")

    lateral_amplitude = ViewField('ampl_lateral', 'ommatidia')
    axial_amplitude = ViewField('ampl_axial', 'ommatidia')

    # Rhabdomeres specifics properties

    is_wired = ViewField('is_wired', 'rhabdomere',
         doc="Whether this rhabdomere is correctly wired to a neighbouring ommatidium.")

    sensitivity = ViewField('sensitivity', 'rhabdomere',
        doc="Rhabdomeres R (or UV), G, B channel sensitivities.")

    diameter = ViewField('diameter_um', 'rhabdomere',
        doc="Rhabdomeres diameters (μm).")

    rest_offsets = ViewField('rest_offset', 'rhabdomere',
        doc="Rhabdomeres positional offsets from the ommatidium optical axis (at rest).")

    wavelength = ViewField('wavelength_um', 'rhabdomere',
        doc="Rhabdomeres peak wavelengths (μm).")

    tau_membrane = ViewField('tau_membrane', 'rhabdomere',
        doc="Rhabdomeres membrane RC.")

    # Neural superposition wiring properties

    @property
    def cartridges(self) -> 'CartridgeView':
        return CartridgeView(self.model, self.omm_indices)

    @property
    def neural_superposition(self) -> bool:
        """Whether this model is superposition eyes."""
        return self.model._superposition_wired

    @property
    def cartridge_indices(self) -> np.ndarray:
        """(N, R) global donor rhabdomere index per slot (= the gather table)."""
        return self._buffer['cartridge_src', self.rhab_indices].reshape(self.shape).astype(np.intp)

    @property
    def cartridge_map(self) -> np.ndarray:
        """(N, R) donor ommatidium per slot (-1 where unwired)."""
        if not self.neural_superposition:
            return np.full((self.N, self.R), -1, dtype=np.intp)

        src = self._buffer['cartridge_src', self.rhab_indices].reshape(self.shape)
        is_wired = self._buffer['is_wired', self.rhab_indices].reshape(self.shape).astype(bool)
        return np.where(is_wired, (src // self.R).astype(np.intp), -1)

    @property
    def has_conflicts(self) -> np.ndarray:
        return self.model._get_conflicts.any[self.omm_indices]

    @property
    def donation_conflicts(self) -> np.ndarray:
        return self.model._get_conflicts.donation[self.omm_indices]

    @property
    def receiving_conflicts(self) -> np.ndarray:
        return self.model._get_conflicts.receiving[self.omm_indices]

    @property
    def unwired_slots(self) -> np.ndarray:
        return self.model._get_conflicts.unwired_slots[self.omm_indices]

    @property
    def unwired_count(self) -> int:
        return self.unwired_slots.sum(-1)


class NeighbourResult:
    """
    Result of a neighbours() query.

    Attributes:
        mask: (Q,) bool, which input queries were inside this view and got results.
        indices: (M, k) int, global ommatidia indices of the k neighbours, M = mask.sum().
        distances: (M, k) float, distances to these neighbours.
        immediate: (M, k) bool or None, True where the neighbour is in the first lattice ring of the query ommatidium.
        same_chirality: (M, k) bool or None, True where neighbour chirality matches query.
    """

    def __init__(self, view, mask, indices, distances, is_immediate=None, same_chirality=None):
        self.view = view
        self.mask = mask
        self.indices = indices
        self.distances = distances
        self.immediate = is_immediate
        self.same_chirality = same_chirality

    def __bool__(self) -> bool:
        return bool(self.indices.size)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    @property
    def ommatidia(self) -> 'OmmatidiumView':
        return OmmatidiumView(self.view.model, self.indices.ravel())

    @property
    def rhabdomeres(self) -> 'RhabdomereView':
        return self.ommatidia.rhabdomeres


class SpatialQueries:
    """
    Spatial-queries for any ommatidium-anchored view.

    KD-trees and neighbour graphs are built lazily and cached on the view,
    keyed by model._spatial_version, so any geometry changes (which bumps that counter) invalidates them.

    Subclasses must provide a _spatial (dict) slot.
    """
    __slots__ = ()

    def _to_local(self, query: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        """
        Map global ommatidium indices to local rows.
        """

        query = np.asarray(query, dtype=np.intp).reshape(-1)
        omm = self.omm_indices
        if omm.size == 0:
            return np.empty(0, dtype=np.intp), np.zeros(query.shape[0], dtype=bool)

        order = np.argsort(omm)

        cand = order[np.clip(np.searchsorted(omm[order], query), 0, omm.size - 1)]
        in_view = omm[cand] == query

        return cand[in_view], in_view

    @property
    def _spatial_store(self) -> dict:
        """Per-view cache dict, cleared when the model's geometry changes."""
        if self._spatial.get('_ver') != self.model._spatial_version:
            self._spatial.clear()
            self._spatial['_ver'] = self.model._spatial_version
        return self._spatial

    # Lazy generation of trees

    @property
    def _local_i_conflictfree(self) -> np.ndarray:
        """
        Local rows of ommatidia that are conflict-free and have no unwired slots.
        """

        if 'conflictfree_local_i' not in self._spatial_store:
            self._spatial_store['conflictfree_local_i'] = np.flatnonzero(~self.has_conflicts & ~self.unwired_slots.any(axis=-1))

        return self._spatial_store['conflictfree_local_i']

    def _get_tree(self, space: str, conflict_free: bool = False) -> Optional[cKDTree]:
        """
        Cached KDtree over this view's points in the given 'space'.
        (space is 'positions' or 'directions')
        """
        key = f'tree:{space}:{int(conflict_free)}'

        if key not in self._spatial_store:
            pts = self.positions if space == 'positions' else self.directions
            if conflict_free:
                pts = pts[self._local_i_conflictfree]

            self._spatial_store[key] = cKDTree(pts) if pts.shape[0] else None

        return self._spatial_store[key]

    # Lazy generation of graphs

    def _get_neighbour_graph(self, k: int) -> dict:

        graph = self._spatial_store.get('neighbours_graph')

        if graph is None or graph.get('k') != k:
            dist, idx = knn(self._get_tree('positions'), None, k)

            graph = {
                'neighbour_indices': idx,
                'neighbour_distances': dist,
                'k': k
            }
            self._spatial_store['neighbours_graph'] = graph

        return graph

    def _get_first_ring_graph(self) -> dict:
        """
        First-ring β-skeleton graph for this view's ommatidia.
        Built once on a stereographic projection of the optical axes, and cached.

        Returns:
            Dict containing:
            - adjacency (list of local-index arrays)
            - max_gap
            - is_boundary
            - degree (n,)
            - pair_keys (set of undirected global-index edge keys)
            - big (key multiplier)
            - points2d (n, 2)
        """

        g = self._spatial_store.get('first_ring_graph')
        if g is not None:
            return g

        n = len(self)
        omm = self.omm_indices
        big = int(self.model.shape[0]) + 1

        if n < 3:
            adj = [np.delete(np.arange(n, dtype=np.intp), i) for i in range(n)]
            pts2d = np.zeros((n, 2))
            # Base boundary mask for tiny clusters
            is_boundary = np.ones(n, dtype=bool)
            max_gap = np.full(n, 2.0 * np.pi)
            edges = np.zeros((0, 2), dtype=int)
            delaunay_simplices = None
        else:
            pts2d, *_ = sphere_to_stereo(self.directions)

            delaunay_simplices = Delaunay(pts2d).simplices

            # Topology
            edges = beta_skeleton_edges(pts2d, beta=self.model.lattice_beta, simplices=delaunay_simplices)

            # Convert edge list to adjacency list
            adj = [[] for _ in range(n)]
            for a, b in edges.tolist():
                adj[a].append(b)
                adj[b].append(a)
            adj = [np.array(nb, dtype=np.intp) for nb in adj]

            # Boundary detection
            is_boundary = identify_boundary_points(pts2d, adj)

            # Still keep max_gap for Pass 1 logic if needed
            max_gap = first_ring_gap(pts2d, adj)

        if edges.size > 0:
            gi = omm[edges[:, 0]].astype(np.int64)
            gj = omm[edges[:, 1]].astype(np.int64)
            lo = np.minimum(gi, gj)
            hi = np.maximum(gi, gj)
            pair_keys = set((lo * big + hi).tolist())
        else:
            pair_keys = set()

        g = {
            # TODO: Several of these cached things are not used as much as they should be
            'adjacency': adj,
            'max_gap': max_gap,
            'simplices': delaunay_simplices,
            'is_boundary': is_boundary,  # cached for confidence & edge propagation
            'degree': np.fromiter((a.size for a in adj), dtype=np.int32, count=n),
            'pair_keys': pair_keys,
            'big': big,
            'points2d': pts2d,
        }
        self._spatial_store['first_ring_graph'] = g
        return g

    def _get_directional_graph(self, k: int = 8) -> dict:
        """
        For each ommatidium, its k nearest neighbours in direction space (chord
        distance on the unit sphere of optical axes), with each neighbour's
        direction projected into the ommatidium's tangent frame.
        """

        graph = self._spatial_store.get('directional_graph')

        if graph is not None and graph.get('k') == k:
            return graph

        n = len(self)
        if n <= 1:
            graph = {
                'proj_x': np.zeros((n, 0), dtype=np.float32),
                'proj_y': np.zeros((n, 0), dtype=np.float32),
                'angular_sep': np.zeros((n, 0), dtype=np.float32),
                'neighbour_local_indices': np.zeros((n, 0), dtype=np.intp),
                'local_x': np.zeros((n, 3), dtype=np.float32),
                'local_y': np.zeros((n, 3), dtype=np.float32),
                'k': 0,
            }

            self._spatial_store['directional_graph'] = graph
            return graph

        dirs = self.directions
        chord_dist, nb_local = knn(self._get_tree('directions'), None, k)

        local_x = self.right
        local_y = self.up

        delta = dirs[nb_local] - dirs[:, None, :]
        proj_x = np.sum(delta * local_x[:, None, :], axis=2)
        proj_y = np.sum(delta * local_y[:, None, :], axis=2)

        graph = {
            'proj_x': proj_x.astype(np.float32),
            'proj_y': proj_y.astype(np.float32),
            'angular_sep': chord_to_angle(chord_dist).astype(np.float32),
            'neighbour_local_indices': nb_local.astype(np.intp),
            'local_x': local_x.astype(np.float32),
            'local_y': local_y.astype(np.float32),
            'k': int(nb_local.shape[1]),
        }

        self._spatial_store['directional_graph'] = graph
        return graph

    # TODO: Should probably have public accessors for the graphs (and they should be SimpleNamespaces instead of dicts)?

    # Some internal helpers

    # TODO: 'immediate' logic should not live here
    @staticmethod
    def _tag_immediate_factor(result: 'NeighbourResult', dists: np.ndarray, factor: Optional[float]) -> None:
        """
        Tag result.immediate where distance <= factor * local scale (mean of nearest 3).
        """
        if factor is None or not dists.size:
            return

        n_ref = min(3, dists.shape[1])
        local_scale = np.mean(dists[:, :n_ref], axis=1, keepdims=True)
        result.immediate = dists <= (local_scale * float(factor))

    def _tag_immediate_beta(self, result: 'NeighbourResult', valid_global: np.ndarray) -> None:
        """
        Tag result.immediate by β-skeleton first-ring membership.
        """

        nb = result.indices

        if not nb.size:
            result.immediate = np.zeros(nb.shape, dtype=bool)
            return

        graph = self._get_first_ring_graph()
        keys, big = graph['pair_keys'], graph['big']
        qi = np.asarray(valid_global, dtype=np.int64)[:, None]

        lo = np.minimum(qi, nb).astype(np.int64)
        hi = np.maximum(qi, nb).astype(np.int64)
        flat = (lo * big + hi).ravel().tolist()

        result.immediate = np.fromiter((kk in keys for kk in flat), dtype=bool, count=len(flat)).reshape(nb.shape)

    def _omm_chirality(self, omm_global: np.ndarray) -> np.ndarray:
        """
        +1 / -1 chirality for the given (global) ommatidium indices (read from rhabdomere slot 0).
        """
        rhab0 = np.asarray(omm_global, dtype=np.intp) * self.R
        neg = np.asarray(self._buffer['chirality_neg', rhab0])

        return np.where(neg == 1, -1.0, 1.0).astype(np.float32)

    def _empty(self) -> 'OmmatidiumView':
        return OmmatidiumView(self.model, np.empty(0, dtype=np.intp))

    # Public queries

    def neighbours(self,
            query: Optional[ArrayLike] = None,
            positions: Optional[ArrayLike] = None,
            k: int = 6,
            immediate_only: bool = False,
            neighbour_dist_factor: float = 1.25,
        ) -> NeighbourResult:
        """
        k-nearest ommatidia neighbours within this view.

        Provide exactly one of:
            - query: global ommatidia indices (only those inside this view are used).
            - positions: (Q, 3) world-space query points.

        immediate_only restricts to the cached first-ring graph.
        neighbour_dist_factor tags 'is_immediate' for points queries (d <= factor * local scale).
            (set to None to skip tagging)
        """

        if (query is None) == (positions is None):
            raise ValueError("Provide either 'query' or 'positions' (but not both)")

        if query is not None:
            qidx_global = np.asarray(query, dtype=np.intp).reshape(-1)
            valid_local, in_this_view = self._to_local(qidx_global)
            valid_global = qidx_global[in_this_view]

            if immediate_only:
                graph = self._get_neighbour_graph(k)
                neighb_indices = graph['neighbour_indices'][valid_local]
                neighb_dists = graph['neighbour_distances'][valid_local]
            else:
                neighb_dists, neighb_indices = knn(self._get_tree('positions'), self.positions[valid_local], k, self_indices=valid_local)

            g_neighb_indices = self.omm_indices[neighb_indices]
            result = NeighbourResult(self, mask=in_this_view, indices=g_neighb_indices, distances=neighb_dists)

            if g_neighb_indices.size:
                q_chir = self._omm_chirality(valid_global)
                nb_chir = self._omm_chirality(g_neighb_indices.ravel()).reshape(g_neighb_indices.shape)
                result.same_chirality = nb_chir == q_chir[:, None]

            self._tag_immediate_beta(result, valid_global)
            return result

        # Points path
        points = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        neighb_dists, neighb_indices = knn(self._get_tree('positions'), points, k, drop_self=False)
        g_neighb_indices = self.omm_indices[neighb_indices]

        result = NeighbourResult(self,
                                 mask=np.ones(points.shape[0], dtype=bool),
                                 indices=g_neighb_indices, distances=neighb_dists)
        self._tag_immediate_factor(result, neighb_dists, neighbour_dist_factor)
        return result

    def _query_nearest(self,
            space: str,
            query: ArrayLike,
            k: int,
            avoid_conflicts: bool,
        ) -> Tuple['OmmatidiumView', np.ndarray]:
        """
        k nearest ommatidia (by space) to each external query point, with distances.
        """

        points = np.asarray(query, dtype=np.float32).reshape(-1, 3)

        tree = self._get_tree(space, avoid_conflicts)
        if tree is None:
            return self._empty(), np.empty((points.shape[0], 0), dtype=np.float32)

        distances, local = knn(tree, points, k, drop_self=False)
        if avoid_conflicts:
            local = self._local_i_conflictfree[local]

        return OmmatidiumView(self.model, self.omm_indices[local].ravel()), distances

    def query_directions(self,
            directions: ArrayLike,
            k: int = 1,
            avoid_conflicts: bool = False,
        ) -> Tuple['OmmatidiumView', np.ndarray]:
        """
        k ommatidia whose optical axes lie closest (chord distance) to each query direction.
        """
        return self._query_nearest('directions', directions, k, avoid_conflicts)

    def query_positions(self,
            positions: ArrayLike,
            k: int = 1,
            avoid_conflicts: bool = False,
        ) -> Tuple['OmmatidiumView', np.ndarray]:
        """
        k ommatidia closest in world-space position to each query point.
        """
        return self._query_nearest('positions', positions, k, avoid_conflicts)

    def query_lookat(self,
            target_positions: ArrayLike,
            k: int = 1,
            avoid_conflicts: bool = False,
        ) -> 'OmmatidiumView':
        """
        k ommatidia best looking at each world-space target.

        Unlike query_directions (optical-axis vectors only), this accounts for
        ommatidium *position*: the score is the dot product of each optical axis
        with the unit vector from that ommatidium to the target.
        """
        if k < 1:
            raise ValueError("k must be >= 1")

        query = np.asarray(target_positions, dtype=np.float32).reshape(-1, 3)
        if self.omm_indices.size == 0:
            return self._empty()

        conflict = self.has_conflicts if avoid_conflicts else None
        best = k_lookat(self.positions, self.directions, query, k, conflict)

        return OmmatidiumView(self.model, self.omm_indices[best].ravel())

    def _query_ball(self,
            space: str,
            center: ArrayLike,
            radius: float,
            avoid_conflicts: bool,
        ) -> 'OmmatidiumView':
        """
        All ommatidia within 'radius' of 'center' in the given space.
        """
        centre = np.asarray(center, dtype=np.float32)

        tree = self._get_tree(space, avoid_conflicts)
        if tree is None:
            return self._empty()

        hits = np.atleast_1d(np.asarray(tree.query_ball_point(centre, r=float(radius)), dtype=np.intp))

        if avoid_conflicts:
            hits = self._local_i_conflictfree[hits]

        return OmmatidiumView(self.model, self.omm_indices[hits])

    def query_cone(self,
            center_direction: ArrayLike,
            angle: float,
            degrees: bool = True,
            avoid_conflicts: bool = False,
        ) -> 'OmmatidiumView':
        """
        All ommatidia whose optical axis lies within 'angle' of 'center_direction'.
        """
        c = np.asarray(center_direction, dtype=np.float32)
        n = float(np.linalg.norm(c))
        if n < 1e-12:
            return self._empty()
        a = np.deg2rad(angle) if degrees else float(angle)
        return self._query_ball('directions', c / n, angle_to_chord(a), avoid_conflicts)

    def query_ball(self,
            center_position: ArrayLike,
            radius: float,
            avoid_conflicts: bool = False,
        ) -> 'OmmatidiumView':
        """
        All ommatidia whose world-space position is within 'radius' of 'center_position'.
        """
        return self._query_ball('positions', center_position, radius, avoid_conflicts)

    def _suggest_k_search(self, distance_rad: float, ring_margin: int = 1, k_min: int = 15) -> int:
        """
        How many direction-space neighbours must be scanned for 'directed_neighbours'
        to reach a stride of 'distance_rad'.

        Note: lenses within m hex-rings ~= 1 + 3 m (m + 1).
        Falls back to 'k_min'.
        """

        n = len(self)
        if distance_rad <= 0 or n <= 1:
            return min(k_min, n - 1)

        ioa = self.interommatidial_angles
        if ioa.size == 0:
            return min(k_min, n - 1)

        # Median minor IOA over the eye gives robust global estimate
        delta_phi = float(np.median(ioa[:, 0]))
        if not np.isfinite(delta_phi) or delta_phi <= 0:
            return min(k_min, n - 1)

        # Rings needed = distance / pitch
        m = int(np.ceil(distance_rad / delta_phi)) + int(ring_margin)
        # Hexagonal packing formula: neighbours in m rings = 1 + 3m(m+1)
        k_needed = 1 + 3 * m * (m + 1)
        return int(np.clip(k_needed, k_min, n - 1))
    
    def directed_neighbours(self,
            direction: ArrayLike,
            query: Optional[ArrayLike] = None,
            k: int = 1,
            distance: float = 0.0,
            coordinate: str = 'spherical',
            return_weights: bool = False,
            k_search: Optional[int] = None,
        ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Finds the k nearest lattice neighbours to a specific vector in the tangent plane.

        Projects a global search direction into the local tangent frame of each ommatidium.
        If 'distance' is provided, it searches for a neighbour at that specific angular offset (a 'stride').

        Args:
            - direction: (2,) or (3,) array. The search direction in world coordinates.
            - query: Optional global indices to restrict the query. If None, all ommatidia in the view are used.
            - k: Number of neighbours to return per ommatidium.
            - distance: Target angular distance (radians). If 0.0, the nearest
                neighbour in that direction is returned regardless of distance.
            - coordinate: 'spherical' (az/el) or 'cartesian' (x/y/z) for the input direction.
            - return_weights: If True, returns a weight (0.0 to 1.0) representing
                how closely the found neighbour matches the target vector.
            - k_search: Number of nearest neighbours to evaluate. Set to None or -1 for auto.

        Returns:
            If return_weights is False:
                indices: (Q,) or (Q, k) array of global ommatidium indices.
            If return_weights is True:
                (indices, weights): Tuple of indices and float32 weights.
        """

        if k_search is None or k_search < 0:
            k_search = self._suggest_k_search(distance)

        graph = self._get_directional_graph(k_search)

        if query is None:
            valid_local = np.arange(len(self), dtype=np.intp)
        else:
            valid_local, _ = self._to_local(query)

        Q = valid_local.size
        if Q == 0 or graph['k'] == 0:
            empty = np.empty(0, dtype=np.intp) if k == 1 else np.empty((0, k), dtype=np.intp)
            return (empty, np.empty(empty.shape, dtype=np.float32)) if return_weights else empty

        local_x = graph['local_x'][valid_local]
        local_y = graph['local_y'][valid_local]
        direction = np.asarray(direction, dtype=np.float32)

        if coordinate.lower() == 'spherical':
            if direction.shape != (2,):
                raise ValueError("Spherical coordinates require 'direction' of shape (2,) for azimuth, elevation")

            d_az, d_el = direction[0], direction[1]
            dirs = self.directions[valid_local]

            az, el = cartesian_to_spherical(dirs)
            az_grad, el_grad = spherical_gradients(az, el)

            target_world = d_az * az_grad + d_el * el_grad
            target_dx = np.sum(target_world * local_x, axis=1)
            target_dy = np.sum(target_world * local_y, axis=1)

        elif coordinate.lower() == 'cartesian':
            if direction.shape != (3,):
                raise ValueError("Cartesian coordinates require 'direction' of shape (3,) for x, y, z")
            target_dx = local_x @ direction
            target_dy = local_y @ direction
        else:
            raise ValueError("Coordinates must be 'spherical' or 'cartesian'")

        # Scale target by distance or normalise to unit vector
        target_norms = np.hypot(target_dx, target_dy)
        safe = np.where(target_norms < 1e-12, 1.0, target_norms)
        target_dx /= safe
        target_dy /= safe

        if distance > 0:
            target_chord = angle_to_chord(distance)
            target_dx *= target_chord
            target_dy *= target_chord

        nb_proj_x = graph['proj_x'][valid_local]
        nb_proj_y = graph['proj_y'][valid_local]
        nb_local = graph['neighbour_local_indices'][valid_local]

        # Calculate distance in tangent plane to all candidates
        dist_sq = (nb_proj_x - target_dx[:, None]) ** 2 + (nb_proj_y - target_dy[:, None]) ** 2

        # Extract k best
        k_eff = min(k, dist_sq.shape[1])
        if k_eff == 1:
            best_idx = np.argmin(dist_sq, axis=1)
            rows = np.arange(Q)
            indices = self.omm_indices[nb_local[rows, best_idx]]
            if return_weights:
                actual_dist = np.sqrt(dist_sq[rows, best_idx])

                # Normalised weight: 1.0 at target, 0.0 at distance away
                w_scale = angle_to_chord(distance) if distance > 0 else 1.0
                weights = np.clip(1.0 - (actual_dist / w_scale), 0.0, 1.0)
        else:

            # Multi-neighbour logic
            top_k_local = np.argpartition(dist_sq, k_eff - 1, axis=1)[:, :k_eff]
            rows = np.arange(Q)[:, None]

            subset_dist = dist_sq[rows, top_k_local]
            sub_order = np.argsort(subset_dist, axis=1)

            final_local_idx = np.take_along_axis(top_k_local, sub_order, axis=1)
            indices = self.omm_indices[np.take_along_axis(nb_local, final_local_idx, axis=1)]

            if k > k_eff:
                indices = np.pad(indices, ((0, 0), (0, k - k_eff)), mode='edge')

            if return_weights:
                actual_dist = np.sqrt(np.take_along_axis(subset_dist, sub_order, axis=1))
                w_scale = angle_to_chord(distance) if distance > 0 else 1.0
                weights = np.clip(1.0 - (actual_dist / w_scale), 0.0, 1.0)
                if k > k_eff:
                    weights = np.pad(weights, ((0, 0), (0, k - k_eff)), mode='constant', constant_values=0.0)

        if return_weights:
            return indices, weights.astype(np.float32)

        return indices


class OmmatidiumView(SpatialQueries, BaseView):
    """Selection indexes the ommatidia axis."""

    __slots__ = ('_model', '_omm_indices', '_spatial')

    def __init__(self, model: 'Model', indices: np.ndarray):
        self._model = model
        self._omm_indices = np.asarray(indices, dtype=np.intp)
        self._spatial = {}

    @property
    def model(self) -> 'Model':
        return self._model

    @property
    def omm_indices(self) -> np.ndarray:
        return self._omm_indices

    @property
    def indices(self) -> np.ndarray:
        return self._omm_indices

    @property
    def rhab_indices(self) -> np.ndarray:
        return self._omm_indices[..., None] * self.R + np.arange(self.R, dtype=np.intp).reshape(-1)

    def __getitem__(self, key) -> 'OmmatidiumView':
        return OmmatidiumView(self._model, self._omm_indices[key])

    def __repr__(self) -> str:
        ind = self._omm_indices
        if ind.ndim == 0:
            info_str = f'n=1, ind=[{int(ind.item())}]'
        else:
            n = len(ind)
            info_str = f'n={n}'

            amin = ind.min()
            amax = ind.max()
            is_contiguous = (amax - amin == n - 1) and (len(np.unique(ind)) == n)
            if n <= 5:
                info_str += f', ind={self._omm_indices.tolist()}'
            else:
                if is_contiguous:
                    info_str += f', ind=[{amin}->{amax} contig.]'
                else:
                    info_str += f', ind=[..., non contig.]'
        return f'{type(self).__name__}({info_str})'

    def __eq__(self, other) -> bool:
        if not isinstance(other, OmmatidiumView):
            return False
        return self._buffer is other._buffer and np.array_equal(self.omm_indices, other.omm_indices)

    def __hash__(self) -> int:
        return hash((id(self._buffer), self._omm_indices.tobytes()))

    def __iter__(self) -> Generator['OmmatidiumView', None, None]:
        for i in self._omm_indices.reshape(-1):
            yield OmmatidiumView(self._model, np.array([i], dtype=np.intp))

    @property
    def rhabdomeres(self) -> 'RhabdomereView':
        return RhabdomereView(self._model, self.rhab_indices)

    def ommatidia_by_chirality(self) -> Dict[int, 'OmmatidiumView']:
        chir = self._omm_chirality(self.omm_indices)  # +1 / -1 per ommatidium
        out = {}
        for sign in (+1.0, -1.0):
            idx = self.omm_indices[chir == sign]
            if idx.size:
                out[int(sign)] = OmmatidiumView(self.model, idx)
        return out


class CartridgeView(OmmatidiumView):
    """Ommatidium-anchored, but .rhabdomeres follow the neural-superposition wiring map."""
    __slots__ = ()

    @property
    def rhab_indices(self) -> np.ndarray:
        phys = self._omm_indices[..., None] * self.R + np.arange(self.R, dtype=np.intp)
        src = self._buffer['cartridge_src', phys].reshape(-1)
        return src.astype(np.intp)


class RhabdomereView(BaseView):
    """Selection indexes the rhabdomeres axis."""

    __slots__ = ('_model', '_rhab_indices')

    def __init__(self, model: 'Model', rhab_indices: np.ndarray):
        self._model = model
        self._rhab_indices = np.asarray(rhab_indices, dtype=np.intp)

    @property
    def shape(self) -> Tuple[int, ...]:
        """The logical shape of this view (1D for rhabdomeres)."""
        return (len(self),)

    @property
    def model(self) -> 'Model':
        return self._model

    @property
    def rhab_indices(self) -> np.ndarray:
        return self._rhab_indices

    @property
    def indices(self) -> np.ndarray:
        return self._rhab_indices

    @property
    def omm_indices(self) -> np.ndarray:
        return np.unique(self._rhab_indices // self.R)

    def __getitem__(self, key) -> 'RhabdomereView':
        return RhabdomereView(self._model, self._rhab_indices[..., key])

    def __repr__(self) -> str:
        ind = self._rhab_indices
        n = int(ind.shape[0]) if ind.ndim else 1
        return f'RhabdomereView(n={n})'

    def __eq__(self, other) -> bool:
        if not isinstance(other, RhabdomereView):
            return False
        return self._buffer is other._buffer and np.array_equal(self.rhab_indices, other.rhab_indices)

    def __hash__(self) -> int:
        return hash((id(self._buffer), self._rhab_indices.tobytes()))

    @property
    def ommatidia(self) -> 'OmmatidiumView':
        return OmmatidiumView(self._model, self.omm_indices)

    def _tip_rel_world(self) -> np.ndarray:
        """Helper: Rhabdomere tip position relative to lens centre in world coordinates."""

        parent_omm = self.rhab_indices // self.R

        focal = np.asarray(self._buffer['focal_um', parent_omm], dtype=np.float32)
        offsets = self._buffer['rest_offset', self.rhab_indices]

        local_pos = np.column_stack([offsets, -focal])
        world_pos = local_to_world(
            local_pos,
            self._buffer['right', parent_omm],
            self._buffer['up', parent_omm],
            self._buffer['forward', parent_omm]
        )
        return world_pos

    @property
    def relative_positions(self) -> np.ndarray:
        """
        Rhabdomere tip positions relative to their parent lens centre in world coordinates, (M, 3).
        """
        return self._tip_rel_world()

    @property
    def positions(self) -> np.ndarray:
        """
        World-space positions of the rhabdomere tips (M, 3).
        """
        parent_omm = self.rhab_indices // self.R
        omm_pos = self._buffer['position', parent_omm]
        return omm_pos + self._tip_rel_world()

    @property
    def directions(self) -> np.ndarray:
        """
        Per-receptor rest viewing directions in world space (M, 3).
        """
        # Vector from tip through lens center (0,0,0 local) is -relative_tip
        return norm_l2(-self._tip_rel_world()).astype(np.float32)


class EyeView(OmmatidiumView):
    """
    One eye: an ommatidium selection + eye identity (index / side).
    """

    DEFAULT_VIRTUAL_ROWS = 3

    __slots__ = ('_eye_index', '_side')

    def __init__(self, model: 'Model', eye_index: int, indices: np.ndarray, side: str = 'left'):
        super().__init__(model, indices)
        self._eye_index = int(eye_index)
        self._side = str(side)

    def __repr__(self) -> str:
        return f"Eye(id={self._eye_index}, side={self._side!r}, ommatidia={len(self)})"

    @property
    def eye_index(self) -> int:
        return self._eye_index

    @property
    def side(self) -> str:
        """'left', 'right', or 'midline'."""
        return self._side

    @property
    def side_sign(self) -> float:
        """Sign of the side: Left/Midline = +1.0, Right = -1.0."""
        return -1.0 if self._side == 'right' else 1.0

    @property
    def ghost_positions(self) -> np.ndarray:
        """(G, 3) virtual lens positions grown outward from this eye's boundary (cached)."""
        return self._ensure_ghosts()[0]

    @property
    def ghost_directions(self) -> np.ndarray:
        """(G, 3) virtual optical axes matching ghost_positions (cached)."""
        return self._ensure_ghosts()[1]

    # TODO: Decide where ghost creation logic should live, and add a full cloud accessor (real + ghosts)
    def build_ghosts(self, n_rows: Optional[int] = None, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """(Re)grow and cache the ghost ring(s)."""
        rows = self.DEFAULT_VIRTUAL_ROWS if n_rows is None else int(n_rows)

        ghosts = ghosts_from_growth_3d(
            positions=self.positions,
            directions=self.directions,
            is_edge=self.is_edge,
            ioa_angles=self.interommatidial_angles,
            n_rows=rows,
            **kwargs,
        )
        self._spatial_store['ghosts'] = ghosts
        return ghosts

    def invalidate_ghosts(self) -> None:
        """Drop the cache explicitly (otherwise invalidated on spatial version bumps)."""
        self._spatial_store.pop('ghosts', None)

    def _ensure_ghosts(self) -> Tuple[np.ndarray, np.ndarray]:
        if 'ghosts' not in self._spatial_store:
            self.build_ghosts()
        return self._spatial_store['ghosts']
