"""
User-facing views of Model data buffers.

Views:
  BaseView      : Abstract base defining the ViewField descriptor properties
  SpatialQueries: Mixin, shared spatial-queries (trees, graphs, knn)
  OmmatidiumView: Indexes the Ommatidia axis
  CartridgeView : Like OmmatidiumView, but receptors follow neural superposition
  ReceptorView  : Indexes the Rhabdomeres axis
  EyeView       : An OmmatidiumView over one eye
"""
import logging
from typing import TYPE_CHECKING, Optional, Tuple, Union
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

from insectvision.compound_eyes.helpers.neural_superposition import UNWIRED_SRC
from insectvision.geometry.linalg import tangent_frames
from insectvision.geometry.neighbours import knn
from insectvision.geometry.spherical import cartesian_to_spherical, spherical_gradients, angle_to_chord, chord_to_angle
from insectvision.geometry.circular import wrap_angle
from insectvision.utils.shared import norm_l2

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
        self.__doc__ = doc

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self

        idx = obj.omm_indices if self.level == 'ommatidia' else obj.rhab_indices
        return obj._buffer[self.field_name, idx]

    def __set__(self, obj, value):
        idx = obj.omm_indices if self.level == 'ommatidia' else obj.rhab_indices
        obj._buffer[(self.field_name, idx)] = value


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

    # Lattice properties

    interommatidial_angles = ViewField('ioa_angles', 'ommatidia')
    interommatidial_tilt = ViewField('ioa_tilt', 'ommatidia')
    acceptance_angles_rest = ViewField('rest_acc_angles', 'rhabdomere')
    acceptance_angles_current = ViewField('curr_acc_angles', 'rhabdomere')

    @property
    def hexatic_order(self) -> np.ndarray:
        """Per-ommatidium hexatic order |Ψ6| of the lens lattice (1 = perfect hex, 0 = isotropic)."""
        return self.model._hexatic_order[self.omm_indices]

    # Per-ommatidium neighbourhood-related properties

    eye_membership = ViewField('eye_id', 'ommatidia')       # TODO: rename this one maybe

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

    # Per-ommatidium specifics (optics

    focal_length = ViewField('focal_um', 'ommatidia',
        doc="Per-ommatidium focal legnth (μm)")

    aperture = ViewField('aperture_um', 'ommatidia',
        doc="Per-ommatidium aperture (lens diameter) (μm)")

    # Per-ommatidium rhabdomeres bundles properties

    chi = ViewField('chi', 'ommatidia',
        doc="Per-ommatidium bundle orientation (chi)")
    bundle_orientation = chi

    @property
    def saccade_field(self) -> np.ndarray:
        """Per-ommatidium microsaccade actuation axis in world coordinates. Shape (N, 3)."""
        saccade = self._buffer['saccade_dxdy', self.omm_indices]
        r = self._buffer['right', self.omm_indices]
        u = self._buffer['up', self.omm_indices]
        return saccade[:, 0][:, None] * r + saccade[:, 1][:, None] * u

    @property
    def chirality(self) -> np.ndarray:
        """Returns +1 or -1 for each ommatidium."""
        neg = self._buffer['chirality_neg', self.rhab_indices]
        return np.where(neg == 1, -1.0, 1.0).astype(np.float32)

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

    # TODO: also expose rhabdomeres positions in world space?

    wavelength = ViewField('wavelength_um', 'rhabdomere',
        doc="Rhabdomeres peak wavelengths (μm).")

    tau_membrane = ViewField('tau_membrane', 'rhabdomere',
        doc="Rhabdomeres membrane RC.")

    # Neural superposition wiring properties

    @property
    def neural_superposition(self) -> bool:
        """Whether this model is superposition eyes."""
        return self.model._superposition_wired

    @property
    def cartridge_map(self) -> np.ndarray:
        """(N, R) donor ommatidium per slot (-1 where unwired)."""
        if not self.neural_superposition:
            return np.full((self.N, self.R), -1, dtype=np.intp)
        src = self.model.cartridge_indices[self.omm_indices]
        return np.where(src != UNWIRED_SRC, (src // self.R).astype(np.intp), -1)

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
    def has_selfwires(self) -> np.ndarray:
        return self.model._get_conflicts.has_selfwires[self.omm_indices]

    @property
    def unwired_count(self) -> int:
        return self.model._get_conflicts.unwired_count[self.omm_indices]


class NeighbourResult:
    """
    Result of a neighbours() query.

    Attributes:
        mask: (Q,) bool, which input queries were inside this view and got results.
        indices: (M, k) int, global ommatidia indices of the k neighbours, M = mask.sum().
        distances: (M, k) float, distances to these neighbours.
        is_immediate: (M, k) bool or None, True where the neighbour is in the first lattice ring of the query ommatidium.
        same_chirality: (M, k) bool or None, True where neighbour chirality matches query.
    """

    def __init__(self, view, mask, indices, distances, is_immediate=None, same_chirality=None):
        self.view = view
        self.mask = mask
        self.indices = indices
        self.distances = distances
        self.is_immediate = is_immediate
        self.same_chirality = same_chirality

    def __bool__(self) -> bool:
        return bool(self.indices.size)

    def __len__(self) -> int:
        return int(self.indices.shape[0])

    @property
    def ommatidia(self) -> 'OmmatidiumView':
        return OmmatidiumView(self.view.model, self.indices.ravel())

    @property
    def receptors(self) -> 'ReceptorView':
        return self.ommatidia.receptors


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
        Local rows of ommatidia that are conflict-free and not self-wired.
        """
    
        if 'conflictfree_local_i' not in self._spatial_store:
            self._spatial_store['conflictfree_local_i'] = np.flatnonzero(~self.has_conflicts & ~self.has_selfwires)

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

    # Some internal helpers

    def _omm_chirality(self, omm_global: np.ndarray) -> np.ndarray:
        """
        +1 / -1 chirality for the given (global) ommatidium indices (read from rhabdomere slot 0).
        """
        rhab0 = np.asarray(omm_global, dtype=np.intp) * self.R
        neg = np.asarray(self._buffer['chirality_neg', rhab0])

        return np.where(neg == 1, -1.0, 1.0).astype(np.float32)

    def _empty(self) -> 'OmmatidiumView':
        return OmmatidiumView(self.model, np.empty(0, dtype=np.intp))

    @staticmethod
    def _tag_immediate(result: 'NeighbourResult', dists: np.ndarray, factor: Optional[float]) -> None:
        """
        Tag result.is_immediate where distance <= factor * local scale (mean of nearest 3).
        """
        if factor is None or not dists.size:
            return

        n_ref = min(3, dists.shape[1])
        local_scale = np.mean(dists[:, :n_ref], axis=1, keepdims=True)
        result.is_immediate = dists <= (local_scale * float(factor))

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

        immediate_only restricts to the cached first-ring graph; otherwise
        neighbour_dist_factor tags 'is_immediate' (d <= factor * local scale),
        or set it to None to skip the tagging.
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

            self._tag_immediate(result, neighb_dists, neighbour_dist_factor)
            return result

        # Points path
        points = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        neighb_dists, neighb_indices = knn(self._get_tree('positions'), points, k, drop_self=False)
        g_neighb_indices = self.omm_indices[neighb_indices]

        result = NeighbourResult(self, mask=np.ones(points.shape[0], dtype=bool),
                                 indices=g_neighb_indices, distances=neighb_dists)
        self._tag_immediate(result, neighb_dists, neighbour_dist_factor)
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
        best = self.model._lookat_top_k(self.positions, self.directions, query, k, conflict)

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
        """All ommatidia whose optical axis lies within 'angle' of 'center_direction'."""
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
        """All ommatidia whose world-space position is within 'radius' of 'center_position'."""
        return self._query_ball('positions', center_position, radius, avoid_conflicts)

    def directed_neighbours(self,
            direction: ArrayLike,
            query: Optional[ArrayLike] = None,
            k: int = 1,
            coordinate: str = 'spherical',
            return_weights: bool = False,
            k_search: int = 8,
        ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        For each ommatidium, the k lattice neighbour(s) lying closest to a search
        direction. See the long-form docstring in the original implementation;
        behaviour is unchanged.
        """
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

        target_norms = np.hypot(target_dx, target_dy)
        zero_mask = target_norms < 1e-12
        target_dx_n = np.where(zero_mask, 1.0, target_dx / np.where(zero_mask, 1.0, target_norms))
        target_dy_n = np.where(zero_mask, 0.0, target_dy / np.where(zero_mask, 1.0, target_norms))
        target_angle = np.arctan2(target_dy_n, target_dx_n)

        nb_proj_x = graph['proj_x'][valid_local]
        nb_proj_y = graph['proj_y'][valid_local]
        nb_local = graph['neighbour_local_indices'][valid_local]

        nb_angles = np.arctan2(nb_proj_y, nb_proj_x)
        score = np.abs(wrap_angle(nb_angles - target_angle[:, None]))
        score = np.where(score > (np.pi / 2.0), 1e6, score)   # reject anything >90 deg off

        k_eff = min(k, score.shape[1])
        if k_eff <= 0:
            indices = np.zeros(Q, dtype=np.intp) if k == 1 else np.zeros((Q, k), dtype=np.intp)
            if return_weights:
                w = target_norms if k == 1 else np.tile(target_norms[:, None], (1, k))
                return indices, w
            return indices

        if k_eff == 1:
            best = np.argmin(score, axis=1)
            indices = self.omm_indices[nb_local[np.arange(Q), best]]
            if k > 1:
                indices = np.tile(indices[:, None], (1, k))
        else:
            top_k_local = np.argpartition(score, k_eff - 1, axis=1)[:, :k_eff]
            order = np.argsort(np.take_along_axis(score, top_k_local, axis=1), axis=1)
            indices = self.omm_indices[np.take_along_axis(nb_local, np.take_along_axis(top_k_local, order, axis=1), axis=1)]

            if k > k_eff:
                pad = np.tile(indices[:, 0:1], (1, k - k_eff))
                indices = np.concatenate([indices, pad], axis=1)

        if return_weights:
            w = target_norms if k == 1 else np.tile(target_norms[:, None], (1, k))
            return indices, w.astype(np.float32)

        return indices

    def max_gap(self) -> float:
        """Largest angular gap between any ommatidium and its nearest neighbour (rad)."""

        if len(self) <= 1:
            return 0.0

        tree = self._get_tree('directions')
        chord_dists, _ = tree.query(self.directions, k=2)
        return float(chord_to_angle(np.max(chord_dists[:, 1])))



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
    def indices(self):
        return self._omm_indices

    @property
    def rhab_indices(self) -> np.ndarray:
        return self._omm_indices[..., None] * self.R + np.arange(self.R, dtype=np.intp).reshape(-1)

    def __getitem__(self, key):
        return OmmatidiumView(self._model, self._omm_indices[key])

    def __repr__(self):
        return f'{type(self).__name__}(ommatidia={self._omm_indices.shape})'

    def __eq__(self, other):
        if not isinstance(other, OmmatidiumView):
            return False
        return self._buffer is other._buffer and np.array_equal(self.omm_indices, other.omm_indices)

    def __iter__(self):
        for i in self._omm_indices.reshape(-1):
            yield OmmatidiumView(self._model, np.array([i], dtype=np.intp))

    @property
    def receptors(self) -> 'ReceptorView':
        return ReceptorView(self._model, self.rhab_indices)


class CartridgeView(OmmatidiumView):
    """Ommatidium-anchored, but .receptors follow the neural-superposition wiring map."""
    __slots__ = ()

    @property
    def rhab_indices(self) -> np.ndarray:
        return self._buffer['cartridge_src', self._omm_indices].reshape(-1)


class ReceptorView(BaseView):
    """Selection indexes the rhabdomeres axis."""

    __slots__ = ('_model', '_rhab_indices')

    def __init__(self, model: 'Model', rhab_indices: np.ndarray):
        self._model = model
        self._rhab_indices = np.asarray(rhab_indices, dtype=np.intp)

    @property
    def shape(self) -> Tuple[int, ...]:
        """The logical shape of this view (1D for receptors)."""
        return (len(self),)

    @property
    def N(self) -> int:
        idx = self._rhab_indices
        return int(idx.shape[0]) if idx.ndim else 1

    @property
    def model(self) -> 'Model':
        return self._model

    @property
    def rhab_indices(self) -> np.ndarray:
        return self._rhab_indices

    @property
    def indices(self):
        return self._rhab_indices

    @property
    def omm_indices(self) -> np.ndarray:
        return np.unique(self._rhab_indices // self.R)

    def __getitem__(self, key):
        return ReceptorView(self._model, self._rhab_indices[..., key])

    def __repr__(self):
        return f'ReceptorView(receptors={self._rhab_indices.shape})'

    def __eq__(self, other):
        if not isinstance(other, ReceptorView):
            return False
        return self._buffer is other._buffer and np.array_equal(self.rhab_indices, other.rhab_indices)

    @property
    def ommatidia(self) -> OmmatidiumView:
        return OmmatidiumView(self._model, self.omm_indices)


class EyeView(OmmatidiumView):
    """
    One eye: an ommatidium selection + eye identity (index / side).
    """

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