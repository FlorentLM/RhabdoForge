"""
User-facing views of ReceptorArray data buffers.

Views (LensView, ReceptorView, Cartridge) are slice-like handles into the array.
They hold a reference to the parent ReceptorArray and a '_gi' array of global indices into its data buffers.

Hierarchy:
    LensView                 # M lenses, iterable yields size-1 LensViews
                             # size-1 LensViews expose singular accessors
    ReceptorView             # M receptors
    Cartridge                # central lens + R1-R6 members from neighbours
    Eye                      # KDtrees + neighbour graph for one anatomical eye,
                             # with a 'side' label ('left' / 'right' / 'midline')
    VisualOutput             # per-receptor output array with reshape helpers
                             # (per-lens, per-cartridge, plus colours / radiance slicing)

Notes:
    - A LensView spanning multiple eyes raises on eye-dependent operations,
    use 'eye.lenses' directly for eye-scoped queries.

    - LensView geometry is read-only. Only global transforms with ra.translate/scale/rotate are possible.

    - ReceptorView allows setting cell-level properties (sensitivity, tau_membrane, acceptance)
    and the actuated 'directions' field (it is dynamic state).

    - Receptor positions and rest directions are derived from lens position and
    the orientation pipeline, so they can't be set directly.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Union, TYPE_CHECKING
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree

from insectvision.compound_eyes.datatypes import get_metadata_field

if TYPE_CHECKING:
    from insectvision.compound_eyes.receptor_array import ReceptorArray
    from insectvision.compound_eyes.kernel import RhabdomereKernel


@dataclass
class NeighbourResult:
    """
    Result of an Eye.neighbours() query.

    Attributes:

    mask: (Q,) bool array, Which input query points were inside this eye and got results.
    indices: (M, k) int array, Global lens indices of the k nearest neighbours, M = mask.sum().
    distances: (M, k) float array, Distances to those neighbours.
    is_immediate: (M, k) bool array (optional), True where the neighbour is in the
        first lattice ring of the query lens (distance <= neighbour_dist_factor * closest).
        Only populated when Eye.neighbours() is called with neighbour_dist_factor != None,
        or when immediate_only=True (in which case all entries are True).
    same_chirality: (M, k) bool array (optional), True where the neighbour's chirality
        matches the query lens's chirality. Only populated when Eye.neighbours() is
        called with a chirality array AND in lens_indices mode (so the query has an
        identifiable lens id).
    """
    mask: np.ndarray
    indices: np.ndarray
    distances: np.ndarray
    is_immediate: Optional[np.ndarray] = None
    same_chirality: Optional[np.ndarray] = None

    def __bool__(self) -> bool:
        return bool(self.indices.size)

    def __len__(self) -> int:
        return int(self.indices.shape[0])


class LensView:
    """
    A subset of M lenses in a ReceptorArray.

    A size-1 LensView exposes 'singular' convenience properties (position,
    direction, chirality, ...) that return scalars / 1D vectors rather than
    arrays of shape (1, ...). The plural properties (positions, directions,
    chiralities, ...) always return arrays.
    """

    __slots__ = ('_ra', '_gi')
    __hash__ = None     # disable hashing, views are mutable handles, not values

    def __init__(self, ra: 'ReceptorArray', indices: ArrayLike):
        self._ra = ra
        self._gi = np.asarray(indices, dtype=np.intp).reshape(-1)

        if self._gi.size > 0:
            if int(self._gi.min()) < 0 or int(self._gi.max()) >= ra.lens_count:
                raise IndexError(f"Lens indices out of range for ReceptorArray with {ra.lens_count} lenses")

    # Basics

    def __len__(self) -> int:
        return int(self._gi.size)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n={len(self)}, indices={self._gi[:8]}{'...' if len(self) > 8 else ''})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, LensView):
            return NotImplemented
        return self._ra is other._ra and np.array_equal(self._gi, other._gi)

    def __iter__(self):
        # Iterating yields size-1 LensViews (i.e. ommatidia)
        for i in self._gi:
            yield LensView(self._ra, np.array([i], dtype=np.intp))

    def __getitem__(self, idx) -> 'LensView':
        return LensView(self._ra, self._gi[idx])

    def _require_single(self, name: str) -> int:
        if len(self) != 1:
            raise ValueError(f"'{name}' requires a size-1 LensView")
        return int(self._gi[0])

    @property
    def global_indices(self) -> np.ndarray:
        """Global lens indices into the ReceptorArray (read-only copy)."""
        return self._gi.copy()

    @property
    def indices(self) -> np.ndarray:
        """Alias to global_indices."""
        return self.global_indices

    # Geometry (read-only)

    @property
    def positions(self) -> np.ndarray:
        """(M, 3) lens world positions."""
        return self._ra._lens_positions[self._gi].copy()

    @property
    def directions(self) -> np.ndarray:
        """(M, 3) lens optical-axis (forward) unit vectors."""
        return self._ra._lens_directions[self._gi].copy()

    @property
    def right_axes(self) -> np.ndarray:
        """(M, 3) tangent 'right' vectors (lens-local x)."""
        return self._ra._local_right[self._gi].copy()

    @property
    def up_axes(self) -> np.ndarray:
        """(M, 3) tangent 'up' vectors (lens-local y)."""
        return self._ra._local_up[self._gi].copy()

    # Lattice geometry

    @property
    def ioa_axes(self) -> np.ndarray:
        """(M, 2) per-lens (minor, major) interommatidial angles (rad)."""
        return self._ra.lens_static_data['ioa_axes'][self._gi].copy()

    @property
    def ioa_tilts(self) -> np.ndarray:
        """(M,) per-lens hex-lattice tilts (rad)."""
        return self._ra.lens_static_data['ioa_tilt'][self._gi].copy()

    @property
    def lens_diameters_um(self) -> np.ndarray:
        """(M,) lens apertures (μm)."""
        return self._ra.lens_static_data['lens_diameter_um'][self._gi].copy()

    @property
    def nodal_distances_um(self) -> np.ndarray:
        """(M,) lens-to-rhabdomere lever arms (μm)."""
        return self._ra.lens_static_data['nodal_distance_um'][self._gi].copy()

    # Bundle orientation

    @property
    def bundle_orientations(self) -> np.ndarray:
        """(M,) per-lens bundle yaw (chi, rad)."""
        return self._ra._bundle_orientation[self._gi].copy()

    @property
    def chiralities(self) -> np.ndarray:
        """(M,) per-lens chirality (+1 or -1)."""
        return self._ra._chirality_arr[self._gi].copy()

    @property
    def saccade_axes_local(self) -> np.ndarray:
        """(M, 2) saccade actuation axis in each lens (right, up) tangent frame."""
        d = self._ra.lens_static_data
        return np.column_stack([d['sacc_x'][self._gi], d['sacc_y'][self._gi]])

    @property
    def saccade_axes(self) -> np.ndarray:
        """(M, 3) saccade actuation axis in world coordinates."""
        sx = self._ra.lens_static_data['sacc_x'][self._gi][:, None]
        sy = self._ra.lens_static_data['sacc_y'][self._gi][:, None]
        return sx * self._ra._local_right[self._gi] + sy * self._ra._local_up[self._gi]

    # Photomechanical biophysics (per-lens, broadcast from kernel)

    @property
    def tau_rises(self) -> np.ndarray:
        return self._ra.lens_static_data['tau_rise'][self._gi].copy()

    @property
    def tau_relaxes(self) -> np.ndarray:
        return self._ra.lens_static_data['tau_relax'][self._gi].copy()

    @property
    def tau_fasts(self) -> np.ndarray:
        return self._ra.lens_static_data['tau_fast'][self._gi].copy()

    @property
    def tau_adapts(self) -> np.ndarray:
        return self._ra.lens_static_data['tau_adapt'][self._gi].copy()

    @property
    def gain_lat_ums(self) -> np.ndarray:
        return self._ra.lens_static_data['gain_lat_um'][self._gi].copy()

    @property
    def gain_ax_ums(self) -> np.ndarray:
        return self._ra.lens_static_data['gain_ax_um'][self._gi].copy()

    # Dynamic state

    @property
    def adapted_luminances(self) -> np.ndarray:
        return self._ra.lens_dynamic_data['adapted_lum'][self._gi].copy()

    @property
    def fast_luminances(self) -> np.ndarray:
        return self._ra.lens_dynamic_data['fast_lum'][self._gi].copy()

    @property
    def lateral_displacements_um(self) -> np.ndarray:
        return self._ra.lens_dynamic_data['lateral_um'][self._gi].copy()

    @property
    def axial_displacements_um(self) -> np.ndarray:
        return self._ra.lens_dynamic_data['axial_um'][self._gi].copy()

    # TODO: Setters for indirect buffer access?

    # Diagnostics, conflicts

    @property
    def donation_conflicts(self) -> np.ndarray:
        """(M,) True if any peripheral receptor in this lens is donated to != 1 cartridge."""
        return self._ra.donation_conflicts[self._gi].copy()

    @property
    def receiving_conflicts(self) -> np.ndarray:
        """(M,) True if this lens's cartridge failed to gather a neighbour for any slot."""
        return self._ra.receiving_conflicts[self._gi].copy()

    @property
    def have_conflicts(self) -> np.ndarray:
        """(M,) True if the lens has a donation or receiving conflict."""
        return self._ra.have_conflicts[self._gi].copy()

    # Linking to receptors / eye

    @property
    def receptors(self) -> 'ReceptorView':
        """The R*M receptors behind these lenses (R = receptors_per_lens)."""
        R = self._ra.receptors_per_lens
        rcpt_indices = (self._gi[:, None] * R + np.arange(R, dtype=np.intp)[None, :]).ravel()
        return ReceptorView(self._ra, rcpt_indices)

    @property
    def eye_ids(self) -> np.ndarray:
        """(M,) eye id (0-7) of each lens."""
        return self._ra._lens_eye_ids[self._gi].copy()

    @property
    def sides(self) -> np.ndarray:
        """(M,) per-lens side string ('left' / 'right' / 'midline') from each parent eye."""
        side_by_eid = {e.eye_id: e.side for e in self._ra._eyes}
        ids = self._ra._lens_eye_ids[self._gi]
        return np.array([side_by_eid.get(int(i), 'unknown') for i in ids], dtype=object)

    @property
    def eye(self) -> 'Eye':
        """The single Eye these lenses belong to. Raises if the view spans multiple eyes."""
        ids = self._ra._lens_eye_ids[self._gi]
        if ids.size == 0:
            raise ValueError("Empty LensView has no eye")
        first = ids[0]
        if not np.all(ids == first):
            raise ValueError("LensView spans multiple eyes")
        return self._ra.eye(int(first))

    # Singular accessors (require size-1 LensView)

    @property
    def lens_index(self) -> int:
        return self._require_single('lens_index')

    @property
    def position(self) -> np.ndarray:
        return self._ra._lens_positions[self._require_single('position')].copy()

    @property
    def direction(self) -> np.ndarray:
        return self._ra._lens_directions[self._require_single('direction')].copy()

    @property
    def right_axis(self) -> np.ndarray:
        return self._ra._local_right[self._require_single('right_axis')].copy()

    @property
    def up_axis(self) -> np.ndarray:
        return self._ra._local_up[self._require_single('up_axis')].copy()

    @property
    def ioa_axis(self) -> np.ndarray:
        return self._ra.lens_static_data['ioa_axes'][self._require_single('ioa_axis')].copy()

    @property
    def ioa_tilt(self) -> float:
        return float(self._ra.lens_static_data['ioa_tilt'][self._require_single('ioa_tilt')])

    @property
    def lens_diameter_um(self) -> float:
        return float(self._ra.lens_static_data['lens_diameter_um'][self._require_single('lens_diameter_um')])

    @property
    def nodal_distance_um(self) -> float:
        return float(self._ra.lens_static_data['nodal_distance_um'][self._require_single('nodal_distance_um')])

    @property
    def bundle_orientation(self) -> float:
        return float(self._ra._bundle_orientation[self._require_single('bundle_orientation')])

    @property
    def chirality(self) -> float:
        return float(self._ra._chirality_arr[self._require_single('chirality')])

    @property
    def saccade_axis(self) -> np.ndarray:
        self._require_single('saccade_axis')
        return self.saccade_axes[0]

    @property
    def saccade_axis_local(self) -> np.ndarray:
        self._require_single('saccade_axis_local')
        return self.saccade_axes_local[0]

    @property
    def eye_id(self) -> int:
        return int(self._ra._lens_eye_ids[self._require_single('eye_id')])

    @property
    def side(self) -> str:
        """'left' / 'right' / 'midline' of the parent eye (size-1 view only)."""
        self._require_single('side')
        return self.eye.side

    @property
    def has_conflicts(self) -> bool:
        return bool(self._ra.have_conflicts[self._require_single('has_conflicts')])

    @property
    def cartridge(self) -> Optional['Cartridge']:
        """The cartridge anchored at this lens's central rhabdomere (size-1 view only, if wired)."""
        idx = self._require_single('cartridge')
        if not self._ra._cartridges_wired:
            return None
        return Cartridge(self._ra, idx)

    # Directional neighbour search (delegates to parent Eye)

    def directed_neighbours(self,
        direction: ArrayLike,
        k: int = 1,
        coordinate: str = 'spherical',
        return_weights: bool = False,
        k_search: int = 8,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        For each lens in this view, find the k neighbour(s) lying closest to
        the given search direction. See Eye.directed_neighbours for details
        on 'direction' / 'coordinate'.

        The view must be within a single eye (otherwise it will raise).
        """
        if self._gi.size == 0:
            empty = np.empty(0, dtype=np.intp) if k == 1 else np.empty((0, k), dtype=np.intp)
            if return_weights:
                return empty, np.empty(empty.shape, dtype=np.float32)
            return empty
        eye_ids = self._ra._lens_eye_ids[self._gi]

        if np.unique(eye_ids).size > 1:
            raise ValueError("directed_neighbours() requires a single-eye LensView")

        eye = self._ra.eye(int(eye_ids[0]))
        return eye.directed_neighbours(
            direction=direction,
            query_lens_indices=self._gi,
            k=k,
            coordinate=coordinate,
            return_weights=return_weights,
            k_search=k_search,
        )


class ReceptorView:
    """
    A subset of M receptors in a ReceptorArray.

    Read-only: positions (derived from parent lens), all metadata fields.
    Settable: sensitivities, tau_membrane, rest acceptance, acceptance tilt, actuated direction.
    """

    __slots__ = ('_ra', '_gi')
    __hash__ = None     # disable hashing, views are mutable handles, not values

    def __init__(self, ra: 'ReceptorArray', indices: ArrayLike):
        self._ra = ra
        self._gi = np.asarray(indices, dtype=np.intp).reshape(-1)

        if self._gi.size > 0:
            if int(self._gi.min()) < 0 or int(self._gi.max()) >= ra.total_receptors:
                raise IndexError("Receptor indices out of range")

    def __len__(self) -> int:
        return int(self._gi.size)

    def __repr__(self) -> str:
        return f"ReceptorView(n={len(self)}, indices={self._gi[:8]}{'...' if len(self) > 8 else ''})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, ReceptorView):
            return NotImplemented
        return self._ra is other._ra and np.array_equal(self._gi, other._gi)

    def __getitem__(self, idx) -> 'ReceptorView':
        return ReceptorView(self._ra, self._gi[idx])

    @property
    def global_indices(self) -> np.ndarray:
        """Global receptor indices into the ReceptorArray (read-only copy)."""
        return self._gi.copy()

    @property
    def indices(self) -> np.ndarray:
        """Alias to global_indices."""
        return self.global_indices

    # Read-only derived / structural

    @property
    def positions(self) -> np.ndarray:
        """(M, 3) world positions (= parent lens positions). Read-only."""
        return self._ra.rcpt_static_data['position'][self._gi].copy()

    @property
    def lens_indices(self) -> np.ndarray:
        """(M,) parent lens global index for each receptor."""
        meta = self._ra.rcpt_static_data['metadata'][self._gi]
        return get_metadata_field(meta, 'lens_id').astype(np.intp)

    @property
    def types(self) -> np.ndarray:
        """(M,) receptor type within bundle (R1=0, R2=1, ..., R7/8=6)."""
        meta = self._ra.rcpt_static_data['metadata'][self._gi]
        return get_metadata_field(meta, 'rcpt_type').astype(np.intp)

    @property
    def eye_ids(self) -> np.ndarray:
        meta = self._ra.rcpt_static_data['metadata'][self._gi]
        return get_metadata_field(meta, 'eye_id').astype(np.intp)

    @property
    def neighbour_counts(self) -> np.ndarray:
        """(M,) number of lattice neighbours this receptor's parent lens has."""
        meta = self._ra.rcpt_static_data['metadata'][self._gi]
        return get_metadata_field(meta, 'neighbour_count').astype(np.intp)

    @property
    def chiralities_neg(self) -> np.ndarray:
        """(M,) 1 if parent lens has chirality -1, else 0."""
        meta = self._ra.rcpt_static_data['metadata'][self._gi]
        return get_metadata_field(meta, 'chirality_neg').astype(np.intp)

    @property
    def cartridge_sources(self) -> np.ndarray:
        """(M,) global receptor index of the cartridge source (R7/8) each member targets."""
        return self._ra.rcpt_static_data['cartridge_src'][self._gi].copy()

    @property
    def rest_offsets_um(self) -> np.ndarray:
        """(M, 2) focal-plane offset behind lens (μm), already rotated by chi/chirality."""
        return self._ra.rcpt_static_data['rot_offset'][self._gi].copy()

    @property
    def rhab_diameters_um(self) -> np.ndarray:
        return self._ra.rcpt_static_data['rhab_diameter_um'][self._gi].copy()

    @property
    def wavelengths_um(self) -> np.ndarray:
        return self._ra.rcpt_static_data['wavelength_um'][self._gi].copy()

    # Settable cell-level properties

    @property
    def sensitivities(self) -> np.ndarray:
        """(M, 3) channel multipliers (UV, G, B)."""
        return self._ra.rcpt_static_data['sensitivity'][self._gi].copy()

    @sensitivities.setter
    def sensitivities(self, value: ArrayLike):
        v = np.broadcast_to(np.asarray(value, dtype=np.float32), (len(self), 3))
        self._ra.rcpt_static_data['sensitivity'][self._gi] = v

    @property
    def sensitivities_uv(self) -> np.ndarray:
        return self._ra.rcpt_static_data['sensitivity'][self._gi, 0].copy()

    @sensitivities_uv.setter
    def sensitivities_uv(self, value: ArrayLike):
        self._ra.rcpt_static_data['sensitivity'][self._gi, 0] = np.asarray(value, dtype=np.float32)

    @property
    def sensitivities_g(self) -> np.ndarray:
        return self._ra.rcpt_static_data['sensitivity'][self._gi, 1].copy()

    @sensitivities_g.setter
    def sensitivities_g(self, value: ArrayLike):
        self._ra.rcpt_static_data['sensitivity'][self._gi, 1] = np.asarray(value, dtype=np.float32)

    @property
    def sensitivities_b(self) -> np.ndarray:
        return self._ra.rcpt_static_data['sensitivity'][self._gi, 2].copy()

    @sensitivities_b.setter
    def sensitivities_b(self, value: ArrayLike):
        self._ra.rcpt_static_data['sensitivity'][self._gi, 2] = np.asarray(value, dtype=np.float32)

    @property
    def tau_membranes(self) -> np.ndarray:
        return self._ra.rcpt_static_data['tau_membrane'][self._gi].copy()

    @tau_membranes.setter
    def tau_membranes(self, value: ArrayLike):
        self._ra.rcpt_static_data['tau_membrane'][self._gi] = np.asarray(value, dtype=np.float32)

    @property
    def rest_acceptance_axes(self) -> np.ndarray:
        """(M, 2) acceptance angles (minor, major) at rest (rad)."""
        return self._ra.rcpt_static_data['rest_acc'][self._gi].copy()

    @rest_acceptance_axes.setter
    def rest_acceptance_axes(self, value: ArrayLike):
        v = np.broadcast_to(np.asarray(value, dtype=np.float32), (len(self), 2))
        self._ra.rcpt_static_data['rest_acc'][self._gi] = v

    @property
    def acceptance_tilts(self) -> np.ndarray:
        return self._ra.rcpt_static_data['acc_tilt'][self._gi].copy()

    @acceptance_tilts.setter
    def acceptance_tilts(self, value: ArrayLike):
        self._ra.rcpt_static_data['acc_tilt'][self._gi] = np.asarray(value, dtype=np.float32)

    @property
    def directions(self) -> np.ndarray:
        """(M, 3) current (actuated) viewing directions. Settable."""
        return self._ra.rcpt_dynamic_data['direction'][self._gi].copy()

    @directions.setter
    def directions(self, value: ArrayLike):
        v = np.broadcast_to(np.asarray(value, dtype=np.float32), (len(self), 3))
        self._ra.rcpt_dynamic_data['direction'][self._gi] = v

    @property
    def acceptance_axes(self) -> np.ndarray:
        """(M, 2) current (actuated) acceptance axes (rad)."""
        return self._ra.rcpt_dynamic_data['acc_axes'][self._gi].copy()

    @acceptance_axes.setter
    def acceptance_axes(self, value: ArrayLike):
        v = np.broadcast_to(np.asarray(value, dtype=np.float32), (len(self), 2))
        self._ra.rcpt_dynamic_data['acc_axes'][self._gi] = v

    @property
    def adaptation_states(self) -> np.ndarray:
        return self._ra.rcpt_dynamic_data['adaptation_state'][self._gi].copy()

    @adaptation_states.setter
    def adaptation_states(self, value: ArrayLike):
        self._ra.rcpt_dynamic_data['adaptation_state'][self._gi] = np.asarray(value, dtype=np.float32)


class Cartridge:
    """
    A lamina cartridge in neural-superposition optics.

    The cartridge is anchored at one ommatidium's central rhabdomere (R7/8).
    Its members are the peripheral rhabdomeres (R1-R6) from neighbouring
    ommatidia whose lines of sight converge on this cartridge's direction.

    '.lens' is the home (central) ommatidium. '.sources' gives, per member,
    the global lens index of the ommatidium that contributes that receptor.
    """

    __slots__ = ('_ra', '_central_lens_idx', '_member_indices')
    __hash__ = None

    def __init__(self, ra: 'ReceptorArray', central_lens_idx: int):
        self._ra = ra
        self._central_lens_idx = int(central_lens_idx)
        if not getattr(ra, '_cartridges_wired', False):
            raise ValueError("Cartridges not wired")

        R = ra.receptors_per_lens
        sources = ra._cartridge_map[self._central_lens_idx]
        self._member_indices = (sources * R + np.arange(R, dtype=np.intp))

    def __len__(self) -> int:
        return int(self._member_indices.size)

    def __repr__(self) -> str:
        return f"Cartridge(lens={self._central_lens_idx}, R={len(self)})"

    @property
    def lens(self) -> LensView:
        """The home (central) ommatidium for this cartridge."""
        return LensView(self._ra, np.array([self._central_lens_idx], dtype=np.intp))

    @property
    def receptors(self) -> ReceptorView:
        """All member receptors (the R1-R6 contributors from neighbouring lenses)."""
        return ReceptorView(self._ra, self._member_indices)

    @property
    def sources(self) -> np.ndarray:
        """(R,) global lens index of the source ommatidium for each member slot."""
        return self._ra._cartridge_map[self._central_lens_idx].copy()


class Eye:
    """
    One anatomical eye.

    Owns:
      - The lens-index mask for the eye
      - A 'side' label: 'left', 'right', or 'midline'
      - A position KDtree (lazy)
      - A direction KDtree (lazy)
      - The lattice neighbour graph (lazy)

    Spatial queries are eye-local.
    For animal-wide queries, iterate over eyes or use 'ra.query_directions()' (it dispatches across eyes).
    """

    __slots__ = (
        '_ra', '_eye_id', '_lens_indices', '_side',
        '_position_tree', '_direction_tree',
        '_position_tree_cf', '_direction_tree_cf', '_cf_local_indices',
        '_neighbour_graph', '_neighbour_k', '_directional_graph',
    )

    def __init__(self, ra: 'ReceptorArray', eye_id: int, lens_indices: np.ndarray, side: str = 'left'):
        self._ra = ra
        self._eye_id = int(eye_id)
        self._lens_indices = np.asarray(lens_indices, dtype=np.intp)
        self._side = str(side)
        self._position_tree: Optional[cKDTree] = None
        self._direction_tree: Optional[cKDTree] = None

        self._position_tree_cf: Optional[cKDTree] = None
        self._direction_tree_cf: Optional[cKDTree] = None
        self._cf_local_indices: Optional[np.ndarray] = None

        self._neighbour_graph: Optional[np.ndarray] = None
        self._neighbour_k: int = -1
        self._directional_graph: Optional[dict] = None

    def __repr__(self) -> str:
        return f"Eye(id={self._eye_id}, side={self._side!r}, lenses={len(self._lens_indices)})"

    def __len__(self) -> int:
        return int(self._lens_indices.size)

    @property
    def eye_id(self) -> int:
        return self._eye_id

    @property
    def side(self) -> str:
        """'left', 'right', or 'midline'."""
        return self._side

    @property
    def lens_indices(self) -> np.ndarray:
        """Global indices of the lenses in this eye."""
        return self._lens_indices.copy()

    @property
    def lenses(self) -> LensView:
        return LensView(self._ra, self._lens_indices)

    @property
    def receptors(self) -> ReceptorView:
        R = self._ra.receptors_per_lens
        rcpt_indices = (self._lens_indices[:, None] * R
                        + np.arange(R, dtype=np.intp)[None, :]).ravel()
        return ReceptorView(self._ra, rcpt_indices)

    @property
    def ommatidia(self) -> LensView:
        """Alias for '.lenses': iterate to yield Ommatidia."""
        return self.lenses

    # Cache management

    def _invalidate(self) -> None:
        """
        Invalidate cached trees and neighbour graph. Called when lens geometry changes.
        """
        self._position_tree = None
        self._direction_tree = None
        self._position_tree_cf = None
        self._direction_tree_cf = None
        self._cf_local_indices = None
        self._neighbour_graph = None
        self._neighbour_k = -1
        self._directional_graph = None

    def _ensure_position_tree(self) -> cKDTree:
        if self._position_tree is None:
            self._position_tree = cKDTree(self._ra._lens_positions[self._lens_indices])
        return self._position_tree

    def _ensure_direction_tree(self) -> cKDTree:
        if self._direction_tree is None:
            self._direction_tree = cKDTree(self._ra._lens_directions[self._lens_indices])
        return self._direction_tree

    def _ensure_neighbour_graph(self, k: int) -> np.ndarray:
        """
        Per-lens lattice neighbours (drops self). Rebuilds if k changes.
        """

        if self._neighbour_graph is None or self._neighbour_k != k:
            tree = self._ensure_position_tree()
            positions = self._ra._lens_positions[self._lens_indices]
            n = positions.shape[0]
            actual_k = min(k, n - 1) if n > 1 else 0

            if actual_k == 0:
                self._neighbour_graph = np.empty((n, 0), dtype=np.intp)
            else:
                _, nidx_local = tree.query(positions, k=actual_k + 1)
                # nidx_local has shape (n, k+1); drop the self column (the first)
                if nidx_local.ndim == 1:
                    nidx_local = nidx_local.reshape(-1, 1)
                self._neighbour_graph = nidx_local[:, 1:].astype(np.intp)
            self._neighbour_k = k

        return self._neighbour_graph

    def _ensure_trees_cf(self) -> Tuple[Optional[cKDTree], Optional[cKDTree], np.ndarray]:
        """Lazy-build conflict-free trees (used when exclude_conflicts=True)."""
        if self._direction_tree_cf is None:
            valid_mask = ~self._ra.have_conflicts[self._lens_indices]
            self._cf_local_indices = np.flatnonzero(valid_mask)
            if self._cf_local_indices.size > 0:
                dirs = self._ra._lens_directions[self._lens_indices[self._cf_local_indices]]
                pos = self._ra._lens_positions[self._lens_indices[self._cf_local_indices]]
                self._direction_tree_cf = cKDTree(dirs)
                self._position_tree_cf = cKDTree(pos)
            else:
                self._direction_tree_cf = None
                self._position_tree_cf = None
        return self._direction_tree_cf, self._position_tree_cf, self._cf_local_indices

    # Queries

    def neighbours(self,
            lens_indices: Optional[ArrayLike] = None,
            points: Optional[ArrayLike] = None,
            k: int = 6,
            immediate_only: bool = False,
            neighbour_dist_factor: float = 1.3,
            chirality: Optional[ArrayLike] = None,
        ) -> NeighbourResult:
        """
        k-nearest lens neighbours within this eye.

        Args:
            - lens_indices: global lens query indices (only lenses in this eye are
                queried, others are masked out).
            - points: (Q, 3) world-space query points.
            - k: number of neighbours per query.
            - immediate_only: if True, return only first lattice ring neighbours.
                The result's 'is_immediate' field is all True.
            - neighbour_dist_factor: when immediate_only=False, tags 'is_immediate'
                in the result: a neighbour at distance d is "immediate" if
                d <= neighbour_dist_factor * closest_neighbour_dist. Set to None
                to skip the tagging.
            - chirality: optional (N_animal,) array of per-lens chirality (+/-1).
                When provided and in 'lens_indices' mode, the result's
                'same_chirality' field is populated. Ignored in 'points' mode
                (queries have no lens identity to compare from).

        Returns a NeighbourResult.
        """

        if (lens_indices is None) == (points is None):
            raise ValueError("Provide either 'lens_indices' or 'points', not both")

        # Build per-eye local positions and a lookup global->local to allow mapping results back to global indices
        local_to_global = self._lens_indices

        if lens_indices is not None:
            qidx_global = np.asarray(lens_indices, dtype=np.intp).reshape(-1)
            in_this_eye = np.isin(qidx_global, local_to_global)
            valid_global = qidx_global[in_this_eye]

            # Map global to local (within this eye)
            global_to_local = {g: i for i, g in enumerate(local_to_global)}
            valid_local = np.array([global_to_local[g] for g in valid_global], dtype=np.intp)

            if immediate_only:
                graph = self._ensure_neighbour_graph(k)
                local_nidx = graph[valid_local]
                positions = self._ra._lens_positions[self._lens_indices]

                # Recompute distances because the graph only stores indices
                q_pos = positions[valid_local]
                n_pos = positions[local_nidx]
                distances = np.linalg.norm(q_pos[:, None, :] - n_pos, axis=2)

            else:
                tree = self._ensure_position_tree()
                positions = self._ra._lens_positions[self._lens_indices][valid_local]
                actual_k = min(k, len(local_to_global) - 1)
                distances, local_nidx = tree.query(positions, k=actual_k + 1)
                if local_nidx.ndim == 1:
                    local_nidx = local_nidx.reshape(-1, 1)
                    distances = distances.reshape(-1, 1)
                local_nidx = local_nidx[:, 1:]
                distances = distances[:, 1:]

            global_nidx = local_to_global[local_nidx]
            result = NeighbourResult(mask=in_this_eye, indices=global_nidx, distances=distances)

            if immediate_only:
                # all returned neighbours are by definition first ring
                result.is_immediate = np.ones_like(global_nidx, dtype=bool)
            elif neighbour_dist_factor is not None and distances.size > 0:
                closest = distances[:, 0:1]
                result.is_immediate = distances <= closest * float(neighbour_dist_factor)

            if chirality is not None and valid_global.size > 0:
                chir = np.asarray(chirality)
                query_chir = chir[valid_global]
                neigh_chir = chir[global_nidx]
                result.same_chirality = neigh_chir == query_chir[:, None]

            return result

        # Points path
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        tree = self._ensure_position_tree()
        actual_k = min(k, len(local_to_global))
        distances, local_nidx = tree.query(pts, k=actual_k)

        if local_nidx.ndim == 1:
            local_nidx = local_nidx.reshape(-1, 1)
            distances = distances.reshape(-1, 1)
        global_nidx = local_to_global[local_nidx]

        result = NeighbourResult(
            mask=np.ones(pts.shape[0], dtype=bool),
            indices=global_nidx,
            distances=distances,
        )

        if neighbour_dist_factor is not None and distances.size > 0:
            closest = distances[:, 0:1]
            result.is_immediate = distances <= closest * float(neighbour_dist_factor)
        # not possible in points mode (no query lens identity)

        return result

    def query_directions(self,
        directions: ArrayLike,
        k: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the k lenses whose optical axes lie closest (Euclidean) to each
        of the given query directions.

        Returns (indices, distances) where indices are global lens indices
        and distances are chord distances on the unit sphere.
        """

        dirs = np.asarray(directions, dtype=np.float32).reshape(-1, 3)
        tree = self._ensure_direction_tree()
        actual_k = min(k, len(self._lens_indices))
        distances, local_idx = tree.query(dirs, k=actual_k)
        if local_idx.ndim == 1:
            local_idx = local_idx.reshape(-1, 1)
            distances = distances.reshape(-1, 1)

        return self._lens_indices[local_idx], distances

    def query_positions(self,
        positions: ArrayLike,
        k: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find the k lenses closest in world-space position to each query point.

        Returns (indices, distances) of shape (Q, k) with global lens indices
        and Euclidean distances in world units.
        """
        pts = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        tree = self._ensure_position_tree()
        actual_k = min(k, len(self._lens_indices))
        distances, local_idx = tree.query(pts, k=actual_k)
        if local_idx.ndim == 1:
            local_idx = local_idx.reshape(-1, 1)
            distances = distances.reshape(-1, 1)
        return self._lens_indices[local_idx], distances

    def query_lookat(self,
        targets: ArrayLike,
        k: int = 1,
    ) -> np.ndarray:
        """
        Find the k lenses best looking at world-space target points.

        Unlike query_directions (which only matches optical-axis vectors),
        this accounts for lens *position*: the score is the dot product of
        each lens's optical axis with the unit vector from that lens to the
        target.

        Returns global lens indices of shape (Q, k), best-first.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        q = np.asarray(targets, dtype=np.float32).reshape(-1, 3)
        Q = q.shape[0]
        if len(self._lens_indices) == 0:
            return np.empty((Q, k), dtype=np.intp)

        pos = self._ra._lens_positions[self._lens_indices]
        dirs = self._ra._lens_directions[self._lens_indices]

        desired = q[:, None, :] - pos[None, :, :]
        norms = np.linalg.norm(desired, axis=-1, keepdims=True)
        np.divide(desired, norms, out=desired, where=norms > 0)
        dots = np.einsum('jk,ijk->ij', dirs, desired)

        k_eff = min(k, dots.shape[1])
        part = np.argpartition(dots, -k_eff, axis=1)[:, -k_eff:]
        top = np.take_along_axis(dots, part, axis=1)
        order = np.argsort(top, axis=1)[:, ::-1]
        best = np.take_along_axis(part, order, axis=1)
        return self._lens_indices[best]

    def query_cone(self,
        center_direction: ArrayLike,
        angle: float,
        degrees: bool = True,
    ) -> np.ndarray:
        """
        All lenses whose optical axis lies within 'angle' of 'center_direction'.

        Returns global lens indices (in arbitrary order).
        """
        c = np.asarray(center_direction, dtype=np.float32)
        n = float(np.linalg.norm(c))
        if n < 1e-12:
            return np.empty(0, dtype=np.intp)
        c = c / n
        a = np.deg2rad(angle) if degrees else float(angle)
        radius = 2.0 * np.sin(a / 2.0)  # chord on unit sphere
        tree = self._ensure_direction_tree()
        hits = tree.query_ball_point(c, r=radius)
        local = np.atleast_1d(np.asarray(hits, dtype=np.intp))
        return self._lens_indices[local]

    def query_ball(self,
        center_position: ArrayLike,
        radius: float,
    ) -> np.ndarray:
        """
        All lenses whose world-space position is within 'radius' of
        'center_position'.

        Returns global lens indices (arbitrary order).
        """
        c = np.asarray(center_position, dtype=np.float32)
        tree = self._ensure_position_tree()
        hits = tree.query_ball_point(c, r=float(radius))
        local = np.atleast_1d(np.asarray(hits, dtype=np.intp))
        return self._lens_indices[local]

    def max_gap(self) -> float:
        """
        Largest angular gap (in radians) between any lens and its nearest neighbour
        in this eye.
        """
        if len(self._lens_indices) <= 1:
            return 0.0
        tree = self._ensure_direction_tree()
        dirs = self._ra._lens_directions[self._lens_indices]
        chord_dists, _ = tree.query(dirs, k=2)
        max_chord = float(np.max(chord_dists[:, 1]))
        # Convert chord to great-circle angle
        return float(np.arccos(np.clip(1.0 - (max_chord ** 2) / 2.0, -1.0, 1.0)))

    def neighbours_by_bearing(self,
        query_lens_indices: Optional[ArrayLike] = None,
        k: int = 6,
    ) -> NeighbourResult:
        """
        First-ring lattice neighbours, ordered around each lens by their
        angular bearing in the lens's tangent frame.

        Useful when downstream code expects a consistent CCW ordering of the
        hex ring (e.g. for indexing into a per-direction filter bank).

        Args:
            - query_lens_indices: lenses to query (defaults to all lenses in this eye).
            - k: number of neighbours per query (typically 6 for a hex lattice).
        """
        result = self.neighbours(
            lens_indices=query_lens_indices, k=k, immediate_only=True
        )
        if not result:
            return result

        query_global = (np.asarray(query_lens_indices, dtype=np.intp)[result.mask]
                        if query_lens_indices is not None
                        else self._lens_indices)

        right = self._ra._local_right[query_global]
        up = self._ra._local_up[query_global]
        centre_pos = self._ra._lens_positions[query_global]

        neigh_pos = self._ra._lens_positions[result.indices]
        delta = neigh_pos - centre_pos[:, None, :]

        # Project into each home lens's tangent plane
        proj_x = np.einsum('ijk,ik->ij', delta, right)
        proj_y = np.einsum('ijk,ik->ij', delta, up)
        bearings = np.arctan2(proj_y, proj_x)

        # Sort each row by bearing
        order = np.argsort(bearings, axis=1)
        rows = np.arange(result.indices.shape[0])[:, None]
        sorted_idx = result.indices[rows, order]
        sorted_dist = result.distances[rows, order]

        is_imm = (result.is_immediate[rows, order]
                  if result.is_immediate is not None else None)
        same_chir = (result.same_chirality[rows, order]
                     if result.same_chirality is not None else None)

        return NeighbourResult(
            mask=result.mask,
            indices=sorted_idx,
            distances=sorted_dist,
            is_immediate=is_imm,
            same_chirality=same_chir,
        )

    # Directional neighbour search (per-lens, "find the neighbour along this direction")

    def _ensure_directional_graph(self, k_search: int = 8) -> dict:
        """
        Build (and cache) a directional neighbour graph for this eye.

        For each lens in this eye, finds its k_search nearest neighbours in
        direction space (chord distance on the unit sphere of optical axes)
        and projects each neighbour's direction into the lens's tangent
        frame.

        Cached on the Eye. Invalidated by _invalidate().
        """
        cached = getattr(self, '_directional_graph', None)
        if cached is not None and cached.get('k_search') == k_search:
            return cached

        n = self._lens_indices.size
        if n <= 1:
            empty = {
                'proj_x': np.zeros((n, 0), dtype=np.float32),
                'proj_y': np.zeros((n, 0), dtype=np.float32),
                'angular_sep': np.zeros((n, 0), dtype=np.float32),
                'neighbour_local_indices': np.zeros((n, 0), dtype=np.intp),
                'local_x': np.zeros((n, 3), dtype=np.float32),
                'local_y': np.zeros((n, 3), dtype=np.float32),
                'k_search': 0,
            }
            self._directional_graph = empty
            return empty

        k_eff = min(k_search, n - 1)
        tree = self._ensure_direction_tree()
        dirs = self._ra._lens_directions[self._lens_indices]
        dists, kd_idx = tree.query(dirs, k=k_eff + 1)
        if kd_idx.ndim == 1:
            kd_idx = kd_idx.reshape(-1, 1)
            dists = dists.reshape(-1, 1)
        nb_idx = kd_idx[:, 1:]
        nb_chord = dists[:, 1:]
        angular_sep = 2.0 * np.arcsin(np.clip(nb_chord / 2.0, -1.0, 1.0))

        # Tangent frame for each lens
        local_x = self._ra._local_right[self._lens_indices]
        local_y = self._ra._local_up[self._lens_indices]

        nb_dirs = dirs[nb_idx]
        delta = nb_dirs - dirs[:, None, :]
        proj_x = np.sum(delta * local_x[:, None, :], axis=2)
        proj_y = np.sum(delta * local_y[:, None, :], axis=2)

        graph = {
            'proj_x': proj_x.astype(np.float32),
            'proj_y': proj_y.astype(np.float32),
            'angular_sep': angular_sep.astype(np.float32),
            'neighbour_local_indices': nb_idx.astype(np.intp),
            'local_x': local_x.astype(np.float32),
            'local_y': local_y.astype(np.float32),
            'k_search': k_eff,
        }
        self._directional_graph = graph
        return graph

    def directed_neighbours(self,
        direction: ArrayLike,
        query_lens_indices: Optional[ArrayLike] = None,
        k: int = 1,
        coordinate: str = 'spherical',
        return_weights: bool = False,
        k_search: int = 8,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        For each lens, find the k lattice neighbour(s) lying closest to the
        given search direction.

        Useful for spatial/temporal correlation filters: "for each
        lens, get the neighbour that looks at the world point just ahead of where I look".

        Args:
            - direction: the search direction. Depends on 'coordinate':
                  - 'spherical' (default): (d_az, d_el) in radians. The
                    search direction is built from the gradient of each
                    lens's azimuth & elevation.
                    +d_az points right (towards larger azimuth) in the tangent plane
                    +d_el points up
                  - 'cartesian': (dx, dy, dz) world-space direction. The
                    world vector is projected into each lens's tangent
                    plane.
            - query_lens_indices: lenses to query (defaults to all in this
                eye). Returned indices are global.
            - k: number of neighbours per lens.
            - coordinate: 'spherical' or 'cartesian'. (see above).
            - return_weights: if True, also return the magnitude of the target direction
                in the tangent plane (small magnitude = the search direction was nearly
                parallel to the lens's optical axis, so the projected target is poorly defined).
            - k_search: number of candidate lattice neighbours to consider
                per lens before angular filtering. Higher = more thorough, slightly slower.

        Returns:
            indices: (Q,) if k == 1 else (Q, k) global lens indices of the
                chosen neighbours.
            weights: (Q,) or (Q, k), only if return_weights=True.
        """
        graph = self._ensure_directional_graph(k_search)
        n_eye = self._lens_indices.size

        if query_lens_indices is None:
            valid_local = np.arange(n_eye, dtype=np.intp)
        else:
            qidx_global = np.asarray(query_lens_indices, dtype=np.intp).reshape(-1)
            global_to_local = {int(g): i for i, g in enumerate(self._lens_indices)}
            in_eye_mask = np.array([int(g) in global_to_local for g in qidx_global], dtype=bool)
            valid_global = qidx_global[in_eye_mask]
            valid_local = np.array([global_to_local[int(g)] for g in valid_global], dtype=np.intp)

        Q = valid_local.size
        if Q == 0 or graph['k_search'] == 0:
            if k == 1:
                empty = np.empty(0, dtype=np.intp)
            else:
                empty = np.empty((0, k), dtype=np.intp)
            if return_weights:
                return empty, np.empty(empty.shape, dtype=np.float32)
            return empty

        local_x = graph['local_x'][valid_local]
        local_y = graph['local_y'][valid_local]

        # target direction(s) in each lens's tangent plane
        direction = np.asarray(direction, dtype=np.float32)

        if coordinate == 'spherical':
            if direction.shape != (2,):
                raise ValueError(
                    f"spherical 'direction' must have shape (2,) = (d_az, d_el), got {direction.shape}"
                )
            d_az, d_el = float(direction[0]), float(direction[1])
            dirs = self._ra._lens_directions[self._lens_indices][valid_local]

            # Convention: az = arctan2(x, -z) (i.e. +z = forward, +x = right, +y = up)
            # TODO: This is wrong, coords are OpenGL, forward is -Z
            az = np.arctan2(dirs[:, 0], -dirs[:, 2])
            el = np.arcsin(np.clip(dirs[:, 1], -1.0, 1.0))
            cos_az, sin_az = np.cos(az), np.sin(az)
            cos_el, sin_el = np.cos(el), np.sin(el)

            # Gradients of position with respect to az and el on the unit sphere
            az_grad = np.column_stack([cos_az * cos_el, np.zeros(Q), sin_az * cos_el])
            el_grad = np.column_stack([-sin_az * sin_el, cos_el, cos_az * sin_el])
            target_world = d_az * az_grad + d_el * el_grad

            target_dx = np.sum(target_world * local_x, axis=1)
            target_dy = np.sum(target_world * local_y, axis=1)

        elif coordinate == 'cartesian':
            if direction.shape != (3,):
                raise ValueError(
                    f"Cartesian 'direction' must have shape (3,), got {direction.shape}"
                )
            target_dx = local_x @ direction
            target_dy = local_y @ direction

        else:
            raise ValueError(
                f"Coordinate must be 'spherical' or 'cartesian', got {coordinate!r}"
            )

        target_norms = np.sqrt(target_dx ** 2 + target_dy ** 2)
        zero_mask = target_norms < 1e-12
        target_dx_n = np.where(zero_mask, 1.0, target_dx / np.where(zero_mask, 1.0, target_norms))
        target_dy_n = np.where(zero_mask, 0.0, target_dy / np.where(zero_mask, 1.0, target_norms))
        target_angle = np.arctan2(target_dy_n, target_dx_n)

        # Score candidate neighbours by angular deviation from the target
        nb_proj_x = graph['proj_x'][valid_local]
        nb_proj_y = graph['proj_y'][valid_local]
        nb_local = graph['neighbour_local_indices'][valid_local]

        nb_angles = np.arctan2(nb_proj_y, nb_proj_x)
        angle_diff = (nb_angles - target_angle[:, None] + np.pi) % (2 * np.pi) - np.pi
        score = np.abs(angle_diff)
        # Anything more than 90 deg off is rejected
        score = np.where(score > (np.pi / 2.0), 1e6, score)

        k_eff = min(k, score.shape[1])
        if k_eff <= 0:
            # No candidates available, return zeros
            if k == 1:
                indices = np.zeros(Q, dtype=np.intp)
            else:
                indices = np.zeros((Q, k), dtype=np.intp)
            if return_weights:
                w = target_norms if k == 1 else np.tile(target_norms[:, None], (1, k))
                return indices, w
            return indices

        if k_eff == 1:
            best = np.argmin(score, axis=1)
            local_winners = nb_local[np.arange(Q), best]
            indices = self._lens_indices[local_winners]
            if k > 1:
                # Pad with the same winner (degenerate case, k_search<k)
                indices = np.tile(indices[:, None], (1, k))
        else:
            top_k_local = np.argpartition(score, k_eff - 1, axis=1)[:, :k_eff]
            top_scores = np.take_along_axis(score, top_k_local, axis=1)
            order = np.argsort(top_scores, axis=1)
            top_k_sorted = np.take_along_axis(top_k_local, order, axis=1)
            local_winners = np.take_along_axis(nb_local, top_k_sorted, axis=1)
            indices = self._lens_indices[local_winners]
            if k > k_eff:
                # Pad missing columns with the first (best) winner
                pad = np.tile(indices[:, 0:1], (1, k - k_eff))
                indices = np.concatenate([indices, pad], axis=1)

        if return_weights:
            w = target_norms if k == 1 else np.tile(target_norms[:, None], (1, k))
            return indices, w.astype(np.float32)
        return indices


class VisualOutput:
    """
    Per-receptor output array with reshaping via the (N, R) cartridge map.

    The renderers return a (N, 4) float array where N is the total
    receptor count and the last axis is (R/UV, G, B, radiance), with radiance the
    mean of the colour channels.

    Layouts:
        .per_lens       -> (n_lenses, R, ...)      regular grid (physical grouping)
        .per_cartridge  -> (n_cartridges, R, ...)  regular grid (neural grouping)
        .colours        -> data[..., :3]           on any layout via the channel helpers
        .radiance       -> data[..., 3]            on any layout via the channel helpers
    """
    __slots__ = ('_data', '_ra', '_R', '_N')

    def __init__(self, data: np.ndarray, ra: 'ReceptorArray'):
        if data.shape[0] % ra.receptors_per_lens != 0:
            raise ValueError(f"data length {data.shape[0]} not divisible by R={ra.receptors_per_lens}")

        self._data = data
        self._ra = ra
        self._R = ra.receptors_per_lens
        self._N = data.shape[0] // self._R

    @property
    def data(self) -> np.ndarray:
        """The raw per-receptor array."""
        return self._data

    @property
    def lens_count(self) -> int:
        return self._N

    @property
    def receptors_per_lens(self) -> int:
        return self._R

    @property
    def cartridge_count(self) -> int:
        return self._N if self._ra._cartridges_wired else 0

    # Channel helpers

    @property
    def colours(self) -> np.ndarray:
        if self._data.ndim < 2 or self._data.shape[-1] < 3:
            raise ValueError(f"Colours requires last axis >= 3, got shape {self._data.shape}")
        return self._data[..., :3]

    @property
    def radiance(self) -> np.ndarray:
        if self._data.ndim < 2 or self._data.shape[-1] < 4:
            raise ValueError(f"Radiance requires last axis >= 4, got shape {self._data.shape}")
        return self._data[..., 3]

    # Group layouts

    @property
    def per_lens(self) -> np.ndarray:
        """Reshape to (N, R, ...). One block of R rows per physical lens."""
        return self._data.reshape(self._N, self._R, *self._data.shape[1:])

    @property
    def per_cartridge(self) -> np.ndarray:
        """
        (N, R, ...) gather of receptor outputs by cartridge.
        """
        if not self._ra._cartridges_wired:
            # return self.per_lens # fallback to physical grouping if not wired
            raise ValueError("VisualOutput has no cartridge wiring.")
        return self._data[self._ra.cartridge_indices]

    @property
    def cartridge_central_indices(self) -> np.ndarray:
        """(N,) global receptor index of each cartridge's central (R7/8)."""
        c = self._ra._kernel.center_index
        return np.arange(self._N, dtype=np.intp) * self._R + c

    @property
    def central_per_cartridge(self) -> np.ndarray:
        """(N, ...) output of just the central receptor of each cartridge."""
        c = self._ra._kernel.center_index
        return self.per_cartridge[:, c, ...]

    def central(self, kernel: 'RhabdomereKernel') -> np.ndarray:
        """Just the central rhabdomere output per lens. Shape (N, ...)."""
        return self.per_lens[:, kernel.center_index]

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self) -> int:
        return int(self._data.shape[0])

    def __repr__(self) -> str:
        c_str = f", cartridges={self._N}" if self._ra._cartridges_wired else ""
        return f"VisualOutput(N={self._N}, R={self._R}{c_str}, shape={self._data.shape})"