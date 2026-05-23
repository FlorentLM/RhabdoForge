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
from typing import Optional, Tuple, TYPE_CHECKING
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
    """
    mask: np.ndarray
    indices: np.ndarray
    distances: np.ndarray

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

    def __len__(self) -> int:
        return self._gi.size

    def _require_single(self, name: str) -> int:
        if len(self) != 1:
            raise ValueError(
                f"'{name}' is a singular accessor and requires a size-1 LensView "
                f"(got n={self._gi.size}). Use the plural '{name}s' or index "
                f"with an int first."
            )
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
            raise ValueError(
                f"LensView spans {len(np.unique(ids))} eyes, "
                "use ra.eyes or 'lens.eye_ids' for mixed-eye views"
            )
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
    def cartridge(self) -> Optional['Cartridge']:
        """The cartridge anchored at this lens's central rhabdomere (size-1 view only, if wired)."""
        idx = self._require_single('cartridge')
        if not self._ra._cartridges_wired:
            return None
        return Cartridge(self._ra, idx)


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
                raise IndexError(
                    f"receptor indices out of range for ReceptorArray with "
                    f"{ra.total_receptors} receptors"
                )

    # Basics

    def __len__(self) -> int:
        return int(self._gi.size)

    def __repr__(self) -> str:
        return f"ReceptorView(n={len(self)}, indices={self._gi[:8]}{'...' if len(self) > 8 else ''})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, ReceptorView):
            return NotImplemented
        return self._ra is other._ra and np.array_equal(self._gi, other._gi)

    def __getitem__(self, idx) -> 'ReceptorView':
        if isinstance(idx, (int, np.integer)):
            return ReceptorView(self._ra, self._gi[int(idx):int(idx) + 1])
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

        # Uses precomputed inverse mapping if wire_cartridges has filled it in
        # or falls back to a linear scan otherwise.
        my_central = self.central_receptor_index
        cached = getattr(self._ra, '_cartridge_members', None)
        if cached is not None and my_central in cached:
            self._member_indices = cached[my_central]
        else:
            self._member_indices = np.flatnonzero(
                self._ra.rcpt_static_data['cartridge_src'] == my_central
            ).astype(np.intp)

    def __len__(self) -> int:
        return int(self._member_indices.size)

    def __repr__(self) -> str:
        return f"Cartridge(lens={self._central_lens_idx}, members={len(self)})"

    @property
    def lens(self) -> LensView:
        """The home (central) ommatidium for this cartridge."""
        return LensView(self._ra, np.array([self._central_lens_idx], dtype=np.intp))

    @property
    def central_receptor_index(self) -> int:
        R = self._ra.receptors_per_lens
        c = self._ra._kernel.center_index
        return self._central_lens_idx * R + c

    @property
    def central_receptor(self) -> ReceptorView:
        return ReceptorView(
            self._ra, np.array([self.central_receptor_index], dtype=np.intp)
        )

    @property
    def receptors(self) -> ReceptorView:
        """All member receptors (the R1-R6 contributors from neighbouring lenses)."""
        return ReceptorView(self._ra, self._member_indices)

    @property
    def sources(self) -> np.ndarray:
        """
        (n_members,) global lens index of the source ommatidium for each member.

        That is, receptor 'members[i]' lives in 'ra.lenses[sources[i]]'.
        """
        R = self._ra.receptors_per_lens
        return (self._member_indices // R).astype(np.intp)


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
        '_position_tree', '_direction_tree', '_neighbour_graph', '_neighbour_k',
    )

    def __init__(self,
        ra: 'ReceptorArray',
        eye_id: int,
        lens_indices: np.ndarray,
        side: str = 'left',
    ):
        self._ra = ra
        self._eye_id = int(eye_id)
        self._lens_indices = np.asarray(lens_indices, dtype=np.intp)
        self._side = str(side)
        self._position_tree: Optional[cKDTree] = None
        self._direction_tree: Optional[cKDTree] = None
        self._neighbour_graph: Optional[np.ndarray] = None
        self._neighbour_k: int = -1

    def __repr__(self) -> str:
        return f"Eye(id={self._eye_id}, side={self._side!r}, n_lenses={len(self._lens_indices)})"

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
        self._neighbour_graph = None
        self._neighbour_k = -1

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

    # Queries

    def neighbours(self,
            lens_indices: Optional[ArrayLike] = None,
            points: Optional[ArrayLike] = None,
            k: int = 6,
            immediate_only: bool = False,
        ) -> NeighbourResult:
        """
        k-nearest lens neighbours within this eye.

        Args:
            - lens_indices: global lens query indices (only those in this eye are queried, others are masked out)
            - points: (Q, 3) world-space query points
            - immediate_only: Equivalent to passing the precomputed lattice graph for 'k'.
                Useful for downstream code that wants "hexagonal-immediate" semantics for boundary lenses too.

        Returns a NeighbourResult with .mask, .indices (global), .distances.
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
            return NeighbourResult(mask=in_this_eye, indices=global_nidx, distances=distances)

        # Points path
        pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        tree = self._ensure_position_tree()
        actual_k = min(k, len(local_to_global))
        distances, local_nidx = tree.query(pts, k=actual_k)

        if local_nidx.ndim == 1:
            local_nidx = local_nidx.reshape(-1, 1)
            distances = distances.reshape(-1, 1)
        global_nidx = local_to_global[local_nidx]

        return NeighbourResult(
            mask=np.ones(pts.shape[0], dtype=bool),
            indices=global_nidx,
            distances=distances,
        )

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

    def directed_neighbours(self,
        query_lens_indices: Optional[ArrayLike] = None,
        k: int = 6,
    ) -> NeighbourResult:
        """
        Lattice-aware neighbours: KNN in position space, but with results
        ordered by angular bearing within each lens's tangent frame.

        Useful when downstream code expects a consistent neighbour ordering around the hex.
        """
        result = self.neighbours(lens_indices=query_lens_indices, k=k, immediate_only=True)
        if not result:
            return result

        # Compute angular bearing of each neighbour in the home lens's tangent frame

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

        return NeighbourResult(mask=result.mask, indices=sorted_idx, distances=sorted_dist)


# TODO: VisualOutput probably should be a thinner wrapper and just take a ra and a data array to return the correct slices

class VisualOutput:
    """
    Per-receptor output array with convenience reshape and indexing.

    The renderers return a (N, 4) float array where N is the total
    receptor count and the last axis is (R/UV, G, B, radiance), with radiance the
    mean of the colour channels.

    Layouts:
        .per_lens       -> (n_lenses, R, ...)        regular grid
        .per_cartridge  -> (n_cartridges, K, ...)    K = max members, padded
        .cartridge_mask -> (n_cartridges, K)         bool, True where valid
        .colours        -> data[..., :3]             on any layout via the channel helpers
        .radiance       -> data[..., 3]              on any layout via the channel helpers

    Convenience:
        .for_lens(i)       -> single lens, shape (R, ...)
        .for_cartridge(i)  -> single cartridge, ragged, shape (n_members_i, ...)
        .central(kernel)   -> central rhabdomere per lens, shape (n_lenses, ...)
    """

    __slots__ = (
        '_data', '_N', '_R',
        '_cartridge_index_map', '_cartridge_mask',
        '_cartridge_centrals',
    )

    def __init__(self,
        data: np.ndarray,
        receptors_per_lens: int,
        cartridge_index_map: Optional[np.ndarray] = None,
        cartridge_mask: Optional[np.ndarray] = None,
        cartridge_centrals: Optional[np.ndarray] = None,
    ):
        """
        Args:
            - data: (N_total, ...) per-receptor output. Typically (N_total, 4).
            - receptors_per_lens: R, so n_lenses = N_total / R.
            - cartridge_index_map: (n_cartridges, K) global receptor indices
                per cartridge, padded with 0 in unused slots. Optional.
            - cartridge_mask: (n_cartridges, K) bool, True where index_map is real.
            - cartridge_centrals: (n_cartridges,) global receptor index of the
                central (R7/8) member of each cartridge. Used to pick out the
                anchor in central_per_cartridge().
        """
        if data.shape[0] % receptors_per_lens != 0:
            raise ValueError(
                f"data length {data.shape[0]} not divisible by R={receptors_per_lens}"
            )
        self._data = data
        self._R = int(receptors_per_lens)
        self._N = data.shape[0] // self._R
        self._cartridge_index_map = cartridge_index_map
        self._cartridge_mask = cartridge_mask
        self._cartridge_centrals = cartridge_centrals

    @classmethod
    def from_receptor_array(cls,
        data: np.ndarray,
        ra: 'ReceptorArray',
    ) -> 'VisualOutput':
        """
        Build a VisualOutput from a renderer output and the parent ReceptorArray,
        with cartridge wiring (if present) baked in.
        """
        R = ra.receptors_per_lens
        if not getattr(ra, '_cartridges_wired', False):
            return cls(data, R)

        members_by_central = ra._cartridge_members  # dict[int -> (k,) intp]

        # Order cartridges by central receptor index for deterministic layout
        centrals = np.array(sorted(members_by_central.keys()), dtype=np.intp)
        max_k = max((members_by_central[int(c)].size for c in centrals), default=0)
        n_carts = centrals.size

        idx_map = np.zeros((n_carts, max_k), dtype=np.intp)
        mask = np.zeros((n_carts, max_k), dtype=bool)
        for i, c in enumerate(centrals):
            members = members_by_central[int(c)]
            idx_map[i, :members.size] = members
            mask[i, :members.size] = True

        return cls(
            data,
            receptors_per_lens=R,
            cartridge_index_map=idx_map,
            cartridge_mask=mask,
            cartridge_centrals=centrals,
        )

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
        return 0 if self._cartridge_index_map is None else int(self._cartridge_index_map.shape[0])

    # Channel helpers

    @property
    def colours(self) -> np.ndarray:
        if self._data.ndim < 2 or self._data.shape[-1] < 3:
            raise ValueError(f"colours requires last axis >= 3, got shape {self._data.shape}")
        return self._data[..., :3]

    @property
    def radiance(self) -> np.ndarray:
        if self._data.ndim < 2 or self._data.shape[-1] < 4:
            raise ValueError(f"radiance requires last axis >= 4, got shape {self._data.shape}")
        return self._data[..., 3]

    # Group layouts

    @property
    def per_lens(self) -> np.ndarray:
        """Reshape to (N, R, ...). One block of R rows per lens."""
        return self._data.reshape(self._N, self._R, *self._data.shape[1:])

    @property
    def per_cartridge(self) -> np.ndarray:
        """
        Padded (n_cartridges, K, ...) gather of receptor outputs by cartridge.
        Slots beyond a cartridge's true member count are zero; check 'cartridge_mask'.

        Raises if cartridge wiring was not supplied at construction.
        """
        if self._cartridge_index_map is None:
            raise ValueError(
                "VisualOutput has no cartridge wiring. Build via "
                "VisualOutput.from_receptor_array(data, ra) or pass "
                "cartridge_index_map at construction."
            )
        gathered = self._data[self._cartridge_index_map]
        # Zero the padded slots so reductions (sum / mean with care) behave
        if self._cartridge_mask is not None and gathered.ndim > self._cartridge_mask.ndim:
            extra = gathered.ndim - self._cartridge_mask.ndim
            m = self._cartridge_mask.reshape(self._cartridge_mask.shape + (1,) * extra)
            gathered = gathered * m
        return gathered

    @property
    def cartridge_mask(self) -> np.ndarray:
        """(n_cartridges, K) bool: True where per_cartridge holds a real member."""
        if self._cartridge_mask is None:
            raise ValueError("VisualOutput has no cartridge wiring.")
        return self._cartridge_mask

    @property
    def cartridge_central_indices(self) -> np.ndarray:
        """(n_cartridges,) global receptor index of each cartridge's central (R7/8)."""
        if self._cartridge_centrals is None:
            raise ValueError("VisualOutput has no cartridge wiring.")
        return self._cartridge_centrals

    @property
    def central_per_cartridge(self) -> np.ndarray:
        """
        (n_cartridges, ...) output of just the central receptor of each cartridge.
        Equivalent to 'data[cartridge_central_indices]'.
        """
        return self._data[self.cartridge_central_indices]

    # Per-element gather

    def for_lens(self, lens_indices) -> np.ndarray:
        """Output for lens(es) 'lens_indices' (int or array). Shape (R, ...) or (M, R, ...)."""
        return self.per_lens[lens_indices]

    def for_cartridge(self, cartridge_index: int) -> np.ndarray:
        """
        Output for one cartridge, with padding stripped. Shape (n_members_i, ...).
        """
        if self._cartridge_index_map is None:
            raise ValueError("VisualOutput has no cartridge wiring.")
        i = int(cartridge_index)
        if self._cartridge_mask is not None:
            idx = self._cartridge_index_map[i][self._cartridge_mask[i]]
        else:
            idx = self._cartridge_index_map[i]
        return self._data[idx]

    def central(self, kernel: 'RhabdomereKernel') -> np.ndarray:
        """Just the central rhabdomere output per lens. Shape (N, ...)."""
        return self.per_lens[:, kernel.center_index]

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self) -> int:
        return int(self._data.shape[0])

    def __repr__(self) -> str:
        carts = f", cartridges={self.cartridge_count}" if self._cartridge_index_map is not None else ""
        return f"VisualOutput(N={self._N}, R={self._R}{carts}, shape={self._data.shape})"
