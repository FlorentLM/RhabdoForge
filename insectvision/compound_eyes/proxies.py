"""
User-facing views of CompoundEyeModel data buffers.

Views (LensView, ReceptorView, Cartridge) are slice-like handles into the array.
They hold a reference to the parent CompoundEyeModel and a '_gi' array of global indices into its data buffers.

Hierarchy:
    LensView                 # M lenses, iterable yields size-1 LensViews
                             # size-1 LensViews expose singular accessors
    ReceptorView             # M receptors
    Cartridge                # central lens + R1-R6 members from neighbours
    Eye                      # KDtrees + neighbour graph for one anatomical eye,
                             # with a 'side' label ('left' / 'right' / 'midline')
    VisualOutput             # per-receptor output array with reshape helpers
                             # (per-lens, per-cartridge, plus colours / radiance slicing)
    CompoundEyeModel         # The complete model view (whole-animal, multiple eyes)

Notes:
    - A LensView spanning multiple eyes raises on eye-dependent operations,
    use 'eye.lenses' directly for eye-scoped queries.

    - LensView geometry is read-only. Only global transforms with model.translate/scale/rotate are possible.

    - ReceptorView allows setting cell-level properties (sensitivity, tau_membrane, acceptance)
    and the actuated 'directions' field (it is dynamic state).

    - Receptor positions and rest directions are derived from lens position and
    the orientation pipeline, so they can't be set directly.
"""
import logging
from typing import Optional, Tuple, Union, List
import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.optimize import linear_sum_assignment, milp, LinearConstraint, Bounds

from insectvision.compound_eyes.buffers import EyesBuffer, get_metadata_field, set_metadata_field
from insectvision.compound_eyes.orientation import (
    BundlesAligner, trivial_orientation, apply_chirality, OrientationResult
)
from insectvision.compound_eyes.acceptance import (
    AcceptanceModel, LensOptics, ReceptorOptics,
    SnyderAcceptance, SamplingAcceptance, ExplicitAcceptance, report_acceptance,
)
from insectvision.engine.world_utils import WORLD_UP, WORLD_FORWARD
from insectvision.utils.math import normalise_vectors, tangent_frames, icosphere, fibonacci_sphere
from insectvision.utils.hexatic import hexatic_phasor, hexatic_axis_angle, hexatic_order
from insectvision.compound_eyes.lattice import (
    facet_diameters, angular_neighbour_median,
)
from insectvision.compound_eyes.rhabdomeres import RhabdomereBundle

logger = logging.getLogger(__name__)


class NeighbourResult:
    """
    Result of an Eye.neighbours() query.

    Attributes:

    mask: (Q,) bool array, Which input query points were inside this eye and got results.
    indices: (M, k) int array, Global lens indices of the k nearest neighbours, M = mask.sum().
    distances: (M, k) float array, Distances to those neighbours.
    is_immediate: (M, k) bool array (optional), True where the neighbour is in the
        first lattice ring of the query lens (distance <= neighbour_dist_factor * closest).
    same_chirality: (M, k) bool array (optional), True where the neighbour's chirality
        matches the query lens's chirality.
    """

    def __init__(self, eye, mask, indices, distances, is_immediate=None, same_chirality=None):
        self.eye = eye
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
    def lenses(self) -> 'LensView':
        """Flattened LensView of all neighbours in the result set."""
        return LensView(self._model, self.indices.ravel())

    @property
    def receptors(self) -> 'ReceptorView':
        """Flattened ReceptorView of all receptors in all neighbour lenses."""
        return self.lenses.receptors

class LensView:
    """
    A subset of M lenses in a CompoundEyeModel.
    """

    __slots__ = ('_model', '_gi')
    __hash__ = None     # disable hashing, views are mutable handles, not values

    def __init__(self, model: 'CompoundEyeModel', indices: ArrayLike):
        self._model = model
        self._gi = np.asarray(indices, dtype=np.intp).reshape(-1)

        if self._gi.size > 0:
            if int(self._gi.min()) < 0 or int(self._gi.max()) >= model.lens_count:
                raise IndexError(f"Lens indices out of range for CompoundEyeModel with {model.lens_count} lenses")

    # Basics

    def __len__(self) -> int:
        return int(self._gi.size)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n={len(self)}, indices={self._gi[:8]}{'...' if len(self) > 8 else ''})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, LensView):
            return NotImplemented
        return self._model is other._model and np.array_equal(self._gi, other._gi)

    def __iter__(self):
        # Iterating yields size-1 LensViews (i.e. ommatidia)
        for i in self._gi:
            yield LensView(self._model, np.array([i], dtype=np.intp))

    def __getitem__(self, idx) -> 'LensView':
        sub_indices = self._gi[idx]
        # Handle single integer access: numpy returns a scalar, but we want
        # LensView to always contain an array for the constructor
        if np.isscalar(sub_indices):
            sub_indices = np.array([sub_indices], dtype=np.intp)
        return LensView(self._model, sub_indices)

    # Helpers

    def _validate_write(self):
        if not self._model.buffer._allow_lens_writes:
            raise RuntimeError("Attempted to modify LensView but CPU-side lens writes are locked.")

    def _mark_dirty(self):
        self._model.buffer.lens_dirty = True
        self._model.buffer.lens_dirty_mask[self._gi] = True

    # Properties

    @property
    def global_indices(self) -> np.ndarray:
        """Global lens indices into the CompoundEyeModel (read-only copy)."""
        return self._gi.copy()

    @property
    def indices(self) -> np.ndarray:
        """Alias to global_indices."""
        return self.global_indices

    # Geometry (read-only)

    @property
    def position(self) -> np.ndarray:
        """(M, 3) lens world positions."""
        return self._model._lens_positions[self._gi].copy()

    @property
    def direction(self) -> np.ndarray:
        """(M, 3) lens optical-axis (forward) unit vectors."""
        return self._model._lens_directions[self._gi].copy()

    @property
    def right_local(self) -> np.ndarray:
        """(M, 3) tangent 'right' vectors (lens-local x)."""
        return self._model._local_right[self._gi].copy()

    @property
    def up_local(self) -> np.ndarray:
        """(M, 3) tangent 'up' vectors (lens-local y)."""
        return self._model._local_up[self._gi].copy()

    @property
    def azimuth_rad(self) -> np.ndarray:
        """(M,) lens azimuths (rad)."""
        d = self.direction
        return np.arctan2(d[:, 0], -d[:, 2])

    @property
    def azimuth_deg(self) -> np.ndarray:
        return np.degrees(self.azimuth_rad)

    @property
    def elevation_rad(self) -> np.ndarray:
        """(M,) lens elevations (rad)."""
        return np.arcsin(np.clip(self.direction[:, 1], -1.0, 1.0))

    @property
    def elevation_deg(self) -> np.ndarray:
        return np.degrees(self.elevation_rad)

    # Lattice geometry

    @property
    def ioa_angles(self) -> np.ndarray:
        """(M, 2) per-lens (minor, major) interommatidial angles (rad)."""
        return self._model.lens_static_data['ioa_axes'][self._gi].copy()

    @ioa_angles.setter
    def ioa_angles(self, value: ArrayLike):
        self._validate_write()
        self._model.lens_static_data['ioa_axes'][self._gi] = value
        self._mark_dirty()

    @property
    def ioa_tilt(self) -> np.ndarray:
        """(M,) per-lens hex-lattice tilts (rad)."""
        return self._model.lens_static_data['ioa_tilt'][self._gi].copy()

    @ioa_tilt.setter
    def ioa_tilt(self, value: ArrayLike):
        self._validate_write()
        self._model.lens_static_data['ioa_tilt'][self._gi] = value
        self._mark_dirty()

    @property
    def diameter_um(self) -> np.ndarray:
        """(M,) lens apertures (μm)."""
        return self._model.lens_static_data['lens_diameter_um'][self._gi].copy()

    @diameter_um.setter
    def diameter_um(self, value: ArrayLike):
        self._validate_write()
        self._model.lens_static_data['lens_diameter_um'][self._gi] = value
        self._mark_dirty()

    @property
    def focal_um(self) -> np.ndarray:
        """(M,) lens-to-rhabdomere lever arms (μm)."""
        return self._model.lens_static_data['focal_um'][self._gi].copy()

    @focal_um.setter
    def focal_um(self, value: ArrayLike):
        self._validate_write()
        self._model.lens_static_data['focal_um'][self._gi] = value
        self._mark_dirty()

    # Bundle orientation
    # TODO: Setters

    @property
    def bundle_orientation(self) -> np.ndarray:
        """(M,) per-lens bundle yaw (chi, rad)."""
        return self._model._bundle_orientation[self._gi].copy()

    @property
    def chirality(self) -> np.ndarray:
        """(M,) per-lens chirality (+1 or -1)."""
        return self._model._chirality_arr[self._gi].copy()

    @property
    def saccade_axes_local(self) -> np.ndarray:
        """(M, 2) saccade actuation axis in each lens (right, up) tangent frame."""
        d = self._model.lens_static_data
        return np.column_stack([d['sacc_x'][self._gi], d['sacc_y'][self._gi]])

    @property
    def saccade_axes(self) -> np.ndarray:
        """(M, 3) saccade actuation axis in world coordinates."""
        sx = self._model.lens_static_data['sacc_x'][self._gi][:, None]
        sy = self._model.lens_static_data['sacc_y'][self._gi][:, None]
        return sx * self._model._local_right[self._gi] + sy * self._model._local_up[self._gi]

    # Photomechanical biophysics (per-lens, broadcast from bundle)

    @property
    def tau_rise(self) -> np.ndarray:
        return self._model.lens_static_data['tau_rise'][self._gi].copy()

    @tau_rise.setter
    def tau_rise(self, value: ArrayLike):
        self._validate_write()
        self._model.lens_static_data['tau_rise'][self._gi] = value
        self._mark_dirty()

    @property
    def tau_relax(self) -> np.ndarray:
        return self._model.lens_static_data['tau_relax'][self._gi].copy()

    @tau_relax.setter
    def tau_relax(self, value: ArrayLike):
        self._validate_write()
        self._model.lens_static_data['tau_relax'][self._gi] = value
        self._mark_dirty()

    @property
    def tau_fast(self) -> np.ndarray:
        return self._model.lens_static_data['tau_fast'][self._gi].copy()

    @tau_fast.setter
    def tau_fast(self, value: ArrayLike):
        self._validate_write()
        self._model.lens_static_data['tau_fast'][self._gi] = value
        self._mark_dirty()

    @property
    def tau_adapt(self) -> np.ndarray:
        return self._model.lens_static_data['tau_adapt'][self._gi].copy()

    @tau_adapt.setter
    def tau_adapt(self, value: ArrayLike):
        self._validate_write()
        self._model.lens_static_data['tau_adapt'][self._gi] = value
        self._mark_dirty()

    @property
    def ampl_lat_um(self) -> np.ndarray:
        return self._model.lens_static_data['ampl_lat_um'][self._gi].copy()

    @ampl_lat_um.setter
    def ampl_lat_um(self, value: ArrayLike):
        self._validate_write()
        self._model.lens_static_data['ampl_lat_um'][self._gi] = value
        self._mark_dirty()

    @property
    def ampl_ax_um(self) -> np.ndarray:
        return self._model.lens_static_data['ampl_ax_um'][self._gi].copy()

    @ampl_ax_um.setter
    def ampl_ax_um(self, value: ArrayLike):
        self._validate_write()
        self._model.lens_static_data['ampl_ax_um'][self._gi] = value
        self._mark_dirty()

    # Dynamic state

    @property
    def adapted_luminance(self) -> np.ndarray:
        return self._model.lens_dynamic_data['adapted_lum'][self._gi].copy()

    @adapted_luminance.setter
    def adapted_luminance(self, value: ArrayLike):
        self._validate_write()
        self._model.lens_dynamic_data['adapted_lum'][self._gi] = value
        self._mark_dirty()

    @property
    def fast_luminance(self) -> np.ndarray:
        return self._model.lens_dynamic_data['fast_lum'][self._gi].copy()

    @fast_luminance.setter
    def fast_luminance(self, value: ArrayLike):
        self._validate_write()
        self._model.lens_dynamic_data['fast_lum'][self._gi] = value
        self._mark_dirty()

    @property
    def lateral_displacement_um(self) -> np.ndarray:
        return self._model.lens_dynamic_data['lateral_um'][self._gi].copy()

    @lateral_displacement_um.setter
    def lateral_displacement_um(self, value: ArrayLike):
        self._validate_write()
        self._model.lens_dynamic_data['lateral_um'][self._gi] = value
        self._mark_dirty()

    @property
    def axial_displacement_um(self) -> np.ndarray:
        return self._model.lens_dynamic_data['axial_um'][self._gi].copy()

    @axial_displacement_um.setter
    def axial_displacement_um(self, value: ArrayLike):
        self._validate_write()
        self._model.lens_dynamic_data['axial_um'][self._gi] = value
        self._mark_dirty()

    # Diagnostics, conflicts

    @property
    def donation_conflicts(self) -> np.ndarray:
        """(M,) True if any peripheral receptor in this lens is donated to != 1 cartridge."""
        return self._model.donation_conflicts[self._gi].copy()

    @property
    def receiving_conflicts(self) -> np.ndarray:
        """(M,) True if this lens's cartridge failed to gather a neighbour for any slot."""
        return self._model.receiving_conflicts[self._gi].copy()

    @property
    def have_conflicts(self) -> np.ndarray:
        """(M,) True if the lens has a donation or receiving conflict."""
        return self._model.have_conflicts[self._gi].copy()

    @property
    def is_binocular(self) -> np.ndarray:
        """(M,) bool mask identifying which lenses in this view are binocular."""
        if self._gi.size == 0:
            return np.empty(0, dtype=bool)

        # look up metadata for the 1st receptor of each lens in the view
        R = self._model.receptors_per_lens
        meta = self._model.rcpt_static_data['metadata'][self._gi * R]
        return get_metadata_field(meta, 'binocular_area').astype(bool)

    @property
    def binocular_fraction(self) -> float:
        """Fraction of lenses in this specific view that are binocular (0.0 to 1.0)."""
        mask = self.is_binocular
        return float(np.mean(mask)) if mask.size > 0 else 0.0

    # Linking to receptors / eye

    @property
    def receptors(self) -> 'ReceptorView':
        """The R*M receptors behind these lenses (R = receptors_per_lens)."""
        R = self._model.receptors_per_lens
        rcpt_indices = (self._gi[:, None] * R + np.arange(R, dtype=np.intp)[None, :]).ravel()
        return ReceptorView(self._model, rcpt_indices)

    @property
    def eye_index(self) -> np.ndarray:
        """(M,) eye id (0-7) of each lens."""
        return self._model._lens_eye_index[self._gi].copy()

    @property
    def side(self) -> np.ndarray:
        """(M,) per-lens side string ('left' / 'right' / 'midline') from each parent eye."""
        side_by_eid = {e.eye_index: e.side for e in self._model._eyes}
        ids = self._model._lens_eye_index[self._gi]
        return np.array([side_by_eid.get(int(i), 'unknown') for i in ids], dtype=object)

    @property
    def side_sign(self) -> np.ndarray:
        """(M,) signs of the lenses (Left=1, Right=-1)."""
        return np.array([self._model.eye(int(i)).side_sign for i in self.eye_index], dtype=np.float32)

    @property
    def eye(self) -> 'Eye':
        """The single Eye these lenses belong to. Raises if the view spans multiple eyes."""
        ids = self._model._lens_eye_index[self._gi]
        if ids.size == 0:
            raise ValueError("Empty LensView has no eye")
        first = ids[0]
        if not np.all(ids == first):
            raise ValueError("LensView spans multiple eyes")
        return self._model.eye(int(first))

    @property
    def cartridges(self) -> List['Cartridge']:
        """The lamina cartridges anchored at these lenses (empty list if not wired)."""
        if not self._model._cartridges_wired:
            return []
        return [Cartridge(self._model, i) for i in self._gi]

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
        eye_indices = self._model._lens_eye_index[self._gi]

        if np.unique(eye_indices).size > 1:
            raise ValueError("directed_neighbours() requires a single-eye LensView")

        eye = self._model.eye(int(eye_indices[0]))
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
    A subset of M receptors in a CompoundEyeModel.
    """

    __slots__ = ('_model', '_gi')
    __hash__ = None     # disable hashing, views are mutable handles, not values

    def __init__(self, model: 'CompoundEyeModel', indices: ArrayLike):
        self._model = model
        self._gi = np.asarray(indices, dtype=np.intp).reshape(-1)

        if self._gi.size > 0:
            if int(self._gi.min()) < 0 or int(self._gi.max()) >= model.total_receptors:
                raise IndexError("Receptor indices out of range")

    def __len__(self) -> int:
        return int(self._gi.size)

    def __repr__(self) -> str:
        return f"ReceptorView(n={len(self)}, indices={self._gi[:8]}{'...' if len(self) > 8 else ''})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, ReceptorView):
            return NotImplemented
        return self._model is other._model and np.array_equal(self._gi, other._gi)

    def __getitem__(self, idx) -> 'ReceptorView':
        return ReceptorView(self._model, self._gi[idx])

    # Helpers

    def _validate_write(self):
        if not self._model.buffer._allow_rcpt_writes:
            raise RuntimeError("Attempted to modify ReceptorView but CPU-side receptor writes are locked.")

    def _mark_dirty(self):
        self._model.buffer.rcpt_dirty = True
        self._model.buffer.rcpt_dirty_mask[self._gi] = True

    # Properties

    @property
    def global_indices(self) -> np.ndarray:
        """Global receptor indices into the CompoundEyeModel (read-only copy)."""
        return self._gi.copy()

    @property
    def indices(self) -> np.ndarray:
        """Alias to global_indices."""
        return self.global_indices

    @property
    def lenses(self) -> 'LensView':
        """Unique parent lenses of these receptors, sorted by index."""
        return LensView(self._model, np.unique(self.lens_indices))

    # Read-only derived / structural
    # TODO: Add setters (with checks)

    @property
    def position(self) -> np.ndarray:
        """(M, 3) world positions (= parent lens positions). Read-only."""
        return self._model.rcpt_static_data['position'][self._gi].copy()

    @property
    def lens_indices(self) -> np.ndarray:
        """(M,) parent lens global index for each receptor."""
        meta = self._model.rcpt_static_data['metadata'][self._gi]
        return get_metadata_field(meta, 'lens_id').astype(np.intp)

    @property
    def receptor_type(self) -> np.ndarray:
        """(M,) receptor type within bundle (R1=0, R2=1, ..., R7/8=6)."""
        meta = self._model.rcpt_static_data['metadata'][self._gi]
        return get_metadata_field(meta, 'rcpt_type').astype(np.intp)

    @property
    def is_wired(self) -> np.ndarray:
        """(M,) bool mask: True if this receptor is part of a valid cartridge."""
        meta = self._model.rcpt_static_data['metadata'][self._gi]
        return get_metadata_field(meta, 'is_wired').astype(bool)

    @property
    def eye_index(self) -> np.ndarray:
        meta = self._model.rcpt_static_data['metadata'][self._gi]
        return get_metadata_field(meta, 'eye_id').astype(np.intp)

    @property
    def neighbour_count(self) -> np.ndarray:
        """(M,) number of lattice neighbours this receptor's parent lens has."""
        meta = self._model.rcpt_static_data['metadata'][self._gi]
        return get_metadata_field(meta, 'neighbour_count').astype(np.intp)

    @property
    def chirality(self) -> np.ndarray:
        """(M,) Per-receptor chirality (+1.0 or -1.0)."""
        meta = self._model.rcpt_static_data['metadata'][self._gi]
        return (1.0 - 2.0 * get_metadata_field(meta, 'chirality_neg')).astype(np.float32)

    @property
    def cartridge_source(self) -> np.ndarray:
        """(M,) global receptor index of the cartridge source (R7/8) each member targets."""
        return self._model.rcpt_static_data['cartridge_src'][self._gi].copy()

    @property
    def rest_offset_um(self) -> np.ndarray:
        """(M, 2) focal-plane offset behind lens (μm), already rotated by chi/chirality."""
        return self._model.rcpt_static_data['rot_offset'][self._gi].copy()

    @property
    def diameter_um(self) -> np.ndarray:
        return self._model.rcpt_static_data['rhab_diameter_um'][self._gi].copy()

    @property
    def wavelength_um(self) -> np.ndarray:
        return self._model.rcpt_static_data['wavelength_um'][self._gi].copy()

    @property
    def azimuth_rad(self) -> np.ndarray:
        """(M,) current viewing azimuths (rad)."""
        d = self.direction
        return np.arctan2(d[:, 0], -d[:, 2])

    @property
    def azimuth_deg(self) -> np.ndarray:
        return np.degrees(self.azimuth_rad)

    @property
    def elevation_rad(self) -> np.ndarray:
        """(M,) current viewing elevations (rad)."""
        return np.arcsin(np.clip(self.direction[:, 1], -1.0, 1.0))

    @property
    def elevation_deg(self) -> np.ndarray:
        return np.degrees(self.elevation_rad)

    # Settable cell-level properties

    @property
    def sensitivity(self) -> np.ndarray:
        """(M, 3) channel multipliers (UV, G, B)."""
        return self._model.rcpt_static_data['sensitivity'][self._gi].copy()

    @sensitivity.setter
    def sensitivity(self, value: ArrayLike):
        self._validate_write()
        v = np.broadcast_to(np.asarray(value, dtype=np.float32), (len(self), 3))
        self._model.rcpt_static_data['sensitivity'][self._gi] = v
        self._mark_dirty()

    @property
    def sensitivity_uv(self) -> np.ndarray:
        return self._model.rcpt_static_data['sensitivity'][self._gi, 0].copy()

    @sensitivity_uv.setter
    def sensitivity_uv(self, value: ArrayLike):
        self._validate_write()
        self._model.rcpt_static_data['sensitivity'][self._gi, 0] = np.asarray(value, dtype=np.float32)
        self._mark_dirty()

    @property
    def sensitivity_r(self) -> np.ndarray:
        """Alias to sensitivities_uv."""
        return self.sensitivity_uv

    @sensitivity_r.setter
    def sensitivity_r(self, value: ArrayLike):
        self.sensitivity_uv(value)

    @property
    def sensitivity_g(self) -> np.ndarray:
        return self._model.rcpt_static_data['sensitivity'][self._gi, 1].copy()

    @sensitivity_g.setter
    def sensitivity_g(self, value: ArrayLike):
        self._validate_write()
        self._model.rcpt_static_data['sensitivity'][self._gi, 1] = value
        self._mark_dirty()

    @property
    def sensitivity_b(self) -> np.ndarray:
        return self._model.rcpt_static_data['sensitivity'][self._gi, 2].copy()

    @sensitivity_b.setter
    def sensitivity_b(self, value: ArrayLike):
        self._validate_write()
        self._model.rcpt_static_data['sensitivity'][self._gi, 2] = np.asarray(value, dtype=np.float32)
        self._mark_dirty()

    @property
    def tau_membrane(self) -> np.ndarray:
        return self._model.rcpt_static_data['tau_membrane'][self._gi].copy()

    @tau_membrane.setter
    def tau_membrane(self, value: ArrayLike):
        self._validate_write()
        self._model.rcpt_static_data['tau_membrane'][self._gi] = value
        self._mark_dirty()

    @property
    def rest_acceptance_angles(self) -> np.ndarray:
        """(M, 2) acceptance angles (minor, major) at rest (rad)."""
        return self._model.rcpt_static_data['rest_acc'][self._gi].copy()

    @rest_acceptance_angles.setter
    def rest_acceptance_angles(self, value: ArrayLike):
        self._validate_write()
        v = np.broadcast_to(np.asarray(value, dtype=np.float32), (len(self), 2))
        self._model.rcpt_static_data['rest_acc'][self._gi] = v
        self._mark_dirty()

    @property
    def acceptance_tilt(self) -> np.ndarray:
        return self._model.rcpt_static_data['acc_tilt'][self._gi].copy()

    @acceptance_tilt.setter
    def acceptance_tilt(self, value: ArrayLike):
        self._validate_write()
        self._model.rcpt_static_data['acc_tilt'][self._gi] = value
        self._mark_dirty()

    # Dynamics

    @property
    def direction(self) -> np.ndarray:
        """(M, 3) current (actuated) viewing directions. Settable."""
        return self._model.rcpt_dynamic_data['direction'][self._gi].copy()

    @direction.setter
    def direction(self, value: ArrayLike):
        self._validate_write()
        v = np.broadcast_to(np.asarray(value, dtype=np.float32), (len(self), 3))
        self._model.rcpt_dynamic_data['direction'][self._gi] = v
        self._mark_dirty()

    @property
    def acceptance_angles(self) -> np.ndarray:
        """(M, 2) current (actuated) acceptance angles (rad)."""
        return self._model.rcpt_dynamic_data['acc_axes'][self._gi].copy()

    @acceptance_angles.setter
    def acceptance_angles(self, value: ArrayLike):
        self._validate_write()
        v = np.broadcast_to(np.asarray(value, dtype=np.float32), (len(self), 2))
        self._model.rcpt_dynamic_data['acc_axes'][self._gi] = v
        self._mark_dirty()

    @property
    def adaptation_state(self) -> np.ndarray:
        return self._model.rcpt_dynamic_data['adaptation_state'][self._gi].copy()

    @adaptation_state.setter
    def adaptation_state(self, value: ArrayLike):
        self._validate_write()
        self._model.rcpt_dynamic_data['adaptation_state'][self._gi] = value
        self._mark_dirty()


class Cartridge:
    """
    A lamina cartridge in neural-superposition optics.

    The cartridge is anchored at one ommatidium's central rhabdomere (R7/8).
    Its members are the peripheral rhabdomeres (R1-R6) from neighbouring
    ommatidia whose lines of sight converge on this cartridge's direction.

    '.lens' is the home (central) ommatidium. '.sources' gives, per member,
    the global lens index of the ommatidium that contributes that receptor.
    """

    __slots__ = ('_model', '_central_lens_idx', '_member_indices')
    __hash__ = None

    def __init__(self, model: 'CompoundEyeModel', central_lens_idx: int):
        self._model = model
        self._central_lens_idx = int(central_lens_idx)
        if not getattr(model, '_cartridges_wired', False):
            raise ValueError("Cartridges not wired")

        R = model.receptors_per_lens
        sources = model._cartridge_map[self._central_lens_idx].copy()

        sources[sources == -1] = self._central_lens_idx
        self._member_indices = (sources * R + np.arange(R, dtype=np.intp))

    def __len__(self) -> int:
        return int(self._member_indices.size)

    def __repr__(self) -> str:
        return f"Cartridge(lens={self._central_lens_idx}, R={len(self)})"

    @property
    def lens(self) -> LensView:
        """The home (central) ommatidium for this cartridge."""
        return LensView(self._model, np.array([self._central_lens_idx], dtype=np.intp))

    @property
    def receptors(self) -> ReceptorView:
        """All member receptors (the R1-R6 contributors from neighbouring lenses)."""
        return ReceptorView(self._model, self._member_indices)

    @property
    def sources(self) -> np.ndarray:
        """(R,) global lens index of the source ommatidium for each member slot."""
        return self._model._cartridge_map[self._central_lens_idx].copy()


class Eye:
    """
    One eye.

    Owns:
      - The lens-index mask for the eye
      - A 'side' label: 'left', 'right', or 'midline'
      - A position KDtree (lazy)
      - A direction KDtree (lazy)
      - The lattice neighbour graph (lazy)

    Spatial queries are eye-local.
    For animal-wide queries, iterate over eyes or use 'model.query_directions()' (it dispatches across eyes).
    """

    __slots__ = (
        '_model', '_eye_index', '_lens_indices', '_side',
        '_position_tree', '_direction_tree',
        '_position_tree_cf', '_direction_tree_cf', '_cf_local_indices',
        '_neighbour_graph', '_neighbour_k', '_directional_graph',
    )

    def __init__(self, model: 'CompoundEyeModel', eye_index: int, lens_indices: np.ndarray, side: str = 'left'):
        self._model = model
        self._eye_index = int(eye_index)
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
        return f"Eye(id={self._eye_index}, side={self._side!r}, lenses={len(self._lens_indices)})"

    def __len__(self) -> int:
        return int(self._lens_indices.size)

    @property
    def eye_index(self) -> int:
        return self._eye_index

    @property
    def side(self) -> str:
        """'left', 'right', or 'midline'."""
        return self._side

    @property
    def side_sign(self) -> float:
        """Sign of the side: Left/Midline = 1.0, Right = -1.0."""
        return -1.0 if self._side == 'right' else 1.0

    @property
    def lens_indices(self) -> np.ndarray:
        """Global indices of the lenses in this eye."""
        return self._lens_indices.copy()

    @property
    def lenses(self) -> LensView:
        return LensView(self._model, self._lens_indices)

    @property
    def receptors(self) -> ReceptorView:
        R = self._model.receptors_per_lens
        rcpt_indices = (self._lens_indices[:, None] * R
                        + np.arange(R, dtype=np.intp)[None, :]).ravel()
        return ReceptorView(self._model, rcpt_indices)

    @property
    def ommatidia(self) -> LensView:
        """Alias for '.lenses': iterate to yield Ommatidia."""
        return self.lenses

    # TODO: Rename this one?
    @property
    def binocular_mask(self) -> np.ndarray:
        return self.lenses.is_binocular

    @property
    def binocular_fraction(self) -> float:
        return self.lenses.binocular_fraction

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
            self._position_tree = cKDTree(self._model._lens_positions[self._lens_indices])
        return self._position_tree

    def _ensure_direction_tree(self) -> cKDTree:
        if self._direction_tree is None:
            self._direction_tree = cKDTree(self._model._lens_directions[self._lens_indices])
        return self._direction_tree

    def _ensure_neighbour_graph(self, k: int) -> np.ndarray:
        """
        Per-lens lattice neighbours (drops self). Rebuilds if k changes.
        """

        if self._neighbour_graph is None or self._neighbour_k != k:
            tree = self._ensure_position_tree()
            positions = self._model._lens_positions[self._lens_indices]
            n = positions.shape[0]
            actual_k = min(k, n - 1) if n > 1 else 0

            if actual_k == 0:
                self._neighbour_graph = np.empty((n, 0), dtype=np.intp)
            else:
                _, nidx_local = tree.query(positions, k=actual_k + 1)
                # nidx_local has shape (n, k+1), drop the self column (the first)
                if nidx_local.ndim == 1:
                    nidx_local = nidx_local.reshape(-1, 1)
                self._neighbour_graph = nidx_local[:, 1:].astype(np.intp)
            self._neighbour_k = k

        return self._neighbour_graph

    def _ensure_trees_cf(self) -> Tuple[Optional[cKDTree], Optional[cKDTree], np.ndarray]:
        """Lazy-build conflict-free trees (used when avoid_conflicts=True)."""
        if self._direction_tree_cf is None:
            conflict_free = ~self._model.have_conflicts[self._lens_indices]
            fully_wired = ~self._model.has_selfwires[self._lens_indices]
            valid_mask = conflict_free & fully_wired
            self._cf_local_indices = np.flatnonzero(valid_mask)
            if self._cf_local_indices.size > 0:
                dirs = self._model._lens_directions[self._lens_indices[self._cf_local_indices]]
                pos = self._model._lens_positions[self._lens_indices[self._cf_local_indices]]
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
        neighbour_dist_factor: float = 1.25,
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

        Returns a NeighbourResult.
        """

        if (lens_indices is None) == (points is None):
            raise ValueError("Provide either 'lens_indices' or 'points' (but not both)")

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
                positions = self._model._lens_positions[self._lens_indices]

                # Recompute distances because the graph only stores indices
                q_pos = positions[valid_local]
                n_pos = positions[local_nidx]
                distances = np.linalg.norm(q_pos[:, None, :] - n_pos, axis=2)

            else:
                tree = self._ensure_position_tree()
                positions = self._model._lens_positions[self._lens_indices][valid_local]
                actual_k = min(k, len(local_to_global) - 1)
                distances, local_nidx = tree.query(positions, k=actual_k + 1)
                if local_nidx.ndim == 1:
                    local_nidx = local_nidx.reshape(-1, 1)
                    distances = distances.reshape(-1, 1)
                local_nidx = local_nidx[:, 1:]
                distances = distances[:, 1:]

            global_nidx = local_to_global[local_nidx]
            result = NeighbourResult(self, mask=in_this_eye, indices=global_nidx, distances=distances)

            # Populate chirality match if the model has initialised it
            chir = getattr(self._model, '_chirality_arr', None)
            if chir is not None:
                result.same_chirality = chir[global_nidx] == chir[valid_global][:, None]

            # Populate immediacy
            d_factor = float(neighbour_dist_factor if neighbour_dist_factor is not None else 1.25)
            if distances.size > 0:
                result.is_immediate = distances <= (distances[:, 0:1] * d_factor)

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
            eye=self,
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
        avoid_conflicts: bool = False
        ) -> Tuple['LensView', np.ndarray]:
        """
        Find the k lenses whose optical axes lie closest (Euclidean) to each
        of the given query directions.

        Returns (LensView, distances) where distances are chord distances on the unit sphere.
        """

        dirs = np.asarray(directions, dtype=np.float32).reshape(-1, 3)
        Q = dirs.shape[0]

        if avoid_conflicts:
            tree_cf, _, cf_local = self._ensure_trees_cf()
            if tree_cf is None:
                return np.empty((Q, 0), dtype=np.intp), np.empty((Q, 0), dtype=np.float32)

            actual_k = min(k, len(cf_local))
            distances, local_idx_cf = tree_cf.query(dirs, k=actual_k)
            if local_idx_cf.ndim == 1:
                local_idx_cf = local_idx_cf.reshape(-1, 1)
                distances = distances.reshape(-1, 1)
            local_idx = cf_local[local_idx_cf]
        else:
            tree = self._ensure_direction_tree()
            actual_k = min(k, len(self._lens_indices))
            distances, local_idx = tree.query(dirs, k=actual_k)
            if local_idx.ndim == 1:
                local_idx = local_idx.reshape(-1, 1)
                distances = distances.reshape(-1, 1)

        view = LensView(self._model, self._lens_indices[local_idx].ravel())
        return view, distances

    def query_positions(self,
        positions: ArrayLike,
        k: int = 1,
        avoid_conflicts: bool = False
    ) -> Tuple['LensView', np.ndarray]:
        """
        Find the k lenses closest in world-space position to each query point.

        Returns (LensView, distances) where distances are Euclidean distances in world units.
        """
        pts = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        Q = pts.shape[0]

        if avoid_conflicts:
            _, tree_cf, cf_local = self._ensure_trees_cf()
            if tree_cf is None:
                return np.empty((Q, 0), dtype=np.intp), np.empty((Q, 0), dtype=np.float32)

            actual_k = min(k, len(cf_local))
            distances, local_idx_cf = tree_cf.query(pts, k=actual_k)
            if local_idx_cf.ndim == 1:
                local_idx_cf = local_idx_cf.reshape(-1, 1)
                distances = distances.reshape(-1, 1)
            local_idx = cf_local[local_idx_cf]
        else:
            tree = self._ensure_position_tree()
            actual_k = min(k, len(self._lens_indices))
            distances, local_idx = tree.query(pts, k=actual_k)
            if local_idx.ndim == 1:
                local_idx = local_idx.reshape(-1, 1)
                distances = distances.reshape(-1, 1)

        view = LensView(self._model, self._lens_indices[local_idx].ravel())
        return view, distances

    def query_lookat(self,
         targets: ArrayLike,
         k: int = 1,
         avoid_conflicts: bool = False
         ) -> 'LensView':
        """
        Find the k lenses best looking at world-space target points.

        Unlike query_directions (which only matches optical-axis vectors),
        this accounts for lens *position*: the score is the dot product of
        each lens's optical axis with the unit vector from that lens to the
        target.

        Returns a LensView (best first).
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        q = np.asarray(targets, dtype=np.float32).reshape(-1, 3)
        Q = q.shape[0]
        if len(self._lens_indices) == 0:
            return LensView(self._model, np.empty(0, dtype=np.intp))

        pos = self._model._lens_positions[self._lens_indices]
        dirs = self._model._lens_directions[self._lens_indices]

        desired = q[:, None, :] - pos[None, :, :]
        norms = np.linalg.norm(desired, axis=-1, keepdims=True)
        np.divide(desired, norms, out=desired, where=norms > 0)
        dots = np.einsum('jk,ijk->ij', dirs, desired)

        if avoid_conflicts:
            conflicts = self._model.have_conflicts[self._lens_indices]
            dots[:, conflicts] = -np.inf

        k_eff = min(k, dots.shape[1])
        part = np.argpartition(dots, -k_eff, axis=1)[:, -k_eff:]
        top = np.take_along_axis(dots, part, axis=1)
        order = np.argsort(top, axis=1)[:, ::-1]
        best = np.take_along_axis(part, order, axis=1)
        return LensView(self._model, self._lens_indices[best].ravel())

    def query_cone(self,
        center_direction: ArrayLike,
        angle: float,
        degrees: bool = True,
        avoid_conflicts: bool = False
        ) -> 'LensView':
        """
        All lenses whose optical axis lies within 'angle' of 'center_direction'.

        Returns a LensView.
        """
        c = np.asarray(center_direction, dtype=np.float32)
        n = float(np.linalg.norm(c))
        if n < 1e-12:
            return np.empty(0, dtype=np.intp)
        c = c / n
        a = np.deg2rad(angle) if degrees else float(angle)
        radius = 2.0 * np.sin(a / 2.0)

        if avoid_conflicts:
            tree_cf, _, cf_local = self._ensure_trees_cf()
            if tree_cf is None:
                return np.empty(0, dtype=np.intp)
            hits = tree_cf.query_ball_point(c, r=radius)
            local_cf = np.atleast_1d(np.asarray(hits, dtype=np.intp))
            local = cf_local[local_cf]
        else:
            tree = self._ensure_direction_tree()
            hits = tree.query_ball_point(c, r=radius)
            local = np.atleast_1d(np.asarray(hits, dtype=np.intp))

        return LensView(self._model, self._lens_indices[local])

    def query_ball(self,
        center_position: ArrayLike,
        radius: float,
        avoid_conflicts: bool = False
        ) -> 'LensView':
        """
        All lenses whose world-space position is within 'radius' of
        'center_position'.

        Returns a LensView.
        """
        c = np.asarray(center_position, dtype=np.float32)

        if avoid_conflicts:
            _, tree_cf, cf_local = self._ensure_trees_cf()
            if tree_cf is None:
                return np.empty(0, dtype=np.intp)
            hits = tree_cf.query_ball_point(c, r=float(radius))
            local_cf = np.atleast_1d(np.asarray(hits, dtype=np.intp))
            local = cf_local[local_cf]
        else:
            tree = self._ensure_position_tree()
            hits = tree.query_ball_point(c, r=float(radius))
            local = np.atleast_1d(np.asarray(hits, dtype=np.intp))

        return LensView(self._model, self._lens_indices[local])

    def max_gap(self) -> float:
        """
        Largest angular gap (in radians) between any lens and its nearest neighbour
        in this eye.
        """
        if len(self._lens_indices) <= 1:
            return 0.0
        tree = self._ensure_direction_tree()
        dirs = self._model._lens_directions[self._lens_indices]
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

        right = self._model._local_right[query_global]
        up = self._model._local_up[query_global]
        centre_pos = self._model._lens_positions[query_global]

        neigh_pos = self._model._lens_positions[result.indices]
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
            eye=self,
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
        dirs = self._model._lens_directions[self._lens_indices]
        dists, kd_idx = tree.query(dirs, k=k_eff + 1)
        if kd_idx.ndim == 1:
            kd_idx = kd_idx.reshape(-1, 1)
            dists = dists.reshape(-1, 1)
        nb_idx = kd_idx[:, 1:]
        nb_chord = dists[:, 1:]
        angular_sep = 2.0 * np.arcsin(np.clip(nb_chord / 2.0, -1.0, 1.0))

        # Tangent frame for each lens
        local_x = self._model._local_right[self._lens_indices]
        local_y = self._model._local_up[self._lens_indices]

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
            dirs = self._model._lens_directions[self._lens_indices][valid_local]

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

    def pull_retina(self, amount_um: float = 0.0):
        """
        Commands a muscular pull of the retina for this eye.
        The sign is handled automatically: positive values always move the
        retina symmetrically relative to the body midline.
        """
        self._model._retina_pulls[self._eye_index] = float(amount_um) * self.side_sign


class VisualOutput:
    """
    Per-receptor output array with various biological pathway mappings conveniences.

    The renderers return a float array where the last axis is (R/UV, G, B, radiance).
    This class supports both single snapshots (shape: N, 4) and timeseries (shape: T, N, 4).

    Layouts / pathways:
        .per_lens          -> (..., N, R, 4) Physical ommatidia grouping.
        .per_cartridge     -> (..., N, R, 4) Neural superposition grouping.
        .per_receptor(i)   -> (..., N, 4)    Specific receptor type across all cartridges.
        .peripheral_signal -> (..., N, 4)    Pooled R1-R6 (LMC pathway for motion).
        .central_signal    -> (..., N, 4)    Central R7/R8 (Medulla colour pathway).
        .lmc_input         -> (..., N, 4)    Alias for peripheral_signal.
        .pale_input        -> (..., N, 4)    Alias for central_signal.

    Signal analysis:
        .colours      -> (..., 3) The adapted spectral response.
        .adaptation   -> (..., )  The gain factor (state of the biological system).
        .radiance     -> (..., )  Adapted intensity (mean of adapted RGB).
        .raw_radiance -> (..., )  Physical light intensity recovered by 'un-baking'
                                  the adaptation factor.
    """
    __slots__ = ('_data', '_model', '_R', '_N', '_T')

    def __init__(self, data: np.ndarray, model: 'CompoundEyeModel'):
        shape = data.shape
        if shape[-2] % model.receptors_per_lens != 0:
            raise ValueError(f"data length {shape[-2]} not divisible by R={model.receptors_per_lens}")

        self._data = data
        self._model = model
        self._R = model.receptors_per_lens
        self._N = shape[-2] // self._R
        self._T = shape[0] if data.ndim == 3 else None

    @classmethod
    def from_history(cls, history: list['VisualOutput']) -> 'VisualOutput':
        """Stacks a list of single-frame VisualOutputs into one timeseries VisualOutput."""
        if not history:
            raise ValueError("History list is empty")
        stacked_data = np.stack([vo.data for vo in history], axis=0)
        return cls(stacked_data, history[0]._model)

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
        return self._N if self._model._cartridges_wired else 0

    # Channel helpers

    @property
    def colours(self) -> np.ndarray:
        """The adapted spectral response (Photoreceptor output)."""
        return self._data[..., :3]

    @property
    def adaptation(self) -> np.ndarray:
        """
        The adaptation state (gain factor) of the receptors.
        This is the value calculated by the Naka-Rushton equations (0.0 to 1.0+).
        """
        return self._data[..., 3]

    @property
    def gain(self) -> np.ndarray:
        """Alias to adaptation"""
        return self.adaptation

    @property
    def raw_radiance(self) -> np.ndarray:
        """
        The physical light intensity hitting the eye before adaptation.
        Recovered by 'un-baking' the adaptation factor.
        """
        return np.mean(self.colours, axis=-1) / (self.adaptation + 1e-6)

    @property
    def radiance(self) -> np.ndarray:
        """
        The mean intensity of the adapted signal.
        Calculated as the mean of the RGB channels.
        """
        return np.mean(self.colours, axis=-1)

    # Level 1: raw grids
    @property
    def per_lens(self) -> np.ndarray:
        """Returns (..., N, R, 4) array of all receptor outputs, per lens."""
        if self._data.ndim == 2:
            return self._data.reshape(self._N, self._R, 4)
        return self._data.reshape(self._T, self._N, self._R, 4)

    @property
    def per_ommatidium(self) -> np.ndarray:
        """Alias to per_lens."""
        return self.per_lens

    @property
    def per_cartridge(self) -> np.ndarray:
        """Returns (..., N, R, 4) array of all receptor outputs, per cartridge."""
        if not self._model._cartridges_wired:
            return self.per_lens   # fallback for R=1 models
        if self._data.ndim == 2:
            return self._data[self._model.cartridge_indices]
        return self._data[:, self._model.cartridge_indices, :]

    # Level 2: type-based access
    def per_receptor(self, index: int) -> np.ndarray:
        """Returns (..., N, 4) array for a specific receptor index (e.g. 0 for R1)."""
        return self.per_cartridge[..., index, :]

    # Level 3: biological pathways
    @property
    def peripheral_signal(self) -> np.ndarray:
        """The pooled response of all peripheral rhabdomeres (LMC-pathway)."""
        if self._R == 1:
            return self.per_lens[..., 0, :]

        indices = getattr(self._model.bundle, 'peripheral_indices', None)
        if indices is None or len(indices) == 0:
            return self.per_cartridge[..., 0, :]

        return np.mean(self.per_cartridge[..., indices, :], axis=-2)

    @property
    def central_signal(self) -> np.ndarray:
        """The response of the central rhabdomere (Medulla color pathway)."""
        if self._R == 1:
            return self.per_lens[..., 0, :]

        center_idx = getattr(self._model.bundle, 'center_index', 0)
        return self.per_cartridge[..., center_idx, :]

    @property
    def lmc_input(self):
        """Alias to peripheral_signal"""
        return self.peripheral_signal

    @property
    def pale_input(self):
        """Alias to central_signal"""
        return self.central_signal

    def __getitem__(self, idx):
        return self._data[idx]

    def __len__(self) -> int:
        return self._T if self._T is not None else self._data.shape[0]

    def __repr__(self) -> str:
        c_str = f", cartridges={self._N}" if self._model._cartridges_wired else ""
        t_str = f", T={self._T}" if self._T is not None else ""
        return f"VisualOutput(N={self._N}, R={self._R}{c_str}{t_str}, shape={self._data.shape})"

    # Plotting methods

    def plot(self, pathway: str = 'all', projection: str = 'equirectangular',
             ax=None, false_colors: bool = False, uv_encoding: bool = False,
             draw_edges: bool = False, dark_mode: bool = False):
        """
        Displays the visual output as a gapless Voronoi tessellation (like in the first person shader).
        If this VisualOutput contains multiple timesteps, this plots the last one.
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        from scipy.spatial import Voronoi

        if ax is None:
            if projection == 'equirectangular':
                fig, ax = plt.subplots(figsize=(10, 5))
            else:
                fig = plt.figure(figsize=(10, 5))
                ax = fig.add_subplot(111, projection=projection)

        is_ts = self._data.ndim == 3

        # Extract spatial data and colour
        if pathway == 'all':
            rgb = self.colours[-1] if is_ts else self.colours
            az = self._model.receptors.azimuth_rad
            el = self._model.receptors.elevation_rad

        elif pathway in ('peripheral', 'central', 'lens', 'cartridge'):
            if pathway == 'peripheral':
                sig = self.peripheral_signal
            elif pathway == 'central':
                sig = self.central_signal
            elif pathway == 'lens':
                sig = np.mean(self.per_lens, axis=-2)
            elif pathway == 'cartridge':
                sig = np.mean(self.per_cartridge, axis=-2)

            rgb = sig[-1, :, :3] if is_ts else sig[:, :3]
            az = self._model.lenses.azimuth_rad
            el = self._model.lenses.elevation_rad

            # Peripheral is greyscale
            if pathway == 'peripheral':
                gray = np.mean(rgb, axis=-1, keepdims=True)     # TODO: use weights?
                rgb = np.repeat(gray, 3, axis=-1)
        else:
            raise ValueError("pathway must be 'all', 'peripheral', 'central', 'lens', or 'cartridge'")

        # False colour modes
        rgb = np.clip(rgb, 0.0, 1.0).copy()
        if uv_encoding:
            rgb[:, 2] = np.clip(rgb[:, 2] + rgb[:, 0], 0.0, 1.0)
        elif false_colors:
            rgb[:, 0] = 0.0

        # Voronoi cells on the unwrapped cylinder
        pts = np.column_stack((az, el))
        pts_left = np.column_stack((az - 2 * np.pi, el))
        pts_right = np.column_stack((az + 2 * np.pi + 1e-6, el))
        cap_x = np.linspace(-2 * np.pi, 2 * np.pi, max(100, len(az) // 10))
        pts_top = np.column_stack((cap_x, np.full_like(cap_x, np.pi / 2 + 0.5)))
        pts_bot = np.column_stack((cap_x, np.full_like(cap_x, -np.pi / 2 - 0.5)))

        all_pts = np.vstack((pts, pts_left, pts_right, pts_top, pts_bot))
        vor = Voronoi(all_pts)

        polygons = []
        for i in range(len(pts)):
            polygons.append(vor.vertices[vor.regions[vor.point_region[i]]])

        if projection == 'equirectangular':
            polygons = [np.degrees(p) for p in polygons]
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_xlabel('Azimuth (deg)')
            ax.set_ylabel('Elevation (deg)')
            ax.set_aspect('equal', adjustable='box')
        else:
            for p in polygons:
                p[:, 0] = (p[:, 0] + np.pi) % (2 * np.pi) - np.pi
            ax.grid(True, alpha=0.3)

        # Anti-aliasing workaround: 'face' draws a mini border of the polygon's
        # own color over the pixel gaps left by matplotlib
        edge_c = ('white' if dark_mode else 'black') if draw_edges else 'face'

        collection = PolyCollection(
            polygons,
            facecolors=rgb,
            edgecolors=edge_c,
            linewidths=0.5,
            antialiaseds=True
        )
        ax.add_collection(collection)

        if dark_mode:
            ax.set_facecolor('black')
            if ax.figure is not None:
                ax.figure.patch.set_facecolor('black')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('white')

        return ax

    def plot_time_series(self, pathway: str = 'peripheral',
                         max_items: int = 100, sort_by: str = 'azimuth',
                         false_colors: bool = False, uv_encoding: bool = False,
                         dark_mode: bool = False, ax=None):
        """
        Plots a spatio-temporal heatmap of the visual output over time.
        """
        import matplotlib.pyplot as plt

        if self._data.ndim != 3:
            raise ValueError("This requires a VisualOutput containing multiple timesteps. "
                             "Use VisualOutput.from_history() to combine a list of frames.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(10, max(4, min(10, max_items * 0.1))))

        if pathway == 'all':
            data = self.colours
            az = self._model.receptors.azimuth_rad
        elif pathway in ('peripheral', 'central', 'lens', 'cartridge'):
            if pathway == 'peripheral':
                sig = self.peripheral_signal

                # Grayscale for peripheral
                data = sig[..., :3]
                gray = np.mean(data, axis=-1, keepdims=True)
                data = np.repeat(gray, 3, axis=-1)

            elif pathway == 'central':
                data = self.central_signal[..., :3]

            elif pathway == 'lens':
                data = np.mean(self.per_lens, axis=-2)[..., :3]

            elif pathway == 'cartridge':
                data = np.mean(self.per_cartridge, axis=-2)[..., :3]

            az = self._model.lenses.azimuth_rad
        else:
            raise ValueError("Pathway must be 'all', 'peripheral', 'central', 'lens', or 'cartridge'")

        data = np.clip(data, 0.0, 1.0)
        if uv_encoding:
            data[..., 2] = np.clip(data[..., 2] + data[..., 0], 0.0, 1.0)
        elif false_colors:
            data[..., 0] = 0.0

        # Transpose to (Space, Time, RGB) for imshow
        data = np.transpose(data, (1, 0, 2))

        if sort_by == 'azimuth':
            order = np.argsort(az)
            data = data[order]

        # downsample (spatially) if needed
        S, T, C = data.shape
        if S > max_items:
            block_size = S // max_items
            end = block_size * max_items
            data = data[:end].reshape(max_items, block_size, T, C).mean(axis=1)

        im = ax.imshow(data, aspect='auto', origin='lower', interpolation='none')
        ax.set_xlabel("Time step")

        ylabel = "Items"
        if pathway == 'all':
            ylabel = "Receptors"
        elif pathway in ('cartridge', 'peripheral', 'central'):
            ylabel = "Cartridges"
        elif pathway == 'lens':
            ylabel = "Lenses"

        if sort_by == 'azimuth':
            ylabel += " (sorted Left to Right)"
        ax.set_ylabel(ylabel)

        if dark_mode:
            ax.set_facecolor('black')
            if ax.figure is not None:
                ax.figure.patch.set_facecolor('black')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            for spine in ax.spines.values():
                spine.set_color('white')

        return ax

class CompoundEyeModel:
    """
    A compound eye specified as N lens positions / directions and a bundle
    of R rhabdomeres per ommatidium.

    The packed per-lens / per-receptor data lives in an EyesBuffer class.

    This class owns that buffer and adds everything the model needs:
      - the RhabdomereBundle and the list of per-eye objects
        (KDtrees, neighbour graphs, side-aware queries),
      - the orientation / chirality / saccade fields produced by the
        BundlesAligner pipeline,
      - neural-superposition wiring,
      - world-frame transforms (translate / scale / rotate),
      - across-eye spatial queries (query_directions, query_positions, ...).

    Views (LensView, ReceptorView, Cartridge, Eye) hold a ref to this model and read/write the buffer through it.

    Args:
        - directions: (N, 3) array_like, Lens optical-axis (forward) directions. Will be normalised.
        - positions: (N, 3) array_like, Lens world positions.
        - bundle: RhabdomereBundle (optional), Per-species bundle model. Defaults to a single panchromatic receptor.
        - eye_index: (N,) array_like of uint (optional), Eye membership for each lens (0-7).
            If None, splits in two eyes on x: x < 0 -> 0 (left), x ≥ 0 -> 1 (right).
        - lens_diameter_um: float or (N,) array_like or None, Lens aperture (μm).
            If None (default), auto-derived per-lens from the local lattice spacing.
        - interommatidial_angles_rad: (2,), (N, 2), or None, Per-lens (minor, major) IOA.
            If None, computed from the local lattice (sorted mean over first ring,
            tilt from the hexatic Ψ6 order parameter).
        - acceptance: AcceptanceModel or None, Provider of the per-receptor rest Δρ field.
            Any callable (lens: LensOptics, rcpt: ReceptorOptics) -> (N, R, 2) half-widths
            (minor, major), radians. See insectvision.compound_eyes.acceptance.
            If None (default): SnyderAcceptance() when bundle.focal_um is set, else
            SamplingAcceptance(). Pass ExplicitAcceptance(array) to supply Δρ verbatim.
        - bundle_orientations: (N,) array_like or None, Override chi from the orientation pipeline (rad).
        - chiralities: (N,) array_like or None, Override chirality from the orientation pipeline (±1).
        - orientation: BundlesAligner or None, Explicit pipeline configuration.
            Required when R > 1 unless both 'bundle_orientations' and 'chiralities' are supplied.
        - flow_direction: (3,) array_like or None, Shorthand for 'orientation=BundlesAligner(flow_direction)'.
            Defaults to forward-facing optic flow.
    """

    HEX_PACKING_FACTOR = 0.9 # Fraction spacing that the lens diameter occupies (1.0 = fully
    # touching, 0.9 leaves a small interommatidial cuticle gap, etc)

    def __init__(self,
                 directions: ArrayLike,
                 positions: ArrayLike,
                 bundle: Optional['RhabdomereBundle'] = None,
                 eye_indices: Optional[ArrayLike] = None,
                 lens_diameter_um: Optional[Union[float, ArrayLike]] = None,
                 interommatidial_angles_rad: Optional[ArrayLike] = None,
                 acceptance: Optional['AcceptanceModel'] = None,
                 bundle_orientations: Optional[ArrayLike] = None,
                 chiralities: Optional[ArrayLike] = None,
                 orientation: Optional['BundlesAligner'] = None,
                 flow_direction: Optional[ArrayLike] = None,
                 retina_muscle_direction: Optional[ArrayLike] = None,
                 ):

        # Validate geometry

        dirs = np.asarray(directions, dtype=np.float32).reshape(-1, 3)
        pos = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        if dirs.shape != pos.shape:
            raise ValueError(
                f"Directions and Positions must have the same shape, "
                f"got {dirs.shape} and {pos.shape}"
            )
        N = dirs.shape[0]
        if N == 0:
            raise ValueError("CompoundEyeModel needs at least 1 lens")

        self._lens_directions = normalise_vectors(dirs).astype(np.float32)
        self._lens_positions = pos.astype(np.float32).copy()

        # Bundle
        if bundle is None:
            bundle = RhabdomereBundle()  # default R=1
        self._bundle = bundle
        R = bundle.count

        # Acceptance model: per-receptor rest Δρ
        if acceptance is None:
            acceptance = SnyderAcceptance() if bundle.focal_um is not None else SamplingAcceptance()
        self._acceptance = acceptance

        # Allocate the packed buffer
        self._buffer = EyesBuffer(n_lenses=N, receptors_per_lens=R)

        with self.unlock():

            # Tangent frames
            self._local_right, self._local_up = tangent_frames(self._lens_directions)
            self._local_right = self._local_right.astype(np.float32)
            self._local_up = self._local_up.astype(np.float32)

            # Buffer for full retinal movement (per eye)
            self._retina_pulls = np.zeros(8, dtype=np.float32)

            # Eye membership (island detection if eye_indices is None)
            self._lens_eye_index = self._resolve_eye_indices(self._lens_positions, eye_indices, N)

            # Fill lens static data
            self._buffer.lens_static_data['right'] = self._local_right
            self._buffer.lens_static_data['up'] = self._local_up
            self._buffer.lens_static_data['forward'] = self._lens_directions
            self._buffer.lens_static_data['focal_um'] = bundle.focal_um or 1.0

            # Broadcast bundle-level photomechanical biophysics to every lens
            self._buffer.lens_static_data['tau_rise'] = bundle.tau_rise
            self._buffer.lens_static_data['tau_relax'] = bundle.tau_relax
            self._buffer.lens_static_data['tau_fast'] = bundle.tau_fast
            self._buffer.lens_static_data['tau_adapt'] = bundle.tau_adapt
            self._buffer.lens_static_data['ampl_lat_um'] = bundle.ampl_lat_um
            self._buffer.lens_static_data['ampl_ax_um'] = bundle.ampl_ax_um

            self._eyes: List[Eye] = []
            self._build_eyes()

            # Lattice properties: IOA (minor, major), tilt, and per-lens lens spacing
            baseline_axes, baseline_tilts, lens_spacing = self._compute_ioa_baseline()
            if interommatidial_angles_rad is not None:
                ioa_axes, ioa_tilts = self._broadcast_ioa(interommatidial_angles_rad, N)
            else:
                ioa_axes, ioa_tilts = baseline_axes, baseline_tilts
            self._buffer.lens_static_data['ioa_axes'] = ioa_axes
            self._buffer.lens_static_data['ioa_tilt'] = ioa_tilts

            if lens_diameter_um is None:
                spacing_est = (self.HEX_PACKING_FACTOR * lens_spacing).astype(np.float32)
                spacing_est = np.where(spacing_est > 0, spacing_est, np.float32(20.0))

                lens_diameters = spacing_est.copy()
                for eid in np.unique(self._lens_eye_index):
                    m = np.where(self._lens_eye_index == eid)[0]
                    if m.size > 19:
                        try:
                            lens_diameters[m] = facet_diameters(self._lens_positions[m], self._lens_directions[m])
                            continue
                        except Exception:
                            pass  # fall through to denoised spacing estimate

                    # Voronoi estimator unavailable: denoise the single-edge readout
                    lens_diameters[m] = angular_neighbour_median(spacing_est[m], self._lens_directions[m])

                self._buffer.lens_static_data['lens_diameter_um'] = lens_diameters.astype(np.float32)

            else:
                ld = np.atleast_1d(np.asarray(lens_diameter_um, dtype=np.float32))
                if ld.size == 1:
                    self._buffer.lens_static_data['lens_diameter_um'] = ld.item()
                elif ld.size == N:
                    self._buffer.lens_static_data['lens_diameter_um'] = ld
                else:
                    raise ValueError(f"lens_diameter_um size {ld.size} must be 1 or N={N}")

            # Per-lens focal length scales with facet diameter at a constant F-number,
            # so larger facets focus longer and see narrower RFs.

            # Note: this feeds both the Snyder acceptance model and the microsaccade lever arm,
            # so it is computed whenever focal_um is available, no matter of which acceptance model is used
            if bundle.focal_um is not None:
                D = self._buffer.lens_static_data['lens_diameter_um'].astype(np.float32)
                pos_D = D[D > 0]
                D_med = float(np.median(pos_D)) if pos_D.size else 1.0

                f_number = float(bundle.focal_um) / max(D_med, 1e-6)
                f_per_lens = (f_number * D).astype(np.float32)
                f_per_lens = np.where(
                    f_per_lens > 1e-3, f_per_lens, np.float32(bundle.focal_um)
                ).astype(np.float32)
                self._buffer.lens_static_data['focal_um'] = f_per_lens

            # Fill receptors static data
            self._buffer.rcpt_static_data['position'] = np.repeat(self._lens_positions, R, axis=0)
            self._buffer.rcpt_static_data['sensitivity'] = np.tile(bundle.sensitivity, (N, 1))
            self._buffer.rcpt_static_data['tau_membrane'] = bundle.tau_membrane
            self._buffer.rcpt_static_data['rhab_diameter_um'] = np.tile(bundle.diameters_um, N)
            self._buffer.rcpt_static_data['wavelength_um'] = np.tile(bundle.wavelengths_nm * 1e-3, N)

            # Acceptance angles (rest + initial dynamic)
            acc = self._resolve_acceptance()
            self._buffer.rcpt_static_data['rest_acc'] = acc
            self._buffer.rcpt_dynamic_data['acc_axes'] = acc

            # Pack metadata bits (chirality_neg filled in _apply_orientation)
            lens_indices = np.repeat(np.arange(N, dtype=np.uint32), R)
            rcpt_types = np.tile(np.arange(R, dtype=np.uint32), N)
            eye_indices_per_rcpt = np.repeat(self._lens_eye_index, R)
            neighbour_counts = np.zeros(N * R, dtype=np.uint32)
            binoc_area = np.repeat(self._compute_binocular_mask(), R)

            self._buffer.pack_metadata(
                eye_indices=eye_indices_per_rcpt,
                receptor_types=rcpt_types,
                neighbour_counts=neighbour_counts,
                lens_indices=lens_indices,
                chirality_neg=0,
                binocular_area=binoc_area,
            )

            # Fill neighbours count from neighbour graph
            for eye in self._eyes:
                graph = eye._ensure_neighbour_graph(k=6)
                n_in_eye_per_lens = graph.shape[1]
                rcpt_indices_eye = (
                    eye._lens_indices[:, None] * R + np.arange(R, dtype=np.intp)[None, :]
                ).ravel()

                self._buffer.rcpt_static_data['metadata'][rcpt_indices_eye] = set_metadata_field(
                    self._buffer.rcpt_static_data['metadata'][rcpt_indices_eye],
                    'neighbour_count',
                    n_in_eye_per_lens,
                )

            # Orientation pipeline

            # placeholders (_apply_orientation will overwrite)
            self._bundle_orientation = np.zeros(N, dtype=np.float32)
            self._chirality_arr = np.ones(N, dtype=np.float32)
            self._saccade_cache = np.zeros((N, 3), dtype=np.float32)

            # Use the real pipeline if explicitly requested or if R > 1
            use_pipeline = (orientation is not None) or (flow_direction is not None) or (R > 1)

            if bundle_orientations is not None and chiralities is not None:
                result = apply_chirality(self, bundle_orientations, chiralities)
            elif use_pipeline:
                if orientation is None:
                    if flow_direction is not None:
                        orientation = BundlesAligner(flow_direction)
                    else:
                        orientation = BundlesAligner(-WORLD_FORWARD)
                result = orientation.compute(
                    self,
                    override_chi=bundle_orientations,
                    override_chirality=chiralities,
                )
            else:
                result = trivial_orientation(N)
                result.saccade_phasor = self._local_up.copy()
                # result.saccade_phasor = self._local_right.copy()

            self._apply_orientation(result)

            # Cartridge mapping and diagnostics
            # (buffer.cartridge_map starts as identity, buffer.cartridges_wired starts False)
            self.donation_conflicts = np.zeros(N, dtype=bool)
            self.receiving_conflicts = np.zeros(N, dtype=bool)
            self.have_conflicts = np.zeros(N, dtype=bool)
            self._buffer.lens_dirty = True
            self._buffer.receptors_dirty = True

            if R > 1:
                self.wire_cartridges_adaptive()
            else:
                self._buffer.rcpt_static_data['cartridge_src'] = np.arange(N * R, dtype=np.uint32)

            self.set_retinal_direction(muscle_direction=retina_muscle_direction or WORLD_UP)

    def unlock(self, lenses: Optional[bool] = None, receptors: Optional[bool] = None):
        """
        Context manager to temporarily allow CPU-side modifications to the buffers.
        Delegates to the underlying EyesBuffer.
        If neither is specified, both are unlocked.
        If one is specified, the other remains locked.
        """
        return self._buffer.unlock(lenses=lenses, receptors=receptors)

    # Factory methods

    @classmethod
    def from_sphere(cls,
        n: int = 2000,
        eye_radius: float = 0.01,
        method: str = 'icosphere',
        force_isotropic: bool = False,
        **kwargs,
        ) -> 'CompoundEyeModel':
        """
        Construct a uniform spherical compound eye.

        Args:
            - n: int, Approximate number of ommatidia.
            - eye_radius: float, Sphere radius in world units (default 0.01 m = 10 mm).
            - method: {'icosphere', 'fibonacci'}, Spherical sampling method.
            - **kwargs: Forwarded to __init__.
        """
        method = method.lower()

        if method == 'icosphere':
            dirs = icosphere(n)
        elif method == 'fibonacci':
            dirs = fibonacci_sphere(n)
        else:
            raise ValueError("Method must be 'icosphere' or 'fibonacci'")

        if force_isotropic:
            # Theoretical IOA for N lenses tiled hexagonally on a sphere
            theoretical_ioa = np.sqrt((4.0 * np.pi) / (n * np.sqrt(3.0) / 2.0))
            kwargs['interommatidial_angles_rad'] = [theoretical_ioa, theoretical_ioa]

        positions = (dirs * float(eye_radius)).astype(np.float32)
        return cls(directions=dirs, positions=positions, **kwargs)

    @classmethod
    def from_lenses(cls,
        directions: ArrayLike,
        positions: ArrayLike,
        **kwargs,
        ) -> 'CompoundEyeModel':
        """
        Explicit lens placement. Forwards to __init__.
        """
        return cls(directions=directions, positions=positions, **kwargs)

    @classmethod
    def from_file(cls, path: str, **kwargs) -> 'CompoundEyeModel':
        """
        Load a species model from a .npz archive of raw geometry.

        This is a geometry-level loader: it reads lens positions and
        directions (plus a few optional fields) and re-runs the full
        construction pipeline.

        To save/load an already-built model (post-orientation, post-wiring),
        use EyesBuffer.to_file() / EyesBuffer.from_file().

        Required fields (any of these names accepted, in order of preference):
            - positions: (N, 3), lens world positions
                aliases: 'positions', 'pos', 'lens_positions'
            - directions: (N, 3), lens optical axes
                aliases: 'directions', 'dirs', 'lens_directions', 'forward'

        Optional fields (used when present, otherwise defaults apply):
            - eye_indices: 'eye_indices', 'eye_ids', 'eye_id', 'eye_idx', 'eye_index'
            - left / right: (N,) bool, fallback if no eye_indices
            - lens_diameter_um: 'lens_diameter_um', 'lens_diameter', 'aperture_um'
            - interommatidial_angles_rad: 'interommatidial_angles_rad', 'ioa', 'ioa_axes'
            - acceptance (rest half-widths): 'acceptance_angles', 'acceptance', 'rho'
            - bundle_orientations: 'bundle_orientations', 'chi'
            - chiralities: 'chiralities', 'chirality'

        Any keyword argument forwarded explicitly via **kwargs overrides what's found in the file.
        """

        with np.load(path, allow_pickle=False) as data:
            available = [s.lower() for s in data.files]

            def first_present(candidates):
                for name in candidates:
                    if name in available:
                        return np.asarray(data[name])
                return None

            positions = first_present(['positions', 'pos', 'lens_positions'])
            directions = first_present(['directions', 'dirs', 'lens_directions', 'forward'])
            if positions is None or directions is None:
                raise ValueError(
                    f"{path}: required 'positions' and 'directions' not found. "
                    f"Available keys: {available}"
                )

            # Resolve eye ids: explicit field, else infer from is_left / is_right.
            # If neither is present, falls back to island detection, and auto-assign sides from centroid x.
            if 'eye_indices' not in kwargs:
                eye_indices = first_present(['eye_idx', 'eye_ids', 'eye_id', 'eye_index'])
                if eye_indices is None:
                    is_left = first_present(['l', 'left', 'is_left'])
                    if is_left is not None:
                        # Convention: 0 = left, 1 = right
                        eye_indices = (~is_left.astype(bool)).astype(np.uint32)
                    else:
                        is_right = first_present(['r', 'right', 'is_right'])
                        if is_right is not None:
                            eye_indices = is_right.astype(np.uint32)
                if eye_indices is not None:
                    kwargs['eye_indices'] = eye_indices

            # Optional geometry fields: only set if not overridden in kwargs.
            optional_field_aliases = {
                'lens_diameter_um': ['lens_diameter_um', 'lens_diameter', 'aperture_um'],
                'interommatidial_angles_rad': ['interommatidial_angles_rad', 'interommatidial_angles', 'ioa', 'ioa_axes'],
                'bundle_orientations': ['bundle_orientations', 'chi'],
                'chiralities': ['chiralities', 'chirality'],
            }
            for param, aliases in optional_field_aliases.items():
                if param in kwargs:
                    continue
                value = first_present(aliases)
                if value is not None:
                    kwargs[param] = value

            if 'acceptance' not in kwargs:
                rho = first_present(['acceptance_angles_rad', 'acceptance_angles', 'acceptance', 'rho'])
                if rho is not None:
                    kwargs['acceptance'] = ExplicitAcceptance(rho)

        logger.info(
            "Loaded %d lenses from %s (fields: %s)",
            positions.shape[0], path, available,
        )
        return cls(directions=directions, positions=positions, **kwargs)

    # Some basic stuff

    def __repr__(self) -> str:
        return (
            f"CompoundEyeModel(N={self.lens_count}, R={self.receptors_per_lens}, "
            f"eyes={len(self._eyes)}, bundle={self._bundle.name!r})"
        )

    def __len__(self) -> int:
        return int(self._lens_directions.shape[0])

    # Locks

    @property
    def allow_lens_writes(self) -> bool:
        return self._buffer._allow_lens_writes

    @allow_lens_writes.setter
    def allow_lens_writes(self, value: bool):
        self._buffer._allow_lens_writes = bool(value)

    @property
    def allow_receptor_writes(self) -> bool:
        return self._buffer._allow_rcpt_writes

    @allow_receptor_writes.setter
    def allow_receptor_writes(self, value: bool):
        self._buffer._allow_rcpt_writes = bool(value)

    # Properties

    @property
    def buffer(self) -> EyesBuffer:
        """The underlying packed-data buffer (GPU-ready)."""
        return self._buffer

    @property
    def lens_count(self) -> int:
        return len(self)

    @property
    def receptors_per_lens(self) -> int:
        return self._bundle.count

    @property
    def total_receptors(self) -> int:
        return self.lens_count * self.receptors_per_lens

    @property
    def bundle(self) -> 'RhabdomereBundle':
        return self._bundle

    @property
    def eyes(self) -> List[Eye]:
        return list(self._eyes)

    def eye(self, eye_index: int) -> Eye:
        for e in self._eyes:
            if e.eye_index == int(eye_index):
                return e
        raise KeyError(f"No eye with index {eye_index}")

    def eyes_by_side(self, side: str) -> List[Eye]:
        """All eyes on a given side ('left', 'right', or 'midline')."""
        s = str(side)
        return [e for e in self._eyes if e.side == s]

    @property
    def left_eyes(self) -> List[Eye]:
        return self.eyes_by_side('left')

    @property
    def right_eyes(self) -> List[Eye]:
        return self.eyes_by_side('right')

    @property
    def midline_eyes(self) -> List[Eye]:
        return self.eyes_by_side('midline')

    def lenses_by_side(self, side: str) -> LensView:
        """All lenses belonging to eyes on the given side."""
        eyes = self.eyes_by_side(side)
        if not eyes:
            return LensView(self, np.empty(0, dtype=np.intp))
        idx = np.concatenate([e.lens_indices for e in eyes])
        return LensView(self, idx)

    @property
    def left_lenses(self) -> LensView:
        return self.lenses_by_side('left')

    @property
    def right_lenses(self) -> LensView:
        return self.lenses_by_side('right')

    @property
    def midline_lenses(self) -> LensView:
        return self.lenses_by_side('midline')

    @property
    def lenses(self) -> LensView:
        return LensView(self, np.arange(self.lens_count, dtype=np.intp))

    @property
    def ommatidia(self) -> LensView:
        """Alias of .lenses, iterating yields Ommatidia."""
        return self.lenses

    @property
    def receptors(self) -> ReceptorView:
        return ReceptorView(self, np.arange(self.total_receptors, dtype=np.intp))

    @property
    def cartridges(self) -> List[Cartridge]:
        if not self._buffer.cartridges_wired:
            return []
        return [Cartridge(self, i) for i in range(self.lens_count)]

    @property
    def binocular_mask(self) -> np.ndarray:
        return self.lenses.is_binocular

    @property
    def binocular_fraction(self) -> float:
        return self.lenses.binocular_fraction

    # Buffer passthroughs

    @property
    def lens_static_data(self) -> np.ndarray:
        return self._buffer.lens_static_data

    @property
    def lens_dynamic_data(self) -> np.ndarray:
        return self._buffer.lens_dynamic_data

    @property
    def rcpt_static_data(self) -> np.ndarray:
        return self._buffer.rcpt_static_data

    @property
    def rcpt_dynamic_data(self) -> np.ndarray:
        return self._buffer.rcpt_dynamic_data

    @property
    def lens_dirty(self) -> bool:
        """True if lens-level data has been modified since the last renderer upload."""
        return self._buffer.lens_dirty

    @lens_dirty.setter
    def lens_dirty(self, value: bool) -> None:
        self._buffer.lens_dirty = bool(value)

    @property
    def cartridge_indices(self) -> np.ndarray:
        """(N, R) global receptor indices grouped by cartridge mapping."""
        return self._buffer.cartridge_indices

    # TODO: These legacy aliases (used by proxies.py & orientation.py) can be removed now
    @property
    def _cartridge_map(self) -> np.ndarray:
        return self._buffer.cartridge_map

    @property
    def _cartridges_wired(self) -> bool:
        return self._buffer.cartridges_wired

    @property
    def _lens_dirty(self) -> bool:
        return self._buffer.lens_dirty

    @_lens_dirty.setter
    def _lens_dirty(self, value: bool) -> None:
        self._buffer.lens_dirty = bool(value)

    # Animal-wide spatial queries (dispatch across eyes)

    def query_directions(self,
         directions: ArrayLike,
         k: int = 1,
         avoid_conflicts: bool = False
         ) -> Tuple['LensView', np.ndarray]:
        """
        Find the k lenses (across all eyes) whose optical axes lie closest to
        each query direction.

        Returns (LensView, distances).
        """
        dirs = np.asarray(directions, dtype=np.float32).reshape(-1, 3)
        Q = dirs.shape[0]
        best_idx = np.full((Q, k), -1, dtype=np.intp)
        best_dist = np.full((Q, k), np.inf, dtype=np.float32)

        for eye in self._eyes:
            view, dist = eye.query_directions(dirs, k=k, avoid_conflicts=avoid_conflicts)
            if len(view) == 0:
                continue

            combined_idx = np.concatenate([best_idx, view.indices.reshape(Q, -1)], axis=1)
            combined_dist = np.concatenate([best_dist, dist], axis=1)

            order = np.argsort(combined_dist, axis=1)[:, :k]
            rows = np.arange(Q)[:, None]
            best_idx = combined_idx[rows, order]
            best_dist = combined_dist[rows, order]

        return LensView(self, best_idx.ravel()), best_dist

    def query_positions(self,
        positions: ArrayLike,
        k: int = 1,
        avoid_conflicts: bool = False
        ) -> Tuple['LensView', np.ndarray]:
        """
        Find the k lenses (across all eyes) closest in world-space position
        to each query point.

        Returns (LensView, distances).
        """
        pts = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        Q = pts.shape[0]
        best_idx = np.full((Q, k), -1, dtype=np.intp)
        best_dist = np.full((Q, k), np.inf, dtype=np.float32)

        for eye in self._eyes:
            view, dist = eye.query_positions(pts, k=k, avoid_conflicts=avoid_conflicts)
            if len(view) == 0:
                continue

            combined_idx = np.concatenate([best_idx, view.indices.reshape(Q, -1)], axis=1)
            combined_dist = np.concatenate([best_dist, dist], axis=1)

            order = np.argsort(combined_dist, axis=1)[:, :k]
            rows = np.arange(Q)[:, None]
            best_idx = combined_idx[rows, order]
            best_dist = combined_dist[rows, order]

        return LensView(self, best_idx.ravel()), best_dist

    def query_lookat(self,
        targets: ArrayLike,
        k: int = 1,
        avoid_conflicts: bool = False
        ) -> 'LensView':
        """
        Find the k lenses (across all eyes) best looking at each world-space
        target point. See Eye.query_lookat for the scoring.

        Returns a LensView (best first).
        """
        if k < 1:
            raise ValueError("k must be >= 1")

        q = np.asarray(targets, dtype=np.float32).reshape(-1, 3)
        if self.lens_count == 0:
            return LensView(self, np.empty(0, dtype=np.intp))

        # Score against all lenses (across all eyes)
        pos = self._lens_positions
        dirs = self._lens_directions
        desired = q[:, None, :] - pos[None, :, :]
        norms = np.linalg.norm(desired, axis=-1, keepdims=True)
        np.divide(desired, norms, out=desired, where=norms > 0)
        dots = np.einsum('jk,ijk->ij', dirs, desired)

        if avoid_conflicts:
            dots[:, self.have_conflicts] = -np.inf

        k_eff = min(k, dots.shape[1])
        part = np.argpartition(dots, -k_eff, axis=1)[:, -k_eff:]
        top = np.take_along_axis(dots, part, axis=1)
        order = np.argsort(top, axis=1)[:, ::-1]
        best = np.take_along_axis(part, order, axis=1)

        return LensView(self, best.ravel().astype(np.intp))

    def query_cone(self,
        center_direction: ArrayLike,
        angle: float,
        degrees: bool = True,
        avoid_conflicts: bool = False
        ) -> 'LensView':
        """
        All lenses (across all eyes) whose optical axis lies within 'angle'
        of 'center_direction'.

        Returns a LensView.
        """
        hits = [eye.query_cone(center_direction, angle, degrees, avoid_conflicts) for eye in self._eyes]
        indices = np.concatenate([h._gi for h in hits]) if hits else np.empty(0, dtype=np.intp)
        return LensView(self, indices)

    def query_ball(self,
        center_position: ArrayLike,
        radius: float,
        avoid_conflicts: bool = False
        ) -> 'LensView':
        """
        All lenses (across all eyes) whose world position lies within 'radius'
        of 'center_position'.

        Returns a LensView.
        """
        hits = [eye.query_ball(center_position, radius, avoid_conflicts) for eye in self._eyes]
        indices = np.concatenate([h._gi for h in hits]) if hits else np.empty(0, dtype=np.intp)
        return LensView(self, indices)

    @property
    def max_gap(self) -> float:
        """
        Largest angular gap (in radians) between any lens and its nearest neighbour,
        over all eyes.
        """
        return float(max((e.max_gap() for e in self._eyes), default=0.0))

    def saccade_field(self) -> np.ndarray:
        """Per-lens microsaccade actuation axis in world coordinates. Shape (N, 3)."""
        return self._saccade_cache.copy()

    # Geometry transforms (on the whole array)

    def translate(self, offset: ArrayLike) -> 'CompoundEyeModel':
        """
        Translate all lens (and receptor) positions by 'offset'.
        """
        off = np.asarray(offset, dtype=np.float32).reshape(3)
        self._lens_positions += off
        self._buffer.rcpt_static_data['position'] += off
        self._invalidate_spatial()
        return self

    def scale(self, factor: float) -> 'CompoundEyeModel':
        """
        Scale all positions about the origin by 'factor'.

        Note: only world-scale geometry scales. Lens and rhabdomere
        diameters (μm) and the focal length (μm) are at ommatidium
        scale and are unchanged.
        """
        f = float(factor)
        self._lens_positions *= f
        self._buffer.rcpt_static_data['position'] *= f
        self._invalidate_spatial()
        return self

    def rotate(self, R: ArrayLike) -> 'CompoundEyeModel':
        """
        Rotate all positions, directions, and tangent frames by the 3x3
        rotation matrix 'R'.

        Cartridge wiring is preserved (relative directions are unchanged).
        """

        Rmat = np.asarray(R, dtype=np.float32)
        if Rmat.shape != (3, 3):
            raise ValueError(f"R must be 3x3, got {Rmat.shape}")

        if not np.allclose(Rmat @ Rmat.T, np.eye(3), atol=1e-3):
            logger.warning("rotate() called with non-orthonormal matrix, results may be off.")

        Rt = Rmat.T.astype(np.float32)

        self._lens_positions = self._lens_positions @ Rt
        self._lens_directions = self._lens_directions @ Rt
        self._local_right = self._local_right @ Rt
        self._local_up = self._local_up @ Rt

        self._buffer.lens_static_data['right'] = self._local_right
        self._buffer.lens_static_data['up'] = self._local_up
        self._buffer.lens_static_data['forward'] = self._lens_directions

        self._buffer.rcpt_static_data['position'] = self._buffer.rcpt_static_data['position'] @ Rt
        self._buffer.rcpt_dynamic_data['direction'] = self._buffer.rcpt_dynamic_data['direction'] @ Rt
        self._saccade_cache = self._saccade_cache @ Rt

        self._invalidate_spatial()
        return self

    # Orientation pipeline backwrite

    def _apply_orientation(self, result: OrientationResult) -> None:
        """
        Write an OrientationResult into corresponding data fields.
        Called by 'BundlesAligner.apply()' and during construction.
        """
        N, R = self.lens_count, self.receptors_per_lens
        chi = result.chi.astype(np.float32)
        chirality = result.chirality.astype(np.float32)
        sacc = result.saccade_phasor.astype(np.float32)

        if chi.shape != (N,) or chirality.shape != (N,) or sacc.shape != (N, 3):
            raise ValueError(
                f"OrientationResult shapes (chi {chi.shape}, chirality "
                f"{chirality.shape}, saccade {sacc.shape}) do not match N={N}"
            )

        # Per-lens
        self._bundle_orientation = chi
        self._chirality_arr = chirality
        self._buffer.lens_static_data['sacc_x'] = np.einsum('ij,ij->i', sacc, self._local_right)
        self._buffer.lens_static_data['sacc_y'] = np.einsum('ij,ij->i', sacc, self._local_up)

        # Per-receptor: rotated focal-plane offsets
        rot_dx, rot_dy = self._bundle.rotated_offsets(chi, chirality)
        rot_offset = np.stack([rot_dx.ravel(), rot_dy.ravel()], axis=-1).astype(np.float32)
        self._buffer.rcpt_static_data['rot_offset'] = rot_offset

        if R > 1:
            # Complex eyes: Tilt follows the bundle yaw (chi)
            self._buffer.rcpt_static_data['acc_tilt'] = np.repeat(chi, R).astype(np.float32)
        else:
            # Ommatidium model (R=1): Tilt follows the hexagonal lattice tiling
            lattice_tilts = self._buffer.lens_static_data['ioa_tilt']
            self._buffer.rcpt_static_data['acc_tilt'] = lattice_tilts.astype(np.float32)

        # Per-receptor: chirality_neg bit in metadata
        is_mirrored = (np.repeat(chirality, R) < 0).astype(np.uint32)
        self._buffer.set_metadata('chirality_neg', is_mirrored)

        # Per-receptor: actuated direction (rest direction post-orientation)
        rec_dirs, _ = self._compute_receptor_geometry()
        self._buffer.rcpt_dynamic_data['direction'] = rec_dirs

        self._saccade_cache = sacc.copy()
        self._buffer.lens_dirty = True

    def _compute_receptor_geometry(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Per-receptor world-space direction and position.

        Direction is the unit vector from the rotated rhabdomere tip back
        through the nodal point (i.e. the receptor's line of sight on the
        outside world).
        """
        bundle = self._bundle
        N, R = self.lens_count, self.receptors_per_lens
        nd = bundle.focal_um
        if nd is None:
            if R > 1:
                raise ValueError("bundle.focal_um is None but R > 1")
            nd = 1.0

        rot_dx, rot_dy = bundle.rotated_offsets(self._bundle_orientation, self._chirality_arr)

        local_tip = np.stack(
            [rot_dx, rot_dy, np.full((N, R), -nd, dtype=np.float32)],
            axis=-1,
        )
        world_tip = (
            local_tip[..., 0:1] * self._local_right[:, None, :]
            + local_tip[..., 1:2] * self._local_up[:, None, :]
            + local_tip[..., 2:3] * self._lens_directions[:, None, :]
        ).reshape(N * R, 3)

        rec_pos = np.repeat(self._lens_positions, R, axis=0)
        rec_dirs = normalise_vectors(-world_tip).astype(np.float32)
        return rec_dirs, rec_pos.astype(np.float32)

    def _compute_binocular_mask(self, angle_threshold_deg: float = None) -> np.ndarray:
        """
        Identifies lenses that have overlapping visual fields with contralateral eyes.
        """
        N = self.lens_count
        is_binocular = np.zeros(N, dtype=bool)

        for eye in self._eyes:
            # Get all lenses *not* in this eye
            other_lenses_idx = np.setdiff1d(np.arange(N), eye.lens_indices)
            if other_lenses_idx.size == 0:
                continue

            # Build a temporary tree for the other eyes
            other_dirs = self._lens_directions[other_lenses_idx]
            other_tree = cKDTree(other_dirs)

            my_dirs = self._lens_directions[eye.lens_indices]

            # If threshold not provided use 1.1 local IOA
            if angle_threshold_deg is None:
                threshold_rad = np.mean(self._buffer.lens_static_data['ioa_axes'][eye.lens_indices, 1]) * 1.1
            else:
                threshold_rad = np.radians(angle_threshold_deg)

            chord_threshold = 2.0 * np.sin(threshold_rad / 2.0)

            # Find if any other eye has a lens within threshold
            indices = other_tree.query_ball_point(my_dirs, r=chord_threshold)

            eye_binoc_mask = np.array([len(hits) > 0 for hits in indices])
            is_binocular[eye.lens_indices] = eye_binoc_mask

        return is_binocular

    # Cartridges (neural-superposition wiring)

    def wire_cartridges_rigid(self,
        assign_radius: float = 0.5,
        angular_dev: float = 35.0,
        scale_dev: float = 0.7,
        min_snap_matches: int = 2,
        neighbour_dist_factor: float = 1.25,
        k_search: int = 20,
        top_k: int = 8,
        identity_bias: float = 0.05,
        slack_cost: float = 1000.0,
        time_limit_s: float = 60.0,
        ) -> None:
        """
        Neural superposition wiring using rigid bundle templates and joint optimization.

        This method assumes the rhabdomere bundle maintains a consistent geometric
        structure. It uses similarity transforms (rotation + scale) to "snap" the
        idealised rhabdomere bundle template onto the ommatidial lattice.

        Args:
            - assign_radius: Max distance (μm) to consider a rhabdomere "hitting" a lens.
            - angular_dev: Max angular deviation (deg) allowed for bundle rotation.
            - scale_dev: Max scaling allowed (fraction) relative to the bundle size.
            - min_snap_matches: Minimum number of rhabdomeres that must align for a valid candidate.
            - neighbour_dist_factor: Multiplier to define "immediate" lattice neighbors.
            - k_search: Number of nearest neighbors to search for candidates.
            - top_k: Number of best-scoring candidates per lens to pass to the MILP solver.
            - slack_cost: Penalty for leaving a lens completely unwired.
            - identity_bias: Preference for the "default" orientation (0 rotation, 1 scale).
            - time_limit_s: Hard timeout for the Mixed-Integer Linear Programming solver.
        """
        N, R = self.lens_count, self.receptors_per_lens
        if R == 1:
            return

        center = self._bundle.center_index
        periph = np.array([i for i in range(R) if i != center], dtype=np.intp)
        P = periph.size

        cartridge_map = np.full((N, R), -1, dtype=np.intp)
        cartridge_map[:, center] = np.arange(N)

        rot_dx, rot_dy = self._bundle.rotated_offsets(
            self._bundle_orientation, self._chirality_arr)
        kp = self._bundle.offsets_um[periph] - self._bundle.offsets_um[center]
        k_scale = float(np.mean(np.linalg.norm(kp, axis=1))) or 1.0
        tpl_dx = (rot_dx - rot_dx[:, center:center + 1]) / k_scale
        tpl_dy = (rot_dy - rot_dy[:, center:center + 1]) / k_scale
        ang_gate = np.radians(angular_dev)

        for eye in self._eyes:
            n_e = len(eye)
            if n_e < 2:
                continue

            neighb = eye.neighbours(
                lens_indices=eye._lens_indices,
                k=min(k_search, n_e - 1),
                neighbour_dist_factor=neighbour_dist_factor,
            )
            if not neighb:
                continue

            gi = eye._lens_indices
            immed = neighb.is_immediate
            same_chi = neighb.same_chirality
            no_self = neighb.indices != gi[:, None]

            with np.errstate(all='ignore'):
                spacing = np.nanmedian(np.where(immed, neighb.distances, np.nan), axis=1)
            spacing = np.where(np.isfinite(spacing) & (spacing > 1e-9), spacing, 1.0)

            dirs = self._lens_directions[gi]
            nbr_dirs = self._lens_directions[neighb.indices]
            delta = nbr_dirs - dirs[:, None, :]
            du = np.einsum('ijk,ik->ij', delta, self._local_right[gi])
            dv = np.einsum('ijk,ik->ij', delta, self._local_up[gi])

            # Normalised by angular spacing per lens
            nb_mag = np.sqrt(du ** 2 + dv ** 2)
            ang_spacing = np.nanmedian(np.where(immed, nb_mag, np.nan), axis=1)
            ang_spacing = np.where(np.isfinite(ang_spacing) & (ang_spacing > 1e-9), ang_spacing, 1.0)

            z_nb = (du + 1j * dv) / ang_spacing[:, None]

            z_tpl = (tpl_dx[gi][:, periph] + 1j * tpl_dy[gi][:, periph])

            # Stage 1: enumerate candidates per lens
            # candidates[i_loc] = list of (cost, donors_global, slots_global)
            candidates = [[] for _ in range(n_e)]

            for i_loc in range(n_e):
                if not immed[i_loc].any():
                    continue
                tpl_i = z_tpl[i_loc]
                nb_i = z_nb[i_loc]
                valid_nb = no_self[i_loc] & same_chi[i_loc]
                if not valid_nb.any():
                    continue

                mag_ok = np.abs(tpl_i) > 1e-8
                if not mag_ok.any():
                    continue

                # Build similarity-transform candidate set
                ws_set = [1.0 + 0j]
                nb_valid_idx = np.flatnonzero(valid_nb)
                for i_a in np.flatnonzero(mag_ok):
                    ws = nb_i[nb_valid_idx] / tpl_i[i_a]
                    ok = (np.abs(np.angle(ws)) <= ang_gate) & (np.abs(np.abs(ws) - 1.0) <= scale_dev)
                    if ok.any():
                        ws_set.extend(ws[ok].tolist())

                # For each w, run a local Hungarian, build a candidate assignment
                seen = set()
                lens_cands = []
                d_inf = np.full_like(np.abs(nb_i[None, :]), np.inf, dtype=float)

                for w in ws_set:
                    z_placed = w * tpl_i
                    d = np.abs(z_placed[:, None] - nb_i[None, :])
                    d = np.where(valid_nb[None, :], d, np.inf)
                    # Pad to square so linear_sum_assignment can run if k > P
                    r_idx, c_idx = linear_sum_assignment(d)
                    md = d[r_idx, c_idx]
                    keep = (md < assign_radius) & np.isfinite(md)
                    if int(keep.sum()) < min_snap_matches:
                        continue
                    donors_g = neighb.indices[i_loc][c_idx[keep]]
                    slots_g = periph[r_idx[keep]]
                    # Dedupe key: assignment as sorted (slot, donor) tuples
                    key = tuple(sorted(zip(slots_g.tolist(), donors_g.tolist())))
                    if key in seen:
                        continue
                    seen.add(key)
                    cost = float(md[keep].sum()) + identity_bias * float(np.abs(w - 1.0))
                    lens_cands.append((cost, donors_g, slots_g))

                lens_cands.sort(key=lambda t: t[0])
                candidates[i_loc] = lens_cands[:top_k]

            # Stage 2: MILP selection
            # One null candidate per lens at slack cost so unwireable rims fall through
            slack_cost = slack_cost * assign_radius * P

            # Variable layout: x[lens_var_start[i_loc] + ci]
            n_vars = sum(len(c) + 1 for c in candidates)  # +1 for null candidate per lens
            if n_vars == 0:
                self._finalize_wiring(cartridge_map)
                return

            costs = np.zeros(n_vars, dtype=float)
            # var -> (i_loc, c_idx, donors, slots), null candidate has empty arrays
            var_meta = []
            lens_var_start = np.zeros(n_e + 1, dtype=np.intp)
            v = 0
            for i_loc in range(n_e):
                lens_var_start[i_loc] = v
                for cost, donors_g, slots_g in candidates[i_loc]:
                    costs[v] = cost
                    var_meta.append((i_loc, donors_g, slots_g))
                    v += 1
                # null candidate
                costs[v] = slack_cost
                var_meta.append((i_loc, np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)))
                v += 1
            lens_var_start[n_e] = v

            # C1: one candidate per lens (equality)
            rows1 = np.repeat(np.arange(n_e), np.diff(lens_var_start))
            cols1 = np.arange(n_vars)
            data1 = np.ones(n_vars, dtype=float)
            A1 = coo_matrix((data1, (rows1, cols1)), shape=(n_e, n_vars)).tocsr()

            # C2: each (donor j, slot r) used at most once
            ds_to_row = {}
            rows2, cols2 = [], []
            for vid, (_, donors_g, slots_g) in enumerate(var_meta):
                for jg, sg in zip(donors_g.tolist(), slots_g.tolist()):
                    key = (jg, sg)
                    if key not in ds_to_row:
                        ds_to_row[key] = len(ds_to_row)
                    rows2.append(ds_to_row[key])
                    cols2.append(vid)
            n_c2 = len(ds_to_row)
            if n_c2 > 0:
                data2 = np.ones(len(rows2), dtype=float)
                A2 = coo_matrix((data2, (rows2, cols2)), shape=(n_c2, n_vars)).tocsr()
                cons = [
                    LinearConstraint(A1, lb=1.0, ub=1.0),
                    LinearConstraint(A2, lb=0.0, ub=1.0),
                ]
            else:
                cons = [LinearConstraint(A1, lb=1.0, ub=1.0)]

            result = milp(
                c=costs,
                constraints=cons,
                bounds=Bounds(0, 1),
                integrality=np.ones(n_vars, dtype=int),
                options={'time_limit': time_limit_s, 'disp': False},
            )

            if not result.success:
                logger.warning(f"Eye {eye.eye_index}: MILP failed ({result.message}); leaving unwired.")
                continue

            chosen = np.flatnonzero(result.x > 0.5)
            for vid in chosen:
                _, donors_g, slots_g = var_meta[vid]
                if donors_g.size == 0:
                    continue
                # var_meta stores the host lens implicitly via lens_var_start; recover it:
                i_loc = int(np.searchsorted(lens_var_start, vid, side='right') - 1)
                cartridge_map[gi[i_loc], slots_g] = donors_g

        self._finalize_wiring(cartridge_map)

    def wire_cartridges_adaptive(self,
        assign_radius: float = 0.5,
        angular_dev: float = 40.0,
        scale_dev: float = 0.75,
        min_snap_matches: int = 2,
        neighbour_dist_factor: float = 1.25,
        k_search: int = 20,
        top_k: int = 6,
        identity_bias: float = 0.05,
        unassigned_penalty: float = 10.0,
        allow_plasticity: bool = True,
        plasticity_radius: float = 2.0,
        time_limit_s: float = 60.0,
        ) -> None:
        """
        Neural superposition wiring allowing for local distortions and missing connections.

        This method is more robust than `wire_rigid`. It allows individual rhabdomere
        slots within a cartridge to be left unwired (with a penalty) and can use
        "plastic" connections in highly distorted lattice regions where a rigid
        transformation would fail.

        Args:
            - assign_radius: Max distance (μm) for standard rhabdomere-to-lens assignment.
            - angular_dev: Max angular deviation (deg) allowed for bundle rotation.
            - scale_dev: Max scaling allowed (fraction) relative to the bundle size.
            - min_snap_matches: Minimum number of rhabdomeres that must align for a valid candidate.
            - neighbour_dist_factor: Multiplier to define "immediate" lattice neighbors.
            - k_search: Number of nearest neighbors to search for candidates.
            - top_k: Number of best-scoring candidates per lens to pass to the MILP solver.
            - identity_bias: Preference for the "default" orientation (0 rotation, 1 scale).
            - unassigned_penalty: Cost incurred for every receptor slot left unwired.
            - allow_plasticity: If True, allows 'best-effort' wiring in distorted regions.
            - plasticity_radius: Increased search radius (μm) used when in plastic mode.
            - time_limit_s: Hard timeout for the MILP solver.
        """
        N, R = self.lens_count, self.receptors_per_lens
        if R == 1:
            return

        center = self._bundle.center_index
        periph = np.array([i for i in range(R) if i != center], dtype=np.intp)

        cartridge_map = np.full((N, R), -1, dtype=np.intp)
        cartridge_map[:, center] = np.arange(N)

        rot_dx, rot_dy = self._bundle.rotated_offsets(self._bundle_orientation, self._chirality_arr)
        kp = self._bundle.offsets_um[periph] - self._bundle.offsets_um[center]
        k_scale = float(np.mean(np.linalg.norm(kp, axis=1))) or 1.0
        tpl_dx = (rot_dx - rot_dx[:, center:center + 1]) / k_scale
        tpl_dy = (rot_dy - rot_dy[:, center:center + 1]) / k_scale
        ang_gate = np.radians(angular_dev)

        for eye in self._eyes:
            n_e = len(eye)
            if n_e < 2:
                continue

            neighb = eye.neighbours(lens_indices=eye._lens_indices, k=min(k_search, n_e - 1),
                                    neighbour_dist_factor=neighbour_dist_factor)
            gi = eye._lens_indices
            with np.errstate(all='ignore'):
                spacing = np.nanmedian(np.where(neighb.is_immediate, neighb.distances, np.nan), axis=1)
            spacing = np.where(np.isfinite(spacing) & (spacing > 1e-9), spacing, 1.0)

            # Stage 1: Enumerate candidates
            candidates = [[] for _ in range(n_e)]
            for i_loc in range(n_e):
                i_glob = gi[i_loc]
                tpl_i = (tpl_dx[i_glob] + 1j * tpl_dy[i_glob])

                # Setup neighbour coordinates in tangent plane (using optical axes)
                nb_dirs = self._lens_directions[neighb.indices[i_loc]] - self._lens_directions[i_glob]
                nb_uv = (nb_dirs @ np.stack([self._local_right[i_glob], self._local_up[i_glob]], axis=1))

                # Normalised by local angular spacing
                nb_mag = np.linalg.norm(nb_uv, axis=1)
                ang_spacing = np.nanmedian(np.where(neighb.is_immediate[i_loc], nb_mag, np.nan))
                if np.isnan(ang_spacing) or ang_spacing < 1e-9:
                    ang_spacing = 1.0

                nb_i = (nb_uv[:, 0] + 1j * nb_uv[:, 1]) / ang_spacing

                valid_nb = (neighb.indices[i_loc] != i_glob) & neighb.same_chirality[i_loc]

                # Standard candidate generation
                ws_set = [1.0 + 0j]
                mag_ok = np.abs(tpl_i) > 1e-8
                if mag_ok.any():
                    nb_valid_idx = np.flatnonzero(valid_nb)
                    for i_a in np.flatnonzero(mag_ok):
                        ws = nb_i[nb_valid_idx] / tpl_i[i_a]
                        ok = (np.abs(np.angle(ws)) <= ang_gate) & (np.abs(np.abs(ws) - 1.0) <= scale_dev)
                        ws_set.extend(ws[ok].tolist())

                lens_cands, seen = [], set()
                for w in ws_set:
                    d = np.abs((w * tpl_i)[:, None] - nb_i[None, :])
                    d = np.where(valid_nb[None, :], d, np.inf)
                    r_idx, c_idx = linear_sum_assignment(d)
                    md = d[r_idx, c_idx]

                    keep = (md < assign_radius) & np.isfinite(md)
                    if int(keep.sum()) >= min_snap_matches:
                        donors_g, slots_g = neighb.indices[i_loc][c_idx[keep]], periph[r_idx[keep]]
                        key = tuple(sorted(zip(slots_g, donors_g)))
                        if key not in seen:
                            seen.add(key)
                            lens_cands.append((identity_bias * float(np.abs(w - 1.0)), slots_g, donors_g, md[keep]))

                # Fallback: if no candidates found, generate one 'plastic' candidate
                if not lens_cands and allow_plasticity:
                    tpl_p = tpl_i[periph]
                    d = np.abs(tpl_p[:, None] - nb_i[None, :])
                    d = np.where(valid_nb[None, :], d, np.inf)

                    r_idx, c_idx = linear_sum_assignment(d)
                    md = d[r_idx, c_idx]

                    keep = (md < plasticity_radius) & np.isfinite(md)
                    if keep.any():
                        lens_cands.append((
                            unassigned_penalty * 0.5,
                            periph[r_idx[keep]],
                            neighb.indices[i_loc][c_idx[keep]],
                            md[keep]
                        ))

                lens_cands.sort(key=lambda t: t[0])
                candidates[i_loc] = lens_cands[:top_k]

            # Stage 2: MILP selection (with graceful drops)
            costs = []
            z_vars, x_vars, d_vars = [], [], []
            var_count = 0

            for i_loc in range(n_e):
                cands = candidates[i_loc]
                if not cands:
                    # Isolated lens fallback
                    cands = [(0.0, np.array([]), np.array([]), np.array([]))]

                i_z, i_x = [], []
                for w_cost, slots_g, donors_g, dists in cands:
                    # Template selection variable (z)
                    i_z.append(var_count)
                    costs.append(w_cost)
                    var_count += 1

                    # Individual slot connection variables (x)
                    c_x = {}
                    for r, j, dist in zip(slots_g, donors_g, dists):
                        c_x[r] = (var_count, j)
                        costs.append(dist)
                        var_count += 1
                    i_x.append(c_x)

                z_vars.append(i_z)
                x_vars.append(i_x)

                # Drop variables (d)
                i_d = {}
                for r in periph:
                    i_d[r] = var_count
                    costs.append(unassigned_penalty)
                    var_count += 1
                d_vars.append(i_d)

            if var_count == 0:
                continue

            rows_eq, cols_eq, data_eq, b_eq = [], [], [], []
            rows_ub, cols_ub, data_ub, b_ub = [], [], [], []
            eq_row, ub_row = 0, 0

            # Constraint 1: Pick exactly one candidate rotation per lens
            for i_loc in range(n_e):
                for z_idx in z_vars[i_loc]:
                    rows_eq.append(eq_row)
                    cols_eq.append(z_idx)
                    data_eq.append(1.0)
                b_eq.append(1.0)
                eq_row += 1

            # Constraint 2: Every slot is either wired (via chosen x) or dropped (d)
            for i_loc in range(n_e):
                for r in periph:
                    for c_x in x_vars[i_loc]:
                        if r in c_x:
                            x_idx, _ = c_x[r]
                            rows_eq.append(eq_row)
                            cols_eq.append(x_idx)
                            data_eq.append(1.0)
                    rows_eq.append(eq_row)
                    cols_eq.append(d_vars[i_loc][r])
                    data_eq.append(1.0)
                    b_eq.append(1.0)
                    eq_row += 1

            # Constraint 3: Can only wire a slot if its parent template candidate is chosen
            for i_loc in range(n_e):
                for c_idx, c_x in enumerate(x_vars[i_loc]):
                    z_idx = z_vars[i_loc][c_idx]
                    for r, (x_idx, _) in c_x.items():
                        rows_ub.append(ub_row)
                        cols_ub.append(x_idx)
                        data_ub.append(1.0)

                        rows_ub.append(ub_row)
                        cols_ub.append(z_idx)
                        data_ub.append(-1.0)

                        b_ub.append(0.0)
                        ub_row += 1

            # Constraint 4: Global donor limit (no donation conflicts)
            jr_to_x = {}
            for i_loc in range(n_e):
                for c_x in x_vars[i_loc]:
                    for r, (x_idx, j) in c_x.items():
                        key = (j, r)
                        if key not in jr_to_x:
                            jr_to_x[key] = []
                        jr_to_x[key].append(x_idx)

            for x_list in jr_to_x.values():
                for x_idx in x_list:
                    rows_ub.append(ub_row)
                    cols_ub.append(x_idx)
                    data_ub.append(1.0)
                b_ub.append(1.0)
                ub_row += 1

            # Build and solve MILP
            constraints = []
            if len(b_eq) > 0:
                A_eq = coo_matrix((data_eq, (rows_eq, cols_eq)), shape=(len(b_eq), var_count))
                constraints.append(LinearConstraint(A_eq, lb=b_eq, ub=b_eq))
            if len(b_ub) > 0:
                A_ub = coo_matrix((data_ub, (rows_ub, cols_ub)), shape=(len(b_ub), var_count))
                constraints.append(LinearConstraint(A_ub, lb=-np.inf, ub=b_ub))

            result = milp(
                c=costs,
                constraints=constraints,
                bounds=Bounds(0, 1),
                integrality=np.ones(var_count, dtype=int),
                options={'time_limit': time_limit_s, 'mip_rel_gap': 0.01, 'disp': False},
            )

            if not result.success:
                import logging
                logging.getLogger(__name__).warning(
                    f"Eye {eye.eye_index}: MILP failed ({result.message}). Falling back to unassigned."
                )
                continue

            # Map the MILP solution back to the cartridge map
            chosen = set(np.flatnonzero(result.x > 0.5))
            for i_loc in range(n_e):
                i_glob = gi[i_loc]
                for c_x in x_vars[i_loc]:
                    for r, (x_idx, j) in c_x.items():
                        if x_idx in chosen:
                            cartridge_map[i_glob, r] = j

        self._finalize_wiring(cartridge_map)

    def _finalize_wiring(self, cartridge_map: np.ndarray) -> None:

        N = self.lens_count
        R = self.receptors_per_lens

        center = self._bundle.center_index
        periph = np.array([i for i in range(R) if i != center], dtype=np.intp)

        self._buffer.cartridge_map = cartridge_map

        cs_signed = cartridge_map * R + np.arange(R)
        unwired_mask = (cartridge_map < 0)

        UNWIRED_SRC = np.uint32(0xFFFFFFFF)
        cs = np.where(unwired_mask, UNWIRED_SRC, cs_signed).astype(np.uint32)
        self._buffer.rcpt_static_data['cartridge_src'] = cs.flatten()

        is_wired_bits = (~unwired_mask).astype(np.uint32).flatten()
        self._buffer.set_metadata('is_wired', is_wired_bits)

        self._buffer.cartridges_wired = True
        self._buffer.lens_dirty = True

        # Receiving conflicts: peripheral slot maps to its own host lens
        recv_conf = np.zeros(N, dtype=bool)
        own = np.arange(N)
        for r in periph:
            recv_conf |= (cartridge_map[:, r] == own)

        # Donation conflicts: a single lens's r-th rhabdomere is claimed by > 1 cartridge
        don_conf = np.zeros(N, dtype=bool)
        for r in periph:
            col = cartridge_map[:, r]
            valid = col >= 0
            if valid.any():
                counts = np.bincount(col[valid], minlength=N)
                don_conf |= counts > 1

        # These should be internal and exposed through properties in model, eye and lensview levels
        self.unwired_slots = (cartridge_map[:, periph] < 0)
        self.unwired_count = int(self.unwired_slots.sum())
        self.has_selfwires = np.any(self.unwired_slots, axis=1)

        self.receiving_conflicts = recv_conf
        self.donation_conflicts = don_conf
        self.have_conflicts = recv_conf | don_conf
        self._invalidate_spatial()

    def set_retinal_direction(self, muscle_direction: ArrayLike = WORLD_UP):
        """
        Bakes the global muscle pull direction into each lens's local coordinate system.

        Args:
            muscle_direction: A (3,) vector in head-space defining the direction
                           the rhabdomere sheet is pulled by muscles.
        """
        m = np.asarray(muscle_direction, dtype=np.float32)
        m /= np.linalg.norm(m)

        # Project the global vector into the tangent plane of every lens
        rx = np.sum(m * self._local_right, axis=1)
        ry = np.sum(m * self._local_up, axis=1)

        self._buffer.lens_static_data['retina_x'] = rx
        self._buffer.lens_static_data['retina_y'] = ry
        self._buffer.lens_dirty = True

    def pull_retina(self, amount_um: float = 0.0):
        """
        Sets the retina pull for all eyes.
        If amount_um is positive, eyes move according to their side_sign.
        Calling without arguments resets the entire retina sheet to base state.
        """
        for eye in self._eyes:
            eye.pull_retina(amount_um)

    # Private helpers

    @staticmethod
    def _detect_eye_islands(
        positions: np.ndarray,
        k: int = 6,
        distance_multiplier: float = 2.5,
        ) -> np.ndarray:
        """
        Spatial connected-components clustering.

        Builds the kNN graph and keeps an edge (i, j) only if its length is
        below 'distance_multiplier * median_edge_length' over the full graph.
        Connected components of the resulting graph are the eye islands.

        Returns:
            (N,) uint32 array of island labels (0-indexed, in arbitrary order).
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import connected_components

        N = positions.shape[0]
        if N == 0:
            return np.empty(0, dtype=np.uint32)
        if N == 1:
            return np.zeros(1, dtype=np.uint32)

        actual_k = min(k, N - 1)
        tree = cKDTree(positions)
        dists, idx = tree.query(positions, k=actual_k + 1)
        dists = dists[:, 1:]   # drop self
        idx = idx[:, 1:]

        median_edge = float(np.median(dists))
        if median_edge <= 0.0:
            # Degenerate: all points coincident, return a single island
            return np.zeros(N, dtype=np.uint32)
        threshold = distance_multiplier * median_edge

        keep = dists < threshold
        rows = np.repeat(np.arange(N), actual_k)[keep.ravel()]
        cols = idx.ravel()[keep.ravel()]
        data = np.ones(rows.size, dtype=np.int8)
        adj = csr_matrix((data, (rows, cols)), shape=(N, N))

        _, labels = connected_components(adj, directed=False, return_labels=True)
        return labels.astype(np.uint32)

    @staticmethod
    def _resolve_eye_indices(
        positions: np.ndarray,
        eye_indices: Optional[ArrayLike],
        N: int,
        max_eyes: int = 8,
        ) -> np.ndarray:
        """
        Resolve per-lens eye membership.

        If 'eye_indices' is provided, use it directly.
        Otherwise, infer islands by spatial clustering on 'positions' and
        relabel them in order of ascending centroid x (so eye 0 is the
        leftmost island). Caps at 'max_eyes' due to the 3-bit metadata field.
        """

        if eye_indices is not None:
            arr = np.asarray(eye_indices, dtype=np.uint32).reshape(-1)
            if arr.size != N:
                raise ValueError(f"eye_indices size {arr.size} must equal N={N}")
            if int(arr.max()) >= max_eyes:
                raise ValueError(f"eye_indices exceed {max_eyes - 1} (3-bit field), got max={arr.max()}")
            return arr

        raw = CompoundEyeModel._detect_eye_islands(positions)
        unique = np.unique(raw)
        if unique.size > max_eyes:
            raise ValueError(
                f"Detected {unique.size} eye islands but the metadata field "
                f"supports at most {max_eyes}. Pass 'eye_indices=' explicitly or "
                f"check that the lens positions are not over-fragmented."
            )

        # Reorder: ascending centroid x. Stable tie-break on centroid y, z.
        centroids = np.stack([positions[raw == u].mean(axis=0) for u in unique])
        order = np.lexsort((centroids[:, 2], centroids[:, 1], centroids[:, 0]))

        # Build map raw -> new index
        remap = np.empty(unique.size, dtype=np.uint32)
        for new_id, raw_pos in enumerate(order):
            remap[unique[raw_pos]] = np.uint32(new_id)
        return remap[raw]

    def _build_eyes(self, midline_fraction: float = 0.1) -> None:
        """
        Build per-eye objects with a side label inferred from the eye centroid.

        Each eye gets side='left' / 'right' / 'midline'. The side is decided
        by comparing the eye's centroid_x to a threshold scaled to the *whole*
        receptor array, not the individual eye, so that small midline eyes
        (e.g. a median ocellus) get classified correctly even when paired eyes
        are present.

        With midline_fraction=0.1, an eye whose centroid is within 10% of the
        most-lateral lens x is considered on the midline.
        """
        self._eyes = []
        unique_ids = np.unique(self._lens_eye_index)
        if self._lens_positions.shape[0] > 0:
            scale = float(np.max(np.abs(self._lens_positions[:, 0])))
        else:
            scale = 0.0
        threshold = midline_fraction * scale

        for eid in unique_ids:
            lens_idx = np.flatnonzero(self._lens_eye_index == eid).astype(np.intp)
            centroid_x = float(self._lens_positions[lens_idx, 0].mean())
            if scale == 0.0:
                side = 'midline'
            elif abs(centroid_x) <= threshold:
                side = 'midline'
            else:
                side = 'left' if centroid_x < 0.0 else 'right'
            self._eyes.append(Eye(self, int(eid), lens_idx, side=side))

    def _invalidate_spatial(self) -> None:
        """
        Invalidates all per-eye spatial caches (KDtrees).
        Should be called whenever positions or directions change.
        """
        for eye in self._eyes:
            eye._invalidate()

        # Geometry changes effectively make the whole buffer dirty
        self._buffer.lens_dirty = True
        self._buffer.lens_dirty_mask.fill(True)
        self._buffer.rcpt_dirty = True
        self._buffer.rcpt_dirty_mask.fill(True)

    @staticmethod
    def _broadcast_ioa(
        value: ArrayLike,
        N: int,
        ) -> Tuple[np.ndarray, np.ndarray]:

        arr = np.asarray(value, dtype=np.float32)
        if arr.shape == (2,):
            axes = np.tile(arr, (N, 1))
        elif arr.shape == (N, 2):
            axes = arr.copy()
        else:
            raise ValueError(
                f"interommatidial_angles_rad must be shape (2,) or ({N}, 2), got {arr.shape}"
            )
        tilts = np.zeros(N, dtype=np.float32)
        return axes, tilts

    def _lens_optics(self) -> LensOptics:
        """Snapshot of per-lens optics for the acceptance model."""
        s = self._buffer.lens_static_data
        ioa = s['ioa_axes'].astype(np.float32)
        return LensOptics(
            focal_um=s['focal_um'].astype(np.float32),
            lens_diameter_um=s['lens_diameter_um'].astype(np.float32),
            ioa_minor=ioa[:, 0],
            ioa_major=ioa[:, 1],
        )

    def _resolve_acceptance(self) -> np.ndarray:
        """Per-receptor rest acceptance half-widths (rad), as (N*R, 2) (minor, major)."""
        N, R = self.lens_count, self.receptors_per_lens
        rcpt = ReceptorOptics(
            rhab_diameter_um=self._bundle.diameters_um.astype(np.float32),
            wavelength_um=(self._bundle.wavelengths_nm * 1e-3).astype(np.float32),
        )
        acc = np.asarray(self._acceptance(self._lens_optics(), rcpt), dtype=np.float32)
        if acc.shape != (N, R, 2):
            raise ValueError(
                f"acceptance model {type(self._acceptance).__name__} returned {acc.shape}, "
                f"expected {(N, R, 2)}"
            )
        return np.ascontiguousarray(acc.reshape(N * R, 2), dtype=np.float32)

    def acceptance_report(self, reference_ratio: Optional[float] = None) -> str:
        N, R = self.lens_count, self.receptors_per_lens
        acc = self._buffer.rcpt_static_data['rest_acc'].reshape(N, R, 2)
        return report_acceptance(
            acc, self._lens_optics(),
            peripheral_indices=self._bundle.peripheral_indices,
            narrowing_ratio=self._bundle.extra_narrowing_ratio,
            reference_ratio=reference_ratio,
        )

    def _compute_ioa_baseline(
            self,
            k_search: int = 8,
            neighbour_dist_factor: float = 1.5,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Per-lens lattice properties from the local first-ring neighbour graph.

        Computes three things:

        - IOA (minor, major): angular separations in optical axis space.
            Sorted over first-ring neighbours.
            minor = mean of the 2 smallest,
            major = mean of the 2 largest
        - Tilt: lattice rotation in the lens's tangent frame, computed via the hexatic order
              Ψ6 over first-ring neighbour bearings.
              Magnitude of Ψ6: 1.0 = perfect hex, 0.0 = isotropic disorder
        - Lens spacing: median first-ring distance in world units

        Args:
            - k_search: number of neighbours to query per lens (~ a few rings).
            - neighbour_dist_factor: a neighbour is first-ring if its
                distance <= neighbour_dist_factor * closest.

        Returns:
            ioa_axes: (N, 2) (minor, major) in radians
            ioa_tilts: (N,) lattice tilt in radians, in [-π/6, +π/6]
            lens_spacing: (N,) median first-ring distance, world units

        Note: receptors get a separate per-receptor 'acc_tilt' written by the orientation pipeline (chi).
        The ioa_tilt here is the lattice's own rotation, the orientation pipeline does not overwrite this.
        """
        N = self.lens_count
        ioa_minor = np.zeros(N, dtype=np.float32)
        ioa_major = np.zeros(N, dtype=np.float32)
        ioa_tilts = np.zeros(N, dtype=np.float32)
        lens_spacing = np.zeros(N, dtype=np.float32)

        for eye in self._eyes:
            n = len(eye)
            if n < 2:
                continue
            k_eff = min(k_search, n - 1)

            neighbours = eye.neighbours(
                lens_indices=eye._lens_indices,
                k=k_eff,
                immediate_only=False,
                neighbour_dist_factor=neighbour_dist_factor,
            )
            if not neighbours:
                continue

            this_eye_lens_ids = eye._lens_indices
            home_dirs = self._lens_directions[this_eye_lens_ids]
            home_pos = self._lens_positions[this_eye_lens_ids]
            home_right = self._local_right[this_eye_lens_ids]
            home_up = self._local_up[this_eye_lens_ids]

            neighb_global = neighbours.indices
            neighb_dirs = self._lens_directions[neighb_global]
            neighb_pos = self._lens_positions[neighb_global]
            is_immediate = neighbours.is_immediate

            # Optical IOA: angular separation in optical-axis space
            dots = np.clip(np.einsum('ik,ijk->ij', home_dirs, neighb_dirs), -1.0, 1.0)
            angular_sep = np.arccos(dots).astype(np.float32)

            # Bearings in home lens's tangent frame
            delta_pos = neighb_pos - home_pos[:, None, :]
            proj_x = np.einsum('ijk,ik->ij', delta_pos, home_right)
            proj_y = np.einsum('ijk,ik->ij', delta_pos, home_up)
            bearings = np.arctan2(proj_y, proj_x)

            # Hexatic order over the first ring
            z_avg = hexatic_phasor(bearings, weights=is_immediate, axis=1)
            e_tilts = hexatic_axis_angle(z_avg).astype(np.float32)
            e_psi6_mag = hexatic_order(z_avg)

            # IOA: mean of 2 smallest / 2 largest first ring separations
            sep_for_min = np.where(is_immediate, angular_sep, np.inf)
            sep_for_max = np.where(is_immediate, -angular_sep, np.inf)

            min2 = np.sort(sep_for_min, axis=1)[:, :2]
            max2 = -np.sort(sep_for_max, axis=1)[:, :2]
            min2_valid = np.where(np.isfinite(min2), min2, np.nan)
            max2_valid = np.where(np.isfinite(max2), max2, np.nan)
            with np.errstate(all='ignore'):
                e_ioa_minor = np.nanmean(min2_valid, axis=1).astype(np.float32)
                e_ioa_major = np.nanmean(max2_valid, axis=1).astype(np.float32)

            # neighbourhood too sparse: fallback to simple mean over whatever is immediate, zero if nothing
            sparse_mask = is_immediate.sum(axis=1) < 2
            if np.any(sparse_mask):
                sep_imm = np.where(is_immediate, angular_sep, np.nan)
                with np.errstate(all='ignore'):
                    fallback = np.nanmean(sep_imm, axis=1)
                fallback = np.where(np.isfinite(fallback), fallback, 0.0).astype(np.float32)
                e_ioa_minor = np.where(sparse_mask, fallback, e_ioa_minor)
                e_ioa_major = np.where(sparse_mask, fallback, e_ioa_major)
                e_tilts = np.where(sparse_mask, 0.0, e_tilts).astype(np.float32)

            e_ioa_minor = np.where(np.isfinite(e_ioa_minor), e_ioa_minor, 0.0)
            e_ioa_major = np.where(np.isfinite(e_ioa_major), e_ioa_major, 0.0)

            # Lens spacing: median first-ring distance (in world units)
            nbr_dist = np.linalg.norm(delta_pos, axis=2)
            nbr_dist_masked = np.where(is_immediate, nbr_dist, np.nan)
            with np.errstate(all='ignore'):
                e_lens_spacing = np.nanmedian(nbr_dist_masked, axis=1).astype(np.float32)
            e_lens_spacing = np.where(np.isfinite(e_lens_spacing), e_lens_spacing, 0.0)

            ioa_minor[this_eye_lens_ids] = e_ioa_minor
            ioa_major[this_eye_lens_ids] = e_ioa_major
            ioa_tilts[this_eye_lens_ids] = e_tilts
            lens_spacing[this_eye_lens_ids] = e_lens_spacing

            logger.debug(
                f"Eye {eye.eye_index} lattice |Ψ6|: {float(np.mean(e_psi6_mag)):.3f} "
                f"(1.0 = perfect hex, 0.0 = isotropic disorder), "
                f"median lens_spacing: {float(np.median(e_lens_spacing)):.3f}"
            )

        ioa_axes = np.stack([ioa_minor, ioa_major], axis=-1).astype(np.float32)
        return ioa_axes, ioa_tilts, lens_spacing


##

if __name__ == '__main__':
    from insectvision.compound_eyes.rhabdomeres import drosophila_bundle

    ra = CompoundEyeModel.from_sphere(n=500)

    print(ra)
    print(f"  Total receptors: {ra.total_receptors}")
    print(f"  Eyes: {ra.eyes}")
    print(f"  Buffer: {ra.buffer}")

    # Drosophila with flow direction
    droso = drosophila_bundle()

    ra2 = CompoundEyeModel.from_sphere(
        n=1600,
        bundle=droso,
        flow_direction=[1.0, 0.0, 0.0],  # Anterior flow
    )
    print(ra2)
    print(f"  Bundle orientations (first 5): {ra2.lenses[:5].bundle_orientations}")
    print(f"  Chiralities (first 5): {ra2.lenses[:5].chirality}")
    print(f"  Cartridges wired: {ra2.buffer.cartridges_wired}")
    print(f"  Cartridge[0]: {ra2.cartridges[0]}")
