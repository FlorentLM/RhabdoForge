"""
User-facing views of CompoundEyeModel data buffers.

Views:
  BaseView       -- Abstract base defining the ViewField descriptor properties.
  OmmatidiumView -- Indexes the Ommatidia axis.
  CartridgeView  -- Like OmmatidiumView, but receptors follow neural superposition.
  ReceptorView   -- Indexes the Rhabdomeres axis.
  EyeView        -- Inherits OmmatidiumView, manages spatial queries for a specific eye.
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

logger = logging.getLogger(__name__)


class ViewField:
    """
    A descriptor that routes property access directly to the underlying Buffer slice.
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
    Abstract base class for all views. Provides unified access to GPU-backed data.
    """

    @property
    def model(self) -> 'Model':
        raise NotImplementedError

    @property
    def _buffer(self) -> 'Buffer':
        return self.model._buffer_object

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
        idx = self.rhab_indices if isinstance(self, ReceptorView) else self.omm_indices
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

    # TODO: Rename these to the correct names

    # Geometry

    @property
    def positions(self) -> np.ndarray:
        return self._buffer['position', self.omm_indices]

    @positions.setter
    def positions(self, value: ArrayLike):
        self._buffer['position', self.omm_indices] = np.asarray(value, dtype=np.float32).reshape(-1, 3)
        self.model._invalidate_spatial()

    @property
    def directions(self) -> np.ndarray:
        return self._buffer['forward', self.omm_indices]

    @directions.setter
    def directions(self, value: ArrayLike):
        fwd = norm_l2(np.asarray(value).reshape(-1, 3)).astype(np.float32)
        self._buffer['forward', self.omm_indices] = fwd
        self._buffer['right', self.omm_indices], self._buffer['up', self.omm_indices] = tangent_frames(fwd)

        self.model._invalidate_spatial()

    forward = directions

    right = ViewField('right', 'ommatidia')
    up = ViewField('up', 'ommatidia')

    # Lattice properties

    interommatidial_angles = ViewField('ioa_angles', 'ommatidia')
    interommatidial_tilt = ViewField('ioa_tilt', 'ommatidia')
    acceptance_angles_rest = ViewField('rest_acc_angles', 'rhabdomere')
    acceptance_angles_current = ViewField('curr_acc_angles', 'rhabdomere')

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

    # chirality = ViewField('chirality_neg', 'rhabdomere',   # TODO: do not return chirality neg
    #    doc="Per-ommatidium bundle chirality")

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



class OmmatidiumView(BaseView):
    """Selection indexes the ommatidia axis."""

    __slots__ = ('_model', '_omm_indices')

    def __init__(self, model: 'Model', indices: np.ndarray):
        self._model = model
        self._omm_indices = np.asarray(indices, dtype=np.intp)

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

    __slots__ = ('_model', '_idx')

    def __init__(self, model: 'Model', rhab_indices: np.ndarray):
        self._model = model
        self._rhab_indices = np.asarray(rhab_indices, dtype=np.intp)

    @property
    def shape(self) -> Tuple[int, ...]:
        """The logical shape of this view (1D for receptors)."""
        return (len(self),)

    @property
    def model(self) -> 'Model':
        return self._model

    @property
    def rhab_indices(self) -> np.ndarray:
        return self._rhab_indices

    @property
    def indices(self):
        return self._rhab_indices     # TODO: not sure about this one

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


class NeighbourResult:
    """Result of an EyeView.neighbours() query."""

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
    def ommatidia(self) -> OmmatidiumView:
        return OmmatidiumView(self.eye.model, self.indices.ravel())

    @property
    def receptors(self) -> ReceptorView:
        return self.ommatidia.receptors


class EyeView(OmmatidiumView):

    def __init__(self, model: 'Model', eye_index: int, indices: np.ndarray, side: str):
        super().__init__(model, indices)
        self._eye_index = eye_index
        self._side = side

        # TODO ...