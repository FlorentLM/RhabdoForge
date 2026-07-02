"""
Packed data container for compound-eye buffers.
"""
from typing import Tuple, Sequence
import numpy as np

# Per-ommatidium data

OMM_STATIC_DTYPE = np.dtype([

    # 16 bytes: ommatidium's position xyz and chi
    ('position',    np.float32, 3),         ('chi',         np.float32),

    # 16 bytes x 3: Frame of ref (12 bytes) + other 4 bytes things float4-align

    # Ommatidium's frame of ref (in world)
    ('forward', np.float32, 3),              ('focal_um',    np.float32),   # focal length (μm) (lens-to-rhabdomere lever arm)
    ('right',   np.float32, 3),              ('aperture_um', np.float32),   # lens aperture (μm) (used for diffraction)
    ('up',      np.float32, 3),              ('ioa_tilt',    np.float32),   # local hexatic lattice angle (rad) = lens anisotropic distortion

    # 16 bytes: saccade dx and dy, lateral amplitude, axial amplitude
    ('saccade_dxdy',  np.float32, 2), ('ampl_lateral', np.float32), ('ampl_axial', np.float32),

    # 16 bytes: Temporal values:            # TODO: use ms instead?
    ('tau_rise',        np.float32),        # mechanical rise time (s)
    ('tau_relax',       np.float32),        # mechanical relaxation time (s)
    ('tau_adapt_fast',  np.float32),        # fast adaptation EMA (s)
    ('tau_adapt_slow',  np.float32),        # slow adaptation EMA (s)

    # 16 bytes: The two remaining 8 bytes things
    ('ioa_angles',      np.float32, 2),     # (minor, major) interommatidial angles (rad)
    ('retina_dxdy',     np.float32, 2),     # retinal shift local dy and dx
])  # 112 bytes


OMM_DYNAMIC_DTYPE = np.dtype([
    ('curr_lum_fast',       np.float32),    # 4 bytes: fast luminance tracker (EMA), for saccade drive
    ('curr_lum_slow',       np.float32),    # 4 bytes: slow luminance baseline (EMA)
    ('curr_lateral_disp',   np.float32),    # 4 bytes: current lateral displacement (μm)
    ('curr_axial_disp',     np.float32),    # 4 bytes: current axial contraction (μm)
])  # 16 bytes


# Per-rhabdomere data

RHAB_STATIC_DTYPE = np.dtype([
    # 16 bytes: 12 bytes (UV, G, B) channel sensitivity multipliers, and 4 bytes peak wavelength (μm)
    ('sensitivity',     np.float32, 3),     ('wavelength_um',   np.float32),

    # 16 bytes: Rest position and acceptance angles
    ('rest_acc_angles', np.float32, 2),     # 8 bytes: acceptance angles (minor, major) at rest (rad)
    ('rest_offset',     np.float32, 2),     # 8 bytes: offset (at rest) from the ommatidium optical axis (μm), post chi/chirality

    # 16 bytes: the 3 remaining 4 bytes fields, and the packed metadata
    ('tau_membrane',    np.float32),        # 4 bytes: Rhabdomere membrane RC (s)
    ('cartridge_src',   np.uint32),         # 4 bytes: Rhabdomere index (global) of the neural-superposition source
    ('diameter_um',     np.float32),        # 4 bytes: Rhabdomere diameter (μm)
    ('metadata',        np.uint32)          # 4 bytes: bit-packed, see _BIT_LAYOUT below
])  # 48 bytes


RHAB_DYNAMIC_DTYPE = np.dtype([
    ('curr_direction',  np.float32, 3),     # 12 bytes: current (actuated) viewing direction
    ('curr_adaptation', np.float32),        #  4 bytes: current adaptation state
    ('curr_acc_angles', np.float32, 2),     #  8 bytes: current (actuated) acceptance angles (rad)
    ('optical_scale',   np.float32),        #  4 bytes # TODO: explain
    ('_pad',            np.float32),        #  4 bytes: pad to 32 bytes
])  # 32 bytes

# TODO: Move most metadata to per-ommatidium ?? Only rhab_R, chirality and is_wired are per-rhabdomere

# Metadata bitfield
#
#   Bits    Field            Width    Notes
#   ------------------------------------------------------------------------------------
#   0-3     eye_id           4        Up to 16 distinct eyes (main L/R, DRA, ocelli...)
#   4-7     rhab_R           4        Rhabdomere type within bundle (R1=0, R2=1, ...)
#   8-11    neighbour_count  4        Number of immediate lattice neighbours
#   12-27   omm_id           16       Parent ommatidium index (up to 65535)
#   28      chirality_neg    1        0 = +1 chirality (normal), 1 = -1 (mirrored)
#   29      is_binocular     1        Whether the rhabdomere is in an ommatidium of the binocular area
#   30      is_wired         1        Whether the rhabdomere is correctly wired in the superposition
#   31      is_edge          1        Whether the rhabdomere is in an ommatidium that is at the edge of the eye

_BIT_LAYOUT = {
    'eye_id':           (0,  4),
    'rhab_R':           (4,  4),
    'neighbour_count':  (8,  4),
    'omm_id':           (12, 16),
    'chirality_neg':    (28,  1),
    'is_binocular':     (29,  1),
    'is_wired':         (30,  1),
    'is_edge':          (31,  1),
}

OMM_PROPS = set(OMM_STATIC_DTYPE.names).union(set(OMM_DYNAMIC_DTYPE.names))
RHAB_PROPS = set(RHAB_STATIC_DTYPE.names).union(set(RHAB_DYNAMIC_DTYPE.names))
STATIC_PROPS = set(OMM_STATIC_DTYPE.names).union(set(RHAB_STATIC_DTYPE.names))
DYNAM_PROPS = set(OMM_DYNAMIC_DTYPE.names).union(set(RHAB_DYNAMIC_DTYPE.names))


def get_metadata_field(metadata: np.ndarray, field: str) -> np.ndarray:
    """Extract one bit-packed field (returned as uint32)."""
    shift, bits = _BIT_LAYOUT[field]
    return (np.asarray(metadata) >> np.uint32(shift)) & np.uint32((1 << bits) - 1)


def set_metadata_field(metadata: np.ndarray, field: str, value) -> np.ndarray:
    """Return 'metadata' with 'field' replaced by 'value' (out-of-range values truncate)."""
    shift, bits = _BIT_LAYOUT[field]
    mask = np.uint32((1 << bits) - 1)
    clear = np.uint32(~(mask << np.uint32(shift)) & np.uint32(0xFFFFFFFF))
    v = (np.asarray(value, dtype=np.uint32) & mask) << np.uint32(shift)
    return (metadata & clear) | v


class Buffer:
    """
    GPU-ready packed buffers for a compound eye.
    """

    def __init__(self, shape: Tuple[int, int] = (1, 1)):

        N, R = shape

        if N < 1:
            raise ValueError('Number of ommatidia must be >= 1')

        if R < 1:
            raise ValueError('Number of rhabdomeres per ommatidium must be >= 1')

        self._shape = int(N), int(R)

        # Mapping: property -> level
        self.levels = {f: 'ommatidium' for f in OMM_PROPS}
        self.levels.update({f: 'rhabdomere' for f in RHAB_PROPS})

        # Mapping: property -> mutability
        self.mutability = {f: 'static' for f in STATIC_PROPS}
        self.mutability.update({f: 'dynamic' for f in DYNAM_PROPS})

        # Mapping: level -> structured arrays (+ GPU sync bookkeeping)
        self.structured_arrays = {
            'ommatidium': {
                'static': np.zeros(self.shape[0], dtype=OMM_STATIC_DTYPE),
                'dynamic': np.zeros(self.shape[0], dtype=OMM_DYNAMIC_DTYPE),
                'stale_mask': np.zeros(self.shape[0], dtype=bool),
                'stale': False
                # TODO: Maybe use a threshold to tell the renderer's sync_cpu whether to update all the buffer or not? ...or drop the surgical update and just always push all the data
            },
            'rhabdomere': {
                'static': np.zeros(self.size, dtype=RHAB_STATIC_DTYPE),
                'dynamic': np.zeros(self.size, dtype=RHAB_DYNAMIC_DTYPE),
                'stale_mask': np.zeros(self.size, dtype=bool),
                'stale': False
            },
        }

    # Sizes and repr

    @property
    def shape(self) -> Tuple[int, int]:
        return self._shape

    @property
    def size(self) -> int:
        return self._shape[0] * self._shape[1]

    def __len__(self) -> int:
        return self._shape[0]

    def __repr__(self) -> str:
        return f'Buffer(shape=({self._shape[0]}x{self._shape[1]}))'

    # Public info methods

    @property
    def fields(self):
        return sorted(OMM_PROPS.union(RHAB_PROPS).union(_BIT_LAYOUT.keys()) - {'_pad', 'metadata'})

    def max_value(self, field: str) -> int:
        """
        Return the maximum possible value for a given metadata bit-field.
        """
        if field not in _BIT_LAYOUT:
            raise KeyError(f"'{field}' is not a valid metadata field.")
        _, bits = _BIT_LAYOUT[field]
        return (1 << bits) - 1

    # Internal helpers

    def _mark_stale(self, level: str, idx) -> None:
        """Flag the written rows of 'level' for re-upload."""
        self.structured_arrays[level]['stale_mask'][idx] = True
        self.structured_arrays[level]['stale'] = True

    def _array_containing(self, field: str):
        return self.structured_arrays[self.levels[field]][self.mutability[field]]

    # Access

    def __getitem__(self, key):
        field, idx = key if isinstance(key, tuple) else (key, slice(None))

        if field in _BIT_LAYOUT:
            meta = self.structured_arrays['rhabdomere']['static']['metadata'][idx]
            out = np.asarray(get_metadata_field(meta, field))
            if isinstance(idx, slice) and idx == slice(None):  # only reshape if looking at the whole array
                out = out.reshape(*self._shape)
            out.flags.writeable = False     # unpacked copy, can be written via buf['field', idx] = v
            return out

        level = self.levels[field]
        arr = self._array_containing(field)[field]

        if isinstance(idx, slice) and idx == slice(None):
            logical_dims = (self.shape[0],) if level == 'ommatidium' else self.shape
            target_shape = logical_dims + arr.shape[1:]
            return arr.reshape(target_shape).squeeze()

        return arr[idx]

    def __setitem__(self, key, values):
        field, idx = key if isinstance(key, tuple) else (key, slice(None))
        values = np.asanyarray(values)

        if field in _BIT_LAYOUT:
            level = 'rhabdomere'
            meta = self.structured_arrays[level]['static']['metadata']

            if isinstance(idx, slice) and idx == slice(None):
                N, R = self._shape
                if values.ndim == 0:
                    grid = np.broadcast_to(values, (N, R))
                elif values.shape == (N, R):
                    grid = values
                elif values.ndim == 1 and values.shape[0] == N:
                    grid = np.broadcast_to(values[:, None], (N, R))
                else:
                    grid = np.broadcast_to(values, (N, R))
                values = grid.reshape(-1)

            meta[idx] = set_metadata_field(meta[idx], field, values)
            self._mark_stale(level, idx)

        else:
            level = self.levels[field]
            arr = self._array_containing(field)[field]

            if isinstance(idx, slice) and idx == slice(None):
                logical_dims = (self.shape[0],) if level == 'ommatidium' else self.shape
                extra_dims = arr.shape[1:]
                logical_shape = logical_dims + extra_dims

                try:
                    values = np.broadcast_to(values, logical_shape)
                except ValueError as e:
                    raise ValueError(
                        f"could not broadcast input array from shape {values.shape} "
                        f"into logical shape {logical_shape} for field '{field}'"
                    ) from e
                values = values.reshape(-1, *extra_dims)

            arr[idx] = values
            self._mark_stale(level, idx)

    def reorder(self, permutation_indices: Sequence[int]) -> None:
        """
        Permute every buffer in place by a given order.

        'permutation_indices' is a length-N array: new row i takes old row permutation_indices[i].
        Rhabdomere rows follow their parent ommatidium. All arrays are left
        C-contiguous and flagged stale for re-upload.
        """
        N, R = self._shape
        perm = np.asarray(permutation_indices, dtype=np.intp).reshape(-1)

        if perm.shape != (N,):
            raise ValueError(f'permutation_indices must have shape ({N},), got {perm.shape}')

        if not np.array_equal(np.bincount(perm, minlength=N), np.ones(N, dtype=np.intp)):
            raise ValueError(f'permutation_indices is not a valid permutation of range(N={N})')

        rperm = (perm[:, None] * R + np.arange(R, dtype=np.intp)).reshape(-1)

        for level, idx in (('ommatidium', perm), ('rhabdomere', rperm)):
            sa = self.structured_arrays[level]
            sa['static'] = np.ascontiguousarray(sa['static'][idx])
            sa['dynamic'] = np.ascontiguousarray(sa['dynamic'][idx])
            sa['stale_mask'] = np.ascontiguousarray(sa['stale_mask'][idx])
            sa['stale'] = True

    @property
    def ommatidia_static(self):
        return self.structured_arrays['ommatidium']['static']

    @property
    def ommatidia_dynamic(self):
        return self.structured_arrays['ommatidium']['dynamic']

    @property
    def rhabdomere_static(self):
        return self.structured_arrays['rhabdomere']['static']

    @property
    def rhabdomere_dynamic(self):
        return self.structured_arrays['rhabdomere']['dynamic']

    @property
    def ommatidia_stale(self) -> bool:
        return self.structured_arrays['ommatidium']['stale']

    @ommatidia_stale.setter
    def ommatidia_stale(self, value: bool):
        value = bool(value)
        self.structured_arrays['ommatidium']['stale_mask'][:] = value
        self.structured_arrays['ommatidium']['stale'] = value

    @property
    def rhabdomeres_stale(self) -> bool:
        return self.structured_arrays['rhabdomere']['stale']

    @rhabdomeres_stale.setter
    def rhabdomeres_stale(self, value: bool):
        value = bool(value)
        self.structured_arrays['rhabdomere']['stale_mask'][:] = value
        self.structured_arrays['rhabdomere']['stale'] = value

    @property
    def ommatidia_stale_mask(self) -> np.ndarray:
        return self.structured_arrays['ommatidium']['stale_mask']

    @ommatidia_stale_mask.setter
    def ommatidia_stale_mask(self, value: bool):
        self.structured_arrays['ommatidium']['stale_mask'][:] = bool(value)

    @property
    def rhabdomeres_stale_mask(self) -> np.ndarray:
        return self.structured_arrays['rhabdomere']['stale_mask']

    @rhabdomeres_stale_mask.setter
    def rhabdomeres_stale_mask(self, value: bool):
        self.structured_arrays['rhabdomere']['stale_mask'][:] = bool(value)