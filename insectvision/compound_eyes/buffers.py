"""
Packed data container for compound-eye buffers.

EyesBuffer is a Data Transfer Object (DTO) holding the four packed structured arrays
that are passed to the GPU, plus the cartridge mapping.

It only knows how to allocate itself, serialize to / from disk, and bitpack the
receptor metadata field.
"""
from contextlib import contextmanager
from typing import Optional
import numpy as np


# Per-lens data

LENS_STATIC_DTYPE = np.dtype([
    ('right',             np.float32, 3),   # 12 bytes: tangent right
    ('sacc_x',            np.float32),      #  4 bytes: saccade local dx
    ('up',                np.float32, 3),   # 12 bytes: tangent up
    ('sacc_y',            np.float32),      #  4 bytes: saccade local dy
    ('forward',           np.float32, 3),   # 12 bytes: optical axis
    ('ioa_tilt',          np.float32),      #  4 bytes: local hexatic lattice rotation (rad)
    ('ioa_axes',          np.float32, 2),   #  8 bytes: (minor, major) interommatidial angles (rad)
    ('focal_um', np.float32),               #  4 bytes: focal length, lens-to-rhabdomere lever arm (μm)
    ('aperture_um',  np.float32),           #  4 bytes: aperture (μm) (used for diffraction)
    ('tau_rise',          np.float32),      #  4 bytes: mechanical rise time (s)
    ('tau_relax',         np.float32),      #  4 bytes: mechanical relaxation time (s)
    ('tau_fast',          np.float32),      #  4 bytes: fast adaptation EMA (s, ~PIP2)
    ('tau_adapt',         np.float32),      #  4 bytes: slow adaptation EMA (s, ~Ca²+)
    ('ampl_lat_um',       np.float32),      #  4 bytes: max lateral displacement at full drive (μm)
    ('ampl_ax_um',        np.float32),      #  4 bytes: max axial contraction at full drive (μm)
    ('retina_x',          np.float32),      #  4 bytes: muscle-driven retinal shift local dx
    ('retina_y',          np.float32),      #  4 bytes: muscle-driven retinal shift local dy
])  # 96 bytes
# TODO: This struct would benefit from a reorganisation


LENS_DYNAMIC_DTYPE = np.dtype([
    ('adapted_lum',  np.float32),           # 4 bytes: slow luminance baseline (~50 ms EMA)
    ('fast_lum',     np.float32),           # 4 bytes: fast luminance tracker (~5 ms EMA), saccade drive
    ('lateral_um',   np.float32),           # 4 bytes: current focal-plane displacement (μm)
    ('axial_um',     np.float32),           # 4 bytes: current axial contraction (μm)
])  # 16 bytes


# Per-receptor data

RCPT_STATIC_DTYPE = np.dtype([
    ('position',         np.float32, 3),   # 12 bytes: world position (= parent lens's position)
    ('metadata',         np.uint32),       #  4 bytes: bit-packed, see _BIT_LAYOUT below
    ('rest_acc',         np.float32, 2),   #  8 bytes: acceptance angles (minor, major) at rest (rad)
    ('rot_offset',       np.float32, 2),   #  8 bytes: focal-plane offset behind lens (μm), post chi/chirality
    ('sensitivity',      np.float32, 3),   # 12 bytes: (UV, G, B) channel multipliers
    ('acc_tilt',         np.float32),      #  4 bytes: acceptance ellipse tilt (rad)
    ('tau_membrane',     np.float32),      #  4 bytes: photoreceptor membrane RC (s)
    ('cartridge_src',    np.uint32),       #  4 bytes: global receptor index of the neural-superposition source
    ('rhab_diameter_um', np.float32),      #  4 bytes: rhabdomere diameter (μm)
    ('wavelength_um',    np.float32),      #  4 bytes: peak wavelength (μm)
])  # 64 bytes


RCPT_DYNAMIC_DTYPE = np.dtype([
    ('direction',        np.float32, 3),   # 12 bytes: current (actuated) viewing direction
    ('adaptation_state', np.float32),      #  4 bytes: neural/biochem adaptation level
    ('acc_axes',         np.float32, 2),   #  8 bytes: current (actuated) acceptance axes (rad)
    ('_pad',             np.float32, 2),   #  8 bytes: pad to 32 bytes
])  # 32 bytes


# Metadata bitfield
#
#   Bits    Field            Width    Notes
#   ------------------------------------------------------------------------------------
#   0-2     eye_id           3        Up to 8 distinct eyes (main L/R, DRA, ocelli...)
#   3-6     rcpt_type        4        Receptor type within bundle (R1=0, R2=1, ...)
#   7-10    neighbour_count  4        Number of immediate lattice neighbours
#   11-26   lens_id          16       Parent ommatidium index (up to 65535)
#   27      chirality_neg    1        0 = +1 chirality (normal), 1 = -1 (mirrored)
#   28-31                    4        pad

_BIT_LAYOUT = {
    'eye_id':          (0,  3),
    'rcpt_type':       (3,  4),
    'neighbour_count': (7,  4),
    'lens_id':         (11, 16),
    'chirality_neg':   (27, 1),
    'binocular_area':  (28, 1),
    'is_wired':        (29, 1),
}


# Some convenience functions

def _mask(bits: int) -> np.uint32:
    return np.uint32((1 << bits) - 1)


def get_metadata_field(metadata: np.ndarray, field: str) -> np.ndarray:
    """
    Extracts one field (returned as uint32).
    """

    shift, bits = _BIT_LAYOUT[field]
    return (metadata >> np.uint32(shift)) & _mask(bits)


def set_metadata_field(metadata: np.ndarray, field: str, value) -> np.ndarray:
    """
    Returns 'metadata' with 'field' replaced by 'value' (out-of-range values truncate).
    """
    shift, bits = _BIT_LAYOUT[field]
    mask = _mask(bits)
    clear = np.uint32(~(mask << np.uint32(shift)) & np.uint32(0xFFFFFFFF))
    v = (np.asarray(value, dtype=np.uint32) & mask) << np.uint32(shift)
    return (metadata & clear) | v


def pack_metadata(
        eye_indices,
        receptor_types,
        neighbour_counts,
        lens_indices,
        chirality_neg,
        binocular_area=0,
        is_wired=1
    ) -> np.ndarray:
    """
    Packs the six fields into a uint32 metadata array, broadcasting as needed.
    """

    ei = np.asarray(eye_indices, dtype=np.uint32)
    rt = np.asarray(receptor_types, dtype=np.uint32)
    nc = np.asarray(neighbour_counts, dtype=np.uint32)
    li = np.asarray(lens_indices, dtype=np.uint32)
    ch = np.asarray(chirality_neg, dtype=np.uint32)
    bi = np.asarray(binocular_area, dtype=np.uint32)
    iw = np.asarray(is_wired, dtype=np.uint32)

    ei, rt, nc, li, ch, bi, iw = np.broadcast_arrays(ei, rt, nc, li, ch, bi, iw)

    out = np.zeros(ei.shape, dtype=np.uint32)
    out = set_metadata_field(out, 'eye_id',          ei)
    out = set_metadata_field(out, 'rcpt_type',       rt)
    out = set_metadata_field(out, 'neighbour_count', nc)
    out = set_metadata_field(out, 'lens_id',         li)
    out = set_metadata_field(out, 'chirality_neg',   ch)
    out = set_metadata_field(out, 'binocular_area',  bi)
    out = set_metadata_field(out, 'is_wired',        iw)
    return out


# Back-compat clear masks
# TODO: Remove these

_CLEAR_EYE_ID        = np.uint32(~(_mask(_BIT_LAYOUT['eye_id'][1])        << _BIT_LAYOUT['eye_id'][0])        & 0xFFFFFFFF)
_CLEAR_RECEPTOR_TYPE = np.uint32(~(_mask(_BIT_LAYOUT['rcpt_type'][1])     << _BIT_LAYOUT['rcpt_type'][0])     & 0xFFFFFFFF)
_CLEAR_NEIGHBOURS    = np.uint32(~(_mask(_BIT_LAYOUT['neighbour_count'][1])      << _BIT_LAYOUT['neighbour_count'][0])      & 0xFFFFFFFF)
_CLEAR_LENS_INDEX    = np.uint32(~(_mask(_BIT_LAYOUT['lens_id'][1])       << _BIT_LAYOUT['lens_id'][0])       & 0xFFFFFFFF)
_CLEAR_CHIRALITY     = np.uint32(~(_mask(_BIT_LAYOUT['chirality_neg'][1]) << _BIT_LAYOUT['chirality_neg'][0]) & 0xFFFFFFFF)


class EyesBuffer:
    """
    GPU-ready packed buffers for a compound eye.

    Four numpy structured arrays:
        - lens_static_data:  (N,)   per-lens static fields
        - lens_dynamic_data: (N,)   per-lens dynamic state
        - rcpt_static_data:  (N*R,) per-receptor static fields
        - rcpt_dynamic_data: (N*R,) per-receptor dynamic state

    Plus an (N, R) cartridge mapping table and a 'lens_dirty' flag
    that downstream GPU upload code can read/clear.
    """

    def __init__(self, n_lenses: int, receptors_per_lens: int):

        if n_lenses < 1:
            raise ValueError(f"n_lenses must be >= 1, got {n_lenses}")
        if receptors_per_lens < 1:
            raise ValueError(f"receptors_per_lens must be >= 1, got {receptors_per_lens}")

        self._n_lenses = int(n_lenses)
        self._receptors_per_lens = int(receptors_per_lens)
        total_rcpt = n_lenses * receptors_per_lens

        self.lens_static_data = np.zeros(n_lenses, dtype=LENS_STATIC_DTYPE)
        self.lens_dynamic_data = np.zeros(n_lenses, dtype=LENS_DYNAMIC_DTYPE)
        self.rcpt_static_data = np.zeros(n_lenses * receptors_per_lens, dtype=RCPT_STATIC_DTYPE)
        self.rcpt_dynamic_data = np.zeros(n_lenses * receptors_per_lens, dtype=RCPT_DYNAMIC_DTYPE)

        # Cartridge mapping: (N, R)
        # Entry (i, r) is the global lens index whose r-th rhabdomere feeds the cartridge centered at lens i.
        # Identity by default
        self.cartridge_map = np.tile(
            np.arange(n_lenses, dtype=np.intp)[:, None],
            (1, receptors_per_lens),
        )
        self.cartridges_wired = False

        # GPU upload tracking: whenever lens-level data is mutated it needs to be reuploaded.
        # The renderer is expected to read and clear the flag
        self.lens_dirty = True
        self.rcpt_dirty = True
        self.lens_dirty_mask = np.ones(self._n_lenses, dtype=bool)
        self.rcpt_dirty_mask = np.ones(total_rcpt, dtype=bool)

        self._allow_lens_writes = False
        self._allow_rcpt_writes = False

    @contextmanager
    def unlock(self, lenses: Optional[bool] = None, receptors: Optional[bool] = None):
        """
        Context manager to temporarily allow CPU-side modifications to the buffers.

        If neither is specified, both are unlocked.
        If one is specified, the other remains locked.
        """

        if lenses is None and receptors is None:
            lenses = True
            receptors = True
        else:
            lenses = lenses or False
            receptors = receptors or False

        prev_lens = self._allow_lens_writes
        prev_rcpt = self._allow_rcpt_writes

        self._allow_lens_writes = prev_lens or lenses
        self._allow_rcpt_writes = prev_rcpt or receptors

        try:
            yield
        finally:
            self._allow_lens_writes = prev_lens
            self._allow_rcpt_writes = prev_rcpt

    # Sizes

    @property
    def n_lenses(self) -> int:
        return self._n_lenses

    @property
    def receptors_per_lens(self) -> int:
        return self._receptors_per_lens

    @property
    def total_receptors(self) -> int:
        return self._n_lenses * self._receptors_per_lens

    def __len__(self) -> int:
        return self._n_lenses

    def __repr__(self) -> str:
        return (f"EyesBuffer(N={self._n_lenses}, R={self._receptors_per_lens}, "
                f"cartridges_wired={self.cartridges_wired})")

    # Bitpacking helpers

    def get_metadata(self, field: str) -> np.ndarray:
        """
        Extract one bitfield from rcpt_static_data['metadata'].

        See datatypes._BIT_LAYOUT for available fields.
        """
        return get_metadata_field(self.rcpt_static_data['metadata'], field)

    def set_metadata(self, field: str, value) -> None:
        """
        In-place update of one bitfield in rcpt_static_data['metadata'].
        """
        if not self._allow_rcpt_writes:
            raise RuntimeError("CPU-side receptor writes are locked.")

        self.rcpt_static_data['metadata'] = set_metadata_field(
            self.rcpt_static_data['metadata'], field, value
        )
        self.rcpt_dirty = True
        self.rcpt_dirty_mask.fill(True)

    def pack_metadata(self,
                      eye_indices,
                      receptor_types,
                      neighbour_counts,
                      lens_indices,
                      chirality_neg,
                      binocular_area
                      ) -> None:
        """
        Replace the entire rcpt_static_data['metadata'] with packed fields.
        """
        if not self._allow_rcpt_writes:
            raise RuntimeError("CPU-side receptor writes are locked.")

        self.rcpt_static_data['metadata'] = pack_metadata(
            eye_indices=eye_indices,
            receptor_types=receptor_types,
            neighbour_counts=neighbour_counts,
            lens_indices=lens_indices,
            chirality_neg=chirality_neg,
            binocular_area=binocular_area,
        )
        self.rcpt_dirty = True
        self.rcpt_dirty_mask.fill(True)

    # Cartridge-index view

    @property
    def cartridge_indices(self) -> np.ndarray:
        """
        (N, R) global receptor indices grouped by cartridge mapping.

        Raises RuntimeError if cartridges have not been wired.
        """
        if not self.cartridges_wired:
            raise RuntimeError("Cartridges not wired")
        return (self.cartridge_map * self._receptors_per_lens
                + np.arange(self._receptors_per_lens, dtype=np.intp))

    # I/O

    def to_file(self, path: str) -> None:
        """
        Serialize the entire buffer state to a .npz archive.

        Writes the four structured arrays, the cartridge map, and the
        cartridges_wired flag.
        """
        np.savez(
            path,
            lens_static_data=self.lens_static_data,
            lens_dynamic_data=self.lens_dynamic_data,
            rcpt_static_data=self.rcpt_static_data,
            rcpt_dynamic_data=self.rcpt_dynamic_data,
            cartridge_map=self.cartridge_map,
            cartridges_wired=np.bool_(self.cartridges_wired),
        )

    @classmethod
    def from_file(cls, path: str) -> 'EyesBuffer':
        """
        Restore a buffer from a .npz archive previously written by to_file().

        Raises ValueError if the structured-array dtypes in the archive
        do not match the current LENS_*_DTYPE / RCPT_*_DTYPE.
        """
        with np.load(path, allow_pickle=False) as data:
            files = set(data.files)
            required = {'lens_static_data', 'lens_dynamic_data',
                        'rcpt_static_data', 'rcpt_dynamic_data',
                        'cartridge_map'}
            missing = required - files
            if missing:
                raise ValueError(f"{path}: missing required keys: {sorted(missing)}")

            ls = data['lens_static_data']
            ld = data['lens_dynamic_data']
            rs = data['rcpt_static_data']
            rd = data['rcpt_dynamic_data']
            cm = data['cartridge_map']

            if ls.dtype != LENS_STATIC_DTYPE:
                raise ValueError(f"{path}: lens_static_data dtype mismatch")
            if ld.dtype != LENS_DYNAMIC_DTYPE:
                raise ValueError(f"{path}: lens_dynamic_data dtype mismatch")
            if rs.dtype != RCPT_STATIC_DTYPE:
                raise ValueError(f"{path}: rcpt_static_data dtype mismatch")
            if rd.dtype != RCPT_DYNAMIC_DTYPE:
                raise ValueError(f"{path}: rcpt_dynamic_data dtype mismatch")

            n_lenses = int(ls.shape[0])
            n_receptors = int(rs.shape[0])
            if n_receptors % n_lenses != 0:
                raise ValueError(
                    f"{path}: receptor count {n_receptors} not divisible by lens count {n_lenses}"
                )
            R = n_receptors // n_lenses

            buf = cls(n_lenses=n_lenses, receptors_per_lens=R)
            buf.lens_static_data[:] = ls
            buf.lens_dynamic_data[:] = ld
            buf.rcpt_static_data[:] = rs
            buf.rcpt_dynamic_data[:] = rd
            buf.cartridge_map = np.asarray(cm, dtype=np.intp).reshape(n_lenses, R)
            buf.cartridges_wired = bool(data['cartridges_wired']) if 'cartridges_wired' in files else False
            buf.lens_dirty = True

        return buf
