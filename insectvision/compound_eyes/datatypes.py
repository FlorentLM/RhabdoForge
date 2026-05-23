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
    ('nodal_distance_um', np.float32),      #  4 bytes: lens-to-rhabdomere lever arm (μm)
    ('lens_diameter_um',  np.float32),      #  4 bytes: aperture (μm) (used for diffraction)

    # Per-lens photomechanical biophysics
    ('tau_rise',          np.float32),      #  4 bytes: mechanical rise time (s)
    ('tau_relax',         np.float32),      #  4 bytes: mechanical relaxation time (s)
    ('tau_fast',          np.float32),      #  4 bytes: fast adaptation EMA (s, ~PIP2)
    ('tau_adapt',         np.float32),      #  4 bytes: slow adaptation EMA (s, ~Ca²+)
    ('gain_lat_um',       np.float32),      #  4 bytes: max lateral displacement at full drive (μm)
    ('gain_ax_um',        np.float32),      #  4 bytes: max axial contraction at full drive (μm)
    ('_pad',              np.float32, 2),   #  8 bytes: pad to 96 bytes
])  # 96 bytes


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


def pack_metadata(eye_id, receptor_types, neighbour_counts, lens_id, chirality_neg) -> np.ndarray:
    """
    Packs the five fields into a uint32 metadata array, broadcasting as needed.
    """

    eid = np.asarray(eye_id, dtype=np.uint32)
    rt  = np.asarray(receptor_types, dtype=np.uint32)
    nc  = np.asarray(neighbour_counts, dtype=np.uint32)
    li  = np.asarray(lens_id, dtype=np.uint32)
    ch  = np.asarray(chirality_neg, dtype=np.uint32)

    eid, rt, nc, li, ch = np.broadcast_arrays(eid, rt, nc, li, ch)

    out = np.zeros(eid.shape, dtype=np.uint32)
    out = set_metadata_field(out, 'eye_id',          eid)
    out = set_metadata_field(out, 'rcpt_type',       rt)
    out = set_metadata_field(out, 'neighbour_count', nc)
    out = set_metadata_field(out, 'lens_id',         li)
    out = set_metadata_field(out, 'chirality_neg',   ch)
    return out


# Back-compat clear masks
# TODO: Remove these

_CLEAR_EYE_ID        = np.uint32(~(_mask(_BIT_LAYOUT['eye_id'][1])        << _BIT_LAYOUT['eye_id'][0])        & 0xFFFFFFFF)
_CLEAR_RECEPTOR_TYPE = np.uint32(~(_mask(_BIT_LAYOUT['rcpt_type'][1])     << _BIT_LAYOUT['rcpt_type'][0])     & 0xFFFFFFFF)
_CLEAR_NEIGHBOURS    = np.uint32(~(_mask(_BIT_LAYOUT['neighbour_count'][1])      << _BIT_LAYOUT['neighbour_count'][0])      & 0xFFFFFFFF)
_CLEAR_LENS_INDEX    = np.uint32(~(_mask(_BIT_LAYOUT['lens_id'][1])       << _BIT_LAYOUT['lens_id'][0])       & 0xFFFFFFFF)
_CLEAR_CHIRALITY     = np.uint32(~(_mask(_BIT_LAYOUT['chirality_neg'][1]) << _BIT_LAYOUT['chirality_neg'][0]) & 0xFFFFFFFF)
