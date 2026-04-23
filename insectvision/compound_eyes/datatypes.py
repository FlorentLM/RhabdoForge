import numpy as np


# Lens data

LENS_STATIC_DTYPE = np.dtype([
    ('right', np.float32, 3),         # 12 bytes: tangent right
    ('sacc_x', np.float32),           # 4 bytes: saccade local dx
    ('up', np.float32, 3),            # 12 bytes: tangent up
    ('sacc_y', np.float32),           # 4 bytes: saccade local dy
    ('forward', np.float32, 3),       # 12 bytes: optical axis
    ('ioa_tilt', np.float32),         # 4 bytes: Local lattice tilt
    ('ioa_axes', np.float32, 2),      # 8 bytes: Local lattice angles (minor, major)
    ('pad', np.float32, 2)            # 8 bytes of pdding
]) # 64 bytes

LENS_DYNAMIC_DTYPE = np.dtype([
    ('adapted_lum', np.float32),      # 4 bytes: slow baseline (~50 ms), gain adaptation
    ('fast_lum', np.float32),         # 4 bytes: fast tracker (~5 ms), saccade drive
    ('lateral_um', np.float32),       # 4 bytes: current focal plane displacement (um)
    ('axial_um', np.float32),         # 4 bytes: current contraction (um)
]) # 16 bytes


# Receptor data

RCPT_STATIC_DTYPE = np.dtype([
    ('position', np.float32, 3),      # 12 bytes: receptor position x, y, z
    ('metadata', np.uint32),          # 4 bytes (see below): eye_id, R_type, neighbour_count, lens_id, chirality
    ('rest_acc', np.float32, 2),      # 8 bytes: acceptance angles (minor, major) at rest
    ('rot_offset', np.float32, 2),    # 8 bytes: focal offset
    ('sensitivity', np.float32, 3),   # 12 bytes: multipliers for UV, blue, green
    ('acc_tilt', np.float32),         # 4 bytes: acceptance angle ellipse tilt
    ('tau', np.float32),              # 4 bytes: EMA integration time for temporal accumulation (ms)
    ('cartridge_src', np.uint32),     # 4 bytes: Neural superposition wiring
    ('pad', np.float32, 2)            # 8 bytes of padding
]) # 64 bytes

# metadata bit layout:
#   bits 0-2: eye ID (0-7)
#   bits 3-6: receptor type (0-15) R1=0, R2=1, ...
#   bits 7-10: neighbour count (0-15)
#   bits 11-26: lens ID (0-65535) parent ommatidium
#   bits 27: chirality (mirror-symmetric kernel, e.g. dorsal vs. ventral)
#   bits 28-31: unused

RCPT_DYNAMIC_DTYPE = np.dtype([
    ('direction', np.float32, 3),     # 12 bytes: Current (actuated) direction
    ('adaptation_state', np.float32), # 4 bytes: Current neural/biochem adaptation
    ('acc_axes', np.float32, 2),      # 8 bytes: Current (actuated) optics
    ('pad2', np.float32, 2)           # 8 bytes of padding
]) # 32 bytes


# Metadata bitfield masks

_CLEAR_EYE_ID = np.uint32(0xFFFFFFF8)
_CLEAR_RECEPTOR_TYPE = np.uint32(0xFFFFFF87)
_CLEAR_NEIGHBOURS = np.uint32(0xFFFFF87F)
_CLEAR_LENS_INDEX = np.uint32(0xF80007FF)
_CLEAR_CHIRALITY = np.uint32(0xF7FFFFFF)