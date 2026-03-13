import numpy as np


RECEPTOR_DTYPE = np.dtype([
    ('position', np.float32, 3),    # 12 bytes: receptor position x, y, z
    ('metadata', np.uint32),        # 4 bytes (see below): eye_id, R_type, neighbour_count, lens_id
    ('direction', np.float32, 3),   # 12 bytes: receptor direction x, y, z
    ('acc_tilt', np.float32),       # 4 bytes: acceptance angle ellipse tilt
    ('acc_axes', np.float32, 2),    # 8 bytes: acceptance angle ellipse minor, major axes
    ('sensitivity', np.float32),    # 4 bytes: photometric response multiplier
    ('tau', np.float32)             # 4 bytes: temporal accumulation (ms)
]) # total 48 bytes

# metadata layout:
#   bits 0-2: eye ID (0-7)
#   bits 3-6: receptor type (0-15) R1=0, R2=1, ...
#   bits 7-10: neighbour count (0-15)
#   bits 11-26: lens ID (0-65535) parent ommatidium
#   bits 27-31: unused

LENS_DTYPE = np.dtype([
    ('ioa_axes', np.float32, 2),    # 8 bytes: lattice geometry axes (ellipse minor, major)
    ('tilt', np.float32),           # 4 bytes: lattice geometry orientation (ellipse tilt)
    ('padding', np.uint32)          # 4 bytes: unused
]) # total 16 bytes


_CLEAR_EYE_ID = np.uint32(0xFFFFFFF8)

_CLEAR_RECEPTOR_TYPE = np.uint32(0xFFFFFF87)

_CLEAR_NEIGHBOURS = np.uint32(0xFFFFF87F)

_CLEAR_LENS_INDEX = np.uint32(0xF80007FF)

DEFAULT_ANGLE = 'deg'   # TODO: get rid of this, and ensure unit consistency everywhere