import numpy as np


GPU_RECEPTOR_DTYPE = np.dtype([
    ('position', np.float32, 4),                  # 16 bytes: x, y, z, w=1
    ('direction', np.float32, 4),               # 16 bytes: x, y, z, w=0
    ('acceptance_angles', np.float32, 2),       #  8 bytes: minor, major
    ('interommatidial_angles', np.float32, 2),  #  8 bytes: minor, major (from parent lens)
    ('tilt', np.float32),                       #  4 bytes: ellipse tilt (lattice geometry)
    ('sensitivity', np.float32),                #  4 bytes: receptor sensitivity
    ('packed_data', np.uint32),                 #  4 bytes: see below
    ('padding', np.uint32)                      #  4 bytes
])  # total = 64 bytes

# packed_data layout:
# bits 0-2: eye ID (0-7)
# bits 3-6: receptor type (0-15) R1=0, R2=1, ...
# bits 7-10: neighbour count (0-15)
# bits 11-26: lens index (0-65535) parent ommatidium
# bits 27-31: unused for now

# TODO: Receptor dtype coul dbe 48 bytes if IOA and tilt were a separate Lens struct

_CLEAR_EYE_ID = np.uint32(0xFFFFFFF8)

_CLEAR_RECEPTOR_TYPE = np.uint32(0xFFFFFF87)

_CLEAR_NEIGHBOURS = np.uint32(0xFFFFF87F)

_CLEAR_LENS_INDEX = np.uint32(0xF80007FF)

DEFAULT_ANGLE = 'deg'   # TODO: get rid of this, and ensure unit consistency everywhere
