from enum import IntEnum, Enum, auto
from typing import Tuple
import numpy as np
from pyglm import glm
import numpy.typing as npt


# Constants

WORLD_RIGHT = WORLD_X = glm.vec3(1.0, 0.0, 0.0)
WORLD_UP = WORLD_Y = glm.vec3(0.0, 1.0, 0.0)
WORLD_FORWARD = WORLD_Z = glm.vec3(0.0, 0.0, -1.0)

WORLD_LEFT = -WORLD_RIGHT
WORLD_DOWN = -WORLD_UP
WORLD_BACKWARD = -WORLD_FORWARD


# Type hint for just 3 arrays (forward, right, up)

ReferenceFrame = Tuple[glm.vec3 | npt.NDArray[np.floating], glm.vec3 | npt.NDArray[np.floating], glm.vec3 | npt.NDArray[np.floating]]



# ______ Custom enums ______

class EyeOutput(IntEnum):
    """
    Eye colours output mode (for visualisation only)
    """
    Raw = 0          # Render receptors individually (scaled down)
    Ommatidium = 1   # One tile per lens (averaging R1-R8)
    Cartridge = 2    # One tile per lens (averaging optically superimposed receptors)


class OmmatidiaProjection(IntEnum):
    """
    Eye rendering projection (acceptance vs. lens position)
    """
    Position = 0     # Positions on the curved eye surface
    OpticalAxis = 1  # Positions from optical axis directions


class OverlayColormap(IntEnum):
    """
    Colours for the overlay view (heatmap)
    """
    Diverging = 0    # Blue, white, red (signed and centred on zero)
    Sequential = 1   # Viridis-like
    Thermal = 2      # Black, red, white


class DisplayMode(IntEnum):
    """
    Camera mode
    """
    Compound = 0
    Panoramic = 1
    Third_person = 2
    Perspective = 3


class RandomnessMode(IntEnum):
    Pseudo = 0      # Standard PCG White Noise
    Halton = 1      # Quasi-random low-discrepancy
    Stratified = 2  # Grid-based jittered sampling
    Fibonacci = 3   # Fibonacci disk (Vogel's method), a spiral pattern based on the golden ratio
    Hammersley = 4  # Hammersley set, similar to Halton but more uniform
    Sobol = 5       # Sobol sequence, Owen scrambling


class SamplingMode(IntEnum):
    Gaussian = 0    # Default approximation
    Airy = 1        # Physical diffraction pattern


class AssetType(Enum):
    """
    Distinguishes between different types of geometry assets.
    """
    Mesh = auto()
    Points = auto()


class LightType(Enum):
    Directional = auto()    # Infinitely distant (sun, moon)
    Point = auto()          # Omnidirectional (with falloff)
    Area = auto()           # Rectangular/disk emitter


def to_enum(val, enum_class):
    """Helper to convert string, int, or enum to the target Enum class."""
    if isinstance(val, enum_class):
        return val
    if isinstance(val, str):
        try:
            return enum_class[val]
        except KeyError:
            try:
                return enum_class[val.capitalize()]
            except KeyError:
                pass
        print(f"Warning: Invalid mode '{val}' for {enum_class.__name__}. Defaulting to {list(enum_class)[0].name}")
        return list(enum_class)[0]
    return enum_class(val)


# ______ Custom dtypes for GPU buffers ______

# Any renderable object instance

RENDERABLE_DTYPE = np.dtype([
    ('transform', np.float32, (4, 4)),
    ('inverse_transform', np.float32, (4, 4)),
    ('blas_node_offset', np.uint32),
    ('vertex_or_point_offset', np.uint32),
    ('index_offset', np.uint32),
    ('material_id', np.uint32),
    ('is_points', np.uint32),
    ('prim_index_offset', np.uint32),
    ('radius_factor', np.float32),
    ('is_srgb', np.uint32),
])  # 160 bytes     # TODO: reorganise this struct


# Lights
# TODO: these three datatypes could be optimised a bit

DIR_LIGHT_DTYPE = np.dtype([
    ('direction', np.float32, 3),
    ('angular_radius', np.float32),
    ('color', np.float32, 3),
    ('intensity', np.float32),
    ('cast_shadows', np.uint32),
    ('_pad', np.uint32, 3),
])  # total 48 bytes


POINT_LIGHT_DTYPE = np.dtype([
    ('position', np.float32, 3),
    ('radius', np.float32),
    ('color', np.float32, 3),
    ('intensity', np.float32),
    ('constant_atten', np.float32),
    ('linear_atten', np.float32),
    ('quadratic_atten', np.float32),
    ('cast_shadows', np.uint32),
])  # total 48 bytes


AREA_LIGHT_DTYPE = np.dtype([
    ('position', np.float32, 3),
    ('width', np.float32),
    ('normal', np.float32, 3),
    ('height', np.float32),
    ('tangent', np.float32, 3),
    ('intensity', np.float32),
    ('bitangent', np.float32, 3),
    ('cast_shadows', np.uint32),
    ('color', np.float32, 3),
    ('two_sided', np.uint32),
])  # total 64 bytes


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

    # 16 bytes: Temporal values (all in seconds, the compute shaders integrate with dt in s)
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
    ('optical_scale',   np.float32),        #  4 bytes: optical RF-narrowing factor Δρ_eff/Δρ_rest, read for photon concentration
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

METADATA_BIT_LAYOUT = {
    'eye_id':           (0,  4),
    'rhab_R':           (4,  4),
    'neighbour_count':  (8,  4),
    'omm_id':           (12, 16),
    'chirality_neg':    (28,  1),
    'is_binocular':     (29,  1),
    'is_wired':         (30,  1),
    'is_edge':          (31,  1),
}

# Sentinel value for a rhabdomere slot left unwired (neural superposition)
UNWIRED_SRC = np.uint32(0xFFFFFFFF)


# ______ Custom colours ______

RHAB_COLOURS = [
    '#ffad13',  # R1
    '#FF1C25',  # R2
    '#880015',  # R3
    '#8000ff',  # R4
    '#008000',  # R5
    '#0000ff',  # R6
    '#aaa712',  # R7/8
]
