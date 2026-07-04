from enum import IntEnum
from typing import Tuple, Callable, Sequence, Optional
import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import Akima1DInterpolator
from scipy.special import j1
from pyglm import glm


WORLD_RIGHT = WORLD_X = glm.vec3(1.0, 0.0, 0.0)
WORLD_UP = WORLD_Y = glm.vec3(0.0, 1.0, 0.0)
WORLD_FORWARD = WORLD_Z = glm.vec3(0.0, 0.0, -1.0)
WORLD_LEFT = -WORLD_RIGHT
WORLD_DOWN = -WORLD_UP
WORLD_BACKWARD = -WORLD_FORWARD


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


class Colormap(IntEnum):
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


# Shared utils

def _match_batch(a: ArrayLike, b: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
    """
    Ensures 'a' and 'b' can broadcast by injecting size-1 dimensions before the last dimension.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    while a.ndim < b.ndim:
        a = np.expand_dims(a, axis=-2)
    while b.ndim < a.ndim:
        b = np.expand_dims(b, axis=-2)
    return a, b


def broadcast_to_shape(
        values: ArrayLike,
        shape: Sequence[int],
        accepted: Sequence[Sequence[int]],
        name: str = 'value', dtype=np.float32
    ) -> np.ndarray:

    arr = np.asarray(values, dtype=dtype)

    target = tuple(shape)
    for in_shape, axes in accepted:
        if arr.shape == tuple(in_shape):
            view = [1] * len(target)
            for src_ax, tgt_ax in enumerate(axes):
                view[tgt_ax] = arr.shape[src_ax]
            return np.ascontiguousarray(np.broadcast_to(arr.reshape(view), target), dtype=dtype)

    allowed = ', '.join(str(tuple(s)) for s, _ in accepted)
    raise ValueError(f'{name} must have one of shapes [{allowed}], got {arr.shape}')


def broadcast_1d(
        values: ArrayLike,
        n: int,
        name: str = 'value'
    ) -> np.ndarray:

    return broadcast_to_shape(
        values=values, shape=(n,),
        accepted=[((), ()), ((1,), (0,)), ((n,), (0,))],
        name=name,
        dtype=np.float32
    )


def akima_interp_fn(x: ArrayLike, y: ArrayLike, fill_value: float) -> 'Callable':
    """
    Akima interpolator that returns 'fill_value' for queries outside range.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    akima_fn = Akima1DInterpolator(x, y)

    def wrapper(query_x):
        query_x = np.asarray(query_x)
        mask_oob = (query_x < x.min()) | (query_x > x.max())
        vals = akima_fn(query_x)
        return np.where(mask_oob, fill_value, vals)

    return wrapper


# TODO: Other luts for other sensitivity profiles?

def airy_sensitivity_lut(size: int = 256, range: float = 6.0) -> np.ndarray:
    """
    Map x from 0.0 (centre) to 4.0 (deep in the tails)
    where x is normalised such that x=0.5 is the half max
    """

    x_vals = np.linspace(0, range, size)
    lut_data = []

    for x_norm in x_vals:
        # Scale for Airy FWHM
        x = 3.232 * x_norm
        if x < 1e-6:
            val = 1.0
        else:
            val = (2.0 * j1(x) / x) ** 2
        lut_data.append(float(val))

    return np.array(lut_data, dtype=np.float32)


def lorentzian_sensitivity_lut(size: int = 256) -> np.ndarray:
    """
    Lorentzian (Cauchy) profile.
    Heavy tails: stays brighter further from the centre compared to a Gaussian.
    """
    x_vals = np.linspace(0, 4.0, size)
    # At x=0.5, val = 1 / (1 + (0.5/0.5)^2) = 0.5
    lut_data = 1.0 / (1.0 + (x_vals / 0.5) ** 2)
    return np.array(lut_data, dtype=np.float32)


def leakage_sensitivity_lut(size: int = 256, pedestal_height: float = 0.05, pedestal_width: float = 3.0) -> np.ndarray:
    """
    Sum-of-Gaussians.
    Simulates a narrow optical core with a wide 'pedestal' caused by
    light leakage between ommatidia (common in insect eye measurements).
    """
    x_vals = np.linspace(0, 4.0, size)
    core = np.exp(-2.77258872224 * x_vals ** 2) # Core Gaussian (standard GAUSS_K)
    wide = np.exp(-2.77258872224 * (x_vals / pedestal_width) ** 2)   # wide Gaussian pedestal
    # re-normalised so peak is 1.0
    combined = (core + pedestal_height * wide) / (1.0 + pedestal_height)
    return np.array(combined, dtype=np.float32)


def norm_minmax(
        array: ArrayLike,
        axis: Optional[int] = None,
        inplace: bool = False,
        eps: float = 1e-9,
) -> np.ndarray:
    """
    Min-max normalise to [0, 1] (NaN-aware).
    """

    if inplace:
        arr = np.asarray(array)
        if not np.issubdtype(arr.dtype, np.floating):
            raise ValueError("In-place normalisation requires a float array dtype.")
    else:
        arr = np.array(array, dtype=float, copy=True)

    keep = (axis is not None)
    vmin = np.nanmin(arr, axis=axis, keepdims=keep)
    vmax = np.nanmax(arr, axis=axis, keepdims=keep)

    np.subtract(arr, vmin, out=arr)
    np.divide(arr, (vmax - vmin) + eps, out=arr)

    return arr


def norm_l2(
        vectors: ArrayLike,
        axis: int = -1,
        inplace: bool = False,
        eps: float = 1e-9,
) -> np.ndarray:
    """
    L2-normalise along 'axis'. Vectors with norm <= eps are left unchanged.
    """
    if inplace:
        arr = np.asarray(vectors)
        if not np.issubdtype(arr.dtype, np.floating):
            raise ValueError("In-place normalisation requires a float array dtype.")
    else:
        arr = np.array(vectors, dtype=float, copy=True)

    norms = np.linalg.norm(arr, axis=axis, keepdims=True)
    np.divide(arr, norms, out=arr, where=norms > eps)

    return arr