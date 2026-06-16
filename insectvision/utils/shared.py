from enum import IntEnum
import numpy as np
from numpy.typing import ArrayLike


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


class SamplingMode(IntEnum):
    Gaussian = 0    # Default approximation
    Airy = 1        # Physical diffraction pattern



# TODO: These below will live here for now...


from scipy.special import j1

def airy_sensitivity_lut(size=256):
    # map x from 0.0 (centre) to 4.0 (deep in the tails)
    # where x is normalised such that x=0.5 is the half-max

    x_vals = np.linspace(0, 4.0, size)
    lut_data = []

    for x_norm in x_vals:
        # scale for Airy FWHM logic
        x = 3.232 * x_norm
        if x < 1e-6:
            val = 1.0
        else:
            val = (2.0 * j1(x) / x) ** 2
        lut_data.append(float(val))

    return np.array(lut_data, dtype=np.float32)
# TODO: Other sensitivity profiles?


def norm_minmax(
        array: ArrayLike,
        axis=None,
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


def broadcast_to_shape(values, shape, accepted, name='value', dtype=np.float32):
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


def broadcast_1d(values, n, name='value'):
    return broadcast_to_shape(
        values=values, shape=(n,),
        accepted=[((), ()), ((1,), (0,)), ((n,), (0,))],
        name=name,
        dtype=np.float32
    )
