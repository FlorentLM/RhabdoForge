import inspect
import logging
from typing import Tuple, Sequence, Optional
import numpy as np
from numpy.typing import ArrayLike


logger = logging.getLogger(__name__)


# Broadcasting helpers

def match_batch(a: ArrayLike, b: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
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
        name: str = 'value',
        dtype=np.float32
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


# Normalisation helpers

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


def norm_rms(array: ArrayLike, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    """
    Center points on centroid and scale to unit RMS radius.
    Preserves aspect ratio (uniform scaling). Works for 2D or 3D.

    Args:
        - array: (N, D) array of spatial coordinates
        - axis: The axis representing the points (default 0)
        - eps: Small value to prevent division by zero
    """
    p = np.asarray(array, dtype=np.float64)
    p = p - p.mean(axis=axis, keepdims=True)
    rms = np.sqrt((p ** 2).sum(axis=-1).mean())
    return p / np.maximum(rms, eps)


# General stuff


def pretty_size(size_bytes: float) -> str:
    # Source - https://stackoverflow.com/a/45846841
    # Posted by rtaft
    # Retrieved 2026-06-22, License - CC BY-SA 3.0
    size_bytes = float(f'{size_bytes:.3g}')
    magnitude = 0
    while abs(size_bytes) >= 1000:
        magnitude += 1
        size_bytes /= 1000.0
    return f'{size_bytes:f}'.rstrip('0').rstrip('.') + ['', 'K', 'M', 'B', 'T'][magnitude]


def infer_name(obj: object, depth: int = 2, fallback: str = '') -> str:
    """
    Attempt to find the variable for an object in the caller's stack.

    Args:
        - obj: The object to search for
        - depth: How many frames to look back
               1 = the function calling infer_name
               2 = the function calling the function that calls infer_name
        - fallback: String to return if no name is found
    """
    try:
        frame = inspect.currentframe()
        for _ in range(depth):
            if frame is None:
                break
            frame = frame.f_back

        if frame:
            for name, val in frame.f_locals.items():
                if val is obj and not name.startswith('_'):
                    return name
    except Exception as e:
        logger.debug(f'Could not infer name: {e}')

    return fallback