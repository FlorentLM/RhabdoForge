from typing import Tuple, Sequence, Optional
import numpy as np
from numpy.typing import ArrayLike


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
