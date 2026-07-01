from typing import Tuple, Callable
import numpy as np
from numpy._typing import ArrayLike
from numpy.typing import ArrayLike
from scipy.interpolate import Akima1DInterpolator


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
