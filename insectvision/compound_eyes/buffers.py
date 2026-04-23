import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *

from typing import Optional, Dict
import numpy as np


class BufferHandle:
    """
    A wrapper around a GPU SSBO.

    Notes:
        - `read()` is synchronous (with glGetBufferSubData).
            Cheap for small buffers (lens_dynamic, rcpt_dynamic),
            slow for large ones (colors, ema_state).
            Prefer `read_async()` on buffers that support it.
        - `write()` uploads the full buffer contents (with glBufferSubData).
            Partial writes can be done with `write(data, start=...)`.
        - `reset()` zeroes the GPU buffer directly, bypassing any CPU mirror
          or dirty-tracking.
    """

    def __init__(
            self,
            name: str,
            binding: int,
            dtype: np.dtype,
            count: int,
            supports_async: bool = False,
            _async_reader: Optional[callable] = None,
    ):

        self.name = name
        self.binding = binding
        self.dtype = np.dtype(dtype)
        self.count = count
        self._supports_async = supports_async
        self._async_reader = _async_reader

    def __repr__(self):
        return (
            f"<BufferHandle name={self.name!r} SSBO bind={self.binding} "
            f"count={self.count} dtype={self.dtype} "
            f"{'(async)' if self._supports_async else '(sync)'}>"
        )

    @property
    def itemsize(self) -> int:
        return int(self.dtype.itemsize)

    @property
    def nbytes(self) -> int:
        return self.count * self.itemsize

    def read(self, start: int = 0, count: Optional[int] = None) -> np.ndarray:
        """
        Synchronous read of [start:start+count] elements. Blocks until GPU has finished pending writes.
        """

        if count is None:
            count = self.count - start

        if start < 0 or count < 0 or (start + count) > self.count:
            raise IndexError(f"read range [{start}:{start + count}] out of bounds for {self.name} (count={self.count})")

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.binding)
        data_bytes = glGetBufferSubData(
            GL_SHADER_STORAGE_BUFFER,
            start * self.itemsize,
            count * self.itemsize
        )
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        return np.frombuffer(data_bytes, dtype=self.dtype).copy()

    def read_async(self) -> np.ndarray:
        """
        Asynchronous read using the double-buffered PBO path. Only available
        for buffers registered with `supports_async=True` (currently: colors).
        Returns the previous frame's data — may be zeros on the first frame.
        """
        if not self._supports_async or self._async_reader is None:
            raise RuntimeError(f"Buffer '{self.name}' does not support async read. Use read() instead.")
        return self._async_reader()

    def write(self, data: np.ndarray, start: int = 0) -> None:
        """
        Upload data to the buffer starting at element index `start` (default 0).
        """

        arr = np.ascontiguousarray(data)
        if arr.dtype != self.dtype:
            # allow raw bytes of matching size, raise on mismatch
            if arr.nbytes != len(arr) * self.itemsize:
                raise TypeError(
                    f"write(): data dtype {arr.dtype} does not match buffer dtype {self.dtype}."
                )

        n = arr.size if arr.dtype == self.dtype else arr.nbytes // self.itemsize
        if start < 0 or (start + n) > self.count:
            raise IndexError(
                f"write range [{start}:{start + n}] out of bounds for {self.name} (count={self.count})"
            )

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.binding)
        glBufferSubData(
            GL_SHADER_STORAGE_BUFFER,
            start * self.itemsize,
            arr.nbytes,
            arr.tobytes(),
        )
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def reset(self) -> None:
        """
        Zero-out the entire GPU buffer immediately.
        """
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.binding)
        try:
            # glClearBufferSubData is the fast path
            glClearBufferSubData(
                GL_SHADER_STORAGE_BUFFER,
                GL_R32F,              # internal format for the clear (arbitrary for zero)
                0, self.nbytes,
                GL_RED, GL_FLOAT,
                None,                       # passing None clears to zero
            )
        except Exception:
            # if that didn't work just upload zeroes
            zeros = np.zeros(self.count, dtype=self.dtype)
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.nbytes, zeros)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)


class BufferRegistry:

    def __init__(self):
        self._buffers: Dict[str, BufferHandle] = {}

    def __getitem__(self, name: str) -> BufferHandle:
        if name not in self._buffers:
            raise KeyError(
                f"No buffer named {name!r}. Available: {sorted(self._buffers)}"
            )
        return self._buffers[name]

    def __contains__(self, name: str) -> bool:
        return name in self._buffers

    def __iter__(self):
        return iter(self._buffers)

    def __repr__(self):
        return f"<BufferRegistry {list(self._buffers)}>"

    def keys(self):
        return self._buffers.keys()

    def items(self):
        return self._buffers.items()

    def values(self):
        return self._buffers.values()

    def allocate(self,
        name: str,
        dtype: np.dtype,
        count: int,
        data: Optional[np.ndarray] = None,
        usage: int = GL_STATIC_DRAW,
        supports_async: bool = False,
        _async_reader: Optional[callable] = None,

    ) -> BufferHandle:
        """Create a new SSBO, allocate GPU memory, and register it."""

        itemsize = np.dtype(dtype).itemsize
        size_bytes = count * itemsize

        binding = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, binding)

        if data is not None:
            data_arr = np.ascontiguousarray(data)
            glBufferData(GL_SHADER_STORAGE_BUFFER, size_bytes, data_arr.tobytes(), usage)
        else:
            glBufferData(GL_SHADER_STORAGE_BUFFER, size_bytes, None, usage)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        handle = BufferHandle(name, binding, dtype, count, supports_async, _async_reader)
        if handle.name in self._buffers:
            raise KeyError(f"Buffer {handle.name!r} already registered.")

        self._buffers[handle.name] = handle

        return handle
