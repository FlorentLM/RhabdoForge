import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *

from contextlib import contextmanager, ExitStack
from typing import Optional, Dict, Callable, Sequence
import numpy as np


class BufferObject:
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

    BUFFER_TYPES = {
        GL_SHADER_STORAGE_BUFFER: 'SSBO',
        GL_PIXEL_PACK_BUFFER: 'PBO',
        GL_ARRAY_BUFFER: 'VBO',
        GL_ELEMENT_ARRAY_BUFFER: 'EBO'
    }

    def __init__(
            self,
            name: str,
            handle: int,
            dtype: np.dtype,
            count: int,
            target: int = GL_SHADER_STORAGE_BUFFER,
            usage: int = GL_STATIC_DRAW,
            supports_async: bool = False,
            _async_reader: Optional['Callable'] = None,
    ):

        self.name = name
        self.handle = handle
        self.dtype = np.dtype(dtype)
        self.count = count
        self.target = target
        self.usage = usage
        self._supports_async = supports_async
        self._async_reader = _async_reader

    def __repr__(self):

        return (
            f"<{self.BUFFER_TYPES.get(self.target, 'Unknown buffer')} "
            f"{'(async)' if self._supports_async else '(sync)'} "
            f'name={self.name!r} handle={self.handle} '
            f'count={self.count} dtype={self.dtype}>'
        )

    @contextmanager
    def bind(self, mode_override: Optional[int] = None):
        """Standard binding."""
        target = mode_override if mode_override is not None else self.target
        glBindBuffer(target, self.handle)
        try:
            yield self
        finally:
            glBindBuffer(target, 0)

    @contextmanager
    def bind_base(self, binding: int):
        """Binding for Indexed buffers (SSBOs)."""
        glBindBufferBase(self.target, binding, self.handle)
        try:
            yield self
        finally:
            glBindBufferBase(self.target, binding, 0)

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

        with self.bind():
            data_bytes = glGetBufferSubData(self.target, start * self.itemsize, count * self.itemsize)
            ret = np.frombuffer(data_bytes, dtype=self.dtype).copy()

        return ret

    def read_async(self) -> np.ndarray:
        """
        Asynchronous read using the double-buffered PBO path. Only available
        for buffers registered with `supports_async=True`.
        Returns the previous frame's data (zeros on the first frame).
        """
        if not self._supports_async or self._async_reader is None:
            raise RuntimeError(f"Buffer '{self.name}' does not support async read. Use read() instead.")
        return self._async_reader()

    def write(self, data: np.ndarray, start: int = 0) -> None:
        """
        Upload data to the buffer starting at element index `start` (default 0).
        """

        arr = np.ascontiguousarray(data)
        # allow raw bytes of matching size, raise on mismatch
        if arr.dtype != self.dtype and arr.nbytes != len(arr) * self.itemsize:
            raise TypeError(f"write(): data dtype {arr.dtype} does not match buffer dtype {self.dtype}.")

        n = arr.size if arr.dtype == self.dtype else arr.nbytes // self.itemsize
        if start < 0 or (start + n) > self.count:
            raise IndexError(f"write range [{start}:{start + n}] out of bounds for {self.name} (count={self.count})")

        with self.bind():
            glBufferSubData(self.target, start * self.itemsize, arr.nbytes, arr.tobytes())

    def resize(self, count: int, data: Optional[np.ndarray] = None) -> None:
        """Resize GPU buffer, optionally initialising with new data."""

        self.count = count

        with self.bind():
            if data is not None:
                arr = np.ascontiguousarray(data)
                glBufferData(self.target, self.nbytes, arr.tobytes(), self.usage)
            else:
                glBufferData(self.target, self.nbytes, None, self.usage)

    def reset(self) -> None:
        """
        Zero-out the entire GPU buffer immediately.
        """

        with self.bind():
            try:
                # glClearBufferSubData is the fast path
                glClearBufferSubData(
                    GL_SHADER_STORAGE_BUFFER,
                    GL_R32F,  # internal format for the clear (arbitrary for zero)
                    0, self.nbytes,
                    GL_RED, GL_FLOAT,
                    None,  # passing None clears to zero
                )
            except Exception:
                # if that didn't work just upload zeroes
                zeros = np.zeros(self.count, dtype=self.dtype)
                glBufferSubData(self.target, 0, self.nbytes, zeros)

    def free(self):
        if self.handle:
            glDeleteBuffers(1, [self.handle])
            self.handle = 0


class BufferRegistry:

    def __init__(self):
        self._buffers: Dict[str, BufferObject] = {}

    def __getitem__(self, name: str) -> BufferObject:
        if name not in self._buffers:
            raise KeyError(f"No buffer named {name!r}. Available: {sorted(self._buffers)}")

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
                 target: int = GL_SHADER_STORAGE_BUFFER,
                 data: Optional[np.ndarray] = None,
                 usage: int = GL_STATIC_DRAW,
                 supports_async: bool = False,
                 _async_reader: Optional[callable] = None,
                 ) -> BufferObject:
        """Create a new SSBO, allocate GPU memory, and register it."""

        itemsize = np.dtype(dtype).itemsize
        size_bytes = count * itemsize

        handle = glGenBuffers(1)
        glBindBuffer(target, handle)

        if data is not None:
            data_arr = np.ascontiguousarray(data)
            glBufferData(target, size_bytes, data_arr.tobytes(), usage)
        else:
            glBufferData(target, size_bytes, None, usage)

        glBindBuffer(target, 0)

        buf = BufferObject(
            name=name,
            handle=handle,
            dtype=dtype,
            count=count,
            target=target,
            usage=usage,
            supports_async=supports_async,
            _async_reader=_async_reader
        )

        if buf.name in self._buffers:
            raise KeyError(f"Buffer {buf.name!r} already registered.")

        self._buffers[buf.name] = buf

        return buf

    @contextmanager
    def grouped_bind(self, names: Sequence[str]):
        """
        Binds a group of buffers.
        """
        with ExitStack() as stack:
            for name in names:
                if name in self._buffers:
                    stack.enter_context(self._buffers[name].bind())
            yield

    @contextmanager
    def grouped_bind_base(self, bindings: Dict[str, int]):
        """
        Binds a group of buffers to specific binding points.
        """
        with ExitStack() as stack:
            for name, point in bindings.items():
                if name in self._buffers:
                    stack.enter_context(self._buffers[name].bind_base(point))
            yield

    def free(self):
        for buf in self._buffers.values():
            buf.free()
        self._buffers.clear()