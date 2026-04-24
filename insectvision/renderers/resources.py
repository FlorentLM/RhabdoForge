import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *

from contextlib import contextmanager, ExitStack
from typing import Optional, Dict, Callable, Sequence, Any, Union
import numpy as np


class GPUResourceManager:
    """
    Manages global bindings and texture units to prevent overlapping when combining registries.
    """
    def __init__(self):
        self.ssbo_binding = 0
        self.texture_unit = 0

    def next_ssbo(self) -> int:
        val = self.ssbo_binding
        self.ssbo_binding += 1
        return val

    def next_texture(self) -> int:
        val = self.texture_unit
        self.texture_unit += 1
        return val


class BufferObject:
    """
    A wrapper around a GPU Buffer (SSBO, PBO, etc).
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
            binding: Optional[int] = None,
            supports_async: bool = False,
            _async_reader: Optional['Callable'] = None,
    ):

        self.name = name
        self.handle = handle
        self.dtype = np.dtype(dtype)
        self.count = count
        self.target = target
        self.usage = usage
        self.binding = binding
        self._supports_async = supports_async
        self._async_reader = _async_reader

    def __repr__(self):
        sync_str = ' (async)' if self._supports_async else ''
        binding_str = f' binding={self.binding}' if self.binding else ''
        return (
            f"<{self.BUFFER_TYPES.get(self.target, 'Unknown buffer')}{sync_str} "
            f'name={self.name!r} handle={self.handle} '
            f'count={self.count} dtype={self.dtype}'
            f'{binding_str}>'
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
    def bind_base(self):
        """Binding for Indexed buffers (SSBOs)."""
        if self.binding is None:
            raise ValueError(f"Buffer {self.name} has no binding point assigned.")
        glBindBufferBase(self.target, self.binding, self.handle)
        try:
            yield self
        finally:
            glBindBufferBase(self.target, self.binding, 0)

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

    def __init__(self, resource_manager: Optional[GPUResourceManager] = None):
        self._buffers: Dict[str, BufferObject] = {}
        self._rm = resource_manager

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

    @property
    def shader_defines(self) -> Dict[str, Any]:
        return {f"BINDING_{b.name.upper()}": b.binding for b in self._buffers.values() if b.binding is not None}

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

        binding = None
        if target in (GL_SHADER_STORAGE_BUFFER, GL_UNIFORM_BUFFER) and self._rm:
            binding = self._rm.next_ssbo()

        buf = BufferObject(
            name=name,
            handle=handle,
            dtype=dtype,
            count=count,
            target=target,
            usage=usage,
            binding=binding,
            supports_async=supports_async,
            _async_reader=_async_reader
        )

        if buf.name in self._buffers:
            raise KeyError(f"Buffer {buf.name!r} already registered.")

        self._buffers[buf.name] = buf

        return buf

    @contextmanager
    def grouped_bind(self, names: Optional[Union[str, Sequence[str]]] = None):
        """
        Binds a group of buffers.
        """
        if names is None:
            names =  self._buffers.keys()
        elif isinstance(names, str):
            names = [names]

        with ExitStack() as stack:
            for name in names:
                if name in self._buffers:
                    stack.enter_context(self._buffers[name].bind())
            yield

    @contextmanager
    def grouped_bind_base(self, names: Optional[Union[str, Sequence[str]]] = None):
        """
        Binds a group of buffers to specific binding points.
        """
        if names is None:
            names = self._buffers.keys()
        elif isinstance(names, str):
            names = [names]

        with ExitStack() as stack:
            for name in names:
                if name in self._buffers:
                    stack.enter_context(self._buffers[name].bind_base())
            yield

    def free(self):
        for buf in self._buffers.values():
            buf.free()
        self._buffers.clear()


class TextureObject:
    """Wrapper for a GPU Texture, managed automatically."""
    def __init__(self, name: str, handle: int, target: int, unit: int):
        self.name = name
        self.handle = handle
        self.target = target
        self.unit = unit

    @contextmanager
    def bind(self):
        glActiveTexture(GL_TEXTURE0 + self.unit)
        glBindTexture(self.target, self.handle)
        try:
            yield self
        finally:
            glActiveTexture(GL_TEXTURE0 + self.unit)
            glBindTexture(self.target, 0)

    def free(self):
        if self.handle:
            glDeleteTextures(1, [self.handle])
            self.handle = 0


class TextureRegistry:
    def __init__(self, resource_manager: Optional[GPUResourceManager] = None):
        self._textures: Dict[str, TextureObject] = {}
        self._rm = resource_manager

    def __getitem__(self, name: str) -> TextureObject:
        return self._textures[name]

    def __contains__(self, name: str) -> bool:
        return name in self._textures

    @contextmanager
    def bind_all(self):
        with ExitStack() as stack:
            for tex in self._textures.values():
                stack.enter_context(tex.bind())
            yield

    def allocate_2d(self, name: str, width: int, height: int, image_data=None, repeat=False, dtype=float) -> TextureObject:
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT if repeat else GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT if repeat else GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        if dtype == float:
            bitdepth, typ = GL_RGBA32F, GL_FLOAT
        else:
            bitdepth, typ = GL_SRGB_ALPHA, GL_UNSIGNED_BYTE

        glTexImage2D(GL_TEXTURE_2D, 0, bitdepth, width, height, 0, GL_RGBA, typ, image_data)
        glBindTexture(GL_TEXTURE_2D, 0)

        unit = self._rm.next_texture() if self._rm else 0
        tex = TextureObject(name, tex_id, GL_TEXTURE_2D, unit)
        self._textures[name] = tex
        return tex

    def allocate_array(self, name: str, texture_ids: Sequence[int]) -> TextureObject:
        if not texture_ids:
            raise ValueError("No texture IDs provided for array.")

        glBindTexture(GL_TEXTURE_2D, texture_ids[0])
        tex_w = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH)
        tex_h = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT)
        glBindTexture(GL_TEXTURE_2D, 0)

        layer_count = len(texture_ids)
        tex_array_id = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D_ARRAY, tex_array_id)
        glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_SRGB8_ALPHA8, tex_w, tex_h, layer_count)

        for i, tex_id in enumerate(texture_ids):
            glCopyImageSubData(
                tex_id, GL_TEXTURE_2D, 0, 0, 0, 0,
                tex_array_id, GL_TEXTURE_2D_ARRAY, 0, 0, 0, i,
                tex_w, tex_h, 1
            )

        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)

        unit = self._rm.next_texture() if self._rm else 0
        tex = TextureObject(name, tex_array_id, GL_TEXTURE_2D_ARRAY, unit)
        self._textures[name] = tex
        return tex

    def register_existing(self, name: str, handle: int, target: int) -> TextureObject:
        unit = self._rm.next_texture() if self._rm else 0
        tex = TextureObject(name, handle, target, unit)
        self._textures[name] = tex
        return tex

    def free(self):
        for tex in self._textures.values():
            tex.free()
        self._textures.clear()