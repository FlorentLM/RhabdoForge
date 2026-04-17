import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *

import time
import random
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Optional, Union, Dict, Tuple
from numpy.typing import ArrayLike
import numpy as np
from pyglm import glm

from insectvision.interactive.utils import DisplayMode
from insectvision.geometry.meshes import CONE_VERTICES, SPHERE_VERTICES
from insectvision.compound_eyes import ReceptorArray, VisualOutput, Eye
from insectvision.engine.agent import Agent
from insectvision.engine.scene import Scene
from insectvision.engine.shader_utils import ShaderProgram


# SSBO bindings
BINDING_RECEPTORS         = 0
BINDING_LENSES            = 1
BINDING_COLORS            = 2
BINDING_STATE             = 3
BINDING_RAYS_INTERMEDIATE = 4
BINDING_CARTRIDGES        = 17
BINDING_SACCADES          = 18


class EyeOutput(IntEnum):
    Raw = 0          # Render receptors individually (scaled down)
    Ommatidium = 1   # One tile per lens (averaging R1-R8)
    Cartridge = 2    # One tile per lens (averaging optically superimposed receptors)


class OmmatidiaProjection(IntEnum):
    Position = 0     # Positions on the curved eye surface
    OpticalAxis = 1  # Positions from optical axis directions


class Colormap(IntEnum):
    Diverging = 0    # Blue, white, red (signed and centred on zero)
    Sequential = 1   # Viridis-like
    Thermal = 2      # Black, red, white


##
# Small helpers

def _query_available_VRAM() -> int:
    """Queries available VRAM in MB."""

    extensions = glGetString(GL_EXTENSIONS).decode('utf-8').split()

    # NVIDIA
    if 'GL_NVX_gpu_memory_info' in extensions:
        from OpenGL.raw.GL.NVX.gpu_memory_info import GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX
        return glGetIntegerv(GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX) // 1024

    # AMD
    elif 'GL_ATI_meminfo' in extensions:
        # GL_VBO_FREE_MEMORY_ATI returns [total_free, largest_free_block, total_aux_free, largest_aux_free]
        GL_VBO_FREE_MEMORY_ATI = 0x87FB
        mem_info = (GLint * 4)()
        glGetIntegerv(GL_VBO_FREE_MEMORY_ATI, mem_info)
        return mem_info[0] // 1024

    return 0


def _create_ssbo(data: Optional[ArrayLike] = None, size: Optional[int] = None, usage: int = GL_STATIC_DRAW) -> int:
    """Create a new SSBO, upload data into it, and return the handle."""

    if data is None and size is None:
        raise AttributeError("SSBO size must be provided if data is not passed.")

    if data is not None:
        data = np.asarray(data)
        if size is None:
            size = data.nbytes

    ssbo_bind = glGenBuffers(1)

    glBindBuffer(GL_SHADER_STORAGE_BUFFER, ssbo_bind)
    glBufferData(GL_SHADER_STORAGE_BUFFER, size, data, usage)
    glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
    return ssbo_bind


def _create_pbo(data: Optional[ArrayLike] = None, size: Optional[int] = None, usage: int = GL_STREAM_READ) -> int:
    """Create a new PBO, upload data into it, and return the handle."""

    if data is None and size is None:
        raise AttributeError("PBO size must be provided if data is not passed.")

    if data is not None:
        data = np.asarray(data)
        if size is None:
            size = data.nbytes

    pbo_bind = glGenBuffers(1)

    glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo_bind)
    glBufferData(GL_PIXEL_PACK_BUFFER, size, data, usage)
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)

    return pbo_bind


def _create_vao(vertex_data: np.ndarray):
    """
    Create a VAO + VBO for a flat array of vec3 positions at attribute location 0.
    Returns (vao, vbo, vertex_count).
    """
    vertex_count = len(vertex_data) // 3
    vao = glGenVertexArrays(1)
    vbo = glGenBuffers(1)

    glBindVertexArray(vao)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glBufferData(GL_ARRAY_BUFFER, vertex_data.nbytes, vertex_data, GL_STATIC_DRAW)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
    glBindVertexArray(0)

    return vao, vbo, vertex_count


##

class TextureViewer:
    """A simple helper to render a 2D texture to a full-screen quad."""

    def __init__(self):
        self._shader = None
        self._vao = None

    @property
    def shader(self):
        if self._shader is None:
            print("Compiling fullscreen texture viewer shaders...")
            self._shader = ShaderProgram(vert_path='fullscreen.vert', frag_path='textureSampler.frag')
        return self._shader

    @property
    def vao(self):
        if self._vao is None:
            self._vao = glGenVertexArrays(1)
        return self._vao

    def draw(self, texture_id):
        """Draws the given 2D texture to the screen."""

        self.shader.use()
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, texture_id)
        glUniform1i(self.shader.get_loc("texture_sampler"), 0)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)

        glBindVertexArray(0)
        self.shader.stop()

        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)
        glClear(GL_DEPTH_BUFFER_BIT)

    def free(self):
        if self._shader:
            self._shader.free()
        if self._vao:
            glDeleteVertexArrays(1, [self._vao])
        self._shader = None
        self._vao = None


class BaseRenderer(ABC):
    """
    Abstract base class for an insect eye renderer.
    """

    def __init__(self,
            receptor_array: 'ReceptorArray',
            agent: 'Agent',
            time_dithering: bool = True,
            nb_samples: int = 256,
            quasi_random: bool = False,
            batch_size: int = 1
        ):

        self._ra: 'ReceptorArray' = receptor_array
        self.agent: 'Agent' = agent
        self.scene: 'Scene'

        # Estimate needed sizes and available memory
        frame_bytes = len(self._ra) * 16  # 16 bytes per vec4 (RGBA float)

        avail_VRAM = _query_available_VRAM() * 0.9  # 90% for safety, in Mb
        req_batch_dim = (frame_bytes * max(1, batch_size)) / (1024 * 1024)  # in Mb
        other_VRAM_usage = self._estim_vram_use()
        tot_VRAM_needed = req_batch_dim + other_VRAM_usage

        safe_batch_size = batch_size
        if not avail_VRAM:
            print("WARNING: Could not query available VRAM. Assuming enough memory is available.")

        if avail_VRAM and tot_VRAM_needed >= avail_VRAM:
            safe_batch_size = int((avail_VRAM - other_VRAM_usage) * 1024 * 1024 / frame_bytes)

            if safe_batch_size < batch_size:
                print(f"WARNING: Requested batch size of {req_batch_dim} frames ({req_batch_dim:.2f} MB) exceeds available VRAM.")
                print(f"         Reducing batch size to {safe_batch_size} frames.")

        self._max_ssbo_bytes = glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE)  # in bytes
        self._batch_size = max(1, safe_batch_size)

        # GPU-side history buffer (SSBO)
        self._colours_ssbo = _create_ssbo(data=None, size=frame_bytes * self._batch_size, usage=GL_DYNAMIC_DRAW)

        # 2 Ping-Pong PBOs
        self._pbos = [
            _create_pbo(data=None, size=frame_bytes, usage=GL_STREAM_READ),
            _create_pbo(data=None, size=frame_bytes, usage=GL_STREAM_READ)
        ]
        self._pbo_index = 0
        self._fences = [0, 0]

        # and corresponding CPU buffer
        self._colours_cpu_buffer = np.zeros((len(self._ra), 4), dtype=np.float32)

        # Receptors
        self._receptors_data_ssbo = _create_ssbo(data=self._ra.receptor_data, usage=GL_DYNAMIC_DRAW)
        self._receptors_state_ssbo = _create_ssbo(data=None, size=frame_bytes, usage=GL_DYNAMIC_DRAW)

        # Ommatidia
        self._lens_data_ssbo = _create_ssbo(data=self._ra.lens_data, usage=GL_STATIC_DRAW)

        # Cartridge map
        self._cartridge_ssbo = _create_ssbo(data=self._ra.cartridge_map.astype(np.uint32), usage=GL_STATIC_DRAW)

        # Saccade field
        sacc = self._ra.saccade_field()
        sacc_padded = np.zeros((len(sacc), 4), dtype=np.float32)
        sacc_padded[:, :3] = sacc
        self._saccade_ssbo = _create_ssbo(data=sacc_padded, usage=GL_STATIC_DRAW)

        # Visualisation shaders (lazy-loaded)
        self._lazy_fp_colour_shader: Optional[int] = None    # 1st person colour mode
        self._lazy_fp_overlay_shader: Optional[int] = None   # 1st person overlay mode
        self._lazy_tp_colour_shader: Optional[int] = None    # 3rd person colour mode
        self._lazy_tp_overlay_shader: Optional[int] = None   # 3rd person overlay mode
        self._lazy_overlay_ssbo: Optional[int] = None

        # Geometry VAOs (lazy-loaded)
        self._lazy_lens_cones_vao: Optional[int] = None
        self._lazy_lens_hemisph_vao: Optional[int] = None
        self._lazy_lens_cones_vbo: Optional[int] = None
        self._lazy_lens_hemisph_vbo: Optional[int] = None
        self._lazy_cones_vertices = 0
        self._lazy_hemisph_vertices = 0

        # Overlay stuff
        self._overlay_ssbo_capacity = 0
        self._overlay_colormap = Colormap.Thermal
        self._overlay_range = (0.0, 1.0)
        self._overlay_current_peak: Optional[float] = None
        self._overlay_compression = 1.0   # power exponent for dynamic range compression (1.0 = linear, 0.5 = sqrt, etc)
        self._overlay_autorange_perc = 98 # percentile to reject outliers

        # States flags and other things
        self._overlay_enabled = False
        self.runs_interactive = False
        self.tiled_mode = True

        self.projection_mode = OmmatidiaProjection.Position
        self.output_mode = EyeOutput.Ommatidium

        self._nb_samples = nb_samples
        self._quasi_random = quasi_random  # Halton sampling for direction generation
        self._time_dithering = time_dithering

        # Time keeping
        self._dither_counter: int = 0  # only advanced when time dithering is on
        self._frame_index: int = 0  # advanced at each new rendered frame
        self._last_render_time: float = 0.0
        self._dt = 0.0  # elapsed time (in seconds) since last render

    # Internal properties for lazy loaded resources

    @property
    def _cones_vao(self):
        if self._lazy_lens_cones_vao is None:
            vao, vbo, vertices = _create_vao(CONE_VERTICES)

            self._lazy_lens_cones_vao = vao
            self._lazy_lens_cones_vbo = vbo
            self._lazy_cones_vertices = vertices

        return self._lazy_lens_cones_vao

    @property
    def _hemispheres_vao(self):
        if self._lazy_lens_hemisph_vao is None:
            vao, vbo, vertices = _create_vao(SPHERE_VERTICES)

            self._lazy_lens_hemisph_vao = vao
            self._lazy_lens_hemisph_vbo = vbo
            self._lazy_hemisph_vertices = vertices

        return self._lazy_lens_hemisph_vao

    @property
    def _tp_colour_shader(self):
        if self._lazy_tp_colour_shader is None:
            self._lazy_tp_colour_shader = ShaderProgram(
                vert_path='thirdPersonEye.vert',
                frag_path='thirdPersonEye.frag'
            )
        return self._lazy_tp_colour_shader

    @property
    def _tp_overlay_shader(self):
        if self._lazy_tp_overlay_shader is None:
            self._lazy_tp_overlay_shader = ShaderProgram(
                vert_path='thirdPersonEye.vert',
                frag_path='thirdPersonEye.frag',
                defines={'OVERLAY_MODE'}
            )
        return self._lazy_tp_overlay_shader

    @property
    def _fp_colour_shader(self):
        if self._lazy_fp_colour_shader is None:
            self._lazy_fp_colour_shader = ShaderProgram(
                vert_path='firstPersonEye.vert',
                frag_path='firstPersonEye.frag'
            )
        return self._lazy_fp_colour_shader

    @property
    def _fp_overlay_shader(self):
        if self._lazy_fp_overlay_shader is None:
            self._lazy_fp_overlay_shader = ShaderProgram(
                vert_path='firstPersonEye.vert',
                frag_path='firstPersonEye.frag',
                defines={'OVERLAY_MODE'}
            )
        return self._lazy_fp_overlay_shader

    @property
    def _overlay_ssbo(self) -> int:
        if self._lazy_overlay_ssbo is None:
            self._lazy_overlay_ssbo = glGenBuffers(1)

        return self._lazy_overlay_ssbo

    @property
    def _active_colour_ssbo(self) -> int:
        return self._overlay_ssbo if self.overlay_enabled else self._colours_ssbo

    # Various internal helpers

    def _estim_vram_use(self) -> float:
        # Each subclass implements its own estimation logic
        return 100.0

    def _tick(self):
        now = time.perf_counter()
        self._dt = (now - self._last_render_time) if self._last_render_time > 0.0 else 0.0
        self._last_render_time = now

    # Internal helpers for GL resource binding

    def _bind_eye_ssbos(self):
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_RECEPTORS, self._receptors_data_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_LENSES, self._lens_data_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_COLORS, self._active_colour_ssbo)
        if self.output_mode == EyeOutput.Cartridge:
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_CARTRIDGES, self._cartridge_ssbo)

    def _unbind_eye_ssbos(self):
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_RECEPTORS, 0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_LENSES, 0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_COLORS, 0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_CARTRIDGES, 0)

    def _set_eye_uniforms(self, shader):
        glUniform1i(shader.get_loc('output_mode'), int(self.output_mode))
        glUniform1i(shader.get_loc('receptor_count'), self._ra.receptor_count)
        glUniform1i(shader.get_loc('center_index'), self._ra.kernel.center_index)

        if self.overlay_enabled:
            glUniform1f(shader.get_loc('overlay_data_min'), self._overlay_range[0])
            glUniform1f(shader.get_loc('overlay_data_max'), self._overlay_range[1])
            glUniform1i(shader.get_loc('colormap'), int(self._overlay_colormap))
            glUniform1f(shader.get_loc('compression'), self._overlay_compression)

    # Internal rendering logic and draw calls

    @abstractmethod
    def _compute_colors(self, *args, **kwargs):
        # Each subclass implements its own rendering logic
        raise NotImplementedError

    def _draw_eye_firstperson(self):
        """First-person compound-eye view (colours or scalar overlay)."""

        shader = self._fp_overlay_shader if self.overlay_enabled else self._fp_colour_shader
        shader.use()

        glEnable(GL_DEPTH_TEST)

        # Compute nice-looking dimensions for tiles
        viewport = glGetIntegerv(GL_VIEWPORT)
        aspect_ratio = viewport[2] / viewport[3] if viewport[3] > 0 else 1.0
        receptive_field_scale = 1.0 / (2.0 * np.pi)

        glUniform1f(shader.get_loc('aspect_ratio'), aspect_ratio)
        glUniform1i(shader.get_loc('tiled_mode'), self.tiled_mode)
        glUniform1i(shader.get_loc('projection_mode'), self.projection_mode)
        glUniform1f(shader.get_loc('receptive_field_scale'), receptive_field_scale)

        self._set_eye_uniforms(shader)
        self._bind_eye_ssbos()

        nb_units = len(self._ra) if self.output_mode == EyeOutput.Raw else self._ra.lens_count

        glBindVertexArray(self._cones_vao)
        glDrawArraysInstanced(GL_TRIANGLES, 0, self._lazy_cones_vertices, nb_units)

        glBindVertexArray(0)
        self._unbind_eye_ssbos()

        glDisable(GL_DEPTH_TEST)

        shader.stop()

    def _draw_eye_thirdperson(self, observer_camera):
        """Third-person eye model (colours or scalar overlay)."""

        shader = self._tp_overlay_shader if self.overlay_enabled else self._tp_colour_shader
        shader.use()

        glEnable(GL_BLEND)

        view_matrix_np = np.array(observer_camera.view, dtype=np.float32)
        projection_matrix_np = np.array(observer_camera.projection, dtype=np.float32)
        c2w_mat = glm.inverse(self.agent.view)

        glUniformMatrix4fv(shader.get_loc('view'), 1, True, view_matrix_np)
        glUniformMatrix4fv(shader.get_loc('projection'), 1, True, projection_matrix_np)
        glUniformMatrix4fv(shader.get_loc('eye_to_world'), 1, False, glm.value_ptr(c2w_mat))

        # Compute nice-looking dimensions for acceptance angle cones
        avg_radius = np.mean(np.linalg.norm(self._ra.receptor_data['position'][:, :3], axis=1))
        cone_length = max(0.01, avg_radius) * 0.5
        visualisation_scale = 1.0

        glUniform1i(shader.get_loc('projection_mode'), self.projection_mode)
        glUniform1f(shader.get_loc('cone_length'), cone_length)
        glUniform1f(shader.get_loc('visualisation_scale'), visualisation_scale)

        self._set_eye_uniforms(shader)
        self._bind_eye_ssbos()

        nb_units = len(self._ra) if self.output_mode == EyeOutput.Raw else self._ra.lens_count

        if self.projection_mode == OmmatidiaProjection.Position:
            glBindVertexArray(self._hemispheres_vao)
            glDrawArraysInstanced(GL_TRIANGLES, 0, self._lazy_hemisph_vertices, nb_units)
        else:
            glBindVertexArray(self._cones_vao)
            glDrawArraysInstanced(GL_TRIANGLES, 0, self._lazy_cones_vertices, nb_units)

        glBindVertexArray(0)
        self._unbind_eye_ssbos()

        glEnable(GL_CULL_FACE)
        glDisable(GL_BLEND)
        glDisable(GL_DEPTH_TEST)

        shader.stop()

    # TODO: These two public methods should probably be all under the hood

    def flush(self) -> np.ndarray:
        """
        Blocks until all queued frames on the GPU are rendered, downloads the data, and resets the counter.
        This is used to retrieve a full batch, or the final partial batch at the end of a simulation.
        """

        if self._frame_index == 0:
            return np.array([])

        glFinish()  # Block until all rendering commands are complete

        frames_to_read = int(self._frame_index)

        # Direct synchronous download is ok here
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._colours_ssbo)
        data_bytes = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, len(self._ra) * 16 * frames_to_read)
        data_np = np.frombuffer(data_bytes, dtype=np.float32)

        self._frame_index = 0  # reset counter for next batch

        return data_np.reshape(frames_to_read, len(self._ra), 4)

    def update(self, force_all=False):
        """
        Finds contiguous blocks of changed ommatidia and uploads each block in a single GPU call.
        """

        dirty_indices = np.where(self._ra.dirty_mask)[0]
        if dirty_indices.size == 0:
            return

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._receptors_data_ssbo)

        if force_all:
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self._ra.receptor_data.nbytes, self._ra.receptor_data)

        else:
            # Find contiguous blocks of updated indices
            jumps = np.where(np.diff(dirty_indices) != 1)[0] + 1
            contiguous_blocks = np.split(dirty_indices, jumps)

            item_size = self._ra.receptor_data.itemsize
            for block in contiguous_blocks:
                nb_items = block.size
                if nb_items == 0:
                    continue

                # indexing like this instead of fancy indexing (using [block] directly) avoids a copy
                start_index = block[0]
                data_view = self._ra.receptor_data[start_index:start_index + nb_items]
                glBufferSubData(GL_SHADER_STORAGE_BUFFER, start_index * item_size, data_view.nbytes, data_view)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        self._ra.dirty_mask.fill(False)

    # Main public methods

    @abstractmethod
    def draw(self, view_mode: DisplayMode, point_of_view: Agent):
        # Each subclass implements its own rendering logic
        raise NotImplementedError

    def get_output(self, readback: bool = True) -> Optional['VisualOutput']:
        """
        Runs one frame of simulation.
            - batch_size = 1: Blocks and returns the current frame's data
            - batch_size > 1: Queues the frame on the GPU and returns None
        """

        self.update()

        if self._time_dithering:
            self._dither_counter += 1

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        self._compute_colors()
        glFinish()

        self._frame_index += 1

        if not readback:
            # no cpu readback, we're done here
            return None

        if self.runs_interactive or self._batch_size == 1:
            # Synchronous path

            bytes_to_read = len(self._ra) * 16

            # Copy to current PBO
            current_pbo = self._pbos[self._pbo_index]
            glBindBuffer(GL_COPY_READ_BUFFER, self._colours_ssbo)
            glBindBuffer(GL_COPY_WRITE_BUFFER, current_pbo)
            glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, bytes_to_read)

            # Insert a sync fence to know when this copy finishes
            if self._fences[self._pbo_index]:
                glDeleteSync(self._fences[self._pbo_index])
            self._fences[self._pbo_index] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0)

            # Read from the next PBO (which holds the previous frame)
            next_pbo_index = (self._pbo_index + 1) % 2
            next_pbo = self._pbos[next_pbo_index]
            fence = self._fences[next_pbo_index]

            out_array = np.zeros_like(self._colours_cpu_buffer)

            # If there's a fence, the previous frame has data to read
            if fence:
                glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, 1000000000)

                glBindBuffer(GL_PIXEL_PACK_BUFFER, next_pbo)
                ptr = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, bytes_to_read, GL_MAP_READ_BIT)

                if ptr:
                    ctypes.memmove(self._colours_cpu_buffer.ctypes.data, ptr, bytes_to_read)
                    glUnmapBuffer(GL_PIXEL_PACK_BUFFER)
                    out_array = self._colours_cpu_buffer.copy()

                else:
                    print("Warning: Failed to map PBO. Context lost?")

                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)

            self._pbo_index = next_pbo_index

            # TODO: Currently frame 1 returns zeros because it has no history, maybe return None?
            return VisualOutput(out_array, self._ra)

        else:
            # Asynchronous path

            if self._frame_index < self._batch_size:
                # batch is not full yet, we're done for this frame
                return None

            print(f"  > GPU batch is full. Flushing {self._batch_size} frames...")
            batch_result = self.flush()
            return VisualOutput(batch_result, self._ra)

    def set_overlay(self,
        values: Union[Dict['Eye', np.array], np.array],
        range: Optional[Tuple[float, float]] = None,
        colormap: 'Colormap' = Colormap.Thermal,
        compression: float = 0.5
    ):
        """
        Upload data for overlay visualisation.

        Args:
            values: Either a flat array in global order (total_receptors,)
                    or a dict mapping Eye -> per-eye array
            range: (min, max) bounds for the colourmap. None = auto
            colormap: Which colourmap to use (Colormap enum)
            compression: Power exponent for dynamic range compression
                         1.0 = linear, 0.5 = sqrt, lower = brings out detail
        """

        if isinstance(values, dict):
            merged = np.zeros((len(self._ra)), dtype=np.float32)

            for eye, data in values.items():
                if len(data) == len(eye):
                    data = np.repeat(data, self._ra.receptor_count)
                merged[eye.receptors.global_indices] = data
            values = merged

        buf = np.ascontiguousarray(values, dtype=np.float32).ravel()

        if len(buf) != len(self._ra):
            raise ValueError(
                f"scalar_data has {len(buf)} elements, expected {len(self._ra)}."
            )

        if self._overlay_ssbo_capacity != len(self._ra):
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._overlay_ssbo)
            glBufferData(GL_SHADER_STORAGE_BUFFER, len(self._ra) * 4, None, GL_DYNAMIC_DRAW)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

            self._overlay_ssbo_capacity = len(self._ra)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._overlay_ssbo)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, buf.nbytes, buf)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        # Range
        if range is not None:
            self._overlay_range = float(range[0]), float(range[1])
        else:
            # Auto-normalise: per-frame peak value
            frame_peak = float(np.percentile(np.abs(buf), self._overlay_autorange_perc))
            if self._overlay_current_peak is None:
                self._overlay_current_peak = frame_peak

            # Exponential Moving Average for colourmap scaling
            if frame_peak > self._overlay_current_peak:
                self._overlay_current_peak = 0.5 * self._overlay_current_peak + 0.5 * frame_peak
            else:
                self._overlay_current_peak = 0.98 * self._overlay_current_peak + 0.02 * frame_peak

            range_bound = max(self._overlay_current_peak, 1e-6)

            if colormap == Colormap.Diverging:
                self._overlay_range = (-range_bound, range_bound)
            else:
                self._overlay_range = (0.0, range_bound)

        self._overlay_colormap = colormap
        self._overlay_compression = compression

    # Public properties and methods

    @property
    def nb_samples(self):
        return self._nb_samples

    @nb_samples.setter
    @abstractmethod
    def nb_samples(self, value):
        raise NotImplementedError

    @property
    def time_dithering(self):
        return self._time_dithering

    @time_dithering.setter
    def time_dithering(self, value: bool):
        self._time_dithering = bool(value)
        print(f"Time dithering {'ENABLED' if self._time_dithering else 'DISABLED'}.")

    @property
    def quasi_random(self):
        return self._quasi_random

    @quasi_random.setter
    def quasi_random(self, value: bool):
        self._quasi_random = bool(value)
        print(f"Quasi-random {'ENABLED' if self._quasi_random else 'DISABLED'}.")

    @property
    def overlay_enabled(self) -> bool:
        return self._overlay_enabled and self._overlay_ssbo_capacity > 0

    @overlay_enabled.setter
    def overlay_enabled(self, value: bool):
        self._overlay_enabled = bool(value)

    def dither(self):
        """Dither once (reshuffle the dither counter)"""
        self._dither_counter = random.randint(0, 1024)

    def update_texture(self, asset: 'Asset'):
        """
        Update a texture on the GPU for given Asset.
        Subclasses should override this.
        """
        pass

    # Cleanup

    def free(self):
        """Free GPU resources."""

        for f in self._fences:
            if f:
                try:
                    glDeleteSync(f)
                except Exception:
                    pass

        for buf in (
            self._receptors_data_ssbo,
            self._lens_data_ssbo,
            self._colours_ssbo,
            self._receptors_state_ssbo,
            self._lazy_overlay_ssbo,
            self._cartridge_ssbo,
            self._saccade_ssbo,
            self._lazy_lens_cones_vbo,
            self._lazy_lens_hemisph_vbo,
            *self._pbos
        ):
            if buf:
                glDeleteBuffers(1, [buf])

        for vao in (
            self._lazy_lens_cones_vao,
            self._lazy_lens_hemisph_vao
        ):
            if vao:
                glDeleteVertexArrays(1, [vao])

        for shader in (
                self._lazy_fp_colour_shader,
                self._lazy_fp_overlay_shader,
                self._lazy_tp_colour_shader,
                self._lazy_tp_overlay_shader
        ):
            if shader:
                shader.free()