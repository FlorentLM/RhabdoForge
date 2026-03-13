import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *

import time
import random
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Optional
import numpy as np
from pyglm import glm

from insectvision.geometry import CONE_VERTICES, SPHERE_VERTICES
from insectvision.geometry.compound_eyes import ReceptorArray, VisualOutput

from insectvision.engine.agent import Agent
from insectvision.engine.scene import Scene
from insectvision.engine.utils import ShaderProgram, ViewMode, ProjectionMode


def query_max_SSBO_size() -> int:
    max_size = glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE)
    print(f"[INFO] Max possible SSBO size: {max_size / (1024 * 1024):.2f} MB")
    return max_size


def query_available_VRAM() -> int:
    """
    Checks for NVIDIA extension to query available VRAM. Returns 0 if not supported.
    """
    if b'GL_NVX_gpu_memory_info' in glGetString(GL_EXTENSIONS):
        from OpenGL.raw.GL.NVX.gpu_memory_info import GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX
        return glGetIntegerv(GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX) // 1024
    return 0


class Colormap(IntEnum):
    """
    Colormaps for the heatmap visualisation mode.
    """
    DIVERGING = 0   # Blue -> white -> red  (signed and centred on zero)
    SEQUENTIAL = 1  # Viridis-like          (positive magnitude)
    THERMAL = 2     # Black -> red -> white (positive magnitude)


class BaseRenderer(ABC):
    """
    Abstract base class for an insect eye model.
    """

    def __init__(self,
                 receptor_array: ReceptorArray,
                 time_dithering: bool = True,
                 time_accumulation: float = 0.0,
                 nb_samples: int = 256,
                 quasi_random: bool = False,
                 batch_size: int = 1
                 ):

        self.receptor_array: ReceptorArray = receptor_array
        self.scene: Scene

        self.total_receptors = len(self.receptor_array)
        self._samples_per_receptor = nb_samples

        self._quasi_random = quasi_random   # Halton sampling for direction generation
        self._time_dithering = time_dithering

        self.runs_interactive = False

        # Time keeping
        self._dither_counter: int = 0   # only advanced when time dithering is on
        self._frame_index: int = 0      # advanced at each new rendered frame
        self._last_render_time: float = 0.0
        self._dt = 0.0      # elapsed time (in seconds) since last render

        # Hardware queries
        self._max_ssbo_size_bytes = query_max_SSBO_size()

        # Input receptors SSBO
        self.receptors_input_ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.receptors_input_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.receptor_array.data.nbytes, self.receptor_array.data,
                     GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        # Visualisation resources (lazy-loaded)
        self._voronoi_shader = None     # first person view
        self._eye_model_shader = None   # third person view
        self._heatmap_shader = None     # heatmap mode

        # Heatmap stuff
        self.heatmap_enabled = False
        self._heatmap_ssbo = 0
        self._heatmap_ssbo_capacity = 0
        self._heatmap_colormap = Colormap.THERMAL
        self._heatmap_range = (0.0, 1.0)
        self._heatmap_compression = 1.0  # power exponent for dynamic range compression (1.0 = linear, 0.5 = sqrt, lower = more contrast)
        self._heatmap_auto_range_percentile = 98 # percentile to reject outliers

        self._cones_vao = None
        self._cones_vbo = None
        self._nb_cone_vertices = 0
        self._hemispheres_vao = None
        self._hemispheres_vbo = None
        self._nb_hemisphere_vertices = 0

        self.receptive_field_scale = 1.0 / (2.0 * np.pi)

        # first-person specific stuff
        self.tiled_mode = True
        self.projection_mode: ProjectionMode = ProjectionMode.Physical

        # History buffer state
        self._batch_size = max(1, batch_size)
        self.history_ssbo = None

        # PBO for the synchronous path
        self.sync_pbo = None
        self.sync_cpu_buffer = None

        self._allocate_history_buffers(self._batch_size)

        # Photoreceptor temporal integration
        self.receptor_tau: float = time_accumulation    # TODO: mode this into the receptor dtype

        self.receptor_state_ssbo = glGenBuffers(1)
        receptor_buf_size = self.total_receptors * 16  # vec4 = 16 bytes
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.receptor_state_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, receptor_buf_size, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def _tick(self):
        now = time.perf_counter()
        self._dt = (now - self._last_render_time) if self._last_render_time > 0.0 else 0.0
        self._last_render_time = now

    @property
    def samples_per_receptor(self):
        return self._samples_per_receptor

    @samples_per_receptor.setter
    @abstractmethod
    def samples_per_receptor(self, value):
        # Subclasses may need to re-allocate buffers when this changes
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

    def dither(self):
        """Dither once (reshuffle the dither counter"""
        self._dither_counter = random.randint(0, 1024)

    @abstractmethod
    def _compute_colors(self, *args, **kwargs):
        # Each subclass implements its own rendering logic
        raise NotImplementedError

    @abstractmethod
    def draw(self, view_mode: ViewMode, point_of_view: Agent, agent: Agent):
        # Each subclass implements its own rendering logic
        raise NotImplementedError

    def _allocate_history_buffers(self, requested_frames: int):

        bytes_per_frame = self.total_receptors * 16  # 16 bytes per vec4 (RGBA float)
        requested_history_size_mb = (bytes_per_frame * requested_frames) / (1024 * 1024)

        # Estimate other VRAM usage (this is a rough estimate, subclasses can override)
        other_usage_mb = self.estimate_vram_usage()
        total_requested_mb = requested_history_size_mb + other_usage_mb

        available_vram_mb = query_available_VRAM()

        safe_frames = requested_frames
        if available_vram_mb > 0:
            print(
                f"Available VRAM: {available_vram_mb} MB. Scene VRAM: {other_usage_mb:.2f} MB. Ommatidia views buffer VRAM: {requested_history_size_mb:.2f} MB.")
            if total_requested_mb > available_vram_mb * 0.9:  # 90 % threshold for safety
                safe_history_mb = (available_vram_mb * 0.9) - other_usage_mb
                safe_frames = int(safe_history_mb * 1024 * 1024 / bytes_per_frame)

                if safe_frames < requested_frames:
                    print(
                        f"WARNING: Requested {requested_frames} frames ({requested_history_size_mb:.2f} MB) exceeds available VRAM.")
                    print(f"         Reducing history capacity to {safe_frames} frames.")
        else:
            print("WARNING: Could not query available VRAM. Assuming enough memory is available.")

        self._batch_size = max(1, safe_frames)
        total_buffer_size = self.total_receptors * 16 * self._batch_size

        # GPU-side history buffer (SSBO)
        if self.history_ssbo: glDeleteBuffers(1, [self.history_ssbo])
        self.history_ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.history_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, total_buffer_size, None, GL_DYNAMIC_DRAW)
        self.final_colors_ssbo = self.history_ssbo

        # A single PBO and CPU buffer
        if self.sync_pbo:
            glDeleteBuffers(1, [self.sync_pbo])

        self.sync_pbo = glGenBuffers(1)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, self.sync_pbo)
        glBufferData(GL_PIXEL_PACK_BUFFER, bytes_per_frame, None, GL_STREAM_READ)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        self.sync_cpu_buffer = np.zeros((self.total_receptors, 4), dtype=np.float32)

    def estimate_vram_usage(self) -> float:
        """
        Returns an estimate of VRAM usage in MB, excluding the history buffer
        Subclasses should override this
        """
        return 100.0    # conservative guess

    def get_visual_output(self, agent: Agent, readback: bool = True) -> Optional['VisualOutput']:
        """
        Runs one frame of simulation. Behaviour is determined by the `batch_size`
        - if batch_size = 1: Blocks and returns the current frame's data
        - if batch_size > 1: Queues the frame on the GPU and returns None
        """

        is_sync_mode = getattr(self, 'is_interactive', False) or self._batch_size == 1

        if self._time_dithering:
            self._dither_counter += 1

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        if is_sync_mode:
            # Synchronous path

            self._compute_colors(agent)

            if readback:
                glFinish()

                bytes_to_read = self.total_receptors * 16
                glBindBuffer(GL_COPY_READ_BUFFER, self.history_ssbo)
                glBindBuffer(GL_COPY_WRITE_BUFFER, self.sync_pbo)
                glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, bytes_to_read)

                glBindBuffer(GL_PIXEL_PACK_BUFFER, self.sync_pbo)

                ptr = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, bytes_to_read, GL_MAP_READ_BIT)
                ctypes.memmove(self.sync_cpu_buffer.ctypes.data, ptr, bytes_to_read)

                glUnmapBuffer(GL_PIXEL_PACK_BUFFER)
                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)

                return VisualOutput(self.sync_cpu_buffer.copy(), self.receptor_array)

                # TODO: this would be true zero-copy. should expose it. (but the buffer needs to be unmapped manually after use of raw_array
                # raw_array = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)), shape=(self.num_ommatidia, 4))
                # return VisionResult(raw_array, self.receptor_array)

            else:
                return None

        else:
            # Asynchronous path

            # Submit work for the current frame
            self._compute_colors(agent)
            self._frame_index += 1

            # Check if this frame just completed a batch
            if self._frame_index >= self._batch_size:
                # The buffer is full: block, download, and return the data
                print(f"  > GPU batch is full. Flushing {self._batch_size} frames...")

                batch_result = self.flush()
                return VisualOutput(batch_result, self.receptor_array)

            # if the batch is not yet full, return None
            return None

        # TODO: add third 'streaming async' mode with dual PBOs (ping pong) and GL sync fences

    def flush(self) -> np.ndarray:
        """
        Blocks until all queued frames on the GPU are rendered, downloads the data, and resets the counter.
        This is used to retrieve a full batch, or the final partial batch at the end of a simulation.
        """

        if self._frame_index == 0:
            return np.array([])

        glFinish()  # Block until all rendering commands are complete

        num_frames_to_read = self._frame_index
        bytes_to_read = self.total_receptors * 16 * num_frames_to_read

        # Direct synchronous download is ok here
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.history_ssbo)
        data_bytes = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bytes_to_read)
        data_np = np.frombuffer(data_bytes, dtype=np.float32)

        self._frame_index = 0 # reset counter for next batch

        return data_np.reshape(num_frames_to_read, self.total_receptors, 4)

    def update(self, force_all=False):
        """
        Finds contiguous blocks of changed ommatidia and uploads each block in a single GPU call.
        """

        dirty_indices = np.where(self.receptor_array.dirty_mask)[0]
        if dirty_indices.size == 0:
            return

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.receptors_input_ssbo)

        if force_all:
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.receptor_array.data.nbytes, self.receptor_array.data)

        else:
            # Find contiguous blocks of updated indices
            jumps = np.where(np.diff(dirty_indices) != 1)[0] + 1
            contiguous_blocks = np.split(dirty_indices, jumps)

            item_size = self.receptor_array.data.itemsize  # 48 bytes

            for block in contiguous_blocks:
                nb_items = block.size
                if nb_items == 0:
                    continue

                # indexing like this instead of fancy indexing (using [block] directly) avoids a copy
                start_index = block[0]
                data_view = self.receptor_array.data[start_index: start_index + nb_items]

                glBufferSubData(GL_SHADER_STORAGE_BUFFER, start_index * item_size, data_view.nbytes, data_view)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        self.receptor_array.dirty_mask.fill(False)

    @property
    def eye_model_shader(self):
        if self._eye_model_shader is None:
            self._eye_model_shader = ShaderProgram(vert_path="shaders/eye_model.vert", frag_path="shaders/eye_model.frag")
        return self._eye_model_shader

    @property
    def voronoi_shader(self):
        if self._voronoi_shader is None:
            self._voronoi_shader = ShaderProgram(vert_path='shaders/voronoi.vert', frag_path='shaders/voronoi.frag')
        return self._voronoi_shader

    @property
    def heatmap_shader(self):
        if self._heatmap_shader is None:
            self._heatmap_shader = ShaderProgram(
                vert_path='shaders/voronoi_heatmap.vert',
                frag_path='shaders/voronoi_heatmap.frag',
            )
        return self._heatmap_shader

    # TODO: These two methods might be replaced / reworked. They will do for now
    def set_heatmap_data(
        self,
        values: np.ndarray,
        range: tuple = None,
        colormap: 'Colormap' = Colormap.THERMAL,
        compression: float = 0.5,
    ):
        """
        Upload per-ommatidium scalar data for heatmap visualisation.

        Args:
            values: (num_ommatidia,) float array in global array order.
                         For per-eye data, use set_heatmap_eyes() instead.
            range: (min, max) bounds for the colourmap.
                        None = auto-normalise using a rolling window over recent
                        frames, which adapts smoothly without manual tuning.
            colormap: Which colourmap to use (Colormap enum).
            compression: Power exponent for dynamic range compression.
                         1.0 = linear, 0.5 = sqrt, lower = brings out detail
        """

        buf = np.ascontiguousarray(values, dtype=np.float32).ravel()

        if len(buf) != self.total_receptors:
            raise ValueError(
                f"scalar_data has {len(buf)} elements, expected {self.total_receptors}."
            )

        # (Re)allocate SSBO if needed
        if self._heatmap_ssbo == 0:
            self._heatmap_ssbo = glGenBuffers(1)

        if self._heatmap_ssbo_capacity != self.total_receptors:
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._heatmap_ssbo)
            glBufferData(GL_SHADER_STORAGE_BUFFER, self.total_receptors * 4, None, GL_DYNAMIC_DRAW)
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
            self._heatmap_ssbo_capacity = self.total_receptors

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self._heatmap_ssbo)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, buf.nbytes, buf)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        # Range
        if range is not None:
            self._heatmap_range = (float(range[0]), float(range[1]))
        else:
            # Auto-normalise: track per-frame peak absolute value
            frame_peak = float(np.percentile(np.abs(buf), self._heatmap_auto_range_percentile))

            # Exponential Moving Average for colormap scaling
            if not hasattr(self, '_heatmap_current_range'):
                self._heatmap_current_range = frame_peak
            else:
                if frame_peak > self._heatmap_current_range:
                    # Rise fast to sudden large motion
                    self._heatmap_current_range = 0.5 * self._heatmap_current_range + 0.5 * frame_peak
                else:
                    # Decay slowly
                    self._heatmap_current_range = 0.98 * self._heatmap_current_range + 0.02 * frame_peak

            range_bound = max(self._heatmap_current_range, 1e-6)

            if colormap == Colormap.DIVERGING:
                self._heatmap_range = (-range_bound, range_bound)
            else:
                self._heatmap_range = (0.0, range_bound)

        self._heatmap_colormap = colormap
        self._heatmap_compression = compression

    def set_heatmap_eyes(
        self,
        values_dict: dict,
        range: tuple = None,
        colormap: 'Colormap' = Colormap.THERMAL,
        compression: float = 0.5,
    ):
        """
        Convenience: merge per-eye scalar arrays and upload.

        Args:
            values_dict: dict mapping Eye -> array of per-ommatidium scalars.
                      e.g. {left_eye: left_motion, right_eye: right_motion}
            range: (min, max) for the colourmap. Auto if None.
            colormap: Which colourmap to use.
            compression: Power exponent (see set_heatmap_data).
        """

        merged = np.zeros(self.total_receptors, dtype=np.float32)

        for eye, data in values_dict.items():
            merged[eye.global_indices] = data

        self.set_heatmap_data(merged, range=range, colormap=colormap, compression=compression)

    @property
    def cones_vao(self):

        if self._cones_vao is None or self._cones_vbo is None:
            self._nb_cone_vertices = len(CONE_VERTICES) // 3
            self._cones_vao = glGenVertexArrays(1)
            glBindVertexArray(self._cones_vao)

            self._cones_vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self._cones_vbo)
            glBufferData(GL_ARRAY_BUFFER, CONE_VERTICES.nbytes, CONE_VERTICES, GL_STATIC_DRAW)

            # hardcoded at location 0 so it works for both shaders (first and third person)
            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))

            glBindVertexArray(0)

        return self._cones_vao

    @property
    def hemispheres_vao(self):
        if self._hemispheres_vao is None:
            self._nb_hemisphere_vertices = len(SPHERE_VERTICES) // 3
            self._hemispheres_vao = glGenVertexArrays(1)
            glBindVertexArray(self._hemispheres_vao)

            self._hemispheres_vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, self._hemispheres_vbo)
            glBufferData(GL_ARRAY_BUFFER, SPHERE_VERTICES.nbytes, SPHERE_VERTICES, GL_STATIC_DRAW)

            glEnableVertexAttribArray(0)
            glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
            glBindVertexArray(0)
        return self._hemispheres_vao

    def _draw_voronoi(self):
        """Draws the compound-eye view using final ommatidia colours."""

        if self.heatmap_enabled and self._heatmap_ssbo != 0:
            self._draw_heatmap()
            return

        self.voronoi_shader.use()

        glEnable(GL_DEPTH_TEST)

        # Get current viewport dimensions to calculate aspect ratio
        viewport = glGetIntegerv(GL_VIEWPORT)
        # avoid division by zero if window is not yet setup
        aspect_ratio = viewport[2] / viewport[3] if viewport[3] > 0 else 1.0

        # glUniform1f(self.voronoi_shader.get_loc('aspect_ratio'), 1.0)
        glUniform1f(self.voronoi_shader.get_loc('aspect_ratio'), aspect_ratio)
        glUniform1i(self.voronoi_shader.get_loc('tiled_mode'), self.tiled_mode)
        glUniform1i(self.voronoi_shader.get_loc('projection_mode'), self.projection_mode)
        glUniform1f(self.voronoi_shader.get_loc('receptive_field_scale'), self.receptive_field_scale)

        # Binding 0: Ommatidia geometry (directions, origins, etc)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.receptors_input_ssbo)
        # Binding 1: Final computed colors (from subclass computation)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.final_colors_ssbo)

        glBindVertexArray(self.cones_vao)
        glDrawArraysInstanced(GL_TRIANGLES, 0, self._nb_cone_vertices, self.total_receptors)

        # Unbind everyone
        glBindVertexArray(0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0)
        glDisable(GL_DEPTH_TEST)

        self.voronoi_shader.stop()

    def _draw_heatmap(self):
        """Draws the compound-eye view with scalar data mapped to a colormap."""

        shader = self.heatmap_shader
        shader.use()

        glEnable(GL_DEPTH_TEST)

        viewport = glGetIntegerv(GL_VIEWPORT)
        aspect_ratio = viewport[2] / viewport[3] if viewport[3] > 0 else 1.0

        glUniform1f(shader.get_loc('aspect_ratio'), aspect_ratio)
        glUniform1i(shader.get_loc('tiled_mode'), self.tiled_mode)
        glUniform1i(shader.get_loc('projection_mode'), self.projection_mode)
        glUniform1f(shader.get_loc('receptive_field_scale'), self.receptive_field_scale)

        glUniform1f(shader.get_loc('data_min'), self._heatmap_range[0])
        glUniform1f(shader.get_loc('data_max'), self._heatmap_range[1])
        glUniform1i(shader.get_loc('colormap'), int(self._heatmap_colormap))
        glUniform1f(shader.get_loc('compression'), self._heatmap_compression)

        # Binding 0: Ommatidia geometry (for positioning)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.receptors_input_ssbo)
        # Binding 1: Scalar data (replaces the colour buffer)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self._heatmap_ssbo)

        glBindVertexArray(self.cones_vao)
        glDrawArraysInstanced(GL_TRIANGLES, 0, self._nb_cone_vertices, self.total_receptors)

        glBindVertexArray(0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0)
        glDisable(GL_DEPTH_TEST)

        shader.stop()

    def _draw_eye_model(self, observer_camera, agent):

        self.eye_model_shader.use()

        glEnable(GL_BLEND)
        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # glDisable(GL_CULL_FACE)

        # TODO: Why is this conversion to numpy necessary??
        view_matrix_np = np.array(observer_camera.view, dtype=np.float32)
        projection_matrix_np = np.array(observer_camera.projection, dtype=np.float32)

        # This one works fine
        c2w_mat = glm.inverse(agent.view)

        glUniformMatrix4fv(self.eye_model_shader.get_loc('view'), 1, True, view_matrix_np)
        glUniformMatrix4fv(self.eye_model_shader.get_loc('projection'), 1, True, projection_matrix_np)
        glUniformMatrix4fv(self.eye_model_shader.get_loc('eye_to_world'), 1, False, glm.value_ptr(c2w_mat))

        # Testing something with light

        # sun-like light source high up (+Y), and slightly to the right (+X) and back (+Z)
        # light_dir = glm.normalize(glm.vec3(0.5, 1.0, 0.4))
        # glUniform3fv(self.eye_model_shader.get_loc('light_dir'), 1, glm.value_ptr(light_dir))

        # Compute nice-looking cone length for acceptance angles
        avg_radius = np.mean(np.linalg.norm(self.receptor_array.data['origin'][:, :3], axis=1))
        if avg_radius < 1e-6:
            avg_radius = 0.01

        cone_length_factor = 10.0
        cone_length = avg_radius * cone_length_factor

        visualisation_scale = 1.0

        glUniform1i(self.eye_model_shader.get_loc('projection_mode'), self.projection_mode)
        glUniform1f(self.eye_model_shader.get_loc("cone_length"), cone_length)
        glUniform1f(self.eye_model_shader.get_loc("visualisation_scale"), visualisation_scale)

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.receptors_input_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.final_colors_ssbo)

        if self.projection_mode == ProjectionMode.Physical:
            # Physical layout mode: hemispheres to avoid Z-fighting
            glBindVertexArray(self.hemispheres_vao)
            glDrawArraysInstanced(GL_TRIANGLES, 0, self._nb_hemisphere_vertices, self.total_receptors)
        else:
            # Acceptance angle mode: cones
            glBindVertexArray(self.cones_vao)
            glDrawArraysInstanced(GL_TRIANGLES, 0, self._nb_cone_vertices, self.total_receptors)

        # Unbind everyone
        glBindVertexArray(0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0)

        # Restore state
        glEnable(GL_CULL_FACE)
        glDisable(GL_BLEND)
        glDisable(GL_DEPTH_TEST)

        self.eye_model_shader.stop()

    def free(self):
        """Free GPU resources."""

        if self.receptors_input_ssbo:
            glDeleteBuffers(1, [self.receptors_input_ssbo])

        if self.history_ssbo:
            glDeleteBuffers(1, [self.history_ssbo])

        if self.receptor_state_ssbo:
            glDeleteBuffers(1, [self.receptor_state_ssbo])

        if self._voronoi_shader:
            self._voronoi_shader.free()

        if self._heatmap_shader:
            self._heatmap_shader.free()

        if self._eye_model_shader:
            self._eye_model_shader.free()

        if self._cones_vao:
            glDeleteVertexArrays(1, [self._cones_vao])
        if self._cones_vbo:
            glDeleteBuffers(1, [self._cones_vbo])

        if self._hemispheres_vao:
            glDeleteVertexArrays(1, [self._hemispheres_vao])
        if self._hemispheres_vbo:
            glDeleteBuffers(1, [self._hemispheres_vbo])

        if self._heatmap_ssbo:
            glDeleteBuffers(1, [self._heatmap_ssbo])


class TextureViewer:
    """A simple helper to render a 2D texture to a full-screen quad."""

    def __init__(self):
        self._shader = None
        self._vao = None

    @property
    def shader(self):
        if self._shader is None:
            print("Compiling fullscreen texture viewer shaders...")
            self._shader = ShaderProgram(vert_path='shaders/fullscreen.vert', frag_path='shaders/textureSampler.frag')
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
