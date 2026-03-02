import OpenGL
OpenGL.ERROR_CHECKING = False

import random
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from pyglm import glm

from OpenGL.GL import *
from OpenGL.raw.GL.NVX.gpu_memory_info import GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX

from geometry.compound_eyes import CompoundEye
from geometry.primitives import CONE_VERTICES, SPHERE_VERTICES
from graphics.utils import ShaderProgram, ViewMode, ProjectionMode
from graphics.agent import Agent


def query_max_SSBO_size() -> int:
    max_size = glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE)
    print(f"[INFO] Max possible SSBO size: {max_size / (1024 * 1024):.2f} MB")
    return max_size


def query_available_VRAM() -> int:
    """ Checks for NVIDIA extension to query available VRAM. Returns 0 if not supported. """

    if b'GL_NVX_gpu_memory_info' in glGetString(GL_EXTENSIONS):
        # Value is in KB, convert to MB
        return glGetIntegerv(GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX) // 1024
    return 0


class BaseInsectEyeRenderer(ABC):
    """
    Abstract base class for an insect eye model, handling visualisation and common properties
    """

    def __init__(self, eye_model: CompoundEye, time_dithering: bool = True, nb_samples: int = 256, batch_size: int = 1):
        self.model = eye_model
        self.num_ommatidia = self.model.num_ommatidia
        self.ommatidia_input_data = self.model.data
        self._samples_per_ommatidium = nb_samples
        self._time_dithering = time_dithering
        self._time_counter = 0

        self.runs_interactive = False

        # Hardware queries
        self._max_ssbo_size_bytes = query_max_SSBO_size()

        # Input ommatidia SSBO
        self.input_om_ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.input_om_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.ommatidia_input_data.nbytes, self.ommatidia_input_data,
                     GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        # Visualization resources (lazy-loaded)
        self._voronoi_shader = None     # first person view
        self._eye_model_shader = None       # third person view

        # Cone-specific resources
        self._cones_vao = None
        self._cones_vbo = None
        self._nb_cone_vertices = 0

        # Hemisphere-specific resources
        self._hemispheres_vao = None
        self._hemispheres_vbo = None
        self._nb_hemisphere_vertices = 0

        self.receptive_field_scale = 1.0 / (2.0 * np.pi)

        # first-person specific stuff
        self.tiled_mode = False
        self.projection_mode: ProjectionMode = ProjectionMode.Physical

        # History buffer state
        self._batch_size = max(1, batch_size)
        self._current_frame_index = 0
        self.history_ssbo = None

        # PBOs for the synchronous path
        self.sync_pbo = None
        self.sync_cpu_buffer = None

        self._allocate_history_buffers(self._batch_size)

    @property
    def samples_per_ommatidium(self):
        return self._samples_per_ommatidium

    @samples_per_ommatidium.setter
    @abstractmethod
    def samples_per_ommatidium(self, value):
        # Subclasses may need to re-allocate buffers when this changes
        raise NotImplementedError

    @property
    def time_dithering(self):
        return self._time_dithering

    @time_dithering.setter
    def time_dithering(self, value: bool):
        self._time_dithering = bool(value)
        print(f"Time dithering {'ENABLED' if self._time_dithering else 'DISABLED'}.")

    def dither(self):
        self._time_counter = random.randint(0, 1024)

    @abstractmethod
    def _compute_colors(self, *args, **kwargs):
        # Each subclass implements its own core rendering logic
        raise NotImplementedError

    @abstractmethod
    def draw(self, view_mode: ViewMode, point_of_view: Agent, agent: Agent):
        # Each subclass implements its own core rendering logic
        raise NotImplementedError

    def _allocate_history_buffers(self, requested_frames: int):

        # Smart VRAM Allocation
        bytes_per_frame = self.num_ommatidia * 16  # 16 bytes per vec4 (RGBA float)
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
        total_buffer_size = self.num_ommatidia * 16 * self._batch_size

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
        self.sync_cpu_buffer = np.zeros((self.num_ommatidia, 4), dtype=np.float32)

    def estimate_vram_usage(self) -> float:
        """
        Returns an estimate of VRAM usage in MB, excluding the history buffer
        Subclasses should override this
        """
        return 100.0    # conservative guess

    def get_ommatidia_data(self, agent: Agent, readback: bool = True) -> Optional[np.ndarray]:
        """
        Runs one frame of simulation. Behavior is determined by the `batch_size`
        - If batch_size = 1: Blocks and returns the current frame's data
        - If batch_size > 1: Queues the frame on the GPU and returns None
        """

        is_sync_mode = getattr(self, 'is_interactive', False) or self._batch_size == 1

        if self._time_dithering:
            self._time_counter += 1

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        if is_sync_mode:
            # Synchronous path

            self._compute_colors(agent)

            if readback:
                glFinish()

                bytes_to_read = self.num_ommatidia * 16
                glBindBuffer(GL_COPY_READ_BUFFER, self.history_ssbo)
                glBindBuffer(GL_COPY_WRITE_BUFFER, self.sync_pbo)
                glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, bytes_to_read)

                glBindBuffer(GL_PIXEL_PACK_BUFFER, self.sync_pbo)

                ptr = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, bytes_to_read, GL_MAP_READ_BIT)
                ctypes.memmove(self.sync_cpu_buffer.ctypes.data, ptr, bytes_to_read)

                glUnmapBuffer(GL_PIXEL_PACK_BUFFER)
                glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)

                return self.sync_cpu_buffer.copy()

                # TODO: this would be true zero-copy. should expose it. (but the buffer needs to be unmapped manually after use of raw_array
                # raw_array = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_float)), shape=(self.num_ommatidia, 4))
                # return raw_array

            else:
                return None

        else:
            # Asynchronous path

            # Submit work for the current frame
            self._compute_colors(agent)
            self._current_frame_index += 1

            # Check if this frame just completed a batch
            if self._current_frame_index >= self._batch_size:
                # The buffer is full: block, download, and return the data
                print(f"  > GPU batch is full. Flushing {self._batch_size} frames...")

                return self.flush()

            # if the batch is not yet full, return None
            return None

        # TODO: add third 'streaming async' mode with dual PBOs (ping pong) and GL sync fences

    def flush(self) -> np.ndarray:
        """
        Blocks until all queued frames on the GPU are rendered, downloads the data, and resets the counter.
        This is used to retrieve a full batch, or the final partial batch at the end of a simulation.
        """

        if self._current_frame_index == 0:
            return np.array([])

        glFinish()  # Block until all rendering commands are complete

        num_frames_to_read = self._current_frame_index
        bytes_to_read = self.num_ommatidia * 16 * num_frames_to_read

        # For simplicity and robustness, a direct synchronous download is best here
        # PBOs are most effective when overlapping computation, which isn't happening during a final flush
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.history_ssbo)
        data_bytes = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, bytes_to_read)
        data_np = np.frombuffer(data_bytes, dtype=np.float32)

        # Reset the counter for the next batch
        self._current_frame_index = 0

        return data_np.reshape(num_frames_to_read, self.num_ommatidia, 4)

    def update(self, force_all=False):
        """
        Finds contiguous blocks of changed ommatidia and uploads each block in a single GPU call.
        """

        dirty_indices = np.where(self.model.dirty_mask)[0]
        if dirty_indices.size == 0:
            return

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.input_om_ssbo)

        if force_all:
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.model.data.nbytes, self.model.data)

        else:
            # Find contiguous blocks of updated indices
            jumps = np.where(np.diff(dirty_indices) != 1)[0] + 1
            contiguous_blocks = np.split(dirty_indices, jumps)

            item_size = self.model.data.itemsize  # 48 bytes

            for block in contiguous_blocks:
                nb_items = block.size
                if nb_items == 0:
                    continue

                # indexing like this instead of fancy indexing (using [block] directly) avoids a copy
                start_index = block[0]
                data_view = self.model.data[start_index: start_index + nb_items]

                glBufferSubData(GL_SHADER_STORAGE_BUFFER, start_index * item_size, data_view.nbytes, data_view)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        self.model.dirty_mask.fill(False)

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
        """ Draws the Voronoi visualization using the computed colors """

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
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.input_om_ssbo)
        # Binding 1: Final computed colors (from subclass computation)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.final_colors_ssbo)

        glBindVertexArray(self.cones_vao)
        glDrawArraysInstanced(GL_TRIANGLES, 0, self._nb_cone_vertices, self.num_ommatidia)

        # Unbind everyone
        glBindVertexArray(0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0)
        glDisable(GL_DEPTH_TEST)

        self.voronoi_shader.stop()

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
        avg_radius = np.mean(np.linalg.norm(self.model.data['origin'][:, :3], axis=1))
        if avg_radius < 1e-6:
            avg_radius = 0.01

        cone_length_factor = 10.0
        cone_length = avg_radius * cone_length_factor

        visualisation_scale = 1.0

        glUniform1i(self.eye_model_shader.get_loc('projection_mode'), self.projection_mode)
        glUniform1f(self.eye_model_shader.get_loc("cone_length"), cone_length)
        glUniform1f(self.eye_model_shader.get_loc("visualisation_scale"), visualisation_scale)

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.input_om_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.final_colors_ssbo)

        if self.projection_mode == ProjectionMode.Physical:
            # Physical layout mode: hemispheres to avoid Z-fighting
            glBindVertexArray(self.hemispheres_vao)
            glDrawArraysInstanced(GL_TRIANGLES, 0, self._nb_hemisphere_vertices, self.num_ommatidia)
        else:
            # Acceptance angle mode: cones
            glBindVertexArray(self.cones_vao)
            glDrawArraysInstanced(GL_TRIANGLES, 0, self._nb_cone_vertices, self.num_ommatidia)

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
        """ Free GPU resources """

        glDeleteBuffers(1, [self.input_om_ssbo])
        if self.history_ssbo:
            glDeleteBuffers(1, [self.history_ssbo])

        if self._voronoi_shader:
            self._voronoi_shader.free()

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