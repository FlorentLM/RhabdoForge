from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np
import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *
from geometry.compound_eyes import CompoundEye
from geometry.primitives import CONE_VERTICES
from graphics.utils import ShaderProgram


class EyeRendererBase(ABC):
    """
    Abstract base class for an insect eye model, handling visualization and common properties
    """
    def __init__(self, eye_model: CompoundEye, time_dithering: bool = True, nb_samples: int = 256, window_size: Tuple[int, int] = (1280, 720)):
        self.model = eye_model
        self.num_ommatidia = self.model.num_ommatidia

        self.w, self.h = window_size

        self.ommatidia_input_data = self.model.pack()

        # Default umber of rays to sample per ommatidium
        self._samples_per_ommatidium = nb_samples

        # A counter for time dithering during sampling
        self._time_dithering = time_dithering
        self._time_counter = 0

        # Visualization resources (lazy-loaded)
        self._voronoi_shader = None
        self._voronoi_vao = None
        self._cone_vertex_count = 0

        # A small fixed scale for the receptive field view
        self.receptive_field_scale = 1.0 / (2.0 * np.pi)

        # Dynamic scale for the Voronoi view (needs to fill the quad)
        self.voronoi_scale = self.model.max_gap() * 2.5

        # Query maximum possible size for an SSBO on current GPU
        self._max_ssbo_size_bytes = glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE)
        print(f"Max SSBO size: {self._max_ssbo_size_bytes / (1024 * 1024):.2f} MB")

        # SSBO for input ommatidia geometry (directions, angles, etc)
        self.input_om_ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.input_om_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.ommatidia_input_data.nbytes, self.ommatidia_input_data, GL_STATIC_DRAW)

        # Size of the output buffers in bytes (num_ommatidia * 4 floats * 4 bytes/float)
        buffer_size = self.num_ommatidia * 16

        # Final computed colors (written by subclass, read by draw())
        self.final_colors_ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.final_colors_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, buffer_size, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        # Two PBOs for ping-ponging and doing async reading from the CPU-side
        self.pbo_ids = glGenBuffers(2)
        self.pbo_index = 0

        glBindBuffer(GL_PIXEL_PACK_BUFFER, self.pbo_ids[0])
        glBufferData(GL_PIXEL_PACK_BUFFER, buffer_size, None, GL_STREAM_READ)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, self.pbo_ids[1])
        glBufferData(GL_PIXEL_PACK_BUFFER, buffer_size, None, GL_STREAM_READ)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)

        # CPU-side buffer to return the final data
        self.cpu_read_buffer = np.zeros((self.num_ommatidia, 4), dtype=np.float32)

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

    @abstractmethod
    def _compute_colors(self, *args, **kwargs):
        # Each subclass implements its own core rendering logic
        raise NotImplementedError

    @abstractmethod
    def get_ommatidia_data(self, *args, to_cpu=False, **kwargs):
        # Each subclass implements its own core rendering logic
        raise NotImplementedError

    def _fetch_to_cpu(self):

        # Determine which PBO to read from (current) and which to write to (next)
        current_pbo_idx = self.pbo_index
        next_pbo_idx = (self.pbo_index + 1) % 2

        # This is a GPU-to-GPU copy, so it is asynchronous (the command returns immediately).
        # it initiates the copy from the SSBO to the *next* PBO
        glBindBuffer(GL_COPY_READ_BUFFER, self.final_colors_ssbo)
        glBindBuffer(GL_COPY_WRITE_BUFFER, self.pbo_ids[next_pbo_idx])
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, self.cpu_read_buffer.nbytes)

        # Process data from the *current* PBO (it was filled in the previous frame)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, self.pbo_ids[current_pbo_idx])

        # Map the buffer ('GL_MAP_READ_BIT' is very important!)
        ptr = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, self.cpu_read_buffer.nbytes, GL_MAP_READ_BIT)

        if ptr:
            # Copy the data from the mapped GPU memory to our CPU-side numpy array
            ctypes.memmove(self.cpu_read_buffer.ctypes.data, ptr, self.cpu_read_buffer.nbytes)
            # IMPORTANT: Unmap the buffer to return control to the GPU
            glUnmapBuffer(GL_PIXEL_PACK_BUFFER)
        else:
            # Handle error if mapping fails
            print("Warning: Failed to map PBO for reading.")

        # Unbind all buffers used in the copy and map operations
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        glBindBuffer(GL_COPY_READ_BUFFER, 0)
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0)

        # Swap PBO index for the next frame
        self.pbo_index = next_pbo_idx

    @property
    def voronoi_shader(self):
        if self._voronoi_shader is None:
            self._voronoi_shader = ShaderProgram(vert_path='shaders/voronoi.vert', frag_path='shaders/voronoi.frag')
        return self._voronoi_shader

    @property
    def voronoi_vao(self):
        if self._voronoi_vao is None:
            self._cone_vertex_count = len(CONE_VERTICES) // 3
            vao = glGenVertexArrays(1)
            glBindVertexArray(vao)

            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, CONE_VERTICES.nbytes, CONE_VERTICES, GL_STATIC_DRAW)

            pos_loc = glGetAttribLocation(self.voronoi_shader.program_id, "a_cone_vertex_pos")
            glEnableVertexAttribArray(pos_loc)
            glVertexAttribPointer(pos_loc, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

            glBindVertexArray(0)
            self._voronoi_vao = vao

        return self._voronoi_vao

    def draw(self, tiled_mode=False):
        """ Draws the Voronoi visualization using the computed colors """

        self.voronoi_shader.use()
        glEnable(GL_DEPTH_TEST)

        cone_scale = self.voronoi_scale if tiled_mode else self.receptive_field_scale

        # Get current viewport dimensions to calculate aspect ratio
        viewport = glGetIntegerv(GL_VIEWPORT)
        # avoid division by zero if window is not yet setup
        aspect_ratio = viewport[2] / viewport[3] if viewport[3] > 0 else 1.0

        glUniform1f(self.voronoi_shader.get_loc('u_aspect_ratio'), 1.0)
        # glUniform1f(self.voronoi_shader.get_loc('u_aspect_ratio'), aspect_ratio)
        glUniform1i(self.voronoi_shader.get_loc('u_tiled_mode'), tiled_mode)
        glUniform1f(self.voronoi_shader.get_loc('u_cone_scale'), cone_scale)

        # Binding 0: Ommatidia geometry (directions, origins, etc)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.input_om_ssbo)
        # Binding 1: Final computed colors (from subclass computation)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.final_colors_ssbo)

        glBindVertexArray(self.voronoi_vao)
        glDrawArraysInstanced(GL_TRIANGLES, 0, self._cone_vertex_count, self.num_ommatidia)

        # Unbind everyone
        glBindVertexArray(0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0)
        glDisable(GL_DEPTH_TEST)
        self.voronoi_shader.stop()

    def free(self):
        """ Free GPU resources """

        glDeleteBuffers(4, [self.input_om_ssbo, self.final_colors_ssbo, self.pbo_ids[0], self.pbo_ids[1]])

        if self._voronoi_shader:
            self._voronoi_shader.free()

        if self._voronoi_vao:
            # TODO: also track the associated VBO handle to delete it here
            glDeleteVertexArrays(1, [self._voronoi_vao])
