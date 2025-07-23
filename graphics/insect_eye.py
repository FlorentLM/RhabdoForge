import numpy as np
from OpenGL.GL import *

from graphics.utils import load_shaders, load_compute_shader, DTYPE
from graphics.ommatidia_funcs import ommatidia_builder


class InsectEye:
    def __init__(self, num_ommatidia=500, acceptance_angle_deg=5.0):

        om_dirs, om_lons, om_lats = ommatidia_builder(ommatidia=num_ommatidia)

        self.num_ommatidia = om_dirs.shape[0]
        self.acceptance_angle = np.deg2rad(acceptance_angle_deg)

        self.ommatidia_dirs = np.zeros((self.num_ommatidia, 4), dtype=DTYPE)
        self.ommatidia_dirs[:, :3] = om_dirs

        self._samples_per_ommatidium = 64

        # VBO data for panoramic visualization
        self.vis_vertex_data = np.zeros((self.num_ommatidia, 2), dtype=DTYPE)
        self.vis_vertex_data[:, 0] = om_lons  # Longitude
        self.vis_vertex_data[:, 1] = om_lats  # Latitude

        # Visualisation resources (lazy-loaded)
        self._vis_program = None
        self._vis_vao = None

        # This is the main compute shader that samples for all ommatidia
        self.ommatidia_program = load_compute_shader('shaders/ommatidia.comp')

        # Buffer for reading the ommatidia data back to CPU
        self.cpu_ommatidia_buf = np.zeros((self.num_ommatidia, 4), dtype=DTYPE)

        # Input SSBO for sending ommatidia directions to the compute shader
        self.directions_ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.directions_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.ommatidia_dirs.nbytes, self.ommatidia_dirs, GL_STATIC_DRAW)

        # TODO: Acceptance angle should be per-ommatidia too

        # Output/Input SSBO: Compute shader writes to it, visualization shader reads from it
        self.colors_ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.colors_ssbo)
        # Allocate memory for num_ommatidia * vec4 (16 bytes)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.num_ommatidia * 16, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)  # unbind

    @property
    def samples_per_ommatidium(self):
        return self._samples_per_ommatidium

    @samples_per_ommatidium.setter
    def samples_per_ommatidium(self, value):
        self._samples_per_ommatidium = int(min(4096, max(1, value)))

    @property
    def vis_program(self):
        if self._vis_program is None:
            print("Compiling visualization shaders...")
            self._vis_program = load_shaders(
                'shaders/insect_eye.vert',
                'shaders/insect_eye.frag',
                'shaders/insect_eye.geom'
            )
        return self._vis_program

    @property
    def vis_vao(self):
        if self._vis_vao is None:
            vao = glGenVertexArrays(1)
            glBindVertexArray(vao)

            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, self.vis_vertex_data.nbytes, self.vis_vertex_data, GL_STATIC_DRAW)

            pano_loc = glGetAttribLocation(self.vis_program, 'a_ommatidia_coords')
            glEnableVertexAttribArray(pano_loc)
            glVertexAttribPointer(pano_loc, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

            glBindVertexArray(0)
            self._vis_vao = vao
        return self._vis_vao

    def get_ommatidia_data(self, cubemap_texture_id):
        """ Computes ommatidia data and returns it as a numpy array """

        glUseProgram(self.ommatidia_program)

        # Set uniforms for the data pass
        glUniform1f(glGetUniformLocation(self.ommatidia_program, 'u_acceptance_angle'), self.acceptance_angle)
        glUniform1i(glGetUniformLocation(self.ommatidia_program, 'u_num_ommatidia'), self.num_ommatidia)
        glUniform1i(glGetUniformLocation(self.ommatidia_program, 'u_samples_per_ommatidium'), self.samples_per_ommatidium)

        # Bind input cubemap (texture unit 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap_texture_id)
        glUniform1i(glGetUniformLocation(self.ommatidia_program, 'u_scene_cubemap'), 0)

        # Bind directions SSBO to binding point 0 (for reading)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.directions_ssbo)
        # Bind colors SSBO to binding point 1 (for writing)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.colors_ssbo)

        # Dispatch the compute shader
        # Divide the total number of ommatidia by the workgroup size (64)
        work_groups_x = (self.num_ommatidia + 63) // 64
        glDispatchCompute(work_groups_x, 1, 1)

        # Wait for compute shader to finish writing to the SSBO
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        # Unbind resources
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0)
        glUseProgram(0)

        # Read the data from the GPU SSBO to the CPU buffer

        # Bind the buffer we want to read from
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.colors_ssbo)

        # Map the GPU memory into a CPU-accessible pointer
        ptr = glMapBufferRange(GL_SHADER_STORAGE_BUFFER, 0, self.cpu_ommatidia_buf.nbytes, GL_MAP_READ_BIT)

        # Copy the data from the mapped memory location to the numpy array's memory location
        ctypes.memmove(self.cpu_ommatidia_buf.ctypes.data, ptr, self.cpu_ommatidia_buf.nbytes)

        # Unmap the buffer which invalidates the pointer and returns memory control to the GPU
        glUnmapBuffer(GL_SHADER_STORAGE_BUFFER)

        # Unbind the buffer
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        return self.cpu_ommatidia_buf

    @property
    def shaders(self):
        return self.vis_program

    @property
    def vao(self):
        return self.vis_vao

    def draw(self):
        """ Draws the visualization as a panoramic equirectangular projection """

        glUseProgram(self.vis_program)

        # Data is already on the GPU, just need to bind the SSBO
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.colors_ssbo)

        # Bind the VAO and draw the points
        # The vertex shader will handle the panoramic projection
        glBindVertexArray(self.vis_vao)
        glDrawArrays(GL_POINTS, 0, self.num_ommatidia)

        # Unbind everything
        glBindVertexArray(0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0)
        glUseProgram(0)