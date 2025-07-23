import time
from pathlib import Path

import numpy as np
from OpenGL.GL import *

from geometry.primitives import CONE_VERTICES
from graphics.utils import load_shaders, load_compute_shader, DTYPE
from graphics.ommatidia_funcs import ommatidia_builder


class InsectEye:
    def __init__(self, file_path=None, num_ommatidia=1962, acceptance_angle_deg=None, time_dithering=True):

        if file_path is not None:

            loaded_data = np.load(Path(file_path))

            # TODO: Write proper parsers for the different formats available out there...

            # Data matrix should have shape (N, 3+) with columns:
            # [azimuth_rad, elevation_rad, acceptance_angle_rad, ...]

            self.num_ommatidia = loaded_data.shape[0]

            # Convert spherical coordinates (azimuth/longitude, elevation/latitude) to 3D direction vectors
            om_lons = loaded_data[:, 0]
            om_lats = loaded_data[:, 1]
            om_dirs = np.zeros((self.num_ommatidia, 3), dtype=DTYPE)
            om_dirs[:, 0] = np.cos(om_lats) * np.sin(om_lons)  # x
            om_dirs[:, 1] = np.sin(om_lats)                    # y
            om_dirs[:, 2] = -np.cos(om_lats) * np.cos(om_lons) # z (assuming standard panoramic math)

            # Extract acceptance angles
            acceptance_angles_rad = loaded_data[:, 2]

        else:
            # Use your existing ommatidia_builder for uniform eyes
            om_dirs, om_lons, om_lats = ommatidia_builder(ommatidia=num_ommatidia)
            self.num_ommatidia = om_dirs.shape[0]

            if acceptance_angle_deg is None:
                # TODO: ommatidia builder should be able to do this by itself

                print("Acceptance angle not provided. Calculating a default based on interommatidial angle.")
                # Estimate interommatidial angle from the first two ommatidia (for a uniform eye)
                v1 = om_dirs[0]
                v2 = om_dirs[1]
                interommatidial_angle_rad = np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0))

                # Set the acceptance angle to this calculated value
                acceptance_angle_rad = interommatidial_angle_rad
                print(f"  -> Default acceptance angle set to: {np.rad2deg(acceptance_angle_rad):.2f} degrees")
            else:
                # If provided, use the specified value
                acceptance_angle_rad = np.deg2rad(acceptance_angle_deg)

            # Create a uniform array of acceptance angles
            acceptance_angles_rad = np.full(self.num_ommatidia, acceptance_angle_rad, dtype=DTYPE)

        # Pack all the ommatidia data into a big array
        # Shape is (num_ommatidia, 5) -> [dir_x, dir_y, dir_z, acceptance_angle, padding]
        # Note: vec3 is 12 bytes. float is 4 bytes. Total 16 bytes, which conveniently aligns to vec4 :)
        self.ommatidia_input_data = np.zeros((self.num_ommatidia, 4), dtype=DTYPE)
        self.ommatidia_input_data[:, :3] = om_dirs
        self.ommatidia_input_data[:, 3] = acceptance_angles_rad

        # Number of rays to sample per ommatidium
        self._samples_per_ommatidium = 1024

        # A counter for time dithering during sampling
        self._time_dithering = time_dithering
        self._time_counter = 0

        # Program and VAO for visualisation
        self._voronoi_program = None
        self._voronoi_vao = None
        self._cone_vertex_count = 0

        # Visualisation resources (lazy-loaded)
        self._vis_program = None
        self._vis_vao = None

        # This is the main compute shader that samples for all ommatidia
        self.ommatidia_program = load_compute_shader('shaders/ommatidia.comp')

        # Buffer for reading the ommatidia data back to CPU
        self.cpu_ommatidia_buf = np.zeros((self.num_ommatidia, 4), dtype=DTYPE)

        # Input SSBO for sending ommatidia data to the compute shader
        self.input_ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.input_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.ommatidia_input_data.nbytes, self.ommatidia_input_data, GL_STATIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)  # unbind

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
        self._samples_per_ommatidium = int(min(32768, max(1, value)))

    @property
    def voronoi_program(self):
        if self._voronoi_program is None:
            print("Compiling Voronoi visualization shaders...")
            self._voronoi_program = load_shaders('shaders/voronoi.vert',
                                                 'shaders/voronoi.frag')
        return self._voronoi_program

    @property
    def voronoi_vao(self):
        if self._voronoi_vao is None:
            self._cone_vertex_count = len(CONE_VERTICES) // 3
            vao = glGenVertexArrays(1)
            glBindVertexArray(vao)

            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, CONE_VERTICES.nbytes, CONE_VERTICES, GL_STATIC_DRAW)

            pos_loc = glGetAttribLocation(self.voronoi_program, "a_cone_vertex_pos")
            glEnableVertexAttribArray(pos_loc)
            glVertexAttribPointer(pos_loc, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

            glBindVertexArray(0)
            self._voronoi_vao = vao
        return self._voronoi_vao

    def get_ommatidia_data(self, cubemap_texture_id):
        """ Computes ommatidia data and returns it as a numpy array """

        glUseProgram(self.ommatidia_program)

        # Set uniforms for the data pass
        glUniform1i(glGetUniformLocation(self.ommatidia_program, 'u_num_ommatidia'), self.num_ommatidia)
        glUniform1i(glGetUniformLocation(self.ommatidia_program, 'u_samples_per_ommatidium'), self.samples_per_ommatidium)
        glUniform1f(glGetUniformLocation(self.ommatidia_program, 'u_time'), self._time_counter * 0.01)

        # Bind input cubemap (texture unit 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap_texture_id)
        glUniform1i(glGetUniformLocation(self.ommatidia_program, 'u_scene_cubemap'), 0)

        # Bind directions SSBO to binding point 0 (for reading)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.input_ssbo)
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

        # And update the counter for time dithering
        if self._time_dithering:
            self._time_counter += 1

        return self.cpu_ommatidia_buf

    @property
    def shaders(self):
        return self.voronoi_program

    @property
    def vao(self):
        return self.voronoi_vao

    def draw(self, tiled_mode=False):

        glUseProgram(self.voronoi_program)
        glEnable(GL_DEPTH_TEST)

        glUniform1i(glGetUniformLocation(self.voronoi_program, "u_tiled_mode"), tiled_mode)

        # Bind the SSBOs containing per-ommatidium data
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.input_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.colors_ssbo)

        # Bind the cone's VAO
        glBindVertexArray(self.voronoi_vao)

        # Make the instanced draw call
        glDrawArraysInstanced(GL_TRIANGLES, 0, self._cone_vertex_count, self.num_ommatidia)

        # Unbind everyone
        glBindVertexArray(0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0)
        glDisable(GL_DEPTH_TEST)
        glUseProgram(0)