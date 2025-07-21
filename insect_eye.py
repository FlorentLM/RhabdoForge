import numpy as np
from OpenGL.GL import *
from engine import load_shaders, load_compute_shader
from ommatidia_funcs import ommatidia_builder
from fbo import DataFBO


class InsectEyeAsset:
    def __init__(self, num_ommatidia=500, acceptance_angle_deg=5.0):

        om_dirs, om_lons, om_lats = ommatidia_builder(ommatidia=num_ommatidia)

        self.num_ommatidia = om_dirs.shape[0]
        self.acceptance_angle = np.deg2rad(acceptance_angle_deg)

        self.ommatidia_dirs = np.zeros((self.num_ommatidia, 4), dtype=np.float32)
        self.ommatidia_dirs[:, :3] = om_dirs

        # VBO data for panoramic visualization
        self.vis_vertex_data = np.zeros((self.num_ommatidia, 2), dtype=np.float32)
        self.vis_vertex_data[:, 0] = om_lons  # Longitude
        self.vis_vertex_data[:, 1] = om_lats  # Latitude

        # Visualisation resources (lazy-loaded)
        self._vis_program = None
        self._vis_vao = None

        # Data-gathering resources
        self._data_program = None   # also lazy-loaded
        self.data_fbo = DataFBO(self.num_ommatidia)

        # CPU buffer for holding the data
        self.cpu_data_buffer = np.zeros((self.num_ommatidia, 4), dtype=np.float32)

        # Data Buffers
        # SSBO for sending directions to the data pass shader
        self.directions_ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.directions_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.ommatidia_dirs.nbytes, self.ommatidia_dirs, GL_STATIC_DRAW)

        # SSBO for sending colors to the visualization shader
        self.colors_ssbo = glGenBuffers(1)
        # Bind to GL_UNIFORM_BUFFER here just to allocate the memory
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.colors_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.num_ommatidia * 16, None, GL_DYNAMIC_DRAW)
        # and unbind
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

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
    def data_program(self):
        if self._data_program is None:
            print("Compiling data shader...")
            self._data_program = load_compute_shader('shaders/data_pass.comp')
        return self._data_program

    @property
    def vis_vao(self):
        if self._vis_vao is None:
            vao = glGenVertexArrays(1)
            glBindVertexArray(vao)

            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, self.vis_vertex_data.nbytes, self.vis_vertex_data, GL_STATIC_DRAW)

            pano_loc = glGetAttribLocation(self.vis_program, "a_ommatidia_coords")
            glEnableVertexAttribArray(pano_loc)

            glVertexAttribPointer(pano_loc, 2, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

            glBindVertexArray(0)
            self._vis_vao = vao
        return self._vis_vao

    def get_ommatidia_data(self, cubemap_texture_id):
        """ Computes ommatidia data and returns it as a numpy array """

        glUseProgram(self.data_program)

        # Set uniforms for the data pass
        glUniform1f(glGetUniformLocation(self.data_program, "u_acceptance_angle"), self.acceptance_angle)
        glUniform1i(glGetUniformLocation(self.data_program, "u_num_ommatidia"), self.num_ommatidia)

        # Bind input cubemap (texture unit 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap_texture_id)
        glUniform1i(glGetUniformLocation(self.data_program, "u_scene_cubemap"), 0)

        # Bind input ommatidia directions (SSBO binding point 0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.directions_ssbo)

        # Bind output image texture (image unit 0)
        glBindImageTexture(0, self.data_fbo.data_texture_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F)

        # Dispatch the compute shader
        # Divide the total number of ommatidia by the workgroup size (64)
        work_groups_x = (self.num_ommatidia + 63) // 64
        glDispatchCompute(work_groups_x, 1, 1)

        # Block until the compute shader has finished writing to the image
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        # Unbind resources
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0)
        glBindImageTexture(0, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F)
        glUseProgram(0)

        # Read the data back to the CPU
        self.data_fbo.read_data(self.cpu_data_buffer)

        return self.cpu_data_buffer

    @property
    def shaders(self):
        return self.vis_program

    @property
    def vao(self):
        return self.vis_vao

    def draw(self, data):
        """ Draws the visualization as a panoramic equirectangular projection """

        glUseProgram(self.vis_program)

        # Update and bind the colors SSBO
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.colors_ssbo)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, data.nbytes, data)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.colors_ssbo)

        # Bind the VAO and draw the points
        # The vertex shader will handle the panoramic projection
        glBindVertexArray(self.vis_vao)
        glDrawArrays(GL_POINTS, 0, self.num_ommatidia)

        # Unbind everything
        glBindVertexArray(0)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        glUseProgram(0)