import numpy as np
from OpenGL.GL import *
from engine import load_shaders
from ommatidia_funcs import ommatidia_builder
from fbo import DataFBO


class InsectEyeAsset:
    def __init__(self, num_ommatidia=500, acceptance_angle_deg=5.0):

        om_dirs, om_lats, om_lons = ommatidia_builder(ommatidia=num_ommatidia)

        self.num_ommatidia = om_dirs.shape[0]
        self.acceptance_angle = np.deg2rad(acceptance_angle_deg)

        self.ommatidia_dirs = np.zeros((self.num_ommatidia, 4), dtype=np.float32)
        self.ommatidia_dirs[:, :3] = om_dirs

        # Visualisation resources (lazy-loaded)
        self._vis_program = None
        self._vis_vao = None

        # Data-gathering resources
        self._data_program = None   # also lazy-loaded
        self.data_fbo = DataFBO(self.num_ommatidia)
        self.data_quad_vao = glGenVertexArrays(1)  # A simple VAO for drawing a quad

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
        # The target used during allocation doesn't matter as much as the one used for binding,
        # but we still use the proper target from the start for consistency
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.colors_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.num_ommatidia * 16, None, GL_DYNAMIC_DRAW)

        # Unbind both
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        glBindBuffer(GL_UNIFORM_BUFFER, 0)

    @property
    def vis_program(self):
        if self._vis_program is None:
            print("Compiling visualization shaders...")

            self._vis_program = load_shaders(
                'shaders/insect_eye.vert',
                'shaders/insect_eye.frag',
                'shaders/insect_eye.geom'
            )

            # # Explicitly link the uniform block in the shader to binding point 1
            # block_index = glGetUniformBlockIndex(self._vis_program, "ColorDataBlock")
            # glUniformBlockBinding(self._vis_program, block_index, 1)

        return self._vis_program

    @property
    def data_program(self):
        if self._data_program is None:
            print("Compiling data shaders...")
            self._data_program = load_shaders('shaders/data_pass.vert',
                                              'shaders/data_pass.frag')

            # # Explicitly link the uniform block in the shader to binding point 0
            # block_index = glGetUniformBlockIndex(self._data_program, "OmmatidiaBlock")
            # glUniformBlockBinding(self._data_program, block_index, 0)

        return self._data_program

    @property
    def vis_vao(self):

        if self._vis_vao is None:
            vao = glGenVertexArrays(1)
            glBindVertexArray(vao)
            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, self.ommatidia_dirs.nbytes, self.ommatidia_dirs, GL_STATIC_DRAW)

            # Use the vis_program property to ensure it's compiled
            glUseProgram(self.vis_program)
            pos_loc = glGetAttribLocation(self.vis_program, "a_ommatidium_dir")
            glEnableVertexAttribArray(pos_loc)
            glVertexAttribPointer(pos_loc, 3, GL_FLOAT, GL_FALSE, 4 * self.ommatidia_dirs.itemsize, ctypes.c_void_p(0))

            glBindVertexArray(0)
            self._vis_vao = vao

        return self._vis_vao

    def get_ommatidia_data(self, cubemap_texture_id):
        """ Computes ommatidia data and returns it as a numpy array """

        self.data_fbo.bind()
        glUseProgram(self.data_program)

        # Set uniforms for the data pass
        glUniform1f(glGetUniformLocation(self.data_program, "u_acceptance_angle"), self.acceptance_angle)
        glUniform1i(glGetUniformLocation(self.data_program, "u_num_ommatidia"), self.num_ommatidia)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap_texture_id)
        glUniform1i(glGetUniformLocation(self.data_program, "u_scene_cubemap"), 0)

        # Bind directions UBO to binding point 0
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.directions_ssbo)

        glBindVertexArray(self.data_quad_vao)
        glDrawArrays(GL_TRIANGLE_STRIP, 0, 3)

        # Read the data back to the CPU
        self.data_fbo.read_data(self.cpu_data_buffer)

        self.data_fbo.unbind()
        glUseProgram(0)

        return self.cpu_data_buffer

    @property
    def shaders(self):
        return self.vis_program

    @property
    def vao(self):
        return self.vis_vao

    def draw(self, data):
        """ Draws the visualization using pre-calculated ommatidia data """

        glUseProgram(self.vis_program)

        # Set the visualization scale uniform required by the vertex shader
        glUniform1f(glGetUniformLocation(self.vis_program, "u_vis_scale"), 0.9)

        # Update the colors SSBO with the new data
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.colors_ssbo)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, data.nbytes, data)

        # Bind the colors SSBO to binding point 1
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.colors_ssbo)

        glBindVertexArray(self.vis_vao)
        glDrawArrays(GL_POINTS, 0, self.num_ommatidia)

        # Unbind everything
        glBindVertexArray(0)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)
        glUseProgram(0)