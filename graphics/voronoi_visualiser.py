from OpenGL.GL import *
from geometry.primitives import CONE_VERTICES
from graphics.utils import load_shaders


class VoronoiVisualiser:
    def __init__(self, num_ommatidia):

        self.num_ommatidia = num_ommatidia
        self._program = None
        self._vao = None
        self._cone_vertex_count = 0

    @property
    def program(self):
        if self._program is None:
            print("Compiling Voronoi visualization shaders...")
            self._program = load_shaders('shaders/voronoi.vert',
                                         'shaders/voronoi.frag')
        return self._program

    @property
    def vao(self):
        if self._vao is None:

            self._cone_vertex_count = len(CONE_VERTICES) // 3

            # Create and bind a VAO
            self._vao = glGenVertexArrays(1)
            glBindVertexArray(self._vao)

            # Create and bind a VBO for the cone vertices
            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, CONE_VERTICES.nbytes, CONE_VERTICES, GL_STATIC_DRAW)

            # Configure vertex attributes
            pos_loc = glGetAttribLocation(self.program, "a_cone_vertex_pos")
            glEnableVertexAttribArray(pos_loc)
            glVertexAttribPointer(pos_loc, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

            # No need for instanced attributes here, we get the per-instance data
            # from SSBOs in the shader using gl_InstanceID

            glBindVertexArray(0)
        return self._vao

    def draw(self, insect_eye_instance, tiled_mode=False):
        """ Draws the Voronoi diagram using data from the InsectEye instance """

        glUseProgram(self.program)
        glEnable(GL_DEPTH_TEST)

        # Pass the toggle to the shader
        glUniform1i(glGetUniformLocation(self.program, "u_tiled_mode"), tiled_mode)

        # Bind the SSBOs from the insect eye object
        # Binding point 0: Ommatidia data (positions, acceptance angles)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, insect_eye_instance.input_ssbo)
        # Binding point 1: Output color data
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, insect_eye_instance.colors_ssbo)

        # Bind the cone's VAO
        glBindVertexArray(self.vao)

        # Make the instanced draw call
        glDrawArraysInstanced(GL_TRIANGLES, 0, self._cone_vertex_count, self.num_ommatidia)

        # Unbind everyone
        glBindVertexArray(0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0)
        glDisable(GL_DEPTH_TEST)
        glUseProgram(0)