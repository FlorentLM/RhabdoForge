from OpenGL.GL import *
from graphics.utils import load_shaders


class PanoramicEye:
    """ A simple asset to render a cubemap to the screen as a panoramic (equirectangular) view """

    def __init__(self):
        self._program = None
        self._vao = None

    @property
    def program(self):
        if self._program is None:
            print("Compiling panoramic debug shaders...")
            self._program = load_shaders('shaders/panoramic.vert',
                                         'shaders/panoramic.frag')
        return self._program

    @property
    def vao(self):
        if self._vao is None:
            # A dummy VAO is sufficient as vertices are generated in the vertex shader
            self._vao = glGenVertexArrays(1)
        return self._vao

    def draw(self, cubemap_texture_id):
        """ Draws the panoramic view of the given cubemap """
        glUseProgram(self.program)

        # Bind the cubemap texture we want to inspect
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap_texture_id)
        glUniform1i(glGetUniformLocation(self.program, "u_cubemap"), 0)

        # Draw a full-screen triangle
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)

        # Unbind
        glBindVertexArray(0)
        glUseProgram(0)