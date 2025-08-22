import OpenGL
OpenGL.ERROR_CHECKING = False

from OpenGL.GL import *
from graphics.utils import ShaderProgram


class PanoramicEye:
    """ A simple asset to render a cubemap to the screen as a panoramic (equirectangular) view """

    def __init__(self):
        self._shader = None
        self._vao = None

    @property
    def shader(self):
        if self._shader is None:
            print("Compiling panoramic debug shaders...")
            self._shader = ShaderProgram(vert_path='shaders/panoramic.vert', frag_path='shaders/panoramic.frag')
        return self._shader

    @property
    def vao(self):
        if self._vao is None:
            # A dummy VAO is sufficient as vertices are generated in the vertex shader
            self._vao = glGenVertexArrays(1)
        return self._vao

    def draw(self, cubemap_texture_id):
        """ Draws the panoramic view of the given cubemap """
        self.shader.use()

        # Bind the cubemap texture we want to inspect
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap_texture_id)
        glUniform1i(self.shader.get_loc("u_cubemap"), 0)

        # Draw a full-screen triangle
        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)

        # Unbind
        glBindVertexArray(0)
        self.shader.stop()

    def free(self):
        """ Frees the GPU resources (shader and VAO) """
        if self._shader:
            self._shader.free()
        if self._vao:
            glDeleteVertexArrays(1, [self._vao])
        self._shader = None
        self._vao = None
