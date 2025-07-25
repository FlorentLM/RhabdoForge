from OpenGL.GL import *
from OpenGL.GL import glGenVertexArrays, glGetUniformLocation
from graphics.utils import load_shaders


class CubemapFBO:
    def __init__(self, resolution=256):
        self.resolution = resolution

        # Create Framebuffer object
        self.fbo_id = glGenFramebuffers(1)

        # Create the Color cubemap texture
        self.color_texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.color_texture_id)
        for i in range(6):
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGBA8,
                         self.resolution, self.resolution, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)

        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE)

        # Create the Depth Renderbuffer
        self.depth_buffer_id = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, self.depth_buffer_id)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, self.resolution, self.resolution)

        # Attach textures/buffers to the FBO
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)

        # Attach just one face initially to make the FBO 'complete'
        # The render loop will correctly attach the other faces as needed
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_CUBE_MAP_POSITIVE_X, self.color_texture_id, 0)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, self.depth_buffer_id)

        # Check if the FBO is complete
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            print(f"Framebuffer is not complete: {status}")

        # Unbind everything
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
        glBindRenderbuffer(GL_RENDERBUFFER, 0)
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def bind(self):
        glViewport(0, 0, self.resolution, self.resolution)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)

    def unbind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def free(self):
        glDeleteFramebuffers(1, [self.fbo_id])
        glDeleteTextures(1, [self.color_texture_id])
        glDeleteRenderbuffers(1, [self.depth_buffer_id])


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
