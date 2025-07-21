from OpenGL.GL import *


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
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, self.color_texture_id, 0)
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


class DataFBO:
    def __init__(self, num_ommatidia):

        self.width = num_ommatidia
        self.height = 1

        # Create the Framebuffer Object
        self.fbo_id = glGenFramebuffers(1)
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)

        # Create the 1D Texture
        self.data_texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.data_texture_id)  # treats 1D texture as a 2D texture of height 1

        # we use a float texture (GL_RGBA32F) to store precise color data
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, self.width, self.height, 0,
                     GL_RGBA, GL_FLOAT, None)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        # Attach the texture to the FBO
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, self.data_texture_id, 0)

        # No depth buffer is needed for this data-only pass

        # Check for completeness
        status = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        if status != GL_FRAMEBUFFER_COMPLETE:
            print(f"DataFBO is not complete: {status}")

        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def bind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo_id)
        glViewport(0, 0, self.width, self.height)

    def unbind(self):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

    def read_data(self, buffer):
        """ Reads the texture data into a pre-allocated numpy buffer """

        glBindTexture(GL_TEXTURE_2D, self.data_texture_id)
        glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_FLOAT, buffer)
        glBindTexture(GL_TEXTURE_2D, 0)

    def free(self):
        glDeleteFramebuffers(1, [self.fbo_id])
        glDeleteTextures(1, [self.data_texture_id])