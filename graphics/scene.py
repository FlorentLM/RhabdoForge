import numpy as np
from OpenGL.GL import *
from graphics.utils import load_shaders, load_texture, VEC_DTYPE


class Mesh:
    """ Renderable object with its own shaders, texture, and vertex data """

    def __init__(self, vertex_data, vert_shader_path, frag_shader_path, texture_path):
        self.data = vertex_data
        self.draw_type = GL_TRIANGLES
        self.draw_start = 0
        self.draw_count = len(vertex_data) // 5  # 5 floats per vertex (x, y, z, u, v)

        # Compile GLSL files and load texture
        self.shaders = load_shaders(vert_shader_path, frag_shader_path)
        self.texture = load_texture(texture_path)

        # Create and bind a VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # Create and bind a VBO
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        # Send vertex data to VBO
        glBufferData(GL_ARRAY_BUFFER, self.data.nbytes, self.data, GL_STATIC_DRAW)

        # Configure vertex attributes
        glUseProgram(self.shaders)

        # Position attribute (3 floats)
        pos_loc = glGetAttribLocation(self.shaders, "pos")
        glEnableVertexAttribArray(pos_loc)
        glVertexAttribPointer(pos_loc,
                              3,
                              GL_FLOAT,
                              GL_FALSE,
                              5 * self.data.itemsize,
                              ctypes.c_void_p(0))

        # Texture coordinate attribute (2 floats)
        vertTexCoord_loc = glGetAttribLocation(self.shaders, "vertTexCoord")
        glEnableVertexAttribArray(vertTexCoord_loc)
        glVertexAttribPointer(vertTexCoord_loc,
                              2,
                              GL_FLOAT,
                              GL_FALSE,
                              5 * self.data.itemsize,
                              ctypes.c_void_p(3 * self.data.itemsize))

        # Unbind everything to be safe
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        glUseProgram(0)

    def free(self):
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])
        glDeleteProgram(self.shaders)
        glDeleteTextures(1, [self.texture])


class Instance:
    """ A specific instance of a Mesh in the scene, with its own transform """

    def __init__(self, asset: Mesh, transform=None):
        self.asset = asset
        if transform is None:
            self.transform = np.eye(4, dtype=VEC_DTYPE)
        else:
            self.transform = transform


class Scene:
    """ Container for all objects in the world """

    def __init__(self):
        self.instances = []
        self.assets = {}

    def add_instance(self, instance: Instance):
        self.instances.append(instance)

    def free(self):
        for asset in self.assets.values():
            asset.free()
        self.instances.clear()
        self.assets.clear()