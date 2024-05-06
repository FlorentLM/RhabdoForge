import numpy as np

from OpenGL.GL import *

import engine

##


class CubeAsset:

    _data = np.array((
        # X     Y     Z        U     V
        # bottom
        -1.0, -1.0, -1.0,      0.0, 0.0,
         1.0, -1.0, -1.0,      1.0, 0.0,
        -1.0, -1.0,  1.0,      0.0, 1.0,
         1.0, -1.0, -1.0,      1.0, 0.0,
         1.0, -1.0,  1.0,      1.0, 1.0,
        -1.0, -1.0,  1.0,      0.0, 1.0,

        # top
        -1.0,  1.0, -1.0,      0.0, 0.0,
        -1.0,  1.0,  1.0,      0.0, 1.0,
         1.0,  1.0, -1.0,      1.0, 0.0,
         1.0,  1.0, -1.0,      1.0, 0.0,
        -1.0,  1.0,  1.0,      0.0, 1.0,
         1.0,  1.0,  1.0,      1.0, 1.0,

        # front
        -1.0, -1.0,  1.0,      1.0, 0.0,
         1.0, -1.0,  1.0,      0.0, 0.0,
        -1.0,  1.0,  1.0,      1.0, 1.0,
         1.0, -1.0,  1.0,      0.0, 0.0,
         1.0,  1.0,  1.0,      0.0, 1.0,
        -1.0,  1.0,  1.0,      1.0, 1.0,

        # back
        -1.0, -1.0, -1.0,      0.0, 0.0,
        -1.0,  1.0, -1.0,      0.0, 1.0,
         1.0, -1.0, -1.0,      1.0, 0.0,
         1.0, -1.0, -1.0,      1.0, 0.0,
        -1.0,  1.0, -1.0,      0.0, 1.0,
         1.0,  1.0, -1.0,      1.0, 1.0,

        # left
        -1.0, -1.0,  1.0,      0.0, 1.0,
        -1.0,  1.0, -1.0,      1.0, 0.0,
        -1.0, -1.0, -1.0,      0.0, 0.0,
        -1.0, -1.0,  1.0,      0.0, 1.0,
        -1.0,  1.0,  1.0,      1.0, 1.0,
        -1.0,  1.0, -1.0,      1.0, 0.0,

        # right
         1.0, -1.0,  1.0,      1.0, 1.0,
         1.0, -1.0, -1.0,      1.0, 0.0,
         1.0,  1.0, -1.0,      0.0, 0.0,
         1.0, -1.0,  1.0,      1.0, 1.0,
         1.0,  1.0, -1.0,      0.0, 0.0,
         1.0,  1.0,  1.0,      0.0, 1.0
    ), dtype=np.float32)

    draw_type = GL_TRIANGLES
    draw_start = 0
    draw_count = 36

    def __init__(self):

        # Create and bind a VAO
        self._gVAO = glGenVertexArrays(1)
        glBindVertexArray(self._gVAO)

        # Create and bind a VBO
        self._gVBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._gVBO)

        # Send the vertex coords to the VBO
        glBufferData(GL_ARRAY_BUFFER, self._data.nbytes, self._data, GL_STATIC_DRAW)

        # Compile GLSL files
        self._gProgram = engine.load_shaders('shaders/base.vert', 'shaders/base.frag')
        self._gTexture = engine.load_texture('textures/wood.jpg')

        # Set up the VAO attributes
        glUseProgram(self._gProgram)

        # X, Y, Z coords
        pos_loc = glGetAttribLocation(self._gProgram, "pos")  # Get the location of shader attrib 'pos'
        glEnableVertexAttribArray(pos_loc)
        glVertexAttribPointer(pos_loc,
                              3,                                            # each vertex is 3 items long (X, Y, Z)
                              GL_FLOAT,                                     # datatype size
                              GL_FALSE,                                     # Normalisation on [0.0, 1.0]
                              5 * self._data.itemsize,                      # Stride
                              ctypes.c_void_p(0 * self._data.itemsize)      # Offset - XYZ data starts at the first byte
                              )
        # U, V coords
        vertTexCoord_loc = glGetAttribLocation(self._gProgram, "vertTexCoord")
        glEnableVertexAttribArray(vertTexCoord_loc)
        glVertexAttribPointer(vertTexCoord_loc,
                              2,                                            # each UV is 2 items long (U, V)
                              GL_FLOAT,                                     # datatype size
                              GL_TRUE,                                      # Normalisation on [0.0, 1.0]
                              5 * self._data.itemsize,                      # Stride
                              ctypes.c_void_p(3 * self._data.itemsize)      # Offset - UV data starts after the 3rd byte
                              )

        # Unbind VBO, VAO, shader, and texture
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        glUseProgram(0)
        glBindTexture(GL_TEXTURE_2D, 0)

    @property
    def texture(self):
        return self._gTexture

    @property
    def shaders(self):
        return self._gProgram

    @property
    def vao(self):
        return self._gVAO


class Instance:

    def __init__(self, asset):
        self._asset = asset
        self.transform = np.eye(4, dtype=np.float32)

    @property
    def asset(self):
        return self._asset


##

def render_instance(instance, camera):

    ass = instance.asset

    # Bind shaders and VAO
    glBindVertexArray(ass.vao)
    glUseProgram(ass.shaders)

    # Pass uniform matrices for camera and object transform to the shader
    camera_loc = glGetUniformLocation(ass.shaders, "camera")
    glUniformMatrix4fv(camera_loc, 1, False, camera.matrix)

    model_loc = glGetUniformLocation(ass.shaders, "model")
    glUniformMatrix4fv(model_loc, 1, False, instance.transform)

    # Bind the texture to slot TEXTURE0
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, ass.texture)

    # Draw
    glDrawArrays(ass.draw_type, ass.draw_start, ass.draw_count)

    # Release VAO, shaders and texture
    glBindVertexArray(0)
    glUseProgram(0)
    glBindTexture(GL_TEXTURE_2D, 0)
