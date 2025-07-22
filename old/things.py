import numpy as np

from OpenGL.GL import *


##

class BaseAsset:
    _data = None

    def __init__(self):
        # Create and bind a VAO
        self._gVAO = glGenVertexArrays(1)
        glBindVertexArray(self._gVAO)

        # Create and bind a VBO
        self._gVBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._gVBO)

        # Send the vertex coords to the VBO
        glBufferData(GL_ARRAY_BUFFER, self._data.nbytes, self._data, GL_STATIC_DRAW)

        # Compile GLSL files and use them here
        self._gProgram = engine.load_shaders(f'shaders/base.vert', f'shaders/base.frag')
        glUseProgram(self._gProgram)

        # Load textures
        r_tex = engine.Texture(f'textures/rock.jpg', mode='RGB')
        c_tex = engine.Texture(f'textures/cracks.jpg', mode='RGB')

        # Assign the textures to the two sampler2d in the shader
        loc_tex1 = glGetUniformLocation(self._gProgram, "texture1")
        glUniform1i(loc_tex1, r_tex.unit)

        loc_tex2 = glGetUniformLocation(self._gProgram, "texture2")
        glUniform1i(loc_tex2, c_tex.unit)

        # And keep a ref to the texture objects
        self._gTextures = [r_tex, c_tex]

        # TODO - make a global texture store instead (better if multiple assets share a same texture)

        # X, Y, Z coords
        pos_loc = glGetAttribLocation(self._gProgram, "pos")  # Get the location of shader attrib 'pos'
        glEnableVertexAttribArray(pos_loc)
        glVertexAttribPointer(pos_loc,
                              3,  # each vertex is 3 items long (X, Y, Z)
                              GL_FLOAT,  # datatype size
                              GL_FALSE,  # Normalisation on [0.0, 1.0]
                              5 * self._data.itemsize,  # Stride
                              ctypes.c_void_p(0 * self._data.itemsize)  # Offset - XYZ data starts at the first byte
                              )
        # U, V coords
        vertTexCoord_loc = glGetAttribLocation(self._gProgram, "vertTexCoord")
        glEnableVertexAttribArray(vertTexCoord_loc)
        glVertexAttribPointer(vertTexCoord_loc,
                              2,  # each UV is 2 items long (U, V)
                              GL_FLOAT,  # datatype size
                              GL_TRUE,  # Normalisation on [0.0, 1.0]
                              5 * self._data.itemsize,  # Stride
                              ctypes.c_void_p(3 * self._data.itemsize)  # Offset - UV data starts after the 3rd byte
                              )

        # Unbind VBO, VAO, shader, and texture
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        glUseProgram(0)
        glBindTexture(GL_TEXTURE_2D, 0)

    @property
    def textures(self):
        return self._gTextures

    @property
    def shaders(self):
        return self._gProgram

    @property
    def vao(self):
        return self._gVAO


class CubeAsset(BaseAsset):

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

    # Bind the instance's textures
    for tex in ass.textures:
        glActiveTexture(GL_TEXTURE0 + tex.unit)
        glBindTexture(GL_TEXTURE_2D, tex.idx)

    # Draw
    glDrawArrays(ass.draw_type, ass.draw_start, ass.draw_count)

    # Release VAO, shaders and texture
    glBindVertexArray(0)
    glUseProgram(0)
    glBindTexture(GL_TEXTURE_2D, 0)

    # TODO -  avoid unbinding everything every time if we're rendering several instaces of the same asset

