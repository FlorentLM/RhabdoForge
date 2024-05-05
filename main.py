from datetime import datetime
import time

import glfw
from OpenGL.GL import *
from OpenGL.GLU import *

from collections import deque
import numpy as np

from camera import Camera
import engine
import glm


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


def keyboard_input(window, key: int, scancode: int, action: int, mods: int):
    if key == glfw.KEY_ESCAPE and action == 1:
        print('Quitting')
        glfw.terminate()
        return

    if key == glfw.KEY_W and action == 1:
        print('Going forward')
        # cam_displacement = cam.forward * distance_moved
    if key == glfw.KEY_S and action == 1:
        print('Going backward')
        # cam_displacement = cam.backward * distance_moved
    if key == glfw.KEY_A and action == 1:
        print('Going left')
        # cam_displacement = cam.left * distance_moved
    if key == glfw.KEY_D and action == 1:
        print('Going right')
        # cam_displacement = cam.right * distance_moved
    if key == glfw.KEY_Z and action == 1:
        print('Going up')
        # cam_displacement = engine.WORLD_UP * distance_moved
    if key == glfw.KEY_X and action == 1:
        print('Going down')
        # cam_displacement = engine.WORLD_DOWN * distance_moved

        # cam.fov = event.y * 0.5 + cam.fov

def main(controls=True):
    display = (800, 600)

    if not glfw.init():
        return

    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    window = glfw.create_window(display[0], display[1], "Antworlds", None, None)

    if not window:
        glfw.terminate()
        return

    # Make the window's context current
    glfw.make_context_current(window)

    # Set the wait time for glfwSwapBuffers to 0 (this unlocks FPS)
    glfw.swap_interval(0)
    # The above may not work on all platforms. Another solution is to use single buffer instead of double
    # (add the hint ```glfw.window_hint(glfw.DOUBLEBUFFER, glfw.FALSE)``` before creating the window)

    if controls:
        glfw.set_key_callback(window, keyboard_input)

    # Enable OpenGL depth testing
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    if controls:
        glfw.poll_events()

    # Set up starting camera position and aspect
    cam = Camera()
    cam.pos = 0, 0, 4
    cam.ratio = display[0] / display[1]

    # Load Assets
    c = CubeAsset()

    # Create instances
    crate_0 = Instance(c)
    crate_1 = Instance(c)
    crate_1.transform = glm.translation_mat([-3, 0, 0])
    crate_2 = Instance(c)
    crate_2.transform = glm.translation_mat([3, 0, 0])

    instances = [crate_0, crate_1, crate_2]

    # Initialise some variables
    test_rot_speed = 45.0
    test_rotation = 0

    cam_move_speed = 1

    fps_rolling = deque(maxlen=500)

    tick = time.time_ns()
    while not glfw.window_should_close(window):

        # Update variables according to passed time
        tock = time.time_ns()
        t = (tock - tick) * 1e-9

        distance_moved = cam_move_speed * t
        test_rotated = test_rot_speed * t

        cam_displacement = 0

        # Poll for and process events
        glfw.poll_events()

        # Update transforms
        test_rotation += test_rotated
        while test_rotation > 360.0:
            test_rotation -= 360.0

        crate_0.transform = glm.rotation_mat(np.deg2rad(test_rotation), engine.WORLD_UP)

        # Clear the render with black
        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Render all instances
        for instance in instances:
            render_instance(instance, cam)  # We render to the cam - TODO - render to texture instead?

        # Swap front and back buffers
        glfw.swap_buffers(window)

        tick = tock

    glfw.terminate()


##

if __name__ == "__main__":
    main(controls=True)

