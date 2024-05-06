import time
import sys

import imgui
from imgui.integrations.glfw import GlfwRenderer

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


def init_glfw(width=800, height=600, name='Antworlds'):

    if not glfw.init():
        print("Could not initialize OpenGL context.")
        sys.exit(1)

    # macOS supports only forward-compatible core profiles from 3.2+
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 1)
    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)

    glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)

    # Create a windowed mode window and its OpenGL context
    window = glfw.create_window(int(width), int(height), name, None, None)
    glfw.make_context_current(window)

    # Set the wait time for glfwSwapBuffers to 0 (this unlocks FPS)
    glfw.swap_interval(0)
    # The above may not work on all platforms. Another solution is to use single buffer instead of double
    # (add the hint ```glfw.window_hint(glfw.DOUBLEBUFFER, glfw.FALSE)``` before creating the window)

    if not window:
        glfw.terminate()
        print("Could not initialize window...")
        sys.exit(1)

    return window


def main(controls=True, use_imgui=True):

    display = 800, 600

    # -- Initialise window GLFW and optionally hook it to imgui
    if use_imgui:
        imgui.create_context()

    window = init_glfw(width=display[0], height=display[1], name='Antworlds')

    if use_imgui:
        impl = GlfwRenderer(window)
    # --

    # Enable OpenGL depth testing
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    if controls:
        glfw.set_key_callback(window, keyboard_input)

    # --

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

    len_fps_rolling = 100
    fps_rolling = deque(maxlen=len_fps_rolling)

    tick = time.time_ns()
    running_t = 0
    while not glfw.window_should_close(window):

        # Clear the render with black
        glClearColor(0, 0, 0, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Per the documentation: GLFW needs to poll the window system for events both to provide input to the
        # application and to prove to the window system that the application hasn't locked up
        glfw.poll_events()

        # Update variables according to passed time
        tock = time.time_ns()
        t = (tock - tick) * 1e-9
        running_t += t
        fps_rolling.append(1.0/t)

        fps = sum(fps_rolling) / len_fps_rolling  # This will be wrong for a few frames but it's faster than using len()

        distance_moved = cam_move_speed * t
        test_rotated = test_rot_speed * t

        cam_displacement = 0

        # Update transforms
        test_rotation += test_rotated
        while test_rotation > 360.0:
            test_rotation -= 360.0

        crate_0.transform = glm.rotation_mat(np.deg2rad(test_rotation), engine.WORLD_UP)

        # Render all instances
        for instance in instances:
            render_instance(instance, cam)  # We render to the cam - TODO - render to texture instead?

        # imgui main loop - needs to be done *after* rendering the scene
        if use_imgui:
            # Process inputs from imgui
            impl.process_inputs()

            # Start new frame context
            imgui.new_frame()

            # Draw text label inside of current window
            imgui.text(f'{fps:.2f} fps')

            # Pass all drawing comands to the rendering pipeline and close frame context
            imgui.render()
            impl.render(imgui.get_draw_data())
            imgui.end_frame()

        else:
            if running_t >= 1:
                print(f'{fps:.2f} fps')
                running_t = 0

        # Swap front and back buffers
        glfw.swap_buffers(window)

        # And finally update the time for current frame
        tick = tock

    if use_imgui:
        impl.shutdown()
    glfw.terminate()


##

if __name__ == "__main__":
    main(use_imgui=True)

