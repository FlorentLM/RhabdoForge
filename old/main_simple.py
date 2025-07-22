import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

from collections import deque
import numpy as np

from graphics.camera import Camera
from graphics import glm


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

        # Compile GLSL files
        self._gProgram = engine.load_shaders('shaders/base.vert', 'shaders/base.frag')
        self._gTexture = engine.load_texture('textures/wood.jpg')

        # Create and bind a VAO
        self._gVAO = glGenVertexArrays(1)
        glBindVertexArray(self._gVAO)

        # Create and bind a VBO
        self._gVBO = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self._gVBO)

        # Send the vertex coords to the VBO
        glBufferData(GL_ARRAY_BUFFER, self._data.nbytes, self._data, GL_STATIC_DRAW)

        # Set up the VAO
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
    glUseProgram(ass.shaders)

    # Pass uniform matrices for camera and object transform to the shader
    camera_loc = glGetUniformLocation(ass.shaders, "camera")
    glUniformMatrix4fv(camera_loc, 1, False, camera.matrix)

    model_loc = glGetUniformLocation(ass.shaders, "model")
    glUniformMatrix4fv(model_loc, 1, False, instance.transform)

    # Bind the texture to slot TEXTURE0
    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, ass.texture)

    # Bind VAO and draw
    glBindVertexArray(ass.vao)
    glDrawArrays(ass.draw_type, ass.draw_start, ass.draw_count)

    # Release VAO, shaders and texture
    glBindVertexArray(0)
    glUseProgram(0)
    glBindTexture(GL_TEXTURE_2D, 0)


def main(controls=True):

    # Prepare PyGame context - this can be replaced by another context
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

    # Enable OpenGL depth testing
    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)

    # Enable seamless cubemap filtering
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS)

    # This is needed to use gl_PointSize in vertex shaders
    glEnable(GL_PROGRAM_POINT_SIZE)

    # Set some PyGame-specific stuff
    font = pygame.font.Font(pygame.font.get_default_font(), 20)

    if controls:
        pygame.key.set_repeat(1, 10)
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

    # Set up starting camera position and aspect
    cam = Camera()
    cam.pos = 0, 0, 4

    cam.ratio = 1.0
    cam.fov = 90.0

    # Load assets
    crate_asset = CubeAsset()

    # Create instances
    crate_0 = Instance(crate_asset)
    crate_1 = Instance(crate_asset)
    crate_1.transform = glm.translation_mat([-3, 0, 0])
    crate_2 = Instance(crate_asset)
    crate_2.transform = glm.translation_mat([3, 0, 0])

    instances = [crate_0, crate_1, crate_2]

    # Initialise some variables
    test_rot_speed = 45.0 * 0.001
    test_rotation = 0

    cam_move_speed = 0.01

    fps_rolling = deque(maxlen=500)

    clock = pygame.time.Clock()
    while True:

        # Update variables according to passed time
        t = clock.tick()
        distance_moved = cam_move_speed * t
        test_rotated = test_rot_speed * t

        cam_displacement = 0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    quit()
                if controls:
                    if event.key == K_w:
                        cam_displacement = cam.forward * distance_moved
                    if event.key == K_s:
                        cam_displacement = cam.backward * distance_moved
                    if event.key == K_a:
                        cam_displacement = cam.left * distance_moved
                    if event.key == K_d:
                        cam_displacement = cam.right * distance_moved
                    if event.key == K_z:
                        cam_displacement = engine.WORLD_UP * distance_moved
                    if event.key == K_x:
                        cam_displacement = engine.WORLD_DOWN * distance_moved

            if controls and event.type == MOUSEWHEEL:
                cam.fov = event.y * 0.5 + cam.fov

        if controls:
            mouse_x, mouse_y = pygame.mouse.get_rel()

            # Update camera
            cam.yaw -= mouse_x * 0.1
            cam.pitch -= mouse_y * 0.1
            cam.pos += cam_displacement

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
            render_instance(instance, cam)

        # Compute and display FPS
        fps_rolling.append(clock.get_fps())
        text_surf = font.render(f'{np.mean(fps_rolling).astype(int)}', True, (255, 255, 255, 255), (0, 0, 0, 255))
        text_surf_dat = pygame.image.tobytes(text_surf, "RGBA", True)
        glWindowPos2d(10, 10)
        glDrawPixels(text_surf.get_width(), text_surf.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_surf_dat)

        # Swap buffers to display
        pygame.display.flip()       # equivalent to glfwSwapBuffers() but for PyGame

        if controls:
            pygame.time.wait(1)


##

if __name__ == "__main__":
    main(controls=True)

