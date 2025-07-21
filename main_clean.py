import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import pygame

from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

from collections import deque
import numpy as np

from camera import Camera
import engine
import glm

from fbo import CubemapFBO
from insect_eye import InsectEyeAsset
from human_eye import HumanEyeAsset

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


def render_instance_to_fbo(instance, camera_matrix):
    ass = instance.asset
    glUseProgram(ass.shaders)

    # Tell OpenGL the matrices are row-major and need to be transposed
    glUniformMatrix4fv(glGetUniformLocation(ass.shaders, "camera"), 1, GL_TRUE, camera_matrix)
    glUniformMatrix4fv(glGetUniformLocation(ass.shaders, "model"), 1, GL_TRUE, instance.transform)

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, ass.texture)
    glUniform1i(glGetUniformLocation(ass.shaders, "u_texture"), 0)

    glBindVertexArray(ass.vao)
    glDrawArrays(ass.draw_type, 0, ass.draw_count)

    glBindVertexArray(0)
    glUseProgram(0)


def main(controls=True):
    pygame.init()
    display = (800, 600)

    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    main_aspect_ratio = display[0] / display[1]

    glEnable(GL_DEPTH_TEST)
    glDepthFunc(GL_LESS)
    glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS)
    glEnable(GL_PROGRAM_POINT_SIZE)

    font = pygame.font.Font(pygame.font.get_default_font(), 20)

    if controls:
        pygame.key.set_repeat(1, 10)
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

    cam = Camera(position=(0, 1.5, 5), ratio=main_aspect_ratio)
    cubemap_fbo = CubemapFBO(resolution=512)

    crate_asset = CubeAsset()
    insect_eye_asset = InsectEyeAsset(num_ommatidia=4096, acceptance_angle_deg=15.0)
    human_eye_asset = HumanEyeAsset()

    crate_0 = Instance(crate_asset)
    crate_1 = Instance(crate_asset)
    crate_1.transform = glm.translation_mat([-3, 0, 0])
    crate_2 = Instance(crate_asset)
    crate_2.transform = glm.translation_mat([3, 0, 0])
    instances = [crate_0, crate_1, crate_2]

    test_rot_speed = 45.0 * 0.001
    test_rotation = 0
    cam_move_speed = 0.01
    fps_rolling = deque(maxlen=500)
    clock = pygame.time.Clock()

    PANORAMIC_DEBUG_MODE = False

    while True:
        t = clock.tick()
        distance_moved = cam_move_speed * t
        test_rotated = test_rot_speed * t
        cam_displacement = np.zeros(3, dtype=np.float32)

        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                pygame.quit()
                quit()

            if event.type == KEYDOWN:

                if event.key == K_p:
                    PANORAMIC_DEBUG_MODE = not PANORAMIC_DEBUG_MODE

                if controls:
                    if event.key == K_w: cam_displacement += cam.forward
                    if event.key == K_s: cam_displacement += cam.backward
                    if event.key == K_a: cam_displacement += cam.left
                    if event.key == K_d: cam_displacement += cam.right
                    if event.key == K_z: cam_displacement += engine.WORLD_UP
                    if event.key == K_x: cam_displacement += engine.WORLD_DOWN
            if controls and event.type == MOUSEWHEEL:
                cam.fov = np.clip(cam.fov - event.y * 0.5, 1, 120)

        if controls:
            mouse_x, mouse_y = pygame.mouse.get_rel()
            cam.yaw -= mouse_x * 0.1
            cam.pitch = np.clip(cam.pitch - mouse_y * 0.1, -89, 89)
            if np.linalg.norm(cam_displacement) > 0:
                cam.pos += np.linalg.norm(cam_displacement) * distance_moved * (
                            cam_displacement / np.linalg.norm(cam_displacement))

        test_rotation = (test_rotation + test_rotated) % 360.0
        crate_0.transform = glm.rotation_mat(np.deg2rad(test_rotation), engine.WORLD_UP)

        # --- PASS 1: RENDER SCENE TO CUBEMAP ---
        cubemap_fbo.bind()
        cubemap_projection = glm.perspective_mat(np.deg2rad(90.0), 1.0, 0.1, 100.0)

        # Get the camera's orientation matrix (row-major)
        # TODO: Everything probably needs to be column-major
        cam_orientation = cam.orientation

        targets_world = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
                                 dtype=np.float32)
        ups_world = np.array([[0, -1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], [0, -1, 0], [0, -1, 0]], dtype=np.float32)

        for i in range(6):
            # Transform vectors using row-major math: vector @ matrix
            target_vec4 = np.array(list(targets_world[i]) + [0.0])
            up_vec4 = np.array(list(ups_world[i]) + [0.0])

            oriented_target = (target_vec4 @ cam_orientation)[:3]
            oriented_up = (up_vec4 @ cam_orientation)[:3]

            view_matrix = glm.lookat_mat(cam.pos, cam.pos + oriented_target, oriented_up)

            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                                   cubemap_fbo.color_texture_id, 0)
            glClearColor(0.2, 0.3, 0.4, 1.0)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            camera_mat_for_shader = view_matrix @ cubemap_projection

            for instance in instances:
                render_instance_to_fbo(instance, camera_mat_for_shader)

        cubemap_fbo.unbind()

        # --- RENDER TO SCREEN ---
        glViewport(0, 0, display[0], display[1])
        glClearColor(0.05, 0.05, 0.05, 1)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        CUBEMAP_ID = cubemap_fbo.color_texture_id

        if PANORAMIC_DEBUG_MODE:
            # --- PASS 3a: HUMAN EYE VISUALISATION ---
            human_eye_asset.draw(CUBEMAP_ID)
        else:

            # --- PASS 2: OMMATIDIA DATA GATHERING ---
            ommatidia_values = insect_eye_asset.get_ommatidia_data(CUBEMAP_ID)

            # --- PASS 3b: INSECT EYE VISUALISATION ---
            insect_eye_asset.draw(ommatidia_values)

        fps_rolling.append(clock.get_fps())
        text_surf = font.render(
            f'FPS: {np.mean(fps_rolling):.0f} | View: {"Human" if PANORAMIC_DEBUG_MODE else "Insect"} (P to switch)',
            True, (255, 255, 255, 255), (0, 0, 0, 255))
        text_surf_dat = pygame.image.tobytes(text_surf, "RGBA", True)
        glWindowPos2d(10, 10)
        glDrawPixels(text_surf.get_width(), text_surf.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_surf_dat)

        pygame.display.flip()


if __name__ == "__main__":
    main(controls=True)