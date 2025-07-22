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

    cam.ratio = 1.0  # cubemaps are square
    cam.fov = 90.0

    # Set up FBO
    debug_cubemap_id = engine.load_cubemap('textures/bright_day')
    cubemap_fbo = CubemapFBO(resolution=512)

    # Load assets
    crate_asset = CubeAsset()
    insect_eye_asset = InsectEyeAsset(num_ommatidia=4096, acceptance_angle_deg=15.0)

    try:
        debug_pano_shader = engine.load_shaders('shaders/panoramic.vert', 'shaders/panoramic.frag')
        debug_pano_vao = glGenVertexArrays(1)  # dummy VAO
    except Exception as e:
        print(f"Could not load debug shader: {e}")
        debug_pano_shader = None

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

    VISUALIZE_MODE = True
    PANORAMIC_DEBUG_MODE = True

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


        # --- PASS 1: RENDER SCENE TO CUBEMAP ---
        cubemap_fbo.bind()

        # Define the 6 camera directions for the cubemap
        agent_position = [0, 0, 0]  # Where the viewer is
        targets = [
            [ 1,  0,  0], [-1,  0,  0],   # +X, -X
            [ 0,  1,  0], [ 0, -1,  0],   # +Y, -Y
            [ 0,  0,  1], [ 0,  0, -1]    # +Z, -Z
        ]
        ups = [
            [ 0, -1,  0], [ 0, -1,  0],   # +X, -X
            [ 0,  0,  1], [ 0,  0, -1],   # +Y, -Y
            [ 0, -1,  0], [ 0, -1,  0]    # +Z, -Z
        ]

        # Use a temporary camera for rendering to the cubemap
        cubemap_cam = Camera(position=agent_position, fov=90.0, ratio=1.0)

        for i in range(6):
            # Point the camera in the correct direction
            cubemap_cam.lookat(np.array(agent_position) + np.array(targets[i]))
            # This lookat might not handle UP vector correctly, a more robust lookat matrix is better
            view_matrix = glm.lookat_mat(agent_position, np.array(agent_position) + targets[i], ups[i])

            # Attach the correct face of the cubemap texture to the FBO
            glFramebufferTexture2D(GL_FRAMEBUFFER,
                                   GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                                   cubemap_fbo.color_texture_id,
                                   0)

            glClearColor(0.1, 0.2, 0.3, 1)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Render all instances with this view
            for instance in instances:
                # We need to modify render_instance to accept a view_matrix
                # Let's make a quick adjustment
                # TODO: fix this properly
                render_instance_to_fbo(instance, cubemap_cam.projection, view_matrix)

        cubemap_fbo.unbind()


        # CUBEMAP_ID = cubemap_fbo.color_texture_id
        CUBEMAP_ID = debug_cubemap_id


        # --- PASS 2: OMMATIDIA DATA GATHERING ---
        if not PANORAMIC_DEBUG_MODE:
            ommatidia_values = insect_eye_asset.get_ommatidia_data(CUBEMAP_ID)

        # --- PASS 3 (OPTIONAL): VISUALISATION ---
        if VISUALIZE_MODE:
            glViewport(0, 0, display[0], display[1])
            glClearColor(0.05, 0.05, 0.05, 1)
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            if PANORAMIC_DEBUG_MODE and debug_pano_shader:
                # Debug draw call
                glUseProgram(debug_pano_shader)

                # Bind the cubemap texture we want to inspect
                glActiveTexture(GL_TEXTURE0)
                glBindTexture(GL_TEXTURE_CUBE_MAP, CUBEMAP_ID)
                glUniform1i(glGetUniformLocation(debug_pano_shader, "u_cubemap"), 0)

                # Draw the full-screen triangle
                glBindVertexArray(debug_pano_vao)
                glDrawArrays(GL_TRIANGLES, 0, 3)

                # Unbind
                glBindVertexArray(0)
                glUseProgram(0)

        elif not PANORAMIC_DEBUG_MODE:
            # Feed the data just gathered back to the visualization renderer
            insect_eye_asset.draw(ommatidia_values)

        else:
            # We already have the data, just clear the screen to show it's running
            glViewport(0, 0, display[0], display[1])
            glClear(GL_COLOR_BUFFER_BIT)

            if pygame.time.get_ticks() % 60 == 0:
                print(f"HEADLESS MODE - Ommatidium 0: {ommatidia_values[0]}")

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


def render_instance_to_fbo(instance, projection_matrix, view_matrix):
    ass = instance.asset
    glUseProgram(ass.shaders)

    # We construct the camera matrix manually from parts
    camera_matrix = view_matrix @ projection_matrix
    glUniformMatrix4fv(glGetUniformLocation(ass.shaders, "camera"), 1, GL_FALSE, camera_matrix)
    glUniformMatrix4fv(glGetUniformLocation(ass.shaders, "model"), 1, GL_FALSE, instance.transform)

    glActiveTexture(GL_TEXTURE0)
    glBindTexture(GL_TEXTURE_2D, ass.texture)
    glBindVertexArray(ass.vao)
    glDrawArrays(ass.draw_type, ass.draw_start, ass.draw_count)
    glBindVertexArray(0)
    glUseProgram(0)

##

if __name__ == "__main__":
    main(controls=True)

