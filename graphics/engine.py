import os

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from collections import deque
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *

from graphics.scene import Scene, Instance, Mesh
from graphics.camera import Camera
from graphics.utils import DTYPE, WORLD_UP, WORLD_DOWN, WORLD_RIGHT, WORLD_FORWARD
from graphics.fbo import CubemapFBO
from graphics.glm import lookat_mat


class Engine:
    def __init__(self, width=800, height=600, headless=False, cubemap_resolution=512):
        self.width = width
        self.height = height
        self.headless = headless

        # Create an OpenGL context
        # TODO: maybe something lighter than pygame? glfw?
        pygame.init()
        flags = DOUBLEBUF | OPENGL
        if self.headless:
            flags |= pygame.HIDDEN
        pygame.display.set_mode((self.width, self.height), flags)

        # initial OpenGL state
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_CULL_FACE)
        glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glClearColor(0.1, 0.1, 0.1, 1.0)

        # Core components
        self.scene = Scene()
        self.camera = Camera(position=(0, 0, 4), ratio=width / height)

        # Cubemap FBO for the insect eye pass
        self.cubemap_fbo = CubemapFBO(resolution=cubemap_resolution)

        # A camera for the 6-sided cubemap render
        self.cubemap_render_cam = Camera(fov=90.0, ratio=1.0)

        # Skybox cubemap references
        self.skybox = None
        self.skybox_texture_id = None

        # For interactive mode
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 20)
        self.fps_rolling = deque(maxlen=500)
        self.is_running_interactive = False

        self.cam_move_step = 0.01  # units per frame
        self.mouse_sensitivity = 0.1

    def load_mesh(self, name, *args, **kwargs) -> Mesh:
        """ Loads a mesh and caches it to avoid redundant loads """

        if name not in self.scene.assets:
            self.scene.assets[name] = Mesh(*args, **kwargs)
        return self.scene.assets[name]

    def add_instance(self, instance: Instance):
        self.scene.add_instance(instance)

    def _render_instance(self, instance, camera, view_matrix=None, projection_matrix=None):
        """ Renders a single instance (with a camera's matrices or explicitly provided ones) """

        mesh = instance.asset

        try:
            # Bind everything
            glUseProgram(mesh.shaders)
            glBindVertexArray(mesh.vao)
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, mesh.texture)

            if view_matrix is not None and projection_matrix is not None:
                # Use explicitly provided matrices (for FBO rendering)
                proj = projection_matrix
                view = view_matrix
            else:
                # Use the camera's matrices (for standard rendering)
                proj = camera.projection
                view = camera.view

            # Column-major, so post-multiply, so final matrix is P * V * M
            camera_matrix = proj @ view

            glUniformMatrix4fv(glGetUniformLocation(mesh.shaders, "camera"),
                               1,
                               True,  # OpenGL expects column-major arrays in COLUMN-MAJOR MEMORY (Fortran style)!!
                               camera_matrix)
            glUniformMatrix4fv(glGetUniformLocation(mesh.shaders, "model"),
                               1,
                               True,  # OpenGL expects column-major arrays in COLUMN-MAJOR MEMORY (Fortran style)!!
                               instance.transform)

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, mesh.texture)

            glBindVertexArray(mesh.vao)
            glDrawArrays(mesh.draw_type, mesh.draw_start, mesh.draw_count)

        finally:
            # Unbind everyone
            glBindTexture(GL_TEXTURE_2D, 0)
            glBindVertexArray(0)
            glUseProgram(0)

    def render_frame(self, camera=None):
        """ Renders a single frame to the currently bound framebuffer (e.g., the screen) """
        if camera is None:
            camera = self.camera

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(0, 0, self.width, self.height)

        # Draw skybox first
        if self.skybox and self.skybox_texture_id is not None:
            self.skybox.draw(camera.projection, camera.view, self.skybox_texture_id)

        # Then the rest of the scene
        for instance in self.scene.instances:
            self._render_instance(instance, camera)

    def render_to_cubemap(self, scene, camera):
        """ Renders the given scene into the cubemap FBO from the perspective of the agent position """

        self.cubemap_fbo.bind()

        # Using the cubemap-specific camera (for its 90-degree FOV projection)
        self.cubemap_render_cam.pos = camera.position
        projection = self.cubemap_render_cam.projection

        # Mapping to align the camera's view directions with the cubemap faces as they are interpreted
        # by the panoramic fragment shader
        lookat_directions = [
            camera.right,       # For +X face (index 0), we look to the camera's right
            camera.left,        # For -X face (index 1), we look to the camera's left
            camera.up,          # For +Y face (index 2), we look up
            camera.down,        # For -Y face (index 3), we look down
            camera.backward,    # For +Z face (index 4), we look backward
            camera.forward,     # For -Z face (index 5), we look forward
        ]

        # The 'up' vectors for each look-at direction
        ups = [
            camera.down,        # Up for looking right/left is camera's down
            camera.down,
            camera.backward,    # Up for looking up is camera's backward
            camera.forward,     # Up for looking down is camera's forward
            camera.down,        # Up for looking backward/forward is camera's down
            camera.down,
        ]

        for i in range(6):
            # Attach the correct face of the cubemap texture for rendering
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                                   GL_TEXTURE_CUBE_MAP_POSITIVE_X + i,
                                   self.cubemap_fbo.color_texture_id, 0)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

            # Generate the specific view matrix for this face
            view = lookat_mat(
                camera.position,
                camera.position + lookat_directions[i],  # target point is eye + direction
                ups[i]
            )

            # Draw skybox first into the cubemap face
            if self.skybox and self.skybox_texture_id is not None:
                self.skybox.draw(projection, view, self.skybox_texture_id)

            # Render all instances in the scene with this view
            for instance in scene.instances:
                self._render_instance(instance, None, view_matrix=view, projection_matrix=projection)

        self.cubemap_fbo.unbind()
        return self.cubemap_fbo.color_texture_id

    def run_interactive(self):
        """ Starts an interactive visualization loop with camera controls """

        if self.headless:
            print("Cannot run interactive mode when initialized as headless.")
            return

        pygame.key.set_repeat(1, 10)
        pygame.mouse.set_visible(False)
        pygame.event.set_grab(True)

        is_running = True
        while is_running:
            self.clock.tick()
            if not self.handle_interactive_events():
                is_running = False

            # ___ Per-frame update stuff here ___
            # (like updating object animations or whatever)

            self.render_frame()
            self._draw_fps()
            pygame.display.flip()

        self.close()

    def handle_interactive_events(self):

        # Handle discrete events
        for event in pygame.event.get():
            if event.type == QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
                return False  # Signal to quit

            if event.type == MOUSEWHEEL:
                self.camera.fov -= event.y * 1.5

        # Handle continuous input for camera movement
        self.update_movement()

        return True  # Signal to continue running

    def update_movement(self):
        """ Processes continuous input (keyboard/mouse) to move the camera """

        # Handle continuous key presses
        keys = pygame.key.get_pressed()
        cam_displacement = np.zeros(3, dtype=DTYPE)

        if keys[K_w]: cam_displacement += self.camera.forward
        if keys[K_s]: cam_displacement += self.camera.backward
        if keys[K_a]: cam_displacement += self.camera.left
        if keys[K_d]: cam_displacement += self.camera.right
        if keys[K_SPACE]: cam_displacement += WORLD_UP
        if keys[K_LSHIFT]: cam_displacement += WORLD_DOWN

        # Normalize (prevents faster diagonal movement) and apply speed
        norm = np.linalg.norm(cam_displacement)
        if norm > 0:
            cam_displacement = (cam_displacement / norm) * self.cam_move_step
            self.camera.pos += cam_displacement

        # Handle mouse look
        mouse_x, mouse_y = pygame.mouse.get_rel()
        if mouse_x != 0 or mouse_y != 0:
            self.camera.yaw += mouse_x * self.mouse_sensitivity
            self.camera.pitch = np.clip(self.camera.pitch + mouse_y * self.mouse_sensitivity, -89.0, 89.0, dtype=DTYPE)

    def _draw_fps(self):

        # Unbind texture to prevent state leakage (text background is otherwise the colour as the last used texture)
        glBindTexture(GL_TEXTURE_2D, 0)
        # And explicitely unbind any program still active (also to avoid state leakage)
        glUseProgram(0)

        self.fps_rolling.append(self.clock.get_fps())
        avg_fps = np.mean(self.fps_rolling) if self.fps_rolling else 0

        text_surf = self.font.render(f'{int(avg_fps)} FPS',
                                     True,
                                     (255, 255, 255, 255),
                                     (0, 0, 0, 255))
        text_data = pygame.image.tobytes(text_surf, 'RGBA', True)

        # Temporarily disable depth testing to ensure text is drawn on top
        glDisable(GL_DEPTH_TEST)
        glWindowPos2d(10, self.height - 30)
        glDrawPixels(text_surf.get_width(), text_surf.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)
        glEnable(GL_DEPTH_TEST)  # Reenable depth testing

    def close(self):
        """ Frees all allocated resources """
        self.scene.free()
        pygame.quit()

