import os

from graphics.compound_eye import CompoundEyeRay

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from collections import deque
import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *

from graphics.scene import Scene, Instance, Mesh
from graphics.camera import Camera
from graphics.utils import VEC_DTYPE, WORLD_UP, WORLD_DOWN, WORLD_RIGHT, WORLD_FORWARD
from graphics.raster_mode import CubemapFBO
from graphics.glm import lookat_mat


class Engine:
    def __init__(self, width=800, height=600, headless=False, cubemap_resolution=512):
        self.width = width
        self.height = height
        self.headless = headless

        self.compound_eye = None

        # Properties for geometry counts
        self._total_scene_vertices = 0
        self._total_scene_triangles = 0

        # -- Pygame OpenGL context --
        # TODO: maybe something lighter than pygame? glfw?
        pygame.init()
        flags = DOUBLEBUF | OPENGL
        if self.headless:
            flags |= pygame.HIDDEN
        pygame.display.set_mode((self.width, self.height), flags)

        # -- OpenGL state --
        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_CULL_FACE)
        glEnable(GL_TEXTURE_CUBE_MAP_SEAMLESS)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glClearColor(0.1, 0.1, 0.1, 1.0)

        # -- Core stuff --
        self.scene = Scene()
        self.camera = Camera(position=(0, 0, 4), ratio=width / height)

        # Cubemap FBO to render the whole scene on (used by InsectEyeRaster in its first pass, and by the PanoramicEye)
        self._cubemap_fbo = None     # lazy initialised, only if needed
        self._cubemap_resolution = cubemap_resolution
        # A camera for the 6-sided cubemap render
        self._cubemap_render_cam = None     # also lazy initialised

        # Skybox cubemap references
        self.skybox = None
        self.skybox_texture_id = None

        # -- Stuff for interactive mode --

        # HUD & text rendering
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(pygame.font.get_default_font(), 22)
        self.fps_rolling = deque(maxlen=5)
        self.show_hud = True

        # Text caching and throttling
        self.hud_update_interval_ms = 500  # update HUD text every 500 ms
        self._last_hud_update_time = 0
        self._cached_hud_surface = None

        self._cached_controls_surfaces = []  # cache for controls text
        self._cache_controls_text()  # render controls text once

        self.move_sensitivity = 0.01  # units per frame
        self.mouse_sensitivity = 0.25
        self.zoom_sensitivity = 0.25

    def _cache_controls_text(self):
        """ Renders the controls text once and caches it """

        sample_label = "Samples"
        if self.compound_eye and isinstance(self.compound_eye, CompoundEyeRay):
            sample_label = "Rays"

        controls = [
            'ESC: Quit',
            'H: Show/hide HUD',
            f'+/-: {sample_label}',
            'T: Time dithering',
            'V: Voronoi view',
            'P: Panoramic view',
            'C: Compound eye view',
            'Mouse: Look',
            'LShift/Space: Down/Up',
            'WASD: Move',
            '',
            'Controls:'
        ]

        self._cached_controls_surfaces.clear()
        for text in controls:
            white_surf = self.font.render(text, True, (255, 255, 255, 255))
            gray_surf = self.font.render(text, True, (0, 0, 0, 180))
            self._cached_controls_surfaces.append((white_surf, gray_surf))

    def _render_text_surface(self, white_surf: pygame.Surface, gray_surf: pygame.Surface, x: int, y: int):
        """ Renders pre-made Pygame surface with a simple outline """

        # This is kinda slow

        w, h = white_surf.get_width(), white_surf.get_height()

        gray_data = pygame.image.tobytes(gray_surf, 'RGBA', True)
        white_data = pygame.image.tobytes(white_surf, 'RGBA', True)

        # Draw the outline in 4 directions
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            glWindowPos2d(x + dx, y + dy)
            glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, gray_data)

        # Draw foreground
        glWindowPos2d(x, y)
        glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, white_data)

    def draw_hud(self):
        """ Renders the HUD text with simulation info in the top-left corner """

        current_time = pygame.time.get_ticks()
        self.fps_rolling.append(self.clock.get_fps())

        # Throttled caching
        if current_time - self._last_hud_update_time > self.hud_update_interval_ms:
            self._last_hud_update_time = current_time
            avg_fps = np.mean(self.fps_rolling) if self.fps_rolling else 0

            if self.show_hud:

                # Gather info
                # TODO: this will not change at runtime, there should be a setter to regiser an eye and set this once
                is_raytracer = isinstance(self.compound_eye, CompoundEyeRay)
                mode_name = "Ray-tracer" if is_raytracer else "Rasterizer"
                sample_label = "Rays" if is_raytracer else "Samples"

                # Gather info
                pos = self.camera.position
                num_om = getattr(self.compound_eye, 'num_ommatidia', 0)
                num_samples = getattr(self.compound_eye, 'samples_per_ommatidium', 1)
                dithering_on = getattr(self.compound_eye, '_time_dithering', False)
                # total_samples = num_om * num_samples

                # Format info text
                info_text = (
                    f'FPS: {avg_fps:>4.2f} | Mode: {mode_name} | Ommatidia: {num_om} | '
                    f'{sample_label}: {num_samples} | Dithering: {"On " if dithering_on else "Off"} | '
                    f'XYZ: [ {pos[0]:>5.3f}, {pos[1]:>5.3f}, {pos[2]:>5.3f} ]'
                )

                # Gather scene geometry info
                # Use the Engine's own pre-calculated counts
                # scene_info_lines = [
                #     f'Scene Vertices: {self._total_scene_vertices:,}',
                #     f'Scene Triangles: {self._total_scene_triangles:,}',
                #     f'Total {sample_label}: {total_samples:,}',
                # ]

                # Re-render and cache the HUD surfaces
                surfaces = []
                # Top info bar
                white_surf = self.font.render(info_text, True, (255, 255, 255, 255))
                gray_surf = self.font.render(info_text, True, (0, 0, 0, 180))
                surfaces.append(((white_surf, gray_surf), 'top_left'))

                # Bottom right info
                # for i, text in enumerate(reversed(scene_info_lines)):
                #     white_surf = self.font.render(text, True, (255, 255, 255, 255))
                #     gray_surf = self.font.render(text, True, (0, 0, 0, 180))
                #     surfaces.append(((white_surf, gray_surf), 'bottom_right', i))

                self._cached_hud_surface = surfaces

            else:
                # if not visible just print the FPS to the console
                print(f'FPS: {avg_fps:.2f}')

        # This block is the expensive OpenGL drawing calls
        # TODO: move to a fully GPU-side render with a font atlas texture... but that's for later

        if self.show_hud:

            # Disable depth test and enable blending for transparent text background
            glDisable(GL_DEPTH_TEST)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

            # Unbind any shaders/textures to prevent state leakage
            glUseProgram(0)
            glBindTexture(GL_TEXTURE_2D, 0)

            # Draw info text
            margin = 10
            line_height = self.font.get_height() + 2

            # Draw the cached controls on the left
            for i, (white_surf, gray_surf) in enumerate(self._cached_controls_surfaces):
                x_pos = margin
                y_pos = margin + (i * line_height)
                self._render_text_surface(white_surf, gray_surf, x_pos, y_pos)

                # Draw the cached info surfaces
                if self._cached_hud_surface:
                    for surface_info in self._cached_hud_surface:
                        (white_surf, gray_surf), position, *rest = surface_info

                        if position == 'top_left':
                            x_pos = margin
                            y_pos = self.height - line_height
                            self._render_text_surface(white_surf, gray_surf, x_pos, y_pos)

                        # elif position == 'bottom_right':
                        #     line_index = rest[0]
                        #     x_pos = self.width - white_surf.get_width() - margin
                        #     y_pos = margin + (line_index * line_height)
                        #     self._render_text_surface(white_surf, gray_surf, x_pos, y_pos)

            # Restore GL state
            glEnable(GL_DEPTH_TEST)
            glDisable(GL_BLEND)

    def load_mesh(self, name, *args, **kwargs) -> Mesh:
        """ Loads a mesh and caches it to avoid redundant loads """

        if name not in self.scene.assets:
            self.scene.assets[name] = Mesh(*args, **kwargs)
        return self.scene.assets[name]

    def add_instance(self, instance: Instance):
        self.scene.add_instance(instance)
        self._update_geometry_counts()

    def _update_geometry_counts(self):
        """ Recalculates total vertices and triangles from all instances in the scene """
        vert_count = 0
        for instance in self.scene.instances:
            vert_count += instance.asset.draw_count
        self._total_scene_vertices = vert_count
        self._total_scene_triangles = vert_count // 3

    def _render_instance(self, instance, view_matrix, projection_matrix):
        """ Renders a single instance using provided view and projection matrices """

        mesh = instance.asset

        try:
            glUseProgram(mesh.shaders)
            glBindVertexArray(mesh.vao)

            # Column-major so final matrix is P * V * M
            camera_matrix = projection_matrix @ view_matrix

            glUniformMatrix4fv(glGetUniformLocation(mesh.shaders, "camera"),
                               1, True, camera_matrix)
            glUniformMatrix4fv(glGetUniformLocation(mesh.shaders, "model"),
                               1, True, instance.transform)

            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, mesh.texture)

            glDrawArrays(mesh.draw_type, mesh.draw_start, mesh.draw_count)

        finally:
            # Unbind everything
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
            self._render_instance(instance, camera.view, camera.projection)

    def render_to_cubemap(self, scene, camera):
        """ Renders the given scene into the cubemap FBO from the perspective of the agent position """

        # Lazy initialise the FBO and Cubemp camera
        if self._cubemap_fbo is None or self._cubemap_render_cam is None:
            self._cubemap_fbo = CubemapFBO(resolution=self._cubemap_resolution)
            self._cubemap_render_cam = Camera(fov=90.0, ratio=1.0)

        self._cubemap_fbo.bind()

        # Using the cubemap-specific camera (for its 90-degree FOV projection)
        self._cubemap_render_cam.pos = camera.position

        projection = self._cubemap_render_cam.projection

        # look-at directions and 'up' vectors for each face must correspond to the OpenGL cubemap coordinate system:
        #  - GL_TEXTURE_CUBE_MAP_POSITIVE_X  ->  Right
        #  - GL_TEXTURE_CUBE_MAP_NEGATIVE_X  ->  Left
        #  - GL_TEXTURE_CUBE_MAP_POSITIVE_Y  ->  Up
        #  - GL_TEXTURE_CUBE_MAP_NEGATIVE_Y  ->  Down
        #  - GL_TEXTURE_CUBE_MAP_POSITIVE_Z  ->  Back
        #  - GL_TEXTURE_CUBE_MAP_NEGATIVE_Z  ->  Front
        #
        # Note: the camera's local vectors are used to maintain its orientation (roll)

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
                                   self._cubemap_fbo.color_texture_id, 0)

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
                self._render_instance(instance, view, projection)

        self._cubemap_fbo.unbind()

        # Regenerate mipmaps after rendering to the cubemap
        glBindTexture(GL_TEXTURE_CUBE_MAP, self._cubemap_fbo.color_texture_id)
        glGenerateMipmap(GL_TEXTURE_CUBE_MAP)
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0)

        # Restore the viewport to the main window's dimensions
        glViewport(0, 0, self.width, self.height)

        return self._cubemap_fbo.color_texture_id

    def update_movement(self):
        """ Processes continuous input (keyboard/mouse) to move the camera """

        # Handle continuous key presses
        keys = pygame.key.get_pressed()
        cam_displacement = np.zeros(3, dtype=VEC_DTYPE)

        if keys[K_w]: cam_displacement += self.camera.forward
        if keys[K_s]: cam_displacement += self.camera.backward
        if keys[K_a]: cam_displacement += self.camera.left
        if keys[K_d]: cam_displacement += self.camera.right
        if keys[K_SPACE]: cam_displacement += WORLD_UP
        if keys[K_LSHIFT]: cam_displacement += WORLD_DOWN

        # Normalize (prevents faster diagonal movement) and apply speed
        norm = np.linalg.norm(cam_displacement)
        if norm > 0:
            cam_displacement = (cam_displacement / norm) * self.move_sensitivity
            self.camera.pos += cam_displacement

        # Handle mouse look
        mouse_x, mouse_y = pygame.mouse.get_rel()
        if mouse_x != 0 or mouse_y != 0:
            self.camera.yaw += mouse_x * self.mouse_sensitivity
            self.camera.pitch = np.clip(self.camera.pitch + mouse_y * self.mouse_sensitivity, -89.0, 89.0, dtype=VEC_DTYPE)

    def close(self):
        """ Frees all allocated resources """
        self.scene.free()
        if self._cubemap_fbo:
            self._cubemap_fbo.free()
        pygame.quit()

