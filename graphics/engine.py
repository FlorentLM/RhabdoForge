import os
import string

from graphics.eye_rendering import EyeRendererRay

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
from collections import deque
import numpy as np
import pygame
from pygame.locals import *
from pyglm import glm
from OpenGL.GL import *

from graphics.scene import Scene, Instance, Mesh, PointCloud
from graphics.camera import Camera
from graphics.utils import VEC_DTYPE, WORLD_UP, WORLD_DOWN, WORLD_RIGHT, WORLD_FORWARD, load_shaders
from graphics.raster_mode import CubemapFBO


class FontRenderer:
    """ Renders text on the GPU using a font atlas """

    def __init__(self, font_name, font_size):
        self.font_name = font_name
        self.font_size = font_size
        self.char_data = {}  # To store metrics and UVs for each character

        self.text_program = load_shaders('shaders/text.vert', 'shaders/text.frag')

        self.proj_loc = glGetUniformLocation(self.text_program, "projection")
        self.color_loc = glGetUniformLocation(self.text_program, "textColor")
        self.atlas_loc = glGetUniformLocation(self.text_program, "fontAtlas")

        # Generate font atlas texture
        self._create_font_atlas()

        # VAO and VBO for text quads
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        # Data is buffered on the fly so just set up attributes here
        # Each vertex has 4 floats: x, y, u, v
        glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def _create_font_atlas(self):
        # Use pygame to render glyphs and build the atlas
        font = pygame.font.SysFont(self.font_name, self.font_size)

        # Characters to include in the atlas
        chars_to_render = string.printable

        # Determine atlas size (fixed grid)
        atlas_cols = 16
        atlas_rows = (len(chars_to_render) + atlas_cols - 1) // atlas_cols

        # Get max char dimensions to size cells
        max_w, max_h = 0, 0
        for char in chars_to_render:
            w, h = font.size(char)
            if w > max_w: max_w = w
            if h > max_h: max_h = h

        cell_w, cell_h = max_w, max_h
        atlas_width = atlas_cols * cell_w
        atlas_height = atlas_rows * cell_h

        atlas_surface = pygame.Surface((atlas_width, atlas_height), pygame.SRCALPHA)
        atlas_surface.fill((0, 0, 0, 0))  # Transparent background

        # Render each character and store its data
        for i, char in enumerate(chars_to_render):
            char_surface = font.render(char, True, (255, 255, 255, 255))
            metrics = font.metrics(char)[0]  # (minx, maxx, miny, maxy, advance)
            advance = metrics[4]

            col = i % atlas_cols
            row = i // atlas_cols
            x, y = col * cell_w, row * cell_h

            atlas_surface.blit(char_surface, (x, y))

            # Store character data: size, uv coords, and advance width
            uv_x0 = x / atlas_width
            uv_y0 = y / atlas_height
            uv_x1 = (x + char_surface.get_width()) / atlas_width
            uv_y1 = (y + char_surface.get_height()) / atlas_height

            self.char_data[char] = {
                'w': char_surface.get_width(),
                'h': char_surface.get_height(),
                'uv_rect': (uv_x0, uv_y0, uv_x1, uv_y1),
                'advance': advance
            }

        # Convert pygame surface to OpenGL Texture
        texture_data = pygame.image.tostring(atlas_surface, "RGBA", False)
        self.atlas_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.atlas_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        # Set pixel alignment
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_ALPHA, atlas_width, atlas_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, texture_data)

        # Unbind and reset default alignment
        glBindTexture(GL_TEXTURE_2D, 0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)

    def get_text_width(self, text, scale=1.0):
        """ Calculates the pixel width of a string based on the font atlas """
        width = 0
        for char in text:
            if char in self.char_data:
                # 'advance' metric is the proper way to measure character width
                width += self.char_data[char]['advance'] * scale
        return width

    def _generate_text_vertices(self, text, x, y, scale=1.0):
        """ Generates vertex data for a string and returns it as a list """
        vertices = []
        cursor_x = x
        for char in text:
            if char in self.char_data:
                data = self.char_data[char]
                w, h = data['w'] * scale, data['h'] * scale

                u0, v0, u1, v1 = data['uv_rect']

                # Define quad corners
                bl = (cursor_x, y, u0, v1)          # Bottom-Left
                br = (cursor_x + w, y, u1, v1)      # Bottom-Right
                tr = (cursor_x + w, y + h, u1, v0)  # Top-Right
                tl = (cursor_x, y + h, u0, v0)      # Top-Left

                # Triangle 1: bl, br, tr
                vertices.extend(bl)
                vertices.extend(br)
                vertices.extend(tr)

                # Triangle 2: bl, tr, tl
                vertices.extend(bl)
                vertices.extend(tr)
                vertices.extend(tl)

                cursor_x += data['advance'] * scale
        return vertices

    def free(self):
        glDeleteProgram(self.text_program)
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])
        glDeleteTextures(1, [self.atlas_texture])


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
        self.clock = pygame.time.Clock()
        self.show_hud = True

        # HUD Rendering
        self.font_renderer = FontRenderer(pygame.font.get_default_font(), 22)
        self.hud_projection_matrix = glm.ortho(0, self.width, 0, self.height, -1.0, 1.0)

        self.fps_rolling = deque(maxlen=5)
        self.hud_update_interval_ms = 250
        self._last_hud_update_time = 0

        # Text strings updated periodically
        self._hud_info_text = ""
        self._controls_text_lines = []
        self._scene_stats_lines = []

        # Cached vertex data for text (np arrays)
        self._info_shadow_verts = None
        self._info_fg_verts = None
        self._controls_shadow_verts = None
        self._controls_fg_verts = None
        self._stats_shadow_verts = None
        self._stats_fg_verts = None

        self._update_controls_text()  # Generate controls text strings once

        self.move_sensitivity = 0.01  # units per frame
        self.mouse_sensitivity = 0.25
        self.zoom_sensitivity = 0.25

    def _update_controls_text(self):
        """ Generates the controls text strings and caches them """
        sample_label = "Samples"
        if self.compound_eye and isinstance(self.compound_eye, EyeRendererRay):
            sample_label = "Rays"

        self._controls_text_lines = [
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
        ]

        # Generate and cache vertex data for the controls
        shadow_verts = []
        fg_verts = []
        margin = 10
        line_height = self.font_renderer.font_size

        for i, text in enumerate(self._controls_text_lines):
            y_pos = margin + (i * line_height)
            shadow_verts.extend(self.font_renderer._generate_text_vertices(text, margin + 1, y_pos - 1))
            fg_verts.extend(self.font_renderer._generate_text_vertices(text, margin, y_pos))

        # Store as arrays for fast concatenation later
        self._controls_shadow_verts = np.array(shadow_verts, dtype=np.float32) if shadow_verts else None
        self._controls_fg_verts = np.array(fg_verts, dtype=np.float32) if fg_verts else None

    def draw_hud(self):
        """ Renders the HUD text using the fast GPU FontRenderer """

        if not self.show_hud:
            return

        current_time = pygame.time.get_ticks()
        self.fps_rolling.append(self.clock.get_fps())

        # Throttled update of the text content (string formatting)
        if current_time - self._last_hud_update_time > self.hud_update_interval_ms:
            self._last_hud_update_time = current_time
            avg_fps = np.mean(self.fps_rolling) if self.fps_rolling else 0

            # Update info string
            is_raytracer = isinstance(self.compound_eye, EyeRendererRay)
            mode_name = "Ray-tracer" if is_raytracer else "Rasterizer"
            sample_label = "Rays" if is_raytracer else "Samples"
            pos = self.camera.position
            num_om = getattr(self.compound_eye, 'num_ommatidia', 0)
            num_samples = getattr(self.compound_eye, 'samples_per_ommatidium', 1)
            total_samples = num_om * num_samples
            self._hud_info_text = (
                f'FPS: {avg_fps:>4.2f} | Mode: {mode_name} | Ommatidia: {num_om} | '
                f'{sample_label}: {num_samples} | XYZ: [ {pos.x:>5.3f}, {pos.y:>5.3f}, {pos.z:>5.3f} ]'
            )

            # Generate vertices for the info string
            margin = 10
            line_height = self.font_renderer.font_size * 1.1
            info_sv = self.font_renderer._generate_text_vertices(self._hud_info_text, margin + 1,
                                                                 self.height - line_height - 1)
            info_fv = self.font_renderer._generate_text_vertices(self._hud_info_text, margin, self.height - line_height)
            self._info_shadow_verts = np.array(info_sv, dtype=np.float32) if info_sv else None
            self._info_fg_verts = np.array(info_fv, dtype=np.float32) if info_fv else None

            #Add scene stats
            if self.scene.point_cloud and self.scene.point_cloud.num_points > 0:
                # if there's a point cloud, label things clearly
                self._scene_stats_lines = [
                    f'Total {sample_label}: {total_samples:,}',
                    f'Scene Triangles: {self._total_scene_triangles:,}',  # will be 0 if only points
                    f'Scene Points: {self.scene.point_cloud.num_points:,}',
                ]
            else:
                # mesh-only scenes
                self._scene_stats_lines = [
                    f'Total {sample_label}: {total_samples:,}',
                    f'Scene Triangles: {self._total_scene_triangles:,}',
                    f'Scene Vertices: {self._total_scene_vertices:,}',
                ]

            stats_sv = []
            stats_fv = []
            for i, text in enumerate(self._scene_stats_lines):
                text_width = self.font_renderer.get_text_width(text)
                x_pos = self.width - text_width - margin
                y_pos = margin + (i * line_height)
                stats_sv.extend(self.font_renderer._generate_text_vertices(text, x_pos + 1, y_pos - 1))
                stats_fv.extend(self.font_renderer._generate_text_vertices(text, x_pos, y_pos))
            self._stats_shadow_verts = np.array(stats_sv, dtype=np.float32) if stats_sv else None
            self._stats_fg_verts = np.array(stats_fv, dtype=np.float32) if stats_fv else None

        # Combine cached vertex arrays for a single draw call per pass
        all_shadow_verts = [v for v in
                            (self._info_shadow_verts, self._controls_shadow_verts, self._stats_shadow_verts) if
                            v is not None]
        all_fg_verts = [v for v in (self._info_fg_verts, self._controls_fg_verts, self._stats_fg_verts) if
                        v is not None]

        if not all_shadow_verts and not all_fg_verts:
            return

        # Setup GL state
        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_CULL_FACE)
        glUseProgram(self.font_renderer.text_program)
        glUniformMatrix4fv(self.font_renderer.proj_loc, 1, False, glm.value_ptr(self.hud_projection_matrix))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.font_renderer.atlas_texture)
        glUniform1i(self.font_renderer.atlas_loc, 0)
        glBindVertexArray(self.font_renderer.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.font_renderer.vbo)

        # Draw shadows
        if all_shadow_verts:
            shadow_data = np.concatenate(all_shadow_verts)
            glUniform4f(self.font_renderer.color_loc, 0.0, 0.0, 0.0, 0.7)
            glBufferData(GL_ARRAY_BUFFER, shadow_data.nbytes, shadow_data, GL_DYNAMIC_DRAW)
            glDrawArrays(GL_TRIANGLES, 0, len(shadow_data) // 4)

        # Draw foreground
        if all_fg_verts:
            fg_data = np.concatenate(all_fg_verts)
            glUniform4f(self.font_renderer.color_loc, 1.0, 1.0, 1.0, 1.0)
            glBufferData(GL_ARRAY_BUFFER, fg_data.nbytes, fg_data, GL_DYNAMIC_DRAW)
            glDrawArrays(GL_TRIANGLES, 0, len(fg_data) // 4)

        # Restore GL state
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        glUseProgram(0)
        glEnable(GL_CULL_FACE)
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

    def add_point_cloud(self, point_cloud: PointCloud):
        self.scene.add_point_cloud(point_cloud)
        self._update_geometry_counts()

    def _update_geometry_counts(self):
        """ Recalculates total primitives from all geometry in the scene """

        self._total_scene_vertices = 0
        self._total_scene_triangles = 0

        # Count vertices and triangles from standard mesh instances
        mesh_vert_count = 0
        for instance in self.scene.instances:
            mesh_vert_count += instance.asset.draw_count

        self._total_scene_vertices += mesh_vert_count
        self._total_scene_triangles = mesh_vert_count // 3

        # If a point cloud exists add its points to the vertex count
        # (for display purposes, treats "points" as a type of "vertex")
        if self.scene.point_cloud:
            self._total_scene_vertices += self.scene.point_cloud.num_points

    def _render_instance(self, instance, view_matrix, projection_matrix):
        """ Renders a single instance using provided view and projection matrices """

        mesh = instance.asset

        try:
            glUseProgram(mesh.shaders)
            glBindVertexArray(mesh.vao)

            # For column-major, final matrix is P * V * M
            camera_matrix = projection_matrix * view_matrix

            glUniformMatrix4fv(glGetUniformLocation(mesh.shaders, "camera"),
                               1, False, glm.value_ptr(camera_matrix))
            glUniformMatrix4fv(glGetUniformLocation(mesh.shaders, "model"),
                               1, False, glm.value_ptr(instance.transform))

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
            view = glm.lookAt(
                glm.vec3(camera.position),
                glm.vec3(camera.position + lookat_directions[i]),  # target point is eye + direction
                glm.vec3(ups[i])
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
        cam_displacement = glm.vec3(0.0)

        if keys[K_w]: cam_displacement += self.camera.forward
        if keys[K_s]: cam_displacement += self.camera.backward
        if keys[K_a]: cam_displacement += self.camera.left
        if keys[K_d]: cam_displacement += self.camera.right
        if keys[K_SPACE]: cam_displacement += WORLD_UP
        if keys[K_LSHIFT]: cam_displacement += WORLD_DOWN

        # Normalize (prevents faster diagonal movement) and apply speed
        if glm.length(cam_displacement) > 0:
            cam_displacement = glm.normalize(cam_displacement) * self.move_sensitivity
            self.camera.pos += cam_displacement

        # Handle mouse look
        mouse_x, mouse_y = pygame.mouse.get_rel()
        if mouse_x != 0 or mouse_y != 0:
            self.camera.yaw -= mouse_x * self.mouse_sensitivity
            self.camera.pitch -= mouse_y * self.mouse_sensitivity
            self.camera.pitch = np.clip(self.camera.pitch, -89.0, 89.0)

    def close(self):
        """ Frees all allocated resources """
        self.scene.free()
        if self._cubemap_fbo:
            self._cubemap_fbo.free()
        pygame.quit()