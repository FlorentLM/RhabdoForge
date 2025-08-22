import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import OpenGL
OpenGL.ERROR_CHECKING = False

import string
from collections import deque

import numpy as np
import pygame

from OpenGL.GL import *
from pyglm import glm

from graphics.utils import load_shaders
from graphics.renderers.raytracer import EyeRendererRay


class FontRenderer:
    """ Renders text on the GPU using a font atlas """

    def __init__(self, font_name, font_size):
        self.font_name = font_name
        self.font_size = font_size
        self.char_data = {}  # to store metrics and UVs for each character

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

    def generate_text_vertices(self, text, x, y, scale=1.0):
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


class HUD:
    """ Manages the rendering of all HUD elements """

    # TODO: Comment this class a bit more

    def __init__(self, context):
        self.ctx = context

        self.width = self.ctx.window_size[0]
        self.height = self.ctx.window_size[1]

        self.show = True

        self.font_renderer = FontRenderer(pygame.font.get_default_font(), 22)
        self.projection_matrix = glm.ortho(0, self.width, 0, self.height, -1.0, 1.0)
        self.fps_rolling = deque(maxlen=5)
        self.update_interval_ms = 250
        self._last_update_time = 0
        self._info_text = ""
        self._controls_text_lines = []
        self._stats_text_lines = []
        self._info_shadow_verts, self._info_fg_verts = None, None
        self._controls_shadow_verts, self._controls_fg_verts = None, None
        self._stats_shadow_verts, self._stats_fg_verts = None, None

    def _update_controls_text(self):
        # The main renderer (not the debug one) determines the "sample" label
        sample_label = "Rays" if self.ctx.renderer and isinstance(self.ctx.renderer, EyeRendererRay) else "Samples"

        self._controls_text_lines = [
            'ESC: Quit', 'H: Show/hide HUD', f'+/-: {sample_label}', 'T: Time dithering',
            'V: Voronoi view', 'P: Panoramic view', 'C: Compound eye view',
            'Mouse: Look', 'LShift/Space: Down/Up', 'WASD: Move',
        ]
        shadow_verts, fg_verts = [], []
        margin, line_height = 10, self.font_renderer.font_size

        for i, text in enumerate(self._controls_text_lines):
            y_pos = margin + (i * line_height)
            shadow_verts.extend(self.font_renderer.generate_text_vertices(text, margin + 1, y_pos - 1))
            fg_verts.extend(self.font_renderer.generate_text_vertices(text, margin, y_pos))

        self._controls_shadow_verts = np.array(shadow_verts, dtype=np.float32) if shadow_verts else None
        self._controls_fg_verts = np.array(fg_verts, dtype=np.float32) if fg_verts else None

    def _update_text_vertices(self):
        current_time = pygame.time.get_ticks()

        self.fps_rolling.append(self.ctx.clock.get_fps())

        if current_time - self._last_update_time > self.update_interval_ms:
            self._last_update_time = current_time

            avg_fps = np.mean(self.fps_rolling) if self.fps_rolling else 0

            active_renderer = self.ctx.active_renderer

            is_raytracer = isinstance(active_renderer, EyeRendererRay)

            mode_name = "Ray-Tracer" if is_raytracer else "Rasterizer"
            sample_label = "Rays" if is_raytracer else "Samples"

            pos = self.ctx.agent.position

            num_om = getattr(active_renderer, 'num_ommatidia', 0)
            num_samples = getattr(active_renderer, 'samples_per_ommatidium', 1)

            total_samples = num_om * num_samples
            self._info_text = (f'FPS: {avg_fps:>4.2f} | Mode: {mode_name} | Ommatidia: {num_om} | '
                             f'{sample_label}: {num_samples} | XYZ: [ {pos.x:>5.3f}, {pos.y:>5.3f}, {pos.z:>5.3f} ]')

            margin, line_height = 10, self.font_renderer.font_size * 1.1
            info_sv = self.font_renderer.generate_text_vertices(self._info_text, margin + 1,
                                                                self.height - line_height - 1)
            info_fv = self.font_renderer.generate_text_vertices(self._info_text, margin, self.height - line_height)
            self._info_shadow_verts = np.array(info_sv, dtype=np.float32) if info_sv else None
            self._info_fg_verts = np.array(info_fv, dtype=np.float32) if info_fv else None

            # Get scene stats from the renderer's scene data representation
            renderer_data = getattr(active_renderer, 'rt_scene', None)
            tri_count = getattr(renderer_data, 'num_total_triangles', 0)
            point_count = getattr(renderer_data, 'num_total_points', 0)


            if point_count > 0:
                self._stats_text_lines = [f'Total {sample_label}: {total_samples:,}',
                                        f'Scene Triangles: {tri_count:,}',
                                        f'Scene Points: {point_count:,}']
            else:
                self._stats_text_lines = [f'Total {sample_label}: {total_samples:,}',
                                        f'Scene Triangles: {tri_count:,}']

            stats_sv, stats_fv = [], []

            for i, text in enumerate(self._stats_text_lines):
                text_width = self.font_renderer.get_text_width(text)
                x_pos = self.width - text_width - margin
                y_pos = margin + (i * line_height)
                stats_sv.extend(self.font_renderer.generate_text_vertices(text, x_pos + 1, y_pos - 1))
                stats_fv.extend(self.font_renderer.generate_text_vertices(text, x_pos, y_pos))

            self._stats_shadow_verts = np.array(stats_sv, dtype=np.float32) if stats_sv else None
            self._stats_fg_verts = np.array(stats_fv, dtype=np.float32) if stats_fv else None

    def draw(self):

        if not self.show:
            return

        self._update_text_vertices()

        all_shadow_verts = [v for v in (self._info_shadow_verts, self._controls_shadow_verts, self._stats_shadow_verts) if v is not None]
        all_fg_verts = [v for v in (self._info_fg_verts, self._controls_fg_verts, self._stats_fg_verts) if v is not None]

        if not all_shadow_verts and not all_fg_verts:
            return

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_CULL_FACE)

        glUseProgram(self.font_renderer.text_program)
        glUniformMatrix4fv(self.font_renderer.proj_loc, 1, False, glm.value_ptr(self.projection_matrix))
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_2D, self.font_renderer.atlas_texture)
        glUniform1i(self.font_renderer.atlas_loc, 0)
        glBindVertexArray(self.font_renderer.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.font_renderer.vbo)

        if all_shadow_verts:
            shadow_data = np.concatenate(all_shadow_verts).flatten()
            glUniform4f(self.font_renderer.color_loc, 0.0, 0.0, 0.0, 0.7)
            glBufferData(GL_ARRAY_BUFFER, shadow_data.nbytes, shadow_data, GL_DYNAMIC_DRAW)
            glDrawArrays(GL_TRIANGLES, 0, len(shadow_data) // 4)

        if all_fg_verts:
            fg_data = np.concatenate(all_fg_verts).flatten()
            glUniform4f(self.font_renderer.color_loc, 1.0, 1.0, 1.0, 1.0)
            glBufferData(GL_ARRAY_BUFFER, fg_data.nbytes, fg_data, GL_DYNAMIC_DRAW)
            glDrawArrays(GL_TRIANGLES, 0, len(fg_data) // 4)

        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        glUseProgram(0)
        glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)

    def free(self):
        self.font_renderer.free()