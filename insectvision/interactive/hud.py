import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *

from pathlib import Path
from collections import deque
import numpy as np
import json

from PIL import Image
from pyglm import glm

from insectvision.engine.utils import load_shaders, generate_font_atlas
from insectvision.renderers.raytracer import Raytracer, Pathtracer


class FontRenderer:
    """Renders text on the GPU using a fonts atlas."""

    def __init__(self):
        self.char_data = {}
        self.text_program = load_shaders('shaders/text.vert', 'shaders/text.frag')

        self.proj_loc = glGetUniformLocation(self.text_program, "projection")
        self.color_loc = glGetUniformLocation(self.text_program, "textColor")
        self.atlas_loc = glGetUniformLocation(self.text_program, "fontAtlas")

        self._load_atlas_data()

        # VAO and VBO for text quads
        self.vao = glGenVertexArrays(1)
        self.vbo = glGenBuffers(1)
        glBindVertexArray(self.vao)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        # Data is buffered on the fly so just set up attributes here
        # Each vertex has x, y, u, v floats
        glBufferData(GL_ARRAY_BUFFER, 0, None, GL_DYNAMIC_DRAW)
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(GLfloat), ctypes.c_void_p(0))
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)

    def _load_atlas_data(self, font_name='freesansbold.ttf'):
        """Loads atlas metadata from JSON and the texture from the associated PNG file."""

        atlas_dir = Path('insectvision/interactive/fonts')

        json_path = (atlas_dir / f'{font_name}.json')
        if not json_path.exists():
            generate_font_atlas(font_name=font_name, font_size=22, output_dir=atlas_dir)

        # Load metadata from JSON
        with json_path.open(encoding="UTF-8") as f:
            data = json.load(f)
            self.char_data = data['char_data']
            self.font_size = data['font_size']
            image_filename = data['atlas_image']

        # Load image texture
        image_path = atlas_dir / image_filename
        image = Image.open(image_path).convert("RGBA")

        image_data = image.tobytes()
        atlas_width, atlas_height = image.size

        # Create OpenGL Texture
        self.atlas_texture = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D, self.atlas_texture)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)

        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, atlas_width, atlas_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image_data)
        glBindTexture(GL_TEXTURE_2D, 0)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4)  # Reset to default

    def get_text_width(self, text, scale=1.0):
        """Calculates the pixel width of a string based on the fonts atlas."""
        width = 0
        for char in text:
            if char in self.char_data:
                width += self.char_data[char]['advance'] * scale
        return width

    def generate_text_vertices(self, text, x, y, scale=1.0):
        """Generates vertex data for a string and returns it as a list."""
        vertices = []
        cursor_x = x
        for char in text:
            if char in self.char_data:
                data = self.char_data[char]
                w, h = data['w'] * scale, data['h'] * scale

                u0, v0, u1, v1 = data['uv_rect']

                # Quad corners
                bl = (cursor_x, y, u0, v1)          # bottom-Left
                br = (cursor_x + w, y, u1, v1)      # bottom-Right
                tr = (cursor_x + w, y + h, u1, v0)  # top-Right
                tl = (cursor_x, y + h, u0, v0)      # top-Left

                # Triangle 1
                vertices.extend(bl)
                vertices.extend(br)
                vertices.extend(tr)

                # Triangle 2
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
    """Manages the rendering of all HUD elements."""

    # TODO: Comment this class a bit more

    def __init__(self, context):
        self.ctx = context

        self.width = self.ctx.window_size[0]
        self.height = self.ctx.window_size[1]
        self.nb_px = self.width * self.height

        self.show = True

        self.font_renderer = FontRenderer()

        self.projection_matrix = glm.ortho(0, self.width, 0, self.height, -1.0, 1.0)

        self.view_modes_names = {0: 'Compound eye', 1: 'Panoramic', 2: 'Third person', 3: 'Perspective'}
        self.proj_modes_names = {0: 'Layout', 1: 'Acceptance'}

        # FPS tracking
        self.frame_times = deque(maxlen=60)
        self.last_fps_update_time = 0

        self.update_interval = 0.25  # update text every 250 ms
        self._last_update_time = 0
        self._info_text = ""
        self._controls_text_lines = []
        self._stats_text_lines = []
        self._info_shadow_verts, self._info_fg_verts = None, None
        self._controls_shadow_verts, self._controls_fg_verts = None, None
        self._stats_shadow_verts, self._stats_fg_verts = None, None

        self._update_controls_text()

    def _update_controls_text(self):

        sample_label = "rays" if self.ctx.renderer and isinstance(self.ctx.renderer, Raytracer) else "samples"

        self._controls_text_lines = reversed([
            'Controls:',
            '------',
            'WASD: Move',
            'Mouse: Yaw & Pitch',
            'Q/E: Roll',
            'Ctrl/Space: Down/Up',
            'C: Change view',
            'P: Projection mode (acceptance / layout)',
            'G: Toggle debug objects',
            'V: Toggle Voronoi view',
            'H: Toggle Heatmap mode',
            'T: Toggle time dithering',
            'L: Toggle Sun control',
            f'+/-: Increase/decrease {sample_label}',
            'X: Dither once',
            'O: Teleport to origin',
            'R: Reset rotation',
            'I: Show/hide HUD',
            'ESC: Quit'
        ])

        shadow_verts, fg_verts = [], []
        margin, line_height = 10, self.font_renderer.font_size

        for i, text in enumerate(self._controls_text_lines):
            y_pos = margin + (i * line_height)
            shadow_verts.extend(self.font_renderer.generate_text_vertices(text, margin + 1, y_pos - 1))
            fg_verts.extend(self.font_renderer.generate_text_vertices(text, margin, y_pos))

        self._controls_shadow_verts = np.array(shadow_verts, dtype=np.float32) if shadow_verts else None
        self._controls_fg_verts = np.array(fg_verts, dtype=np.float32) if fg_verts else None

    def _update_text_vertices(self):

        current_time = self.ctx.current_time

        self.frame_times.append(current_time)

        if current_time - self._last_update_time > self.update_interval:
            self._last_update_time = current_time

            # Calculate FPS based on timestamps
            if len(self.frame_times) > 1:
                time_diff = self.frame_times[-1] - self.frame_times[0]
                avg_fps = (len(self.frame_times) - 1) / time_diff if time_diff > 0 else 0
            else:
                avg_fps = 0

            active_renderer = self.ctx.renderer

            is_ray_based = isinstance(active_renderer, Raytracer)

            if is_ray_based and isinstance(active_renderer, Pathtracer):
                render_mode = "Path-tracer"
            elif is_ray_based:
                render_mode = "Ray-tracer"
            else:
                render_mode = "Rasterizer"
            sample_label = "Rays" if is_ray_based else "Samples"

            view_mode = self.view_modes_names[self.ctx.view_mode]

            has_pixels = view_mode in ('Panoramic', 'Third person')

            pos = self.ctx.agent.position

            nb_om = getattr(active_renderer, 'num_ommatidia', 0)
            nb_om_samples = getattr(active_renderer, 'samples_per_ommatidium', 1)
            nb_px_samples = getattr(active_renderer, 'samples_per_pixel', 1)

            samples_pp_str = f', {nb_px_samples}/px' if has_pixels else ''

            proj_mode = self.proj_modes_names[active_renderer.projection_mode]

            self._info_text = (
                f'FPS: {avg_fps:>4.2f} | '
                f'Mode: {render_mode} | '
                f'Ommatidia: {nb_om} | '
                f'{sample_label}: {nb_om_samples}/om{samples_pp_str} | '
                f'XYZ: [ {pos.x:>5.3f}, {pos.y:>5.3f}, {pos.z:>5.3f} ] | '
                f'View mode: {view_mode} | '
                f'Projection mode: {proj_mode}'
                # f' | Sun: azm={self.ctx.renderer.sun.azimuth:.2f}, elv={self.ctx.renderer.sun.elevation:.2f}'
            )

            margin, line_height = 10, self.font_renderer.font_size * 1.1
            info_sv = self.font_renderer.generate_text_vertices(self._info_text, margin + 1,
                                                                self.height - line_height - 1)
            info_fv = self.font_renderer.generate_text_vertices(self._info_text, margin, self.height - line_height)
            self._info_shadow_verts = np.array(info_sv, dtype=np.float32) if info_sv else None
            self._info_fg_verts = np.array(info_fv, dtype=np.float32) if info_fv else None

            # Scene stats
            tri_count = self.ctx.scene.total_triangles
            point_count = self.ctx.scene.total_points

            total_pp = f' (om) + {nb_px_samples * self.nb_px:,} (px)' if has_pixels else ''

            if point_count > 0:
                self._stats_text_lines = [f'Total {sample_label}: {nb_om * nb_om_samples:,}{total_pp}',
                                        f'Scene Triangles: {tri_count:,}',
                                        f'Scene Points: {point_count:,}']
            else:
                self._stats_text_lines = [f'Total {sample_label}: {nb_om * nb_om_samples:,}{total_pp}',
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