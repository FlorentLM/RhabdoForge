import OpenGL

OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *

from pathlib import Path
from collections import deque
import numpy as np
import json
from PIL import Image
from pyglm import glm

from insectvision.engine.shader_utils import ShaderProgram


def generate_font_atlas(font_name=None, font_size=22, output_dir='interactive/fonts', color=(255, 255, 255, 255)):
    """
    Generates a fonts atlas texture and its corresponding metadata file.
    """
    from os import environ
    environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
    try:
        import pygame
    except ImportError:
        raise ImportError("'pygame' package required for pygame texture generation")
    import string
    import json

    pygame.init()

    if font_name is None:
        font_name = pygame.font.get_default_font()

    print(f"Generating fonts atlas for '{font_name}' (size {font_size})...")

    font = pygame.font.SysFont(font_name, font_size)

    chars_to_render = string.printable
    atlas_cols = 16
    atlas_rows = (len(chars_to_render) + atlas_cols - 1) // atlas_cols
    char_data = {}

    max_w, max_h = 0, 0
    for char in chars_to_render:
        w, h = font.size(char)
        if w > max_w: max_w = w
        if h > max_h: max_h = h

    cell_w, cell_h = max_w, max_h
    atlas_width = atlas_cols * cell_w
    atlas_height = atlas_rows * cell_h

    atlas_surface = pygame.Surface((atlas_width, atlas_height), pygame.SRCALPHA)
    atlas_surface.fill((0, 0, 0, 0))

    for i, char in enumerate(chars_to_render):
        char_surface = font.render(char, True, color)
        metrics = font.metrics(char)[0]
        advance = metrics[4]

        col = i % atlas_cols
        row = i // atlas_cols
        x, y = col * cell_w, row * cell_h

        atlas_surface.blit(char_surface, (x, y))

        # Store character metadata
        uv_x0 = x / atlas_width
        uv_y0 = y / atlas_height
        uv_x1 = (x + char_surface.get_width()) / atlas_width
        uv_y1 = (y + char_surface.get_height()) / atlas_height

        char_data[char] = {
            'w': char_surface.get_width(),
            'h': char_surface.get_height(),
            'uv_rect': (uv_x0, uv_y0, uv_x1, uv_y1),
            'advance': advance
        }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_path = output_dir / f'{font_name}.png'
    json_path = output_dir / f'{font_name}.json'

    pygame.image.save(atlas_surface, image_path)

    with open(json_path, 'w') as f:
        json.dump({
            'font_name': font_name,
            'font_size': font_size,
            'atlas_image': f'{font_name}.png',
            'char_data': char_data
        }, f, indent=4)

    print(f"Saved fonts atlas for '{font_name}' (size {font_size}).")

    pygame.quit()


class FontRenderer:
    """Renders text on the GPU using a fonts atlas."""

    def __init__(self):
        self.char_data = {}
        self.text_program = ShaderProgram(vert_path='text.vert', frag_path='text.frag')

        self.proj_loc = self.text_program.get_loc("projection")
        self.color_loc = self.text_program.get_loc("textColor")
        self.atlas_loc = self.text_program.get_loc("fontAtlas")

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
        """
        Loads atlas metadata from JSON and the texture from the associated PNG file.
        """

        atlas_dir = Path('insectvision/interactive/fonts')

        json_path = (atlas_dir / f'{font_name}.json')
        if not json_path.exists():
            generate_font_atlas(font_name=font_name, font_size=22, output_dir=atlas_dir)

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

    def text_width(self, text, scale=1.0):
        """
        Returns pixel width of a string based on the fonts atlas.
        """
        width = 0
        for char in text:
            if char in self.char_data:
                width += self.char_data[char]['advance'] * scale
        return width

    def text_vertices(self, text, x, y, scale=1.0):
        """
        Generates vertex data for a string and returns it as a list.
        """
        vertices = []
        cursor_x = x
        for char in text:
            if char in self.char_data:
                data = self.char_data[char]
                w, h = data['w'] * scale, data['h'] * scale
                u0, v0, u1, v1 = data['uv_rect']

                bl = (cursor_x, y, u0, v1)
                br = (cursor_x + w, y, u1, v1)
                tr = (cursor_x + w, y + h, u1, v0)
                tl = (cursor_x, y + h, u0, v0)

                vertices.extend(bl)
                vertices.extend(br)
                vertices.extend(tr)
                vertices.extend(bl)
                vertices.extend(tr)
                vertices.extend(tl)

                cursor_x += data['advance'] * scale
        return vertices

    def free(self):
        self.text_program.free()
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])
        glDeleteTextures(1, [self.atlas_texture])


class HUD:
    """
    Manages the rendering of all HUD elements.
    """

    def __init__(self, context):
        self.ctx = context

        self.width = self.ctx.window_size[0]
        self.height = self.ctx.window_size[1]
        self.nb_px = self.width * self.height

        self.show = True

        self.font_renderer = FontRenderer()

        self.projection_matrix = glm.ortho(0, self.width, 0, self.height, -1.0, 1.0)

        self.update_interval = 0.25  # update text every 250 ms
        self._last_update_time = 0

        self._controls_text_lines = []
        self._controls_shadow_verts, self._controls_fg_verts = None, None
        self._info_shadow_verts, self._info_fg_verts = None, None
        self._stats_shadow_verts, self._stats_fg_verts = None, None

        self._update_controls_text()

    def _build_text_buffers(self, text_lines, x_align='left', y_start=10):
        """Helper to generate Shadow and FG vertices for a block of text."""

        shadow_verts, fg_verts = [], []
        line_height = self.font_renderer.font_size * 1.1
        margin = 10

        for i, text in enumerate(text_lines):
            y_pos = y_start + (i * line_height)

            if x_align == 'right':
                text_width = self.font_renderer.text_width(text)
                x_pos = self.width - text_width - margin
            else:
                x_pos = margin

            shadow_verts.extend(self.font_renderer.text_vertices(text, x_pos + 1, y_pos - 1))
            fg_verts.extend(self.font_renderer.text_vertices(text, x_pos, y_pos))

        return (
            np.array(shadow_verts, dtype=np.float32) if shadow_verts else None,
            np.array(fg_verts, dtype=np.float32) if fg_verts else None
        )

    def _update_controls_text(self):

        lines = [
            'Movement:',
            '    WASD: Move',
            '    Mouse: Look / Pan',
            '    Space / Ctrl: Up / Down',
            '    L-Shift (hold): Strafe (3rd person)'
        ]

        categories = ['Rendering', 'Sampling', 'Environment', 'Dynamics', 'Agent', 'UI']

        for cat in categories:
            actions = self.ctx.actions.get_by_category(cat)
            if not actions:
                continue

            lines.append(f'{cat}:')

            for a in actions:
                if a.keyboard_hint:
                    lines.append(f'    {a.keyboard_hint}: {a.name}')

        lines.append('    ESC: Quit')

        self._controls_text_lines = reversed(lines)
        self._controls_shadow_verts, self._controls_fg_verts = self._build_text_buffers(
            self._controls_text_lines, x_align='left', y_start=10
        )

    def _update_text_vertices(self):

        from insectvision.renderers import Raytracer, Pathtracer

        current_time = self.ctx.current_time
        if current_time - self._last_update_time < self.update_interval:
            return
        self._last_update_time = current_time

        pos = self.ctx.agent.position

        is_ray_based = isinstance(self.ctx.renderer, Raytracer)
        if isinstance(self.ctx.renderer, Pathtracer):
            renderer_name = 'Pathtracer'
        elif is_ray_based:
            renderer_name = 'Raytracer'
        else:
            renderer_name = 'Rasterizer'

        sample_label = 'Rays' if is_ray_based else 'Samples'

        # Enums to readable strings
        view_mode_str = self.ctx.view_mode.name.replace('_', ' ')
        proj_mode_str = self.ctx.renderer.projection_mode.name

        # Stats
        nb_om = self.ctx.renderer._ra.lens_count
        nb_om_samples = getattr(self.ctx.renderer, 'nb_samples', 1)
        nb_px_samples = getattr(self.ctx.renderer, 'samples_per_pixel', 1)
        has_pixels = self.ctx.view_mode.name in ('Panoramic', 'Third_person')

        # Bottom info line
        info_text = (
            f'FPS: {self.ctx.fps:5.1f} | '
            f'Renderer: {renderer_name} | '
            f'View: {view_mode_str} | '
            f'Proj: {proj_mode_str} | '
            f'Pos: [{pos.x:5.2f}, {pos.y:5.2f}, {pos.z:5.2f}]'
        )

        line_height = self.font_renderer.font_size * 1.1
        self._info_shadow_verts, self._info_fg_verts = self._build_text_buffers(
            [info_text], x_align='left', y_start=self.height - line_height
        )

        # Top-right scene stats
        samples_pp_str = f", {nb_px_samples}/px" if has_pixels else ""
        total_pp = f" (om) + {nb_px_samples * self.nb_px:,} (px)" if has_pixels else ""

        stats_lines = [
            f'Ommatidia: {nb_om:,}',
            f'{sample_label}: {nb_om_samples}/om{samples_pp_str}',
            f'Total {sample_label}: {nb_om * nb_om_samples:,}{total_pp}'
        ]

        tot_tris = self.ctx.scene.total_triangles
        if tot_tris > 0:
            stats_lines.append(f'Triangles: {tot_tris:,}')

        tot_points = self.ctx.scene.total_points
        if tot_points > 0:
            stats_lines.append(f'Points: {tot_points:,}')

        self._stats_shadow_verts, self._stats_fg_verts = self._build_text_buffers(
            stats_lines, x_align='right', y_start=10
        )

    def draw(self):

        if not self.show:
            return

        self._update_text_vertices()

        all_shadow_verts = [v for v in (self._info_shadow_verts, self._controls_shadow_verts, self._stats_shadow_verts)
                            if v is not None]
        all_fg_verts = [v for v in (self._info_fg_verts, self._controls_fg_verts, self._stats_fg_verts) if
                        v is not None]

        if not all_shadow_verts and not all_fg_verts:
            return

        glDisable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_CULL_FACE)

        self.font_renderer.text_program.use()

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

        self.font_renderer.text_program.stop()

        glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
        glDisable(GL_BLEND)

    def free(self):
        self.font_renderer.free()