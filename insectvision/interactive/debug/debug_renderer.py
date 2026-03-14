import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *

import ctypes
import numpy as np
from pyglm import glm

from insectvision.engine.shader_utils import ShaderProgram


class DebugRenderer:
    """
    Very simple classic renderer to display debug items.
    """

    # Each vertex is 6 floats (xyz rgb)
    VERTEX_STRIDE = 6 * 4  # bytes

    def __init__(self, initial_capacity: int = 65536):

        self._color_shader = ShaderProgram(vert_path='color.vert', frag_path='color.frag')
        self._billboard_shader = ShaderProgram(vert_path='billboard.vert', frag_path='billboard.frag')

        # Dynamic VBO + VAO
        self._vao = glGenVertexArrays(1)
        self._vbo = glGenBuffers(1)
        self._capacity = initial_capacity  # floats

        glBindVertexArray(self._vao)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferData(GL_ARRAY_BUFFER, self._capacity * 4, None, GL_DYNAMIC_DRAW)

        # pos
        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, self.VERTEX_STRIDE, ctypes.c_void_p(0))

        # colour
        glEnableVertexAttribArray(1)
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, self.VERTEX_STRIDE, ctypes.c_void_p(12))

        glBindVertexArray(0)

        # per-frame queues
        self._line_batches: list[tuple[np.ndarray, glm.mat4, float, float]] = []
        self._tri_batches: list[tuple[np.ndarray, glm.mat4, float]] = []
        self._billboard_batches: list[tuple[np.ndarray, glm.vec3, float]] = []

    def submit_lines(self,
            data: np.ndarray,
            model: glm.mat4 = None,
            alpha: float = 1.0,
            line_width: float = 1.5
        ):
        """Queue interleaved [xyz rgb ...] line data for current frame."""

        if data.size == 0:
            return
        if model is None:
            model = glm.mat4(1.0)
        self._line_batches.append((data, model, alpha, line_width))

    def submit_tris(self,
            data: np.ndarray,
            model: glm.mat4 = None,
            alpha: float = 1.0
        ):
        """Queue interleaved [xyz rgb ...] triangle data for current frame."""

        if data.size == 0:
            return
        if model is None:
            model = glm.mat4(1.0)
        self._tri_batches.append((data, model, alpha))

    def submit_billboard_lines(self,
            data: np.ndarray,
            world_pos: glm.vec3,
            scale: float = 0.00
        ):
        """Queue billboard line data at world_pos."""

        if data.size == 0:
            return
        self._billboard_batches.append((data, world_pos, scale))

    def flush(self, view: glm.mat4, proj: glm.mat4, viewport: tuple = None):
        """
        Draw all queued geometry and clear queues
        """
        if not self._line_batches and not self._tri_batches and not self._billboard_batches:
            return

        # Snapshot all GL states that this method touches
        prev_fbo = glGetIntegerv(GL_FRAMEBUFFER_BINDING)
        prev_viewport = glGetIntegerv(GL_VIEWPORT)
        prev_program = glGetIntegerv(GL_CURRENT_PROGRAM)
        prev_vao = glGetIntegerv(GL_VERTEX_ARRAY_BINDING)
        prev_vbo = glGetIntegerv(GL_ARRAY_BUFFER_BINDING)
        prev_line_width = glGetFloat(GL_LINE_WIDTH)
        depth_was_on = glIsEnabled(GL_DEPTH_TEST)
        blend_was_on = glIsEnabled(GL_BLEND)
        prev_blend_src = glGetIntegerv(GL_BLEND_SRC_RGB)
        prev_blend_dst = glGetIntegerv(GL_BLEND_DST_RGB)

        # Force clean
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        if viewport is not None:
            glViewport(int(viewport[0]), int(viewport[1]),
                       int(viewport[2]), int(viewport[3]))

        glEnable(GL_DEPTH_TEST)
        glDepthFunc(GL_LESS)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        cull_was_on = glIsEnabled(GL_CULL_FACE)
        glDisable(GL_CULL_FACE)

        glBindVertexArray(self._vao)

        # Worldspace lines and triangles (colour shader)
        if self._line_batches or self._tri_batches:
            cs = self._color_shader
            cs.use()
            glUniformMatrix4fv(cs.get_loc('uView'), 1, GL_FALSE, glm.value_ptr(view))
            glUniformMatrix4fv(cs.get_loc('uProj'), 1, GL_FALSE, glm.value_ptr(proj))

            for data, model, alpha, lw in self._line_batches:
                glLineWidth(lw)
                self._upload_and_draw(data, GL_LINES, cs, model, alpha)

            for data, model, alpha in self._tri_batches:
                self._upload_and_draw(data, GL_TRIANGLES, cs, model, alpha)

        # billboard lines
        if self._billboard_batches:
            bs = self._billboard_shader
            bs.use()
            glUniformMatrix4fv(bs.get_loc('uView'), 1, GL_FALSE, glm.value_ptr(view))
            glUniformMatrix4fv(bs.get_loc('uProj'), 1, GL_FALSE, glm.value_ptr(proj))

            glDisable(GL_DEPTH_TEST)

            for data, world_pos, scale in self._billboard_batches:
                glUniform3f(bs.get_loc('uWorldPos'), world_pos.x, world_pos.y, world_pos.z)
                glUniform1f(bs.get_loc('uScale'), scale)
                self._upload_and_draw(data, GL_LINES)

        # Restore everything
        glBindVertexArray(int(prev_vao))
        glBindBuffer(GL_ARRAY_BUFFER, int(prev_vbo))
        glUseProgram(int(prev_program))
        glBindFramebuffer(GL_FRAMEBUFFER, int(prev_fbo))
        glViewport(int(prev_viewport[0]), int(prev_viewport[1]),
                   int(prev_viewport[2]), int(prev_viewport[3]))
        glLineWidth(prev_line_width)

        if cull_was_on:
            glEnable(GL_CULL_FACE)

        if depth_was_on:
            glEnable(GL_DEPTH_TEST)
        else:
            glDisable(GL_DEPTH_TEST)

        if blend_was_on:
            glEnable(GL_BLEND)
            glBlendFunc(int(prev_blend_src), int(prev_blend_dst))
        else:
            glDisable(GL_BLEND)

        # clear queues for next frame
        self._line_batches.clear()
        self._tri_batches.clear()
        self._billboard_batches.clear()

    def free(self):
        if self._vao:
            glDeleteVertexArrays(1, [self._vao])
        if self._vbo:
            glDeleteBuffers(1, [self._vbo])
        if self._color_shader:
            self._color_shader.free()
        if self._billboard_shader:
            self._billboard_shader.free()
        self._vao = self._vbo = 0
        self._color_shader = self._billboard_shader = None

    def _ensure_capacity(self, n_floats: int):
        if n_floats > self._capacity:
            self._capacity = max(n_floats, self._capacity * 2)
            glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
            glBufferData(GL_ARRAY_BUFFER, self._capacity * 4, None, GL_DYNAMIC_DRAW)

    def _upload_and_draw(self,
            data: np.ndarray,
            mode: int,
            shader: ShaderProgram = None,
            model: glm.mat4 = None,
            alpha: float = None
        ):

        flat = data.astype(np.float32)
        n_floats = flat.size
        n_verts = n_floats // 6

        if n_verts == 0:
            return

        self._ensure_capacity(n_floats)
        glBindBuffer(GL_ARRAY_BUFFER, self._vbo)
        glBufferSubData(GL_ARRAY_BUFFER, 0, flat.nbytes, flat)

        if model is not None and alpha is not None and shader is not None:
            glUniformMatrix4fv(shader.get_loc('uModel'), 1, GL_FALSE, glm.value_ptr(model))
            glUniform1f(shader.get_loc('uAlpha'), alpha)

        glDrawArrays(mode, 0, n_verts)