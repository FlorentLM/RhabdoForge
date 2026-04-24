import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *

import time
import random
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import TYPE_CHECKING, Optional, Union, Dict, Tuple
import numpy as np
from pyglm import glm

from insectvision.interactive.utils import DisplayMode
from insectvision.geometry.meshes import CONE_VERTICES, SPHERE_VERTICES
from insectvision.engine.agent import Agent
from insectvision.engine.scene import Scene
from insectvision.engine.shader_utils import ShaderProgram
from insectvision.renderers.resources import BufferRegistry

if TYPE_CHECKING:
    from insectvision.compound_eyes import ReceptorArray, Eye


# SSBO bindings
BINDING_RCPT_STATIC       = 0
BINDING_LENS_STATIC       = 1
BINDING_COLOR             = 2
BINDING_EMA_HIST          = 3
BINDING_RCPT_DYNAMIC      = 4
BINDING_RAYS_INTERMEDIATE = 5
BINDING_LENS_DYNAMIC      = 6


class EyeOutput(IntEnum):
    Raw = 0          # Render receptors individually (scaled down)
    Ommatidium = 1   # One tile per lens (averaging R1-R8)
    Cartridge = 2    # One tile per lens (averaging optically superimposed receptors)


class OmmatidiaProjection(IntEnum):
    Position = 0     # Positions on the curved eye surface
    OpticalAxis = 1  # Positions from optical axis directions


class Colormap(IntEnum):
    Diverging = 0    # Blue, white, red (signed and centred on zero)
    Sequential = 1   # Viridis-like
    Thermal = 2      # Black, red, white


##
# Small helpers

def _query_available_VRAM() -> int:
    """Queries available VRAM in MB."""

    extensions = glGetString(GL_EXTENSIONS).decode('utf-8').split()

    # NVIDIA
    if 'GL_NVX_gpu_memory_info' in extensions:
        from OpenGL.raw.GL.NVX.gpu_memory_info import GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX
        return glGetIntegerv(GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX) // 1024

    # AMD
    elif 'GL_ATI_meminfo' in extensions:
        # GL_VBO_FREE_MEMORY_ATI returns [total_free, largest_free_block, total_aux_free, largest_aux_free]
        GL_VBO_FREE_MEMORY_ATI = 0x87FB
        mem_info = (GLint * 4)()
        glGetIntegerv(GL_VBO_FREE_MEMORY_ATI, mem_info)
        return mem_info[0] // 1024

    return 0


##

class TextureViewer:
    """A helper to render a 2D texture (or Cubemap) to a fullscreen quad."""

    def __init__(self):
        self._shader_2d = None
        self._shader_cube = None
        self._vao = None

    @property
    def shader_2d(self):
        if self._shader_2d is None:
            self._shader_2d = ShaderProgram(vert_path='fullscreen.vert', frag_path='textureSampler.frag')
        return self._shader_2d

    @property
    def shader_cube(self):
        if self._shader_cube is None:
            self._shader_cube = ShaderProgram(vert_path='fullscreen.vert', frag_path='cubemapSampler.frag')
        return self._shader_cube

    @property
    def vao(self):
        if self._vao is None:
            self._vao = glGenVertexArrays(1)
        return self._vao

    def draw(self, texture_id, is_cubemap=False, simulate_insect_vision=False, uv_encoded_textures=False):
        """Draws the given texture to the screen."""

        shader = self.shader_cube if is_cubemap else self.shader_2d

        shader.use()
        glDisable(GL_DEPTH_TEST)
        glDepthMask(GL_FALSE)

        glActiveTexture(GL_TEXTURE0)
        if is_cubemap:
            glBindTexture(GL_TEXTURE_CUBE_MAP, texture_id)
            glUniform1i(shader.get_loc("cubemap"), 0)
        else:
            glBindTexture(GL_TEXTURE_2D, texture_id)
            glUniform1i(shader.get_loc("texture_sampler"), 0)

        # Pass toggles to the background shader
        glUniform1i(shader.get_loc('false_colors'), int(simulate_insect_vision and not uv_encoded_textures))
        glUniform1i(shader.get_loc('uv_encodeing'), int(uv_encoded_textures))

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 3)

        glBindVertexArray(0)
        shader.stop()

        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)
        glClear(GL_DEPTH_BUFFER_BIT)

    def free(self):
        if self._shader_2d:
            self._shader_2d.free()

        if self._shader_cube:
            self._shader_cube.free()

        if self._vao:
            glDeleteVertexArrays(1, [self._vao])

        self._shader_2d = None
        self._shader_cube = None
        self._vao = None


class BaseRenderer(ABC):
    """
    Abstract base class for an insect eye renderer.
    """

    def __init__(self,
            receptor_array: 'ReceptorArray',
            agent: 'Agent',
            time_dithering: bool = True,
            nb_samples: int = 256,
            quasi_random: bool = False,
            batch_size: int = 1,
            enable_actuation: bool = False,     #TODO: might rename this
        ):

        self._ra: 'ReceptorArray' = receptor_array
        self.agent: 'Agent' = agent
        self.scene: 'Scene'

        N = len(self._ra)
        nb_lenses = self._ra.lens_count

        self._samples_per_rcpt = nb_samples
        self._samples_per_px = 1

        # Estimate needed sizes and available memory
        bytes_per_frame = N * 16  # 16 bytes per vec4 (RGBA float)
        avail_VRAM = _query_available_VRAM() * 0.9  # 90% for safety, in Mb
        req_batch_dim = (bytes_per_frame * max(1, batch_size)) / (1024 * 1024)  # in Mb
        other_VRAM_usage = self._estim_vram_use()
        tot_VRAM_needed = req_batch_dim + other_VRAM_usage

        safe_batch_size = batch_size
        if not avail_VRAM:
            print("WARNING: Could not query available VRAM. Assuming enough memory is available.")

        if avail_VRAM and tot_VRAM_needed >= avail_VRAM:
            safe_batch_size = int((avail_VRAM - other_VRAM_usage) * 1024 * 1024 / bytes_per_frame)
            if safe_batch_size < batch_size:
                print(f"WARNING: Requested batch size exceeds available VRAM. Reducing batch size to {safe_batch_size} frames.")

        self._max_ssbo_bytes = glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE)
        self._batch_size = max(1, safe_batch_size)

        # Ping-pong PBOs tracking
        self._pbo_index = 0
        self._fences = [0, 0]
        self._colours_cpu_buffer = np.zeros((N, 4), dtype=np.float32)

        # Buffer registry
        self.eye_buffers = BufferRegistry()

        # Ping-pong PBOs
        self.eye_buffers.allocate('pbo_0',
                                  dtype=np.uint8,
                                  count=bytes_per_frame,
                                  target=GL_PIXEL_PACK_BUFFER,
                                  usage=GL_STREAM_READ)
        self.eye_buffers.allocate('pbo_1',
                                  dtype=np.uint8,
                                  count=bytes_per_frame,
                                  target=GL_PIXEL_PACK_BUFFER,
                                  usage=GL_STREAM_READ)

        # SSBOs
        self.eye_buffers.allocate('rcpt_static',
                                  dtype=self._ra.rcpt_static_data.dtype,
                                  count=N,
                                  data=self._ra.rcpt_static_data,
                                  usage=GL_STATIC_DRAW)
        self.eye_buffers.allocate('lens_static',
                                  dtype=self._ra.lens_static_data.dtype,
                                  count=nb_lenses,
                                  data=self._ra.lens_static_data,
                                  usage=GL_STATIC_DRAW)
        self.eye_buffers.allocate('rcpt_dynamic',
                                  dtype=self._ra.rcpt_dynamic_data.dtype,
                                  count=N,
                                  data=self._ra.rcpt_dynamic_data,
                                  usage=GL_DYNAMIC_DRAW)
        self.eye_buffers.allocate('lens_dynamic',
                                  dtype=self._ra.lens_dynamic_data.dtype,
                                  count=nb_lenses,
                                  data=self._ra.lens_dynamic_data,
                                  usage=GL_DYNAMIC_DRAW)
        self.eye_buffers.allocate('ema_state',
                                  dtype=np.dtype((np.float32, 4)),
                                  count=N,
                                  usage=GL_DYNAMIC_DRAW)
        self.eye_buffers.allocate('rays_intermediate',
                                  dtype=np.dtype((np.float32, 4)),
                                  count=N * self._samples_per_rcpt,
                                  usage=GL_DYNAMIC_DRAW)

        # Async SSBO
        self.eye_buffers.allocate('colors',
                                  dtype=np.dtype((np.float32, 4)),
                                  count=N * self._batch_size,
                                  usage=GL_DYNAMIC_DRAW,
                                  supports_async=True,
                                  _async_reader=self._colors_read_async)

        # Reduction and dynamics shaders (shared to rasterizer and raytracer)
        self.reduction_shader = ShaderProgram(comp_path='shaders/reduction.comp')
        self.dynamics_shader = ShaderProgram(comp_path='shaders/eyeDynamics.comp')

        # Shared lighting flags (subclasses may override defaults before calling super)
        self.enable_direct = getattr(self, 'enable_direct', True)
        self.enable_ambient = getattr(self, 'enable_ambient', True)
        self.enable_shadows = getattr(self, 'enable_shadows', True)
        self.ambient_intensity = getattr(self, 'ambient_intensity', 1.0)
        self.sky_intensity = getattr(self, 'sky_intensity', 1.0)

        # Dynamics parameters
        self.gain_lat = 3.0      # saccade lateral gain (μm) per unit drive
        self.gain_ax = 8.0       # saccade axial gain (μm) per unit drive
        self.tau_fast = 0.005    # 5 ms: fast saccade trigger
        self.tau_adapt = 0.050   # 50 ms: light adaptation baseline
        self.tau_relax = 0.080   # 80 ms: Mechanical relaxation (elastic return)
        self.gain_biochem = 0.1

        # Visualisation shaders (lazy-loaded)
        self.__fp_colour_shader: Optional[int] = None    # 1st person colour mode
        self.__fp_overlay_shader: Optional[int] = None   # 1st person overlay mode
        self.__tp_colour_shader: Optional[int] = None    # 3rd person colour mode
        self.__tp_overlay_shader: Optional[int] = None   # 3rd person overlay mode

        # Geometry VAOs (lazy-loaded)
        self.__lens_cones_vao: Optional[int] = None
        self.__lens_hemisph_vao: Optional[int] = None
        self.__cones_vertices = 0
        self.__hemisph_vertices = 0

        # States flags and other things
        self._quasi_random: bool = quasi_random  # Halton sampling for direction generation
        self._time_dithering: bool = time_dithering
        self._gpu_actuation: bool = enable_actuation
        self._overlay_enabled: bool = False
        self.runs_interactive: bool = False
        self.tiled_mode: bool = True
        self.simulate_insect_vision: bool = False  # TODO: expose these (and generate UV-encoded assets for demo)
        self.uv_encoded_textures: bool = False

        # Time keeping
        self._dither_counter: int = 0  # only advanced when time dithering is on
        self._frame_index: int = 0  # advanced at each new rendered frame
        self._last_render_time: float = 0.0
        self._dt: float = 0.0  # elapsed time (in seconds) since last render

        # Visualisation stuff
        self.projection_mode = OmmatidiaProjection.Position
        self.output_mode = EyeOutput.Ommatidium
        self.selected_lens = -1

        # Overlay parameters
        self._overlay_colormap = Colormap.Thermal
        self._overlay_range = (0.0, 1.0)
        self._overlay_current_peak: Optional[float] = None
        self._overlay_compression = 1.0  # power exponent for dynamic range compression (1.0 = linear, 0.5 = sqrt, etc)
        self._overlay_autorange_perc = 98  # percentile to reject outliers

        # Fullscreen texture to draw to
        self._screen_surface: Optional[TextureViewer] = None

    # Internal properties for lazy loaded resources

    @property
    def screen_surface(self):
        if self._screen_surface is None:
            self._screen_surface = TextureViewer()
        return self._screen_surface

    @property
    def _cones_vao(self):
        if self.__lens_cones_vao is None:
            v_data = CONE_VERTICES

            self.__cones_vertices = len(v_data) // 3
            self.eye_buffers.allocate('cones_vbo',
                                      dtype=np.float32,
                                      count=len(v_data),
                                      target=GL_ARRAY_BUFFER,
                                      data=v_data)

            vao = glGenVertexArrays(1)
            glBindVertexArray(vao)
            with self.eye_buffers['cones_vbo'].bind():
                glEnableVertexAttribArray(0)
                glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
                glBindVertexArray(0)
            self.__lens_cones_vao = vao

        return self.__lens_cones_vao

    @property
    def _hemispheres_vao(self):
        if self.__lens_hemisph_vao is None:
            v_data = SPHERE_VERTICES

            self.__hemisph_vertices = len(v_data) // 3
            self.eye_buffers.allocate('hemisph_vbo',
                                      dtype=np.float32,
                                      count=len(v_data),
                                      target=GL_ARRAY_BUFFER,
                                      data=v_data)

            vao = glGenVertexArrays(1)
            glBindVertexArray(vao)
            with self.eye_buffers['hemisph_vbo'].bind():
                glEnableVertexAttribArray(0)
                glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))
                glBindVertexArray(0)
            self.__lens_hemisph_vao = vao

        return self.__lens_hemisph_vao

    @property
    def _tp_colour_shader(self):
        if self.__tp_colour_shader is None:
            self.__tp_colour_shader = ShaderProgram(
                vert_path='thirdPersonEye.vert',
                frag_path='thirdPersonEye.frag'
            )
        return self.__tp_colour_shader

    @property
    def _tp_overlay_shader(self):
        if self.__tp_overlay_shader is None:
            self.__tp_overlay_shader = ShaderProgram(
                vert_path='thirdPersonEye.vert',
                frag_path='thirdPersonEye.frag',
                defines={'OVERLAY_MODE'}
            )
        return self.__tp_overlay_shader

    @property
    def _fp_colour_shader(self):
        if self.__fp_colour_shader is None:
            self.__fp_colour_shader = ShaderProgram(
                vert_path='firstPersonEye.vert',
                frag_path='firstPersonEye.frag'
            )
        return self.__fp_colour_shader

    @property
    def _fp_overlay_shader(self):
        if self.__fp_overlay_shader is None:
            self.__fp_overlay_shader = ShaderProgram(
                vert_path='firstPersonEye.vert',
                frag_path='firstPersonEye.frag',
                defines={'OVERLAY_MODE'}
            )
        return self.__fp_overlay_shader

    # Various internal helpers

    def _estim_vram_use(self) -> float:
        # Each subclass implements its own estimation logic
        return 100.0

    def _tick(self):
        now = time.perf_counter()
        self._dt = (now - self._last_render_time) if self._last_render_time > 0.0 else 0.0
        self._last_render_time = now

    # Internal helpers for GL resource binding

    def _set_eye_uniforms(self, shader):
        glUniform1i(shader.get_loc('output_mode'), int(self.output_mode))
        glUniform1i(shader.get_loc('receptor_count'), self._ra.receptor_count)
        glUniform1i(shader.get_loc('center_index'), self._ra.kernel.center_index)
        glUniform1i(shader.get_loc('selected_id'), self.selected_lens)
        glUniform1i(shader.get_loc('frame_offset'), self._frame_index % self._batch_size)
        glUniform1f(shader.get_loc('albedo_boost'), 1.0)
        glUniform1i(shader.get_loc('false_colors'), int(self.simulate_insect_vision and not self.uv_encoded_textures))
        glUniform1i(shader.get_loc('uv_encoding'), int(self.uv_encoded_textures))

        if self.overlay_enabled:
            glUniform1f(shader.get_loc('overlay_data_min'), self._overlay_range[0])
            glUniform1f(shader.get_loc('overlay_data_max'), self._overlay_range[1])
            glUniform1i(shader.get_loc('colormap'), int(self._overlay_colormap))
            glUniform1f(shader.get_loc('compression'), self._overlay_compression)

    # Internal rendering logic and draw calls

    def _reduction(self):

        N = len(self._ra)

        shader = self.reduction_shader
        shader.use()

        eye_bindings = {
            'rays_intermediate': BINDING_RAYS_INTERMEDIATE,
            'rcpt_static': BINDING_RCPT_STATIC,
            'colors': BINDING_COLOR,
            'ema_state': BINDING_EMA_HIST,
            'rcpt_dynamic': BINDING_RCPT_DYNAMIC
        }

        with self.eye_buffers.grouped_bind_base(eye_bindings):

            glUniform1i(shader.get_loc('nb_samples'), self.nb_samples)
            glUniform1i(shader.get_loc('nb_receptors'), N)
            glUniform1f(shader.get_loc('dt'), self._dt)

            frame_offset = self._frame_index % self._batch_size
            glUniform1i(shader.get_loc('frame_offset'), frame_offset)

            work_groups = (N + 63) // 64
            glDispatchCompute(work_groups, 1, 1)

            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        shader.stop()

    def _eye_dynamics(self):

        k = self._ra.kernel
        center_idx = k.center_index

        if k.nodal_distance_um is not None:
            shader = self.dynamics_shader
            shader.use()

        eye_bindings = {
            'rcpt_static': BINDING_RCPT_STATIC,
            'lens_static': BINDING_LENS_STATIC,
            'colors': BINDING_COLOR,
            'ema_state': BINDING_EMA_HIST,
            'rcpt_dynamic': BINDING_RCPT_DYNAMIC,
            'lens_dynamic': BINDING_LENS_DYNAMIC
        }

        with self.eye_buffers.grouped_bind_base(eye_bindings):

            glUniform1i(shader.get_loc('nb_lenses'), self._ra.lens_count)
            glUniform1i(shader.get_loc('receptors_per_lens'), self._ra.receptor_count)
            glUniform1f(shader.get_loc('dt'), self._dt)

            glUniform1f(shader.get_loc('d_rest'), k.nodal_distance_um)
            lam_um = self._ra._wavelength_nm * 1e-3
            glUniform1f(shader.get_loc('diffraction_sq'), (lam_um / k.lens_diameter_um) ** 2)
            glUniform1f(shader.get_loc('acc_rest_geom_sq'), (k.diameters_um[center_idx] / k.nodal_distance_um) ** 2)

            # Dynamics
            saccades_enabled = self._gpu_actuation

            glUniform1f(shader.get_loc('tau_fast'), self.tau_fast)
            glUniform1f(shader.get_loc('tau_adapt'), self.tau_adapt)
            glUniform1f(shader.get_loc('gain_lat'), self.gain_lat if saccades_enabled else 0.0)
            glUniform1f(shader.get_loc('gain_ax'), self.gain_ax if saccades_enabled else 0.0)
            glUniform1f(shader.get_loc('tau_relax'), self.tau_relax)
            glUniform1f(shader.get_loc('gain_biochem'), self.gain_biochem)

            work_groups = (self._ra.lens_count + 63) // 64
            glDispatchCompute(work_groups, 1, 1)
            glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        shader.stop()

    def _colors_read_async(self) -> np.ndarray:
        """
        Non-blocking colour readback (via ping-pong PBO ring).
        Returns the *previous* frame colours (and zeros on the first frame).
        """
        N = len(self._ra)
        bytes_to_read = N * 16

        # Copy colour SSBO into current PBO
        with (self.eye_buffers['colors'].bind(mode_override=GL_COPY_READ_BUFFER),
              self.eye_buffers[f'pbo_{self._pbo_index}'].bind(mode_override=GL_COPY_WRITE_BUFFER)):

            glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, bytes_to_read)

        # Fence to know when the copy is done
        if self._fences[self._pbo_index]:
            glDeleteSync(self._fences[self._pbo_index])
        self._fences[self._pbo_index] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0)

        # Read from the other PBO (previous frame data)
        next_pbo_index = (self._pbo_index + 1) % 2
        fence = self._fences[next_pbo_index]

        out_array = np.zeros_like(self._colours_cpu_buffer)
        if fence:
            glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, 1000000000)

            with self.eye_buffers[f'pbo_{next_pbo_index}'].bind(mode_override=GL_PIXEL_PACK_BUFFER):
                ptr = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, bytes_to_read, GL_MAP_READ_BIT)
                if ptr:
                    ctypes.memmove(self._colours_cpu_buffer.ctypes.data, ptr, bytes_to_read)
                    glUnmapBuffer(GL_PIXEL_PACK_BUFFER)
                    out_array = self._colours_cpu_buffer.copy()
                else:
                    print("Warning: Failed to map PBO. Context lost?")

        self._pbo_index = next_pbo_index
        return out_array

    def _main_render(self):
        """Shared pipeline: tick -> scene-specific sampling -> reduce -> actuate."""

        self._tick()
        self._sample_scene()  # subclasses override: fill sampling_results_ssbo
        self._reduction()
        self._eye_dynamics()

    @abstractmethod
    def _sample_scene(self):
        """Subclass-specific: prepare scene and populate sampling_results_ssbo."""
        raise NotImplementedError

    def _draw_eye_firstperson(self):
        """First-person compound-eye view (colours or scalar overlay)."""

        shader = self._fp_overlay_shader if self.overlay_enabled else self._fp_colour_shader
        shader.use()

        glEnable(GL_DEPTH_TEST)

        # Compute nice-looking dimensions for tiles
        viewport = glGetIntegerv(GL_VIEWPORT)
        aspect_ratio = viewport[2] / viewport[3] if viewport[3] > 0 else 1.0
        receptive_field_scale = 1.0 / (2.0 * np.pi)

        glUniform1f(shader.get_loc('aspect_ratio'), aspect_ratio)
        glUniform1i(shader.get_loc('tiled_mode'), self.tiled_mode)
        glUniform1i(shader.get_loc('projection_mode'), self.projection_mode)
        glUniform1f(shader.get_loc('receptive_field_scale'), receptive_field_scale)

        self._set_eye_uniforms(shader)

        eye_bindings = {
            'rcpt_static': BINDING_RCPT_STATIC,
            'lens_static': BINDING_LENS_STATIC,
            'rcpt_dynamic': BINDING_RCPT_DYNAMIC,
            'lens_dynamic': BINDING_LENS_DYNAMIC,
            'overlay' if self.overlay_enabled else 'colors': BINDING_COLOR
        }

        with self.eye_buffers.grouped_bind_base(eye_bindings):

            N = len(self._ra)
            nb_units = N if self.output_mode == EyeOutput.Raw else self._ra.lens_count

            glBindVertexArray(self._cones_vao)
            glDrawArraysInstanced(GL_TRIANGLES, 0, self.__cones_vertices, nb_units)

            glBindVertexArray(0)

        glDisable(GL_DEPTH_TEST)

        shader.stop()

    def _draw_eye_thirdperson(self, observer_camera):
        """Third-person eye model (colours or scalar overlay)."""

        shader = self._tp_overlay_shader if self.overlay_enabled else self._tp_colour_shader
        shader.use()

        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDisable(GL_CULL_FACE)

        view_matrix_np = np.array(observer_camera.view, dtype=np.float32)
        projection_matrix_np = np.array(observer_camera.projection, dtype=np.float32)
        c2w_mat = glm.inverse(self.agent.view)

        glUniformMatrix4fv(shader.get_loc('view'), 1, True, view_matrix_np)
        glUniformMatrix4fv(shader.get_loc('projection'), 1, True, projection_matrix_np)
        glUniformMatrix4fv(shader.get_loc('eye_to_world'), 1, False, glm.value_ptr(c2w_mat))

        # Nice-looking factors for visualisation
        avg_radius = np.mean(np.linalg.norm(self._ra.rcpt_static_data['position'][:, :3], axis=1))
        cone_length = max(0.01, avg_radius) * 0.3
        visualisation_scale = 1.0
        saccade_visualisation_gain = 0.025

        glUniform1i(shader.get_loc('projection_mode'), self.projection_mode)
        glUniform1f(shader.get_loc('cone_length'), cone_length)
        glUniform1f(shader.get_loc('visualisation_scale'), visualisation_scale)
        glUniform1f(shader.get_loc('saccade_visualisation_gain'), saccade_visualisation_gain)

        self._set_eye_uniforms(shader)

        eye_bindings = {
            'rcpt_static': BINDING_RCPT_STATIC,
            'lens_static': BINDING_LENS_STATIC,
            'rcpt_dynamic': BINDING_RCPT_DYNAMIC,
            'lens_dynamic': BINDING_LENS_DYNAMIC,
            'overlay' if self.overlay_enabled else 'colors': BINDING_COLOR
        }

        with self.eye_buffers.grouped_bind_base(eye_bindings):

            N = len(self._ra)
            nb_units = N if self.output_mode == EyeOutput.Raw else self._ra.lens_count

            if self.projection_mode == OmmatidiaProjection.Position:
                glBindVertexArray(self._hemispheres_vao)
                glDrawArraysInstanced(GL_TRIANGLES, 0, self.__hemisph_vertices, nb_units)
            else:
                glBindVertexArray(self._cones_vao)
                glDrawArraysInstanced(GL_TRIANGLES, 0, self.__cones_vertices, nb_units)

            glBindVertexArray(0)

        glEnable(GL_CULL_FACE)
        glDisable(GL_BLEND)
        glDisable(GL_DEPTH_TEST)

        shader.stop()

    # TODO: These two public methods should probably be all under the hood

    def flush(self) -> np.ndarray:
        """
        Blocks until all queued frames on the GPU are rendered, downloads the data, and resets the counter.
        This is used to retrieve a full batch, or the final partial batch at the end of a simulation.
        """

        if self._frame_index == 0:
            return np.array([])

        glFinish()  # Block until all rendering commands are complete

        frames_to_read = int(self._frame_index)
        N = len(self._ra)

        # Direct synchronous download is ok here
        with self.eye_buffers['colors'].bind():
            data_bytes = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, N * 16 * frames_to_read)
            data_np = np.frombuffer(data_bytes, dtype=np.float32).reshape(frames_to_read, N, 4)

        self._frame_index = 0  # reset counter for next batch

        # return only first 4 floats (current RGBA) and not the next 4 (state for EMA)
        return data_np.copy()

    def update(self, force_all=False):
        """
        Finds contiguous blocks of changed ommatidia and uploads dynamic states to the GPU.
        """

        # Update lens dynamic state if changed
        if self._ra.lens_dirty or force_all:
            self.eye_buffers['lens_dynamic'].write(self._ra.lens_dynamic_data)
            self._ra.lens_dirty = False

        # Update receptor dynamic state if changed
        dirty_indices = np.where(self._ra.dirty_mask)[0]
        if dirty_indices.size == 0 and not force_all:
            return

        if force_all:
            self.eye_buffers['rcpt_dynamic'].write(self._ra.rcpt_dynamic_data)

        else:
            # Find contiguous blocks of updated indices
            jumps = np.where(np.diff(dirty_indices) != 1)[0] + 1
            contiguous_blocks = np.split(dirty_indices, jumps)

            for block in contiguous_blocks:
                nb_items = block.size
                if nb_items == 0:
                    continue

                # indexing like this instead of fancy indexing (using [block] directly) avoids a copy
                start_index = block[0]
                data_view = self._ra.rcpt_dynamic_data[start_index:start_index + nb_items]
                self.eye_buffers['rcpt_dynamic'].write(data_view, start=start_index)

        self._ra.dirty_mask.fill(False)

    # Main public methods

    @abstractmethod
    def draw(self, view_mode: DisplayMode, point_of_view: Agent):
        # Each subclass implements its own rendering logic
        raise NotImplementedError

    def get_output(self, readback: bool = True) -> Optional['VisualOutput']:
        """
        Runs one frame of simulation.
            - batch_size = 1: Blocks and returns the current frame's data
            - batch_size > 1: Queues the frame on the GPU and returns None
        """
        from insectvision.compound_eyes import VisualOutput

        self.update()

        if self._time_dithering:
            self._dither_counter += 1

        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        self._main_render()
        glFinish()

        self._frame_index += 1

        if not readback:
            # no cpu readback, we're done here
            return None

        if self.runs_interactive or self._batch_size == 1:
            # Synchronous path: ping-pong PBO read
            out_array = self._colors_read_async()

            return VisualOutput(out_array, self._ra)

        else:
            # Asynchronous path

            if self._frame_index < self._batch_size:
                # batch is not full yet, we're done for this frame
                return None

            print(f"  > GPU batch is full. Flushing {self._batch_size} frames...")
            batch_result = self.flush()
            return VisualOutput(batch_result, self._ra)

    def set_overlay(self,
        values: Union[Dict['Eye', np.array], np.array],
        range: Optional[Tuple[float, float]] = None,
        colormap: 'Colormap' = Colormap.Thermal,
        compression: float = 0.5
    ):
        """
        Upload data for overlay visualisation.

        Args:
            values: Either a flat array in global order (total_receptors,)
                    or a dict mapping Eye -> per-eye array
            range: (min, max) bounds for the colourmap. None = auto
            colormap: Which colourmap to use (Colormap enum)
            compression: Power exponent for dynamic range compression
                         1.0 = linear, 0.5 = sqrt, lower = brings out detail
        """

        N = len(self._ra)

        if isinstance(values, dict):
            merged = np.zeros(N, dtype=np.float32)

            for eye, data in values.items():
                if len(data) == len(eye):
                    data = np.repeat(data, self._ra.receptor_count)
                merged[eye.receptors.global_indices] = data
            values = merged

        buf = np.ascontiguousarray(values, dtype=np.float32).ravel()
        buf_size = len(buf)

        if buf_size != N:
            raise ValueError(f"scalar_data has {buf_size} elements, expected {N}.")

        if 'overlay' not in self.eye_buffers:
            self.eye_buffers.allocate('overlay', np.float32, N, usage=GL_DYNAMIC_DRAW)
        elif self.eye_buffers['overlay'].count != N:
            self.eye_buffers['overlay'].resize(N)

        self.eye_buffers['overlay'].write(buf)

        # Range
        if range is not None:
            self._overlay_range = float(range[0]), float(range[1])
        else:
            # Auto-normalise: per-frame peak value
            frame_peak = float(np.percentile(np.abs(buf), self._overlay_autorange_perc))
            if self._overlay_current_peak is None:
                self._overlay_current_peak = frame_peak

            # Exponential Moving Average for colourmap scaling
            if frame_peak > self._overlay_current_peak:
                self._overlay_current_peak = 0.5 * self._overlay_current_peak + 0.5 * frame_peak
            else:
                self._overlay_current_peak = 0.98 * self._overlay_current_peak + 0.02 * frame_peak

            range_bound = max(self._overlay_current_peak, 1e-6)

            if colormap == Colormap.Diverging:
                self._overlay_range = (-range_bound, range_bound)
            else:
                self._overlay_range = (0.0, range_bound)

        self._overlay_colormap = colormap
        self._overlay_compression = compression

    # Public properties and methods

    @property
    def nb_samples(self):
        return self._samples_per_rcpt

    @nb_samples.setter
    def nb_samples(self, value):
        N = len(self._ra)

        max_tot_samples = self._max_ssbo_bytes // 16
        max_per_r = max(1, max_tot_samples // N)
        value_clamped = int(np.clip(value, 1, max_per_r))

        if value_clamped == self._samples_per_rcpt:
            return

        print(f"Warning: Clamped samples per receptor to {value_clamped} (HW limit is {max_per_r}).")

        self._samples_per_rcpt = value_clamped
        req_bytes = N * self._samples_per_rcpt * 16

        self.eye_buffers['rays_intermediate'].resize(N * self._samples_per_rcpt)

    @property
    def time_dithering(self):
        return self._time_dithering

    @time_dithering.setter
    def time_dithering(self, value: bool):
        self._time_dithering = bool(value)
        print(f"Time dithering {'ENABLED' if self._time_dithering else 'DISABLED'}.")

    @property
    def quasi_random(self):
        return self._quasi_random

    @quasi_random.setter
    def quasi_random(self, value: bool):
        self._quasi_random = bool(value)
        print(f"Quasi-random {'ENABLED' if self._quasi_random else 'DISABLED'}.")

    @property
    def overlay_enabled(self) -> bool:
        if 'overlay' not in self.eye_buffers:
            self._overlay_enabled = False
            return False
        return self._overlay_enabled and self.eye_buffers['overlay'].count > 0

    @overlay_enabled.setter
    def overlay_enabled(self, value: bool):
        self._overlay_enabled = bool(value)

    @property
    def actuation(self):
        return self._gpu_actuation

    @actuation.setter
    def actuation(self, value: bool):
        self._gpu_actuation = bool(value)
        print(f"Rhabdomere actuation {'ENABLED' if self._time_dithering else 'DISABLED'}.")

    def dither(self):
        """Dither once (reshuffle the dither counter)"""
        self._dither_counter = random.randint(0, 1024)

    def update_texture(self, asset: 'Asset'):
        """
        Update a texture on the GPU for given Asset.
        Subclasses should override this.
        """
        pass

    # Cleanup

    def free(self):
        """Free GPU resources."""

        for f in self._fences:
            if f:
                try:
                    glDeleteSync(f)
                except Exception:
                    pass

        self.eye_buffers.free()

        for vao in (
            self.__lens_cones_vao,
            self.__lens_hemisph_vao
        ):
            if vao:
                glDeleteVertexArrays(1, [vao])

        for shader in (
            self.__fp_colour_shader,
            self.__fp_overlay_shader,
            self.__tp_colour_shader,
            self.__tp_overlay_shader,
            self.reduction_shader,
            self.dynamics_shader
        ):
            if shader:
                shader.free()