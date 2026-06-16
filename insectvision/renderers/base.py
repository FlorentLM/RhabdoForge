import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *

import random
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Optional, Union, Dict, Tuple, Sequence
import numpy as np
from pyglm import glm

from insectvision.utils.shared import EyeOutput, OmmatidiaProjection, Colormap, DisplayMode, RandomnessMode, SamplingMode
from insectvision.engine.meshes import CONE_VERTICES, SPHERE_VERTICES
from insectvision.engine.agent import Agent
from insectvision.engine.scene import Scene
from insectvision.engine.resources import ShaderProgram, GPUResourceManager, BufferRegistry, UniformRegistry
from insectvision.renderers.helpers import VisualOutput

from insectvision.compound_eyes.buffers import OMM_STATIC_DTYPE, OMM_DYNAMIC_DTYPE, RHAB_STATIC_DTYPE, \
    RHAB_DYNAMIC_DTYPE, _BIT_LAYOUT

if TYPE_CHECKING:
    from insectvision.compound_eyes import Model, Eye
    from insectvision.engine.context import Context


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

        with shader:
            glDisable(GL_DEPTH_TEST)
            glDepthMask(GL_FALSE)

            glActiveTexture(GL_TEXTURE0)
            if is_cubemap:
                glBindTexture(GL_TEXTURE_CUBE_MAP, texture_id)
                glUniform1i(shader.get_loc("cubemap"), 0)
            else:
                glBindTexture(GL_TEXTURE_2D, texture_id)
                glUniform1i(shader.get_loc("texture_sampler"), 0)

            glUniform1i(shader.get_loc('false_colors'), int(simulate_insect_vision and not uv_encoded_textures))
            glUniform1i(shader.get_loc('uv_encodeing'), int(uv_encoded_textures))

            glBindVertexArray(self.vao)
            glDrawArrays(GL_TRIANGLES, 0, 3)

            glBindVertexArray(0)

        glDepthMask(GL_TRUE)
        glEnable(GL_DEPTH_TEST)
        glClear(GL_DEPTH_BUFFER_BIT)

    def free(self):
        if self._shader_2d: self._shader_2d.free()
        if self._shader_cube: self._shader_cube.free()
        if self._vao: glDeleteVertexArrays(1, [self._vao])
        self._shader_2d = None
        self._shader_cube = None
        self._vao = None


class BaseRenderer(ABC):
    """
    Abstract base class for an insect eye renderer.
    """

    def __init__(self,
                 model: 'Model',
                 agent: 'Agent',
                 time_dithering: bool = True,
                 nb_samples: int = 256,
                 randomness_mode: Union[int, str, RandomnessMode] = RandomnessMode.Pseudo,
                 sampling_mode: Union[int, str, SamplingMode] = SamplingMode.Gaussian,
                 batch_size: int = 1,
                 enable_microsaccades: bool = False,
                 resource_manager: Optional[GPUResourceManager] = None,
                 context: Optional['Context'] = None
                 ):

        self._context: Optional['Context'] = None
        if context is not None:
            self.attach_context(context)

        self._model: 'Model' = model
        self.agent: 'Agent' = agent
        self.scene: 'Scene'

        self._samples_per_rhab = 1
        self._samples_per_px = 1
        self._noise_threshold = 0.05
        self._use_hybrid_sampling = False
        self._randomness_mode = self._to_enum(randomness_mode, RandomnessMode)
        self._sampling_mode = self._to_enum(sampling_mode, SamplingMode)

        # Estimate needed sizes and available memory
        bytes_per_frame = self._model.size * 16  # 16 bytes per vec4 (RGBA float)
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
        self._colours_cpu_buffer = np.zeros((self._model.size, 4), dtype=np.float32)

        # Buffer registry
        self.resource_manager = resource_manager or GPUResourceManager()
        self.eye_buffers = BufferRegistry(self.resource_manager)

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
        self.eye_buffers.allocate('rhab_static',
                                  dtype=RHAB_STATIC_DTYPE,
                                  count=self._model.size,
                                  data=self._model.buffer.rhabdomere_static,
                                  usage=GL_STATIC_DRAW)
        self.eye_buffers.allocate('omm_static',
                                  dtype=OMM_STATIC_DTYPE,
                                  count=self._model.N,
                                  data=self._model.buffer.ommatidia_static,
                                  usage=GL_STATIC_DRAW)
        self.eye_buffers.allocate('rhab_dynamic',
                                  dtype=RHAB_DYNAMIC_DTYPE,
                                  count=self._model.size,
                                  data=self._model.buffer.rhabdomere_dynamic,
                                  usage=GL_DYNAMIC_DRAW)
        self.eye_buffers.allocate('omm_dynamic',
                                  dtype=OMM_DYNAMIC_DTYPE,
                                  count=self._model.N,
                                  data=self._model.buffer.ommatidia_dynamic,
                                  usage=GL_DYNAMIC_DRAW)
        self.eye_buffers.allocate('ema_state',
                                  dtype=np.dtype((np.float32, 4)),
                                  count=self._model.size,
                                  usage=GL_DYNAMIC_DRAW)
        self.eye_buffers.allocate('rays_intermediate',
                                  dtype=np.dtype((np.float32, 4)),
                                  count=self._model.size * self._samples_per_rhab,
                                  usage=GL_DYNAMIC_DRAW)

        # Async SSBO
        self.eye_buffers.allocate('colors',
                                  dtype=np.dtype((np.float32, 4)),
                                  count=self._model.size * self._batch_size,
                                  usage=GL_DYNAMIC_DRAW,
                                  supports_async=True,
                                  _async_reader=self._colors_read_async)

        # Base define maps generated automatically from the BufferRegistry allocations
        base_defines = self.eye_buffers.shader_defines

        # Reduction and dynamics shaders (shared to rasterizer and raytracer)
        self.reduction_shader = ShaderProgram(comp_path='shaders/reduction.comp', defines=base_defines)
        self.dynamics_shader = ShaderProgram(comp_path='shaders/eyeDynamics.comp', defines=base_defines)

        # Shared lighting flags (subclasses may override defaults before calling super)
        self.enable_direct = getattr(self, 'enable_direct', True)
        self.enable_ambient = getattr(self, 'enable_ambient', True)
        self.enable_shadows = getattr(self, 'enable_shadows', True)
        self.ambient_intensity = getattr(self, 'ambient_intensity', 1.0)
        self.sky_intensity = getattr(self, 'sky_intensity', 1.0)

        self.lum_ref = 1.0  # target operating-point luminance (should be scene-dependant)

        # Visualisation shaders (lazy-loaded)
        self.__fp_colour_shader: Optional[ShaderProgram] = None    # 1st person colour mode
        self.__fp_overlay_shader: Optional[ShaderProgram] = None   # 1st person overlay mode
        self.__tp_colour_shader: Optional[ShaderProgram] = None    # 3rd person colour mode
        self.__tp_overlay_shader: Optional[ShaderProgram] = None   # 3rd person overlay mode

        # Geometry VAOs (lazy-loaded)
        self.__cones_vao: Optional[int] = None
        self.__hemisph_vao: Optional[int] = None
        self.__cones_vertices = 0
        self.__hemisph_vertices = 0

        # States flags and other things
        self._time_dithering: bool = time_dithering
        self._microsaccades_enabled: bool = enable_microsaccades
        self._overlay_enabled: bool = False

        self._needs_warmup: bool = True
        self.runs_interactive: bool = False
        self.tiled_mode: bool = True
        self.simulate_insect_colours: bool = False  # TODO: expose these (and generate UV-encoded assets for demo)
        self.uv_encoded_textures: bool = False

        # Time keeping
        self._dither_counter: int = 0   # only advanced when time dithering is on
        self._frame_index: int = 0      # advanced at each new rendered frame

        # Visualisation stuff
        self.projection_mode = OmmatidiaProjection.Position
        self.output_mode = EyeOutput.Cartridge
        self._selected_omm_indices = np.full(10, -1, dtype=np.int32)

        # Overlay parameters
        self._overlay_colormap = Colormap.Thermal
        self._overlay_range = (0.0, 1.0)
        self._overlay_current_peak: Optional[float] = None
        self._overlay_compression = 1.0         # power exponent for dynamic range compression (1.0 = linear, 0.5 = sqrt, etc)
        self._overlay_autorange_perc = 98       # percentile to reject outliers

        # TODO: Add multiple selectable overlay modes: luminance, adaptation state, etc , and custom (the set_data one)

        # Fullscreen texture to draw to
        self._screen_surface: Optional[TextureViewer] = None

        self.nb_samples = nb_samples    # via property to apply

        # Initialise uniforms registries
        self._eye_uniforms = UniformRegistry(
            aspect_ratio=1.0,

            # Rhabdomeres and ommatidia (constants during runtime)
            nb_ommatidia=self._model.N,
            nb_rhabdomeres=self._model.size,
            rhab_per_omm=self._model.R,

            # Rhabdomere bundle params (constants during runtime)
            bundle_centre_idx=self._model.bundle.center_index,

            # Various visualisation parameters (currently fixed and not modifiable)
            visualisation_eye_surface_albedo=1.0,
            visualisation_rf_scale=0.5,
            visualisation_omm_length=max(0.01, np.mean(self._model.ommatidia.aperture)) * 0.3,
            visualisation_eyes_scale=1.0,
            visualisation_saccade_scale=1.0,

            # Mofidiable during runtime
            output_mode=self.output_mode,
            tiled_mode=self.tiled_mode,
            projection_mode=self.projection_mode,
            false_colors=self.simulate_insect_colours and not self.uv_encoded_textures,
            uv_encoding=self.uv_encoded_textures,

            # Sampling modes
            nb_samples=self.nb_samples,
            use_hybrid_sampling=self._use_hybrid_sampling,
            sampling_mode=self._sampling_mode,  # 0 = Gaussian, 1 = Airy

            # Visualisation defaults
            selected_ommatidia=self._selected_omm_indices,
            overlay_fallback=True,
            overlay_data_min=self._overlay_range[0],
            overlay_data_max=self._overlay_range[1],
            overlay_colormap=int(self._overlay_colormap),
            overlay_compression=self._overlay_compression,

            # Time tracking
            dt=0.0,
            frame_offset=self._frame_index % self._batch_size,
            dither_counter=self._dither_counter,

            # Rhabdomere dynamics
            noise_threshold=self._noise_threshold,
            enable_actuation=self._microsaccades_enabled,
            photon_concentration_factor=0.0,        # TODO: document this better
            lum_ref=self.lum_ref,
            extra_narrowing_ratio=float(self._model.bundle.extra_narrowing_ratio),
        )

    def __repr__(self):
        loop_mode = 'Open-loop (batched)' if self._batch_size > 1 else 'Closed-loop / Interactive'
        return (f"<{self.__class__.__name__} | Mode: {loop_mode} | "
                f"Batch size: {self._batch_size} | "
                f"{self.nb_samples} samples/rhabdomere>")

    # Internal properties for lazy loaded resources

    @property
    def screen_surface(self):
        if self._screen_surface is None:
            self._screen_surface = TextureViewer()
        return self._screen_surface

    @property
    def _cones_vao(self):
        if self.__cones_vao is None:
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
            self.__cones_vao = vao

        return self.__cones_vao

    @property
    def _hemispheres_vao(self):
        if self.__hemisph_vao is None:
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
            self.__hemisph_vao = vao

        return self.__hemisph_vao

    @property
    def _tp_colour_shader(self):
        if self.__tp_colour_shader is None:
            self.__tp_colour_shader = ShaderProgram(vert_path='thirdPersonEye.vert',
                                                    frag_path='thirdPersonEye.frag',
                                                    defines=self.eye_buffers.shader_defines)
        return self.__tp_colour_shader

    @property
    def _tp_overlay_shader(self):
        if self.__tp_overlay_shader is None:
            defines = self.eye_buffers.shader_defines.copy()
            defines['OVERLAY_MODE'] = 1
            self.__tp_overlay_shader = ShaderProgram(vert_path='thirdPersonEye.vert',
                                                     frag_path='thirdPersonEye.frag',
                                                     defines=defines)
        return self.__tp_overlay_shader

    @property
    def _fp_colour_shader(self):
        if self.__fp_colour_shader is None:
            self.__fp_colour_shader = ShaderProgram(vert_path='firstPersonEye.vert',
                                                    frag_path='firstPersonEye.frag',
                                                    defines=self.eye_buffers.shader_defines)
        return self.__fp_colour_shader

    @property
    def _fp_overlay_shader(self):
        if self.__fp_overlay_shader is None:
            defines = self.eye_buffers.shader_defines.copy()
            defines['OVERLAY_MODE'] = 1
            self.__fp_overlay_shader = ShaderProgram(vert_path='firstPersonEye.vert',
                                                     frag_path='firstPersonEye.frag',
                                                     defines=defines)
        return self.__fp_overlay_shader

    # Various internal helpers

    def _estim_vram_use(self) -> float:
        # Each subclass implements its own estimation logic
        return 100.0

    @staticmethod
    def _to_enum(val, enum_class):
        """Helper to convert string, int, or enum to the target Enum class."""
        if isinstance(val, enum_class):
            return val
        if isinstance(val, str):
            try:
                return enum_class[val.capitalize()]
            except KeyError:
                print(f"Warning: Invalid mode '{val}' for {enum_class.__name__}. Defaulting to {list(enum_class)[0].name}")
                return list(enum_class)[0]
        return enum_class(val)

    # Internal rendering logic and draw calls

    def _reduction(self):

        with self.reduction_shader as shader:
            with self.eye_buffers.grouped_bind(['rays_intermediate', 'rhab_static', 'colors', 'ema_state', 'rhab_dynamic']):

                self._eye_uniforms.apply(shader)
                glDispatchCompute(self._model.size, 1, 1)
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def _eye_dynamics(self):

        with self.dynamics_shader as shader:
            with self.eye_buffers.grouped_bind(['rhab_static', 'omm_static', 'colors', 'ema_state', 'rhab_dynamic', 'omm_dynamic']):

                self._eye_uniforms.apply(shader)

                work_groups = (self._model.N + 63) // 64
                glDispatchCompute(work_groups, 1, 1)
                glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

    def _colors_read_async(self) -> np.ndarray:
        """
        Non-blocking colour readback (via ping-pong PBO ring).
        Returns the *previous* frame colours (and zeros on the first frame).
        """

        N = self._model.size
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
            # glClientWaitSync(fence, GL_SYNC_FLUSH_COMMANDS_BIT, 1000000000)

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

    def _update_uniforms(self):

        # Ticks
        self._eye_uniforms.update(
            dt=self._context.dt,
            frame_offset=self._frame_index % self._batch_size,
            dither_counter=self._dither_counter
        )

        # If output is 'Raw', the shader needs rhabdomeres IDs. Otherwise, it needs ommatidia IDs
        shader_highlight_ids = self._selected_omm_indices.copy()

        if self.output_mode == EyeOutput.Raw:
            # Highlight the center rhabdomere of the selected ommatidia
            for i in range(10):
                if shader_highlight_ids[i] != -1:
                    shader_highlight_ids[i] *= self._model.R
                    shader_highlight_ids[i] += self._model.bundle.center_index

        # Sampling changes
        self._eye_uniforms.update(
            nb_samples=self.nb_samples,
            use_hybrid_sampling=self._use_hybrid_sampling,
            randomness_mode=int(self._randomness_mode),
            sampling_mode=int(self._sampling_mode),
        )

        # Process UI commands
        self._eye_uniforms.update(
            noise_threshold=self._noise_threshold,
            projection_mode=self.projection_mode,
            output_mode=self.output_mode,
            tiled_mode=self.tiled_mode,
            randomness_mode=self._randomness_mode,
            enable_actuation=self._microsaccades_enabled,
            selected_lenses=shader_highlight_ids,
        )

        # Process eventual luminosity ref change
        self._eye_uniforms.update(
            lum_ref=self.lum_ref
        )

    def _main_render(self):
        """Shared pipeline: tick -> upload updated uniforms -> scene-specific sampling -> reduce -> actuate."""

        self._update_uniforms()

        self._sample_scene()  # subclasses override: fill sampling_results_ssbo
        self._reduction()
        self._eye_dynamics()

        if self._needs_warmup:
            self._reduction()
            self._needs_warmup = False

    @abstractmethod
    def _sample_scene(self):
        """Subclass-specific: prepare scene and populate sampling_results_ssbo."""
        raise NotImplementedError

    def _draw_eye_firstperson(self):
        """First-person compound-eye view (colours or scalar overlay)."""

        shader = self._fp_overlay_shader if self.overlay_enabled else self._fp_colour_shader

        with shader:
            glEnable(GL_DEPTH_TEST)
            glDisable(GL_CULL_FACE)
            glDepthFunc(GL_LEQUAL)

            # Need the first person projection to conform to viewport aspect
            viewport = glGetIntegerv(GL_VIEWPORT)
            self._eye_uniforms.update(aspect_ratio=viewport[2] / viewport[3] if viewport[3] > 0 else 1.0)

            self._eye_uniforms.apply(shader)

            to_bind = ['rhab_static', 'omm_static', 'rhab_dynamic', 'omm_dynamic', 'colors']
            if self.overlay_enabled:
                to_bind.append('overlay')
            with self.eye_buffers.grouped_bind(to_bind):
                N = self._model.size
                nb_units = N if self.output_mode == EyeOutput.Raw else self._model.N

                glBindVertexArray(self._cones_vao)
                glDrawArraysInstanced(GL_TRIANGLES, 0, self.__cones_vertices, nb_units)
                glBindVertexArray(0)

            glDisable(GL_DEPTH_TEST)

    def _draw_eye_thirdperson(self, observer_camera):
        """Third-person eye model (colours or scalar overlay)."""

        shader = self._tp_overlay_shader if self.overlay_enabled else self._tp_colour_shader

        with shader:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            glDisable(GL_CULL_FACE)

            self._eye_uniforms.update(
                view=observer_camera.view,
                projection=observer_camera.projection,
                eye_to_world=glm.inverse(self.agent.view)
            )

            self._eye_uniforms.apply(shader)

            to_bind = ['rhab_static', 'omm_static', 'rhab_dynamic', 'omm_dynamic', 'colors']
            if self.overlay_enabled:
                to_bind.append('overlay')

            with self.eye_buffers.grouped_bind(to_bind):

                nb_units = self._model.size if self.output_mode == EyeOutput.Raw else self._model.N

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

    # TODO: The following public methods should probably be all under the hood

    def flush(self) -> np.ndarray:
        """
        Blocks until all queued frames on the GPU are rendered, downloads the data, and resets the counter.
        This is used to retrieve a full batch, or the final partial batch at the end of a simulation.
        """

        if self._frame_index == 0:
            return np.array([])

        glFinish()  # Block until all rendering commands are complete

        frames_to_read = int(self._frame_index)

        # Direct synchronous download is ok here
        with self.eye_buffers['colors'].bind():
            data_bytes = glGetBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self._model.size * 16 * frames_to_read)
            data_np = np.frombuffer(data_bytes, dtype=np.float32).reshape(frames_to_read, self._model.size, 4)

        self._frame_index = 0
        return data_np.copy()       # TODO: Should return a timeseries VisualOutput

    def sync_cpu(self, force_all=False):
        """
        CPU data sync (contiguous block synchronisation).
        Checks the high-level flag first, then walks the mask for surgical uploads.
        """
        buf = self._model.buffer

        # Ommatidia
        if buf.ommatidia_stale or force_all:
            if force_all:
                self.eye_buffers['omm_static'].write(buf.ommatidia_static)
                self.eye_buffers['omm_dynamic'].write(buf.ommatidia_dynamic)
            else:
                dirty_idx = np.where(buf.ommatidia_stale_mask)[0]
                if dirty_idx.size > 0:
                    jumps = np.where(np.diff(dirty_idx) != 1)[0] + 1
                    for block in np.split(dirty_idx, jumps):
                        start, nb = block[0], block.size
                        self.eye_buffers['omm_static'].write(buf.ommatidia_static[start:start + nb], start=start)
                        self.eye_buffers['omm_dynamic'].write(buf.ommatidia_dynamic[start:start + nb], start=start)

            buf.ommatidia_stale = False
            buf.ommatidia_stale_mask.fill(False)

        # Rhabdomeres
        if buf.rhabdomeres_stale or force_all:
            if force_all:
                self.eye_buffers['rhab_static'].write(buf.rhabdomere_static)
                self.eye_buffers['rhab_dynamic'].write(buf.rhabdomere_dynamic)
            else:
                dirty_idx = np.where(buf.rhabdomeres_stale_mask)[0]
                if dirty_idx.size > 0:
                    jumps = np.where(np.diff(dirty_idx) != 1)[0] + 1
                    for block in np.split(dirty_idx, jumps):
                        start, nb = block[0], block.size
                        self.eye_buffers['rhab_static'].write(buf.rhabdomere_static[start:start + nb], start=start)
                        self.eye_buffers['rhab_dynamic'].write(buf.rhabdomere_dynamic[start:start + nb], start=start)

            buf.rhabdomeres_stale = False
            buf.rhabdomeres_stale_mask.fill(False)

    # Main public methods

    @abstractmethod
    def draw(self, view_mode: DisplayMode, point_of_view: Agent):
        # Each subclass implements its own rendering logic
        raise NotImplementedError

    def step(self, readback: bool = True) -> Optional['VisualOutput']:
        """
        Advance biological dynamics by one time step and update the eyes.
        The time step is taken from the attached Context (context.dt).

        Behaviour by batch size:
            - batch_size == 1: blocks (via ping-pong PBO) and returns this frame's VisualOutput.
            - batch_size > 1: queues the frame on the GPU. Returns None until the batch is full,
              then returns a VisualOutput wrapping the whole batch.

        Args:
            readback: If False, skip the CPU readback. Only the colour download is skipped.
            The frame is still rendered and biological state still advances.
        """

        if self._context is None:
            raise RuntimeError("renderer.step() requires an attached Context.")

        # Sync any CPU-side changes to the eye model
        self.sync_cpu()

        # Advance dithering for Monte-Carlo noise decorrelation
        if self._time_dithering:
            self._dither_counter += 1

        # GPU dispatch
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        self._main_render()
        # glFinish()

        self._frame_index += 1

        # No readback, work here is done, return
        if not readback:
            return None

        out_array: Optional[np.ndarray] = None

        if self.runs_interactive or self._batch_size == 1:
            # Interactive path: return previous frame via ping-pong PBO
            out_array = self._colors_read_async()
        else:
            # Batched path: return full block (only when batch is full)
            if self._frame_index >= self._batch_size:
                print(f"  > GPU batch is full. Flushing {self._batch_size} frames...")
                out_array = self.flush()

        if out_array is None or out_array.size == 0:
            return None

        return VisualOutput(out_array, self._model)

    def set_overlay(self,
                    values: Optional[Union[Dict['EyeView', np.array], np.array]] = None,
                    range: Optional[Tuple[float, float]] = None,
                    colormap: 'Colormap' = Colormap.Thermal,
                    compression: float = 0.5,
                    autorange_perc: int = 98
                    ):
        """
        Upload data for overlay visualisation.

        Args:
            values: Either a flat array in global order (total_rhabdomeres,) or a dict mapping Eye -> per-eye array
            range: (min, max) bounds for the colourmap. None = auto
            colormap: Which colourmap to use (Colormap enum)
            compression: Power exponent for dynamic range compression
                         1.0 = linear, 0.5 = sqrt, lower = brings out detail
        """

        siz = self._model.size

        if 'overlay' not in self.eye_buffers:
            self.eye_buffers.allocate('overlay',
                                      dtype=np.float32,
                                      count=siz,
                                      usage=GL_DYNAMIC_DRAW)

        if values is None:
            # Clear and use default luminance
            self._eye_uniforms.update(
                overlay_fallback=True,
                overlay_data_min=0.0,
                overlay_data_max=1.0,
                overlay_colormap=int(Colormap.Thermal),
                overlay_compression=1.0
            )
            return

        if isinstance(values, dict):
            merged = np.zeros(siz, dtype=np.float32)

            for eye, data in values.items():
                if len(data) == len(eye):
                    data = np.repeat(data, self._model.R)
                merged[eye.indices] = data
            values = merged

        buf = np.ascontiguousarray(values, dtype=np.float32).ravel()
        buf_size = len(buf)

        if buf_size != siz:
            raise ValueError(f"scalar_data has {buf_size} elements, expected {siz}.")

        elif self.eye_buffers['overlay'].count != siz:
            self.eye_buffers['overlay'].resize(siz)

        self.eye_buffers['overlay'].write(buf)

        self._overlay_autorange_perc = autorange_perc

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

        self._eye_uniforms.update(
            overlay_fallback=False,
            overlay_data_min=self._overlay_range[0],
            overlay_data_max=self._overlay_range[1],
            overlay_colormap=int(self._overlay_colormap),
            overlay_compression=self._overlay_compression,
        )

    # Public properties and methods

    @property
    def context(self) -> Optional['Context']:
        """The Context this renderer is attached to (if any)."""
        return self._context

    def attach_context(self, context: 'Context') -> None:
        """
        Bind this renderer to a Context. Needed so step() can be called without
        an explicit dt.
        """
        if self._context is context:
            return
        self._context = context
        context.renderer = self

    @property
    def nb_samples(self):
        return self._samples_per_rhab

    @nb_samples.setter
    def nb_samples(self, value):

        max_tot_samples = self._max_ssbo_bytes // 16
        max_per_r = max(1, max_tot_samples // self._model.size)
        value_clamped = int(np.clip(value, 1, max_per_r))

        if value_clamped == self._samples_per_rhab:
            return

        if value_clamped != value:
            print(f"Warning: Clamped samples per rhabdomere to {value_clamped} (HW limit is {max_per_r}).")

        self._samples_per_rhab = value_clamped

        self.eye_buffers['rays_intermediate'].resize(self._model.size * self._samples_per_rhab)

        mc_noise = 0.65 / np.sqrt(max(1, self._samples_per_rhab))
        self._noise_threshold = max(0.05, mc_noise)

    @property
    def hybrid_sampling(self) -> bool:
        """Toggle between Importance Sampling (False) and Hybrid Weighted Sampling (True)."""
        return self._use_hybrid_sampling

    @hybrid_sampling.setter
    def hybrid_sampling(self, value: bool):
        self._use_hybrid_sampling = bool(value)
        self._eye_uniforms.update(use_hybrid_sampling=self._use_hybrid_sampling)

    @property
    def sampling_mode(self) -> SamplingMode:
        """The sensitivity profile used for weighting: 'gaussian' or 'airy'."""
        return self._sampling_mode

    @sampling_mode.setter
    def sampling_mode(self, value: Union[int, str, SamplingMode]):
        self._sampling_mode = self._to_enum(value, SamplingMode)
        self._eye_uniforms.update(sampling_mode=int(self._sampling_mode))
        print(f"Sampling Mode: {self._sampling_mode.name}")

    @property
    def time_dithering(self):
        return self._time_dithering

    @time_dithering.setter
    def time_dithering(self, value: bool):
        self._time_dithering = bool(value)
        print(f"Time dithering: {'Enabled' if self._time_dithering else 'Disabled'}.")

    @property
    def randomness_mode(self) -> RandomnessMode:
        """The randomness mode used for sampling: 'pseudorandom', 'halton' or 'stratified'."""
        return self._randomness_mode

    @randomness_mode.setter
    def randomness_mode(self, value: Union[int, str, RandomnessMode]):
        self._randomness_mode = self._to_enum(value, RandomnessMode)
        self._eye_uniforms.update(randomness_mode=int(self._randomness_mode))
        print(f"Randomness Mode: {self._randomness_mode.name}")

    @property
    def photon_concentration(self):
        return self._eye_uniforms['photon_concentration_factor']

    @photon_concentration.setter
    def photon_concentration(self, value: float):
        self._eye_uniforms.update(photon_concentration_factor=float(value))

    @property
    def overlay_enabled(self) -> bool:
        return self._overlay_enabled and 'overlay' in self.eye_buffers and self.eye_buffers['overlay'].count > 0

    @overlay_enabled.setter
    def overlay_enabled(self, value: bool):
        enable = bool(value)
        self._overlay_enabled = enable
        if enable:
            self._eye_uniforms.update(
                overlay_fallback=True,
                overlay_data_min=0.0,
                overlay_data_max=1.0,
                overlay_colormap=int(Colormap.Thermal),
                overlay_compression=1.0
            )
            # dummy call to initialise the buffer
            self.set_overlay()

    @property
    def selected_ommatidia(self) -> Optional[list]:
        active = [int(x) for x in self._selected_omm_indices if x != -1]
        return active if active else None

    @selected_ommatidia.setter
    def selected_ommatidia(self, values: Optional[Union[int, Sequence[int], np.ndarray]]):
        self._selected_omm_indices.fill(-1)

        if values is None:
            return

        if isinstance(values, np.ndarray):
            vals = values.ravel()
            count = min(len(vals), 10)
            self._selected_omm_indices[:count] = vals[:count].astype(np.int32)
            return

        if isinstance(values, int):
            self._selected_omm_indices[0] = values
            return

        for i, val in enumerate(list(values)[:10]):
            self._selected_omm_indices[i] = int(val)

    # TODO: These should be renamed
    @property
    def microsaccades_enabled(self):
        return self._microsaccades_enabled

    @microsaccades_enabled.setter
    def microsaccades_enabled(self, value: bool):
        self._microsaccades_enabled = bool(value)
        self._eye_uniforms.update(enable_actuation=self._microsaccades_enabled)

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
            self.__cones_vao,
            self.__hemisph_vao
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