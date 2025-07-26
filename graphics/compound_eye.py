from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from OpenGL.GL import *

from geometry.primitives import CONE_VERTICES
from graphics.eye_model import EyeModel
from graphics.scene import RaytracingScene, Scene
from graphics.utils import load_shaders, load_compute_shader, VEC_DTYPE


class CompoundEyeBase(ABC):
    """
    Abstract base class for an insect eye model, handling visualization and common properties
    """
    def __init__(self, eye_model: EyeModel, time_dithering=True, nb_samples=256):
        self.model = eye_model
        self.num_ommatidia = self.model.num_ommatidia

        self.ommatidia_input_data = self.model.pack()

        # Default umber of rays to sample per ommatidium
        self._samples_per_ommatidium = nb_samples

        # A counter for time dithering during sampling
        self._time_dithering = time_dithering
        self._time_counter = 0

        # Visualization resources (lazy-loaded)
        self._voronoi_program = None
        self._voronoi_vao = None
        self._cone_vertex_count = 0

        # A small fixed scale for the receptive field view
        self.receptive_field_scale = 1.0 / (2.0 * np.pi)
        # Dynamic scale for the Voronoi view (needs to fill the quad)
        self.voronoi_scale = self.model.max_gap() * 2.5

        # Query maximum possible size for an SSBO on current GPU
        self._max_ssbo_size_bytes = glGetIntegerv(GL_MAX_SHADER_STORAGE_BLOCK_SIZE)
        print(f"Max SSBO size: {self._max_ssbo_size_bytes / (1024 * 1024):.2f} MB")

        # SSBO for input ommatidia geometry (directions, angles, etc)
        self.input_om_ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.input_om_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.ommatidia_input_data.nbytes, self.ommatidia_input_data, GL_STATIC_DRAW)

        # Size of the output buffers in bytes (num_ommatidia * 4 floats * 4 bytes/float)
        buffer_size = self.num_ommatidia * 16

        # Final computed colors (written by subclass, read by draw())
        self.final_colors_ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.final_colors_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, buffer_size, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        # Two PBOs for ping-ponging and doing async reading from the CPU-side
        self.pbo_ids = glGenBuffers(2)
        self.pbo_index = 0

        glBindBuffer(GL_PIXEL_PACK_BUFFER, self.pbo_ids[0])
        glBufferData(GL_PIXEL_PACK_BUFFER, buffer_size, None, GL_STREAM_READ)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, self.pbo_ids[1])
        glBufferData(GL_PIXEL_PACK_BUFFER, buffer_size, None, GL_STREAM_READ)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)

        # CPU-side buffer to return the final data
        self.cpu_read_buffer = np.zeros((self.num_ommatidia, 4), dtype=np.float32)

    @property
    def samples_per_ommatidium(self):
        return self._samples_per_ommatidium

    @samples_per_ommatidium.setter
    @abstractmethod
    def samples_per_ommatidium(self, value):
        # Subclasses may need to re-allocate buffers when this changes
        raise NotImplementedError

    @property
    def time_dithering(self):
        return self._time_dithering

    @time_dithering.setter
    def time_dithering(self, value: bool):
        self._time_dithering = bool(value)
        print(f"Time dithering {'ENABLED' if self._time_dithering else 'DISABLED'}.")

    @abstractmethod
    def _compute_colors(self, *args, **kwargs):
        # Each subclass implements its own core rendering logic
        raise NotImplementedError

    def get_ommatidia_data(self, *args, **kwargs) -> np.ndarray:

        # Subclass runs its specific compute pass
        self._compute_colors(*args, **kwargs)

        # Determine which PBO to read from (current) and which to write to (next)
        current_pbo_idx = self.pbo_index
        next_pbo_idx = (self.pbo_index + 1) % 2

        # This is a GPU-to-GPU copy, so it is asynchronous (the command returns immediately).
        # it initiates the copy from the SSBO to the *next* PBO
        glBindBuffer(GL_COPY_READ_BUFFER, self.final_colors_ssbo)
        glBindBuffer(GL_COPY_WRITE_BUFFER, self.pbo_ids[next_pbo_idx])
        glCopyBufferSubData(GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER, 0, 0, self.cpu_read_buffer.nbytes)

        # Process data from the *current* PBO (it was filled in the previous frame)
        glBindBuffer(GL_PIXEL_PACK_BUFFER, self.pbo_ids[current_pbo_idx])

        # Map the buffer ('GL_MAP_READ_BIT' is very important!)
        ptr = glMapBufferRange(GL_PIXEL_PACK_BUFFER, 0, self.cpu_read_buffer.nbytes, GL_MAP_READ_BIT)

        if ptr:
            # Copy the data from the mapped GPU memory to our CPU-side numpy array.
            ctypes.memmove(self.cpu_read_buffer.ctypes.data, ptr, self.cpu_read_buffer.nbytes)
            # IMPORTANT: Unmap the buffer to return control to the GPU
            glUnmapBuffer(GL_PIXEL_PACK_BUFFER)
        else:
            # Handle error if mapping fails
            print("Warning: Failed to map PBO for reading.")

        # Unbind all buffers used in the copy and map operations
        glBindBuffer(GL_PIXEL_PACK_BUFFER, 0)
        glBindBuffer(GL_COPY_READ_BUFFER, 0)
        glBindBuffer(GL_COPY_WRITE_BUFFER, 0)

        # Swap PBO index for the next frame
        self.pbo_index = next_pbo_idx

        # And update the counter for time dithering
        if self._time_dithering:
            self._time_counter += 1

        return self.cpu_read_buffer

    @property
    def voronoi_program(self):
        if self._voronoi_program is None:
            # print("Compiling Voronoi visualization shaders...")
            self._voronoi_program = load_shaders('shaders/voronoi.vert', 'shaders/voronoi.frag')
        return self._voronoi_program

    @property
    def voronoi_vao(self):
        if self._voronoi_vao is None:
            self._cone_vertex_count = len(CONE_VERTICES) // 3
            vao = glGenVertexArrays(1)
            glBindVertexArray(vao)

            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, CONE_VERTICES.nbytes, CONE_VERTICES, GL_STATIC_DRAW)

            pos_loc = glGetAttribLocation(self.voronoi_program, "a_cone_vertex_pos")
            glEnableVertexAttribArray(pos_loc)
            glVertexAttribPointer(pos_loc, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

            glBindVertexArray(0)
            self._voronoi_vao = vao
        return self._voronoi_vao

    def draw(self, tiled_mode=False):
        """ Draws the Voronoi visualization using the computed colors """

        glUseProgram(self.voronoi_program)
        glEnable(GL_DEPTH_TEST)

        cone_scale = self.voronoi_scale if tiled_mode else self.receptive_field_scale

        glUniform1i(glGetUniformLocation(self.voronoi_program, 'u_tiled_mode'), tiled_mode)
        glUniform1f(glGetUniformLocation(self.voronoi_program, 'u_cone_scale'), cone_scale)

        # Binding 0: Ommatidia geometry (directions, origins, etc)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.input_om_ssbo)
        # Binding 1: Final computed colors (from subclass computation)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.final_colors_ssbo)

        glBindVertexArray(self.voronoi_vao)
        glDrawArraysInstanced(GL_TRIANGLES, 0, self._cone_vertex_count, self.num_ommatidia)

        # Unbind everyone
        glBindVertexArray(0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0)
        glDisable(GL_DEPTH_TEST)
        glUseProgram(0)

    def free(self):
        """ Free GPU resources """
        glDeleteBuffers(4, [self.input_om_ssbo, self.final_colors_ssbo, self.pbo_ids[0], self.pbo_ids[1]])
        if self._voronoi_program:
            glDeleteProgram(self._voronoi_program)
        if self._voronoi_vao:
            glDeleteVertexArrays(1, [self._voronoi_vao])


class CompoundEyeRaster(CompoundEyeBase):
    def __init__(self, eye_model: EyeModel, time_dithering=True, nb_samples=256):
        super().__init__(eye_model, time_dithering=time_dithering, nb_samples=nb_samples)
        self.ommatidia_program = load_compute_shader('shaders/ommatidia_raster.comp')

    @property
    def samples_per_ommatidium(self):
        return self._samples_per_ommatidium

    @samples_per_ommatidium.setter
    def samples_per_ommatidium(self, value):
        self._samples_per_ommatidium = int(min(32768, max(1, value)))
        # no buffers to reallocate for this implementation

    def _compute_colors(self, cubemap_texture_id):
        """ The core ommatidia rendering logic """

        glUseProgram(self.ommatidia_program)

        # Set uniforms for the data pass
        glUniform1i(glGetUniformLocation(self.ommatidia_program, 'u_num_ommatidia'), self.num_ommatidia)
        glUniform1i(glGetUniformLocation(self.ommatidia_program, 'u_samples_per_ommatidium'), self.samples_per_ommatidium)
        glUniform1f(glGetUniformLocation(self.ommatidia_program, 'u_time'), float(self._time_counter))

        # Bind input cubemap (texture unit 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap_texture_id)
        glUniform1i(glGetUniformLocation(self.ommatidia_program, 'u_scene_cubemap'), 0)

        # Bind directions SSBO to binding point 0 (for reading)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.input_om_ssbo)
        # Bind colors SSBO to binding point 1 (for writing)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.final_colors_ssbo)

        # Dispatch the compute shader
        # Divide the total number of ommatidia by the workgroup size (64)
        work_groups_x = (self.num_ommatidia + 63) // 64
        glDispatchCompute(work_groups_x, 1, 1)

        # Wait for compute shader to finish writing to the SSBO
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        # Unbind resources
        glUseProgram(0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0)

    def free(self):
        glDeleteProgram(self.ommatidia_program)
        super().free()


class CompoundEyeRay(CompoundEyeBase):
    def __init__(self, eye_model, scene: Scene, time_dithering=True, nb_samples=256):
        super().__init__(eye_model, time_dithering=time_dithering, nb_samples=nb_samples)

        # Pack the scene for ray-tracing
        self.rt_scene = RaytracingScene(scene)

        # print("Compiling ray-tracing and reduction shaders...")
        self.raytrace_program = load_compute_shader('shaders/ommatidia_raytracing.comp')
        self.reduction_program = load_compute_shader('shaders/rays_reduction.comp')

        # SSBO to store the scene triangles
        self.triangles_ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.triangles_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.rt_scene.triangles.nbytes, self.rt_scene.triangles, GL_DYNAMIC_DRAW)

        # SSBO for the ommatidia sampling - it is bound and allocated by the samples_per_ommatidium setter
        self.ray_results_ssbo = glGenBuffers(1)

        # And a SSBO to store scene materials
        self.materials_ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.materials_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.rt_scene.materials.nbytes, self.rt_scene.materials, GL_DYNAMIC_DRAW)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)  # unbind after last glBufferData

        # Set the default number of samples with the setter to allocate the SSBO
        self._samples_per_ommatidium = 0    # just to bypass the check for first call
        self.samples_per_ommatidium = nb_samples

        self.scene_texture_array = self._create_texture_array(self.rt_scene.texture_ids)

    def _create_texture_array(self, texture_ids):

        if not texture_ids:
            return 0

        # This assumes all textures have the same dimensions...
        # TODO: check dimensions or resize textures to a common size

        # query first texture to get its properties
        glBindTexture(GL_TEXTURE_2D, texture_ids[0])
        tex_w = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH)
        tex_h = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT)
        glBindTexture(GL_TEXTURE_2D, 0)

        layer_count = len(texture_ids)

        tex_array_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, tex_array_id)

        # Allocate immutable storage for entire array
        glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGBA8, tex_w, tex_h, layer_count)

        for i, tex_id in enumerate(texture_ids):
            # Copy data from the 2D source texture to a layer of the 2D array texture
            glCopyImageSubData(
                tex_id, GL_TEXTURE_2D, 0, 0, 0, 0,  # source
                tex_array_id, GL_TEXTURE_2D_ARRAY, 0, 0, 0, i,  # dest
                tex_w, tex_h, 1
            )

        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)

        return tex_array_id

    def update_geometry(self, instances: list):
        """
        Updates the ray-tracing scene by re-transforming vertex positions and uploading the new data to the GPU
        (This is the fast path for dynamic objects)
        """

        # Update the CPU-side buffer in the RaytracingScene object
        self.rt_scene.update(instances)

        # Upload the updated buffer to the GPU
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.triangles_ssbo)
        # glBufferSubData does not reallocate, only updates the content
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.rt_scene.triangles.nbytes, self.rt_scene.triangles)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def replace_scene(self, scene: Scene):
        """
        Rebuilds the entire scene representation on the GPU (deletes old buffers and allocates new ones).
        Called when objects are added/removed from the scene
        """

        self.rt_scene = RaytracingScene(scene)

        # Re-allocate triangle buffer
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.triangles_ssbo)
        # Uses glBufferData to re-allocate storage with the new size and data
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.rt_scene.triangles.nbytes, self.rt_scene.triangles, GL_DYNAMIC_DRAW)

        # Re-allocate material buffer
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.materials_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.rt_scene.materials.nbytes, self.rt_scene.materials, GL_STATIC_DRAW)

        # Re-create Texture array
        # delete the old texture array to prevent memory leak
        glDeleteTextures(1, [self.scene_texture_array])
        self.scene_texture_array = self._create_texture_array(self.rt_scene.texture_ids)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    @property
    def samples_per_ommatidium(self):
        return self._samples_per_ommatidium

    @samples_per_ommatidium.setter
    def samples_per_ommatidium(self, value):

        # Theoretical max number of samples affordable based on hardware's SSBO size limit
        max_total_samples = self._max_ssbo_size_bytes // 16     # each sample result is a vec4 (16 bytes)
        max_samples_per_om = max(1, max_total_samples // self.num_ommatidia)

        # Clamp the requested value to safe range
        new_value = int(np.clip(value, 1, max_samples_per_om))

        if new_value != value:
            print(f"Warning: Clamped samples per ommatidium to {new_value} (HW limit is {max_samples_per_om}).")

        if new_value == self._samples_per_ommatidium:
            return

        self._samples_per_ommatidium = new_value

        # (Re)allocate the intermediate buffer
        self.total_samples = self.num_ommatidia * self._samples_per_ommatidium

        # SSBO for ommatidia sampling
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ray_results_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.total_samples * 16, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def _compute_colors(self, camera, skybox_texture_id):

        # First pass: Ray tracing
        glUseProgram(self.raytrace_program)

        # Bind resources
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, skybox_texture_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.scene_texture_array)

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.input_om_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.triangles_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self.materials_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self.ray_results_ssbo)

        # Set uniforms
        num_triangles = len(self.rt_scene.triangles) // 20   # stride 20 (80 bytes total, 4 bytes per float)
        num_materials = len(self.rt_scene.materials) // 4  # stride 4 (16 bytes total, 4 bytes per float)

        glUniform1i(glGetUniformLocation(self.raytrace_program, 'u_skybox'), 0)
        glUniform1i(glGetUniformLocation(self.raytrace_program, 'u_scene_textures'), 1)
        glUniform1i(glGetUniformLocation(self.raytrace_program, 'u_num_triangles'), num_triangles)
        glUniform1i(glGetUniformLocation(self.raytrace_program, 'u_num_ommatidia'), self.num_ommatidia)
        glUniform1i(glGetUniformLocation(self.raytrace_program, 'u_samples_per_ommatidium'), self.samples_per_ommatidium)
        glUniform1f(glGetUniformLocation(self.raytrace_program, 'u_time'), float(self._time_counter))
        glUniform3fv(glGetUniformLocation(self.raytrace_program, 'u_camera_position'), 1, camera.position)
        glUniformMatrix4fv(glGetUniformLocation(self.raytrace_program, 'u_camera_orientation'), 1, True, camera.orientation)
        glUniform1i(glGetUniformLocation(self.raytrace_program, 'u_num_materials'), num_materials)

        # Dispatch
        work_groups = (self.total_samples + 255) // 256
        glDispatchCompute(work_groups, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        # Second pass: Reduction
        glUseProgram(self.reduction_program)

        # output of pass 1 is the input of pass 2
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self.ray_results_ssbo)
        # output of pass 2 is the final color buffer
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, self.final_colors_ssbo)

        # Set uniforms
        glUniform1i(glGetUniformLocation(self.reduction_program, 'u_samples_per_ommatidium'), self.samples_per_ommatidium)
        glUniform1i(glGetUniformLocation(self.reduction_program, 'u_num_ommatidia'), self.num_ommatidia)

        # Dispatch
        work_groups = (self.num_ommatidia + 63) // 64
        glDispatchCompute(work_groups, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        # Unbind resources
        glUseProgram(0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, 0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, 0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)

    def free(self):
        glDeleteBuffers(3, [self.triangles_ssbo, self.ray_results_ssbo, self.materials_ssbo])
        glDeleteTextures(1, [self.scene_texture_array])
        glDeleteProgram(self.raytrace_program)
        glDeleteProgram(self.reduction_program)
        super().free()