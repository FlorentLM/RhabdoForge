from abc import ABC, abstractmethod

import numpy as np
from pyglm import glm
from OpenGL.GL import *
import ctypes

from geometry.primitives import CONE_VERTICES
from geometry.compound_eyes import CompoundEye
from graphics.scene import RaytracingScene, Scene
from graphics.utils import ShaderProgram


class EyeRendererBase(ABC):
    """
    Abstract base class for an insect eye model, handling visualization and common properties
    """
    def __init__(self, eye_model: CompoundEye, time_dithering=True, nb_samples=256):
        self.model = eye_model
        self.num_ommatidia = self.model.num_ommatidia

        self.ommatidia_input_data = self.model.pack()

        # Default umber of rays to sample per ommatidium
        self._samples_per_ommatidium = nb_samples

        # A counter for time dithering during sampling
        self._time_dithering = time_dithering
        self._time_counter = 0

        # Visualization resources (lazy-loaded)
        self._voronoi_shader = None
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

    def get_ommatidia_data(self, *args, to_cpu=False, **kwargs) -> np.ndarray:

        # Subclass runs its specific compute pass
        self._compute_colors(*args, **kwargs)

        if to_cpu:
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
    def voronoi_shader(self):
        if self._voronoi_shader is None:
            self._voronoi_shader = ShaderProgram(vert_path='shaders/voronoi.vert', frag_path='shaders/voronoi.frag')
        return self._voronoi_shader

    @property
    def voronoi_vao(self):
        if self._voronoi_vao is None:
            self._cone_vertex_count = len(CONE_VERTICES) // 3
            vao = glGenVertexArrays(1)
            glBindVertexArray(vao)

            vbo = glGenBuffers(1)
            glBindBuffer(GL_ARRAY_BUFFER, vbo)
            glBufferData(GL_ARRAY_BUFFER, CONE_VERTICES.nbytes, CONE_VERTICES, GL_STATIC_DRAW)

            pos_loc = glGetAttribLocation(self.voronoi_shader.program_id, "a_cone_vertex_pos")
            glEnableVertexAttribArray(pos_loc)
            glVertexAttribPointer(pos_loc, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

            glBindVertexArray(0)
            self._voronoi_vao = vao

        return self._voronoi_vao

    def draw(self, tiled_mode=False):
        """ Draws the Voronoi visualization using the computed colors """

        self.voronoi_shader.use()
        glEnable(GL_DEPTH_TEST)

        cone_scale = self.voronoi_scale if tiled_mode else self.receptive_field_scale

        # Get current viewport dimensions to calculate aspect ratio
        viewport = glGetIntegerv(GL_VIEWPORT)
        # avoid division by zero if window is not yet setup
        aspect_ratio = viewport[2] / viewport[3] if viewport[3] > 0 else 1.0

        glUniform1f(self.voronoi_shader.get_loc('u_aspect_ratio'), 1.0)
        glUniform1i(self.voronoi_shader.get_loc('u_tiled_mode'), tiled_mode)
        glUniform1f(self.voronoi_shader.get_loc('u_cone_scale'), cone_scale)

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
        self.voronoi_shader.stop()

    def free(self):
        """ Free GPU resources """

        glDeleteBuffers(4, [self.input_om_ssbo, self.final_colors_ssbo, self.pbo_ids[0], self.pbo_ids[1]])

        if self._voronoi_shader:
            self._voronoi_shader.free()

        if self._voronoi_vao:
            glDeleteVertexArrays(1, [self._voronoi_vao])


class EyeRendererRaster(EyeRendererBase):
    def __init__(self, eye_model: CompoundEye, time_dithering=True, nb_samples=256):
        super().__init__(eye_model, time_dithering=time_dithering, nb_samples=nb_samples)
        self.rasterizer_shader = ShaderProgram(comp_path='shaders/ommatidia_rasterizer.comp')

    @property
    def samples_per_ommatidium(self):
        return self._samples_per_ommatidium

    @samples_per_ommatidium.setter
    def samples_per_ommatidium(self, value):
        self._samples_per_ommatidium = int(min(32768, max(1, value)))
        # no buffers to reallocate for this implementation

    def _compute_colors(self, cubemap_texture_id):
        """ The core ommatidia rendering logic """

        self.rasterizer_shader.use()

        # Set uniforms for the data pass
        glUniform1i(self.rasterizer_shader.get_loc('u_num_ommatidia'), self.num_ommatidia)
        glUniform1i(self.rasterizer_shader.get_loc('u_samples_per_ommatidium'), self.samples_per_ommatidium)
        glUniform1f(self.rasterizer_shader.get_loc('u_time'), float(self._time_counter))

        # Bind input cubemap (texture unit 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap_texture_id)
        glUniform1i(self.rasterizer_shader.get_loc('u_scene_cubemap'), 0)

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
        self.rasterizer_shader.stop()
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, 0)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0)

    def free(self):
        self.rasterizer_shader.free()
        super().free()


class EyeRendererRay(EyeRendererBase):
    def __init__(self, eye_model, scene: Scene, time_dithering=True, nb_samples=256, point_radius=0.1):
        super().__init__(eye_model, time_dithering=time_dithering, nb_samples=nb_samples)

        self.point_radius = point_radius

        print("Compiling ray-tracing and reduction shaders...")
        self.raytrace_shader = ShaderProgram(comp_path='shaders/ommatidia_raytracing.comp')
        self.reduction_shader = ShaderProgram(comp_path='shaders/rays_reduction.comp')

        # Initialise buffer IDs to 0 (to prevent errors if free() is called before creation)
        self.triangles_ssbo = 0
        self.triangle_bvh_ssbo = 0
        self.materials_ssbo = 0
        self.scene_texture_array = 0
        self.point_bvh_ssbo = 0
        self.point_primitives_ssbo = 0

        # Intermediate Ray result SSBO
        self.ray_results_ssbo = glGenBuffers(1)

        # Build initial scene
        self.rt_scene = RaytracingScene(scene)
        self.point_cloud = scene.point_cloud
        self._create_scene_buffers()

        # Set the default number of samples with the setter to allocate the SSBO
        self._samples_per_ommatidium = 0
        self.samples_per_ommatidium = nb_samples

    def _create_scene_buffers(self):
        """ Generates and populates all GPU buffers for the current scene """
        print("Creating and uploading scene data to GPU buffers...")

        # Generate new buffer IDs
        self.triangles_ssbo = glGenBuffers(1)
        self.triangle_bvh_ssbo = glGenBuffers(1)
        self.materials_ssbo = glGenBuffers(1)
        self.point_bvh_ssbo = glGenBuffers(1)
        self.point_primitives_ssbo = glGenBuffers(1)

        # Triangle buffers
        if self.rt_scene.triangles is not None:
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.triangles_ssbo)
            glBufferData(GL_SHADER_STORAGE_BUFFER, self.rt_scene.triangles.nbytes, self.rt_scene.triangles,
                         GL_STATIC_DRAW)

        if self.rt_scene.triangle_bvh_nodes is not None:
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.triangle_bvh_ssbo)
            glBufferData(GL_SHADER_STORAGE_BUFFER, self.rt_scene.triangle_bvh_nodes.nbytes,
                         self.rt_scene.triangle_bvh_nodes, GL_STATIC_DRAW)

        if self.rt_scene.materials is not None:
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.materials_ssbo)
            glBufferData(GL_SHADER_STORAGE_BUFFER, self.rt_scene.materials.nbytes, self.rt_scene.materials,
                         GL_STATIC_DRAW)

        self.scene_texture_array = self._create_texture_array(self.rt_scene.texture_ids)

        # Point cloud bufefrs
        if self.point_cloud:
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.point_bvh_ssbo)
            glBufferData(GL_SHADER_STORAGE_BUFFER, self.point_cloud.bvh_nodes.nbytes, self.point_cloud.bvh_nodes,
                         GL_STATIC_DRAW)

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.point_primitives_ssbo)
            glBufferData(GL_SHADER_STORAGE_BUFFER, self.point_cloud.point_attributes.nbytes,
                         self.point_cloud.point_attributes, GL_STATIC_DRAW)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def _free_scene_buffers(self):
        """ Deletes all GPU buffers associated with current scene """

        buffers_to_delete = [
            self.triangles_ssbo, self.triangle_bvh_ssbo, self.materials_ssbo,
            self.point_bvh_ssbo, self.point_primitives_ssbo
        ]

        # Filter out any buffers that were not created (ID is 0)
        buffers_to_delete = [buf for buf in buffers_to_delete if buf != 0]

        if buffers_to_delete:
            glDeleteBuffers(len(buffers_to_delete), buffers_to_delete)

        if self.scene_texture_array != 0:
            glDeleteTextures(1, [self.scene_texture_array])

        self.triangles_ssbo = self.triangle_bvh_ssbo = self.materials_ssbo = 0
        self.point_bvh_ssbo = self.point_primitives_ssbo = self.scene_texture_array = 0

    def _create_texture_array(self, texture_ids):

        if not texture_ids:
            return 0

        glBindTexture(GL_TEXTURE_2D, texture_ids[0])
        tex_w = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_WIDTH)
        tex_h = glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT)
        glBindTexture(GL_TEXTURE_2D, 0)

        layer_count = len(texture_ids)
        tex_array_id = glGenTextures(1)

        glBindTexture(GL_TEXTURE_2D_ARRAY, tex_array_id)
        glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_RGBA8, tex_w, tex_h, layer_count)

        for i, tex_id in enumerate(texture_ids):
            glCopyImageSubData(
                tex_id, GL_TEXTURE_2D, 0, 0, 0, 0,
                tex_array_id, GL_TEXTURE_2D_ARRAY, 0, 0, 0, i,
                tex_w, tex_h, 1
            )

        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)

        return tex_array_id

    def replace_scene(self, scene: Scene):
        """
        Rebuilds the entire ray tracing scene on the GPU
        """
        self._free_scene_buffers()
        self.rt_scene = RaytracingScene(scene)
        self.point_cloud = scene.point_cloud
        self._create_scene_buffers()

    @property
    def samples_per_ommatidium(self):
        return self._samples_per_ommatidium

    @samples_per_ommatidium.setter
    def samples_per_ommatidium(self, value):
        bytes_per_sample = 16
        max_total_samples = self._max_ssbo_size_bytes // bytes_per_sample
        max_samples_per_om = max(1, max_total_samples // self.num_ommatidia)
        new_value = int(np.clip(value, 1, max_samples_per_om))

        if new_value != value:
            print(f"Warning: Clamped samples per ommatidium to {new_value} (HW limit is {max_samples_per_om}).")

        if new_value == self._samples_per_ommatidium:
            return

        self._samples_per_ommatidium = new_value
        self.total_samples = self.num_ommatidia * self._samples_per_ommatidium
        required_buffer_size = self.total_samples * bytes_per_sample

        print(f"Allocating ray result buffer for {self.total_samples:,} total samples ({required_buffer_size / (1024*1024):.2f} MB).")

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ray_results_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, required_buffer_size, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def _compute_colors(self, camera, skybox_texture_id):
        # First pass: Ray tracing
        self.raytrace_shader.use()

        # Bind common stuff
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, skybox_texture_id)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.scene_texture_array)
        glUniform1i(self.raytrace_shader.get_loc('u_skybox'), 0)
        glUniform1i(self.raytrace_shader.get_loc('u_scene_textures'), 1)

        # Bind input/output buffers
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.input_om_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, self.ray_results_ssbo)

        # Bind triangles data
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.triangles_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self.materials_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self.triangle_bvh_ssbo)

        num_triangle_bvh_nodes = len(
            self.rt_scene.triangle_bvh_nodes) if self.rt_scene.triangle_bvh_nodes is not None else 0
        glUniform1ui(self.raytrace_shader.get_loc('u_num_triangle_bvh_nodes'), num_triangle_bvh_nodes)

        # Bind point cloud data
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, self.point_bvh_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, self.point_primitives_ssbo)

        num_point_bvh_nodes = self.point_cloud.num_nodes if self.point_cloud else 0
        glUniform1ui(self.raytrace_shader.get_loc('u_num_point_bvh_nodes'), num_point_bvh_nodes)

        # other uniforms
        glUniform1i(self.raytrace_shader.get_loc('u_samples_per_ommatidium'), self.samples_per_ommatidium)
        glUniform1f(self.raytrace_shader.get_loc('u_point_radius'), self.point_radius)
        glUniform1f(self.raytrace_shader.get_loc('u_time'), float(self._time_counter))
        # glUniform3fv(self.raytrace_shader.get_loc('u_camera_position'), 1, glm.value_ptr(camera.position))
        # glUniformMatrix4fv(self.raytrace_shader.get_loc('u_camera_orientation'), 1, False, glm.value_ptr(camera.orientation))

        camera_to_world_matrix = glm.inverse(camera.view)
        glUniformMatrix4fv(self.raytrace_shader.get_loc('u_camera_to_world'), 1, False, glm.value_ptr(camera_to_world_matrix))

        # Dispatch ray tracing pass
        work_groups = (self.total_samples + 255) // 256
        glDispatchCompute(work_groups, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        # Second pass: reduction
        self.reduction_shader.use()
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self.ray_results_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, self.final_colors_ssbo)
        glUniform1i(self.reduction_shader.get_loc('u_samples_per_ommatidium'), self.samples_per_ommatidium)
        glUniform1i(self.reduction_shader.get_loc('u_num_ommatidia'), self.num_ommatidia)
        work_groups = (self.num_ommatidia + 63) // 64
        glDispatchCompute(work_groups, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        # Unbind all
        self.reduction_shader.stop()
        for i in range(7): glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)

    def free(self):
        """ Frees all GPU resources, including shaders and all buffers """

        self._free_scene_buffers()

        glDeleteBuffers(1, [self.ray_results_ssbo])

        if self.raytrace_shader:
            self.raytrace_shader.free()

        if self.reduction_shader:
            self.reduction_shader.free()

        super().free()