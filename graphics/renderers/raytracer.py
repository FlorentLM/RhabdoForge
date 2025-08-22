from typing import Tuple

import numpy as np
import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *
from pyglm import glm
from pytinybvh import BVH

from graphics.renderers.base import EyeRendererBase
from graphics.scene import Scene, MeshAsset
from graphics.utils import load_texture, VEC_DTYPE, ShaderProgram


class RaytracingSceneBaker:
    """
    Manages the conversion of a high-level Scene into a GPU-ready format for ray tracing
    This class builds acceleration structures and manages all necessary SSBOs
    """

    def __init__(self, scene: Scene, point_radius: float = 0.1):
        self.scene = scene
        self.point_radius = point_radius

        # OpenGL handles (SSBOs, textures)
        self.skybox_texture = 0
        self.triangles_ssbo = 0
        self.triangle_bvh_ssbo = 0
        self.materials_ssbo = 0
        self.scene_texture_array = 0
        self.point_primitives_ssbo = 0
        self.point_bvh_ssbo = 0

        # CPU-side data buffers
        self.triangle_bvh_nodes = None
        self.point_bvh_nodes = None

        # Build statistics
        self.num_total_triangles = 0
        self.num_total_points = 0

        # Material mapping
        self.material_map = {}

        self._build()

    def _build(self):

        print("Building Ray Tracing scene representation...")

        self.skybox_texture = self.scene.skybox_texture_id

        # Pack materials and textures for mesh instances
        if self.scene.instances:
            self._pack_materials_and_textures()
            self._build_triangle_bvh()

        # Process and build point cloud geometry
        if self.scene.point_cloud:
            self._build_point_cloud_bvh()

        print("Ray Tracing scene build complete.")

    def _pack_materials_and_textures(self):
        """ Creates a texture array and a material buffer from all unique meshes """

        unique_meshes = {inst.asset for inst in self.scene.instances if isinstance(inst.asset, MeshAsset)}

        if not unique_meshes:
            return

        self.material_map = {mesh.id: i for i, mesh in enumerate(unique_meshes)}

        texture_paths = sorted(list({mesh.texture_path for mesh in unique_meshes}))
        texture_map = {path: i for i, path in enumerate(texture_paths)}

        texture_ids = [load_texture(path) for path in texture_paths]
        if texture_ids:
            self.scene_texture_array = self._create_texture_array(texture_ids)
            glDeleteTextures(len(texture_ids), texture_ids)

        num_materials = len(unique_meshes)
        material_data = np.zeros(num_materials * 4, dtype=VEC_DTYPE)
        mat_data_u32 = material_data.view(np.uint32)

        for mesh in unique_meshes:
            mat_idx = self.material_map[mesh.id]
            tex_idx = texture_map[mesh.texture_path]
            mat_data_u32[mat_idx * 4] = tex_idx

        self.materials_ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.materials_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, material_data.nbytes, material_data, GL_STATIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def _build_triangle_bvh(self):
        """ Flattens all scene instances, builds a single BVH, and packs data for the GPU """

        all_verts_pos = []
        all_verts_uv = []
        all_material_indices = []

        for instance in self.scene.instances:
            if not isinstance(instance.asset, MeshAsset):
                continue

            mesh = instance.asset
            interleaved = mesh.vertex_data.reshape(-1, 5)
            positions = interleaved[:, :3]
            positions_h = np.hstack([positions, np.ones((len(positions), 1), dtype=VEC_DTYPE)])

            np_transform = np.asarray(instance.transform)
            transformed_pos_h = (np_transform @ positions_h.T).T

            all_verts_pos.append(transformed_pos_h[:, :3])
            all_verts_uv.append(interleaved[:, 3:5])

            num_tris = len(positions) // 3
            mat_idx = self.material_map[mesh.id]
            all_material_indices.append(np.full(num_tris, mat_idx, dtype=np.uint32))

        if not all_verts_pos:
            return

        flat_verts_pos = np.concatenate(all_verts_pos, axis=0)
        flat_verts_uv = np.concatenate(all_verts_uv, axis=0)
        flat_material_indices = np.concatenate(all_material_indices)

        triangles_for_bvh = flat_verts_pos.reshape(-1, 9)
        self.num_total_triangles = len(triangles_for_bvh)

        print(f"Building BVH for {self.num_total_triangles:,} total triangles...")

        bvh = BVH.from_triangles(triangles_for_bvh)
        bvh_buffers = bvh.get_buffers()
        self.triangle_bvh_nodes = bvh_buffers['nodes']
        prim_indices = bvh_buffers['prim_indices']

        print(f"Triangle BVH built with {bvh.node_count} nodes.")

        reordered_verts_pos = self._reorder_vertices(flat_verts_pos, prim_indices)
        reordered_verts_uv = self._reorder_vertices(flat_verts_uv, prim_indices)
        reordered_mat_indices = flat_material_indices[prim_indices]

        packed_triangles = np.zeros((self.num_total_triangles, 20), dtype=VEC_DTYPE)
        v = reordered_verts_pos.reshape(self.num_total_triangles, 3, 3)
        uv = reordered_verts_uv.reshape(self.num_total_triangles, 3, 2)

        packed_triangles[:, 0:3] = v[:, 0, :]
        packed_triangles[:, 4:7] = v[:, 1, :]
        packed_triangles[:, 8:11] = v[:, 2, :]
        packed_triangles[:, 12:14] = uv[:, 0, :]
        packed_triangles[:, 14:16] = uv[:, 1, :]
        packed_triangles[:, 16:18] = uv[:, 2, :]
        packed_triangles[:, 18] = reordered_mat_indices.view(VEC_DTYPE)

        self.triangles_ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.triangles_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, packed_triangles.nbytes, packed_triangles.flatten(), GL_STATIC_DRAW)

        self.triangle_bvh_ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.triangle_bvh_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.triangle_bvh_nodes.nbytes, self.triangle_bvh_nodes, GL_STATIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def _build_point_cloud_bvh(self):
        """ Builds a BVH for the scene's point cloud and packs data """

        pc = self.scene.point_cloud
        self.num_total_points = pc.num_points

        print(f"Building BVH for {self.num_total_points:,} points with radius {self.point_radius}...")

        bvh = BVH.from_points(pc.points, radius=self.point_radius)
        bvh_buffers = bvh.get_buffers()
        self.point_bvh_nodes = bvh_buffers['nodes']
        prim_indices = bvh_buffers['prim_indices']

        print(f"Point BVH built with {bvh.node_count} nodes.")

        reordered_points = pc.points[prim_indices]
        reordered_normals = pc.normals[prim_indices]
        reordered_colors = pc.colors[prim_indices]

        packed_points = np.zeros((self.num_total_points, 12), dtype=VEC_DTYPE)
        packed_points[:, 0:3] = reordered_points
        packed_points[:, 4:7] = reordered_normals
        packed_points[:, 8:11] = reordered_colors

        self.point_primitives_ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.point_primitives_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, packed_points.nbytes, packed_points.flatten(), GL_STATIC_DRAW)

        self.point_bvh_ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.point_bvh_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, self.point_bvh_nodes.nbytes, self.point_bvh_nodes, GL_STATIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def _reorder_vertices(self, vertices, prim_indices):
        """ Helper to reorder vertex attributes based on primitive indices """

        num_verts_per_prim = vertices.shape[0] // len(prim_indices)
        indices = np.repeat(prim_indices * num_verts_per_prim, num_verts_per_prim) + \
                  np.tile(np.arange(num_verts_per_prim), len(prim_indices))
        return vertices[indices]

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

    def free(self):
        """ Deletes all OpenGL resources managed by this class """

        buffers = [b for b in [
            self.triangles_ssbo, self.triangle_bvh_ssbo, self.materials_ssbo,
            self.point_primitives_ssbo, self.point_bvh_ssbo
        ] if b != 0]
        if buffers:
            glDeleteBuffers(len(buffers), buffers)
        if self.scene_texture_array != 0:
            glDeleteTextures(1, [self.scene_texture_array])

        print("RayTracingSceneManager resources freed.")


class EyeRendererRay(EyeRendererBase):
    def __init__(self, eye_model, scene: Scene, time_dithering: bool = True, nb_samples: int = 256, window_size: Tuple[int, int] = (1280, 720), point_radius: float = 0.1):
        super().__init__(eye_model, window_size=window_size, time_dithering=time_dithering, nb_samples=nb_samples)

        # Store a reference to the scene manager
        self.rt_scene = RaytracingSceneBaker(scene, point_radius=point_radius)
        self.point_radius = point_radius

        print("Compiling ray-tracing and reduction shaders...")
        self.raytrace_shader = ShaderProgram(comp_path='shaders/ommatidia_raytracing.comp')
        self.reduction_shader = ShaderProgram(comp_path='shaders/rays_reduction.comp')

        # Intermediate Ray result SSBO
        self.ray_results_ssbo = glGenBuffers(1)

        # Set the default number of samples with the setter to allocate the SSBO
        self._samples_per_ommatidium = 0
        self.samples_per_ommatidium = nb_samples

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

    def _raytrace(self, camera):

        # Pass 1: Ray-tracing

        self.raytrace_shader.use()

        # Bind Textures
        # Bind skybox cubemap to texture unit 0
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.rt_scene.skybox_texture)

        # Bind the scene's texture array (managed by rt_scene_manager) to texture unit 1
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self.rt_scene.scene_texture_array)

        # Tell the shader which texture unit to use for each sampler
        glUniform1i(self.raytrace_shader.get_loc('u_skybox'), 0)
        glUniform1i(self.raytrace_shader.get_loc('u_scene_textures'), 1)

        # Bind Shader Storage Buffers (SSBOs)
        # Binding 0: Input ommatidia data (owned by this renderer)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.input_om_ssbo)
        # Binding 4: Output for this pass - the raw results for every single ray (owned by this renderer)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, self.ray_results_ssbo)

        # Bind all scene geometry and BVH data
        # Binding 1: Triangle primitive data
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.rt_scene.triangles_ssbo)
        # Binding 2: Material data
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self.rt_scene.materials_ssbo)
        # Binding 3: Triangle BVH nodes
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self.rt_scene.triangle_bvh_ssbo)
        # Binding 5: Point cloud BVH nodes
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, self.rt_scene.point_bvh_ssbo)
        # Binding 6: Point primitive data
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, self.rt_scene.point_primitives_ssbo)

        # Set Uniforms
        num_tri_nodes = len(
            self.rt_scene.triangle_bvh_nodes) if self.rt_scene.triangle_bvh_nodes is not None else 0

        num_point_nodes = len(
            self.rt_scene.point_bvh_nodes) if self.rt_scene.point_bvh_nodes is not None else 0

        glUniform1ui(self.raytrace_shader.get_loc('u_num_triangle_bvh_nodes'), num_tri_nodes)
        glUniform1ui(self.raytrace_shader.get_loc('u_num_point_bvh_nodes'), num_point_nodes)

        # Set renderer-specific uniforms
        glUniform1i(self.raytrace_shader.get_loc('u_samples_per_ommatidium'), self.samples_per_ommatidium)
        glUniform1f(self.raytrace_shader.get_loc('u_point_radius'), self.point_radius)
        glUniform1f(self.raytrace_shader.get_loc('u_time'), float(self._time_counter))

        # Set camera uniforms for transforming rays into world space
        camera_to_world_matrix = glm.inverse(camera.view)
        glUniformMatrix4fv(self.raytrace_shader.get_loc('u_camera_to_world'), 1, False,
                           glm.value_ptr(camera_to_world_matrix))

        # Dispatch compute shader
        # Calculate the number of workgroups needed to process all rays
        work_groups = (self.total_samples + 255) // 256  # Workgroup size is 256 in the shader
        glDispatchCompute(work_groups, 1, 1)

        # This barrier is critical: it ensures all writes to the ray_results_ssbo
        # from this pass are complete before the next pass tries to read from it
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        self.raytrace_shader.stop()

    def _reduction(self):

        # Pass 2: Reduction

        self.reduction_shader.use()

        # Bind buffers
        # Binding 3: Input for this pass - the raw ray results from Pass 1
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self.ray_results_ssbo)
        # Binding 4: Output for this pass - the final averaged color for each ommatidium
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, self.final_colors_ssbo)

        # Set uniforms
        glUniform1i(self.reduction_shader.get_loc('u_samples_per_ommatidium'), self.samples_per_ommatidium)
        glUniform1i(self.reduction_shader.get_loc('u_num_ommatidia'), self.num_ommatidia)

        # Dispatch compute shader
        # Calculate workgroups needed to process all ommatidia
        work_groups = (self.num_ommatidia + 63) // 64  # Workgroup size is 64 in the shader
        glDispatchCompute(work_groups, 1, 1)

        # Another critical barrier: ensures the final_colors_ssbo is fully written
        # before the CPU or the drawing shader tries to read from it in the next frame
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        self.reduction_shader.stop()

    def _compute_colors(self, camera):
        """ The core ommatidia rendering logic """

        # Pass 1: Ray-trace
        self._raytrace(camera)

        # Pass 2: Reduction
        self._reduction()

        # Unbind all resources
        for i in range(7):
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)

    def get_ommatidia_data(self, camera, to_cpu=False):

        self._compute_colors(camera)

        if self._time_dithering:
            self._time_counter += 1

        if to_cpu:
            self._fetch_to_cpu()

        return self.cpu_read_buffer

    def draw(self, view_mode: str, camera, tiled_mode: bool = False):
        """ Renders one of the rasterizer's supported views to the screen """

        if view_mode == 'compound_eye':
            # This calls the draw() method in EyeRendererBase for Voronoi rendering
            super().draw(tiled_mode=tiled_mode)

    def free(self):
        """ Frees all GPU resources, including shaders and all buffers """

        glDeleteBuffers(1, [self.ray_results_ssbo])

        if self.raytrace_shader: self.raytrace_shader.free()
        if self.reduction_shader: self.reduction_shader.free()

        self.rt_scene.free()

        super().free()