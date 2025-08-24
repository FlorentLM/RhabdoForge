from typing import Tuple, List, Dict, Optional

import numpy as np
import OpenGL

from graphics.renderers.panoramic import TextureViewer

OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *
from pyglm import glm
from pytinybvh import BVH, instance_dtype

from graphics.renderers.base import EyeRendererBase
from graphics.scene import Scene, MeshAsset, PointsAsset
from graphics.utils import load_texture, VEC_DTYPE, ShaderProgram


# structured numpy dtype for an instance information SSBO
instance_info_dtype = np.dtype([
    ('transform', np.float32, (4, 4)),
    ('inverse_transform', np.float32, (4, 4)),
    ('blas_node_offset', np.uint32),
    ('primitive_offset', np.uint32),
    ('material_id', np.uint32),
    ('is_point_cloud', np.uint32),
])


class RaytracingSceneBaker:
    """
    Manages the conversion of a Scene into a GPU-ready TLAS/BLAS structure
    """
    def __init__(self, scene: Scene):
        self.scene = scene
        self.TLAS = None
        self.BLASes: List[BVH] = []
        self.dynamic_instance_map: Dict[int, int] = {} # Maps scene instance index to TLAS instance index

        # OpenGL handles
        self.skybox_texture = 0
        self.materials_ssbo = 0
        self.tex_array = 0
        self.triangles_ssbo = 0
        self.points_ssbo = 0
        self.tlas_nodes_ssbo = 0
        self.blas_nodes_ssbo = 0
        self.instances_ssbo = 0

        # CPU-side data
        self.instances_info: Optional[np.ndarray] = None
        self.nb_TLAS_nodes: int = 0
        self.material_map: Dict[int, int] = {}
        self.asset_to_blas_map: Dict[int, Dict] = {}

        self.point_radius_by_asset = {}

        print("Baking ray-tracing scene...")

        self.skybox_texture = self.scene.skybox_texture_id

        self._pack_materials()
        self._build_BLASes()
        self._build_TLAS()

        # self.debug_tlas_setup()

        self._upload_buffers()

        print("Ray-tracing scene baking complete.")

    def _pack_materials(self):

        assets = [inst.asset for inst in self.scene.instances if isinstance(inst.asset, MeshAsset)]
        assets = list(dict.fromkeys(assets))

        if not assets:
            return

        self.material_map = {mesh.id: i for i, mesh in enumerate(assets)}

        tex_paths = sorted(list({mesh.texture_path for mesh in assets}))

        tex_map = {path: i for i, path in enumerate(tex_paths)}
        tex_ids = [load_texture(path) for path in tex_paths]

        if tex_ids:
            self.tex_array = self._create_texture_array(tex_ids)
            glDeleteTextures(len(tex_ids), tex_ids)

        mat_data = np.zeros((len(assets), 4), dtype=np.uint32)
        for mesh in assets:
            mat_data[self.material_map[mesh.id], 0] = tex_map[mesh.texture_path]

        self.materials_ssbo = glGenBuffers(1)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.materials_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, mat_data.nbytes, mat_data, GL_STATIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def _pack_triangles(self, mesh_asset: MeshAsset, prim_indices: np.ndarray):
        num_tris = mesh_asset.num_triangles
        packed = np.empty((num_tris, 20), dtype=VEC_DTYPE)
        interleaved = mesh_asset.vertex_data.reshape(-1, 3, 5)  # 3 vertices per tri, 5 floats per vert

        # Reorder the triangles according to the BVH's primitive indices
        reordered_interleaved = interleaved[prim_indices]

        v = reordered_interleaved[:, :, :3]
        uv = reordered_interleaved[:, :, 3:5]

        packed[:, 0:3] = v[:, 0, :]
        packed[:, 4:7] = v[:, 1, :]
        packed[:, 8:11] = v[:, 2, :]
        packed[:, 12:14] = uv[:, 0, :]
        packed[:, 14:16] = uv[:, 1, :]
        packed[:, 16:18] = uv[:, 2, :]

        mat_idx = self.material_map.get(mesh_asset.id, 0)
        # The material index is per-triangle, so it doesn't need reordering itself,
        # it just gets assigned to the already reordered packed data.
        packed[:, 18] = np.full(num_tris, mat_idx, dtype=np.uint32).view(VEC_DTYPE)

        return packed

    def _pack_points(self, points_asset: PointsAsset, prim_indices: np.ndarray):
        num_points = points_asset.num_points
        packed = np.zeros((num_points, 12), dtype=VEC_DTYPE)

        # Reorder all point attributes based on the BVH primitive indices
        reordered_points = points_asset.points[prim_indices]
        reordered_normals = points_asset.normals[prim_indices]
        reordered_colors = points_asset.colors[prim_indices]

        packed[:, 0:3] = reordered_points
        packed[:, 4:7] = reordered_normals
        packed[:, 8:11] = reordered_colors

        return packed

    def _build_BLASes(self):

        # Master lists to collect all primitive and node data
        packed_tris = []
        packed_points = []
        blas_nodes = []

        # keep track of the current total size to calculate offsets
        current_tri_offset = 0
        current_point_offset = 0
        current_node_offset = 0

        print(f"Building BLASes for {len(self.scene.assets)} unique assets...")
        for asset in self.scene.assets.values():

            # ensure each unique asset gets only one BLAS
            if asset.id in self.asset_to_blas_map:
                continue

            # 'blas_id' is the index in the self.blases list
            blas_id = len(self.BLASes)

            # Create the BLAS for the asset
            if isinstance(asset, MeshAsset):
                interleaved = asset.vertex_data.reshape(-1, 5)
                xyz = interleaved[:, :3]

                verts4 = np.empty((xyz.shape[0], 4), dtype=np.float32)
                verts4[:, :3] = xyz.astype(np.float32, copy=False)
                verts4[:, 3] = 1.0

                # TODO: Use indexed meshes instead
                blas = BVH.from_vertices(verts4)

                bvh_buffers = blas.get_buffers()
                prim_indices = bvh_buffers['prim_indices']

                # Pack this asset's triangles
                newly_packed_tris = self._pack_triangles(asset, prim_indices)
                prim_offset = current_tri_offset
                packed_tris.append(newly_packed_tris)
                current_tri_offset += len(newly_packed_tris)

            elif isinstance(asset, PointsAsset):

                point_inst = next((inst for inst in self.scene.instances if inst.asset.id == asset.id), None)

                radius = point_inst.properties.get('point_radius', 0.1) if point_inst else 0.1
                self.point_radius_by_asset[asset.id] = radius

                blas = BVH.from_points(asset.points.astype(np.float32, copy=False), radius=radius)

                bvh_buffers = blas.get_buffers()
                prim_indices = bvh_buffers['prim_indices']

                # Pack this asset's points
                newly_packed_points = self._pack_points(asset, prim_indices)
                prim_offset = current_point_offset
                packed_points.append(newly_packed_points)
                current_point_offset += len(newly_packed_points)

            else:
                continue

            # Store map from the asset's unique ID to its BLAS index and data offsets
            self.asset_to_blas_map[asset.id] = {
                'id': blas_id,
                'prim_offset': prim_offset,
                'node_offset': current_node_offset
            }

            self.BLASes.append(blas)

            # Append this BLAS's node data to the master node list
            nodes = bvh_buffers['nodes']

            blas_nodes.append(nodes)
            current_node_offset += len(nodes)

            asset_str = 'Mesh' if isinstance(asset, MeshAsset) else 'Points'
            prim_str = 'triangles' if isinstance(asset, MeshAsset) else 'points'
            print(f"Built BLAS {blas_id} for {asset_str} asset '{asset.name}' with {blas.prim_count:,} {prim_str}")

        # Create final flat arrays for GPU upload by concatenating the lists of arrays
        self.cpu_triangles = np.concatenate(packed_tris).ravel() if packed_tris else None
        self.cpu_points = np.concatenate(packed_points).ravel() if packed_points else None
        self.cpu_BLASes = np.concatenate(blas_nodes).astype(np.float32, copy=False) if blas_nodes else None

    def _build_TLAS(self):

        if not self.BLASes:
            return

        nb_instances = len(self.scene.instances)

        tlas_instances_data = np.zeros(nb_instances, dtype=instance_dtype)
        self.instances_info = np.zeros(nb_instances, dtype=instance_info_dtype)

        for i, inst in enumerate(self.scene.instances):
            blas_map_entry = self.asset_to_blas_map[inst.asset.id]

            # Get the blas_id, which is the index into the self.blases list
            blas_id = blas_map_entry['id']

            # we want the transforms as row-major
            transform = np.asarray(inst.transform)
            inverse_transform = np.asarray(glm.inverse(inst.transform))

            tlas_instances_data[i]['transform'] = transform
            tlas_instances_data[i]['blas_id'] = blas_id
            tlas_instances_data[i]['mask'] = 0xFFFFFFFF

            self.instances_info[i]['transform'] = transform
            self.instances_info[i]['inverse_transform'] = inverse_transform

            self.instances_info[i]['blas_node_offset'] = blas_map_entry['node_offset']
            self.instances_info[i]['primitive_offset'] = blas_map_entry['prim_offset']

            self.instances_info[i]['is_point_cloud'] = int(isinstance(inst.asset, PointsAsset))

            if isinstance(inst.asset, MeshAsset):
                self.instances_info[i]['material_id'] = self.material_map.get(inst.asset.id, 0)

            if inst.dynamic:
                self.dynamic_instance_map[i] = i

        self.TLAS = BVH.build_tlas(tlas_instances_data, self.BLASes)

        tlas_bufs = self.TLAS.get_buffers()
        self.cpu_TLAS_nodes = tlas_bufs['nodes'].astype(np.float32, copy=False)
        self.cpu_TLAS_prim_indices = tlas_bufs['prim_indices'].astype(np.uint32, copy=False)
        self.nb_TLAS_nodes = len(self.cpu_TLAS_nodes)

    def _upload_buffers(self):

        def upload(buffer_id, data, usage, dtype, min_elms: int = 1):
            if data is None or getattr(data, "nbytes", 0) == 0:
                data = np.zeros((min_elms,), dtype=dtype)

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer_id)
            glBufferData(GL_SHADER_STORAGE_BUFFER, data.nbytes, data, usage)

        (self.triangles_ssbo, self.points_ssbo, self.blas_nodes_ssbo,
         self.tlas_nodes_ssbo, self.instances_ssbo, self.tlas_indices_ssbo) = glGenBuffers(6)

        upload(self.triangles_ssbo, self.cpu_triangles, GL_STATIC_DRAW, np.float32, 20)
        upload(self.points_ssbo, self.cpu_points, GL_STATIC_DRAW, np.float32, 12)
        upload(self.blas_nodes_ssbo, self.cpu_BLASes, GL_STATIC_DRAW, np.float32, 1)
        upload(self.tlas_nodes_ssbo, self.cpu_TLAS_nodes, GL_DYNAMIC_DRAW, np.float32, 1)
        upload(self.tlas_indices_ssbo, self.cpu_TLAS_prim_indices, GL_STATIC_DRAW, np.uint32, 1)
        upload(self.instances_ssbo, self.instances_info, GL_DYNAMIC_DRAW, instance_info_dtype, 1)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def update(self):
        """ Updates transforms of dynamic instances and refits the TLAS """

        if not self.dynamic_instance_map or self.TLAS is None:
            return

        needs_refit = False
        for scene_idx, tlas_idx in self.dynamic_instance_map.items():
            instance = self.scene.instances[scene_idx]

            transform = np.asarray(instance.transform)
            inverse_transform = np.asarray(glm.inverse(instance.transform))

            # Update TLAS instance data
            self.TLAS.set_instance_transform(tlas_idx, transform)

            self.instances_info[tlas_idx]['transform'] = transform
            self.instances_info[tlas_idx]['inverse_transform'] = inverse_transform

            needs_refit = True

        if needs_refit:
            self.TLAS.refit_tlas()

            # Get new TLAS bounds
            new_tlas_data = self.TLAS.get_buffers()['nodes']

            # Re-upload the updated TLAS data
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.tlas_nodes_ssbo)
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, new_tlas_data.nbytes, new_tlas_data)

            # Re-upload the updated instance data
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.instances_ssbo)
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.instances_info.nbytes, self.instances_info)

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

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
            self.triangles_ssbo, self.points_ssbo, self.materials_ssbo,
            self.tlas_nodes_ssbo, self.blas_nodes_ssbo, self.instances_ssbo,
            self.tlas_indices_ssbo
        ] if b != 0]

        if buffers:
            glDeleteBuffers(len(buffers), buffers)

        if self.tex_array != 0:
            glDeleteTextures(1, [self.tex_array])

        print("RayTracingSceneManager resources freed.")


class EyeRendererRay(EyeRendererBase):
    def __init__(self, eye_model, scene: Scene, time_dithering: bool = True, nb_samples: int = 256, pano_res: Tuple[int, int] = (1024, 512)):
        super().__init__(eye_model, time_dithering=time_dithering, nb_samples=nb_samples)

        # Store a reference to the scene manager
        self.scene = scene   # just for convenience
        self._scene_baked = RaytracingSceneBaker(scene)

        print("Compiling ray-tracing and reduction shaders...")
        self.raytrace_shader = ShaderProgram(comp_path='shaders/ommatidia_raytracing.comp')
        self.reduction_shader = ShaderProgram(comp_path='shaders/rays_reduction.comp')
        self.pano_raytrace_shader = None    # lazily-loaded

        self._pano_res = pano_res
        self._pano_texture_id = 0
        self._texture_viewer = TextureViewer()

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

    def _initialize_pano_resources(self):
        """ Creates all resources needed for the panoramic view """

        print("Lazy-loading panoramic ray-tracer resources...")

        if self.pano_raytrace_shader is None:
            self.pano_raytrace_shader = ShaderProgram(comp_path='shaders/panoramic_raytracing.comp')

        if self._pano_texture_id == 0:

            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)

            # Use a high-precision format for the raytracing output
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, self._pano_res[0], self._pano_res[1], 0, GL_RGBA, GL_FLOAT, None)

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glBindTexture(GL_TEXTURE_2D, 0)

            self._pano_texture_id = texture_id

        if self._texture_viewer is None:
            self._texture_viewer = TextureViewer()

    def _raytrace_panoramic(self, agent):
        """ Dispatches a compute shader to generate a ray-traced panoramic image """

        self.pano_raytrace_shader.use()

        # Bind the output texture to image unit 0 for writing
        glBindImageTexture(0, self._pano_texture_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F)

        # Bind Textures (for reading)
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self._scene_baked.skybox_texture)
        glUniform1i(self.pano_raytrace_shader.get_loc('skybox'), 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self._scene_baked.tex_array)
        glUniform1i(self.pano_raytrace_shader.get_loc('scene_textures'), 1)

        # Bind Scene SSBOs (same as ommatidia shader, but starting at binding 1)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self._scene_baked.triangles_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self._scene_baked.materials_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self._scene_baked.points_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, self._scene_baked.blas_nodes_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, self._scene_baked.tlas_nodes_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, self._scene_baked.instances_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, self._scene_baked.tlas_indices_ssbo)

        # Set Uniforms
        glUniform1ui(self.pano_raytrace_shader.get_loc('nb_tlas_nodes'), self._scene_baked.nb_TLAS_nodes)

        # TODO: Handle point radius better for scenes with multiple point assets
        point_inst = next((inst for inst in self.scene.instances if isinstance(inst.asset, PointsAsset)), None)
        radius = self._scene_baked.point_radius_by_asset.get(point_inst.asset.id, 0.1) if point_inst else 0.1

        glUniform1f(self.pano_raytrace_shader.get_loc('point_radius'), radius)

        c2w_mat = glm.inverse(agent.view)
        glUniformMatrix4fv(self.pano_raytrace_shader.get_loc('cam_to_world'), 1, False, glm.value_ptr(c2w_mat))

        # Dispatch compute shader
        work_groups_x = (self._pano_res[0] + 15) // 16
        work_groups_y = (self._pano_res[1] + 15) // 16
        glDispatchCompute(work_groups_x, work_groups_y, 1)

        # Barrier to ensure imageStore operations are complete before the texture is used for drawing
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        self.pano_raytrace_shader.stop()

    def _raytrace(self, agent):

        # Pass 1: Ray-tracing

        self.raytrace_shader.use()

        # Bind Textures
        # Skybox
        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self._scene_baked.skybox_texture)
        glUniform1i(self.raytrace_shader.get_loc('skybox'), 0)

        # Scene textures
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self._scene_baked.tex_array)
        glUniform1i(self.raytrace_shader.get_loc('scene_textures'), 1)

        # Bind Shader Storage Buffers (SSBOs)

        # Binding 0: Input ommatidia data
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.input_om_ssbo)
        # Binding 1: Triangle primitive data (for triangle intersection, shading)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self._scene_baked.triangles_ssbo)
        # Binding 2: Material data (to fetch texture_idx from material_idx on triangles)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self._scene_baked.materials_ssbo)
        # Binding 3: Points primitive data (for sphere intersection, shading)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self._scene_baked.points_ssbo)
        # Binding 4: Output for this pass - the raw results for every single ray
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, self.ray_results_ssbo)
        # Binding 5: BLAS nodes for traversal
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, self._scene_baked.blas_nodes_ssbo)
        # Binding 6: TLAS nodes for traversal
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, self._scene_baked.tlas_nodes_ssbo)
        # Binding 7: per-instance info: transform, inverse, offsets, flags
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, self._scene_baked.instances_ssbo)
        # Binding 8: TLAS leaf -> instance id lookup
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, self._scene_baked.tlas_indices_ssbo)

        # Set Uniforms
        glUniform1ui(self.raytrace_shader.get_loc('nb_tlas_nodes'), self._scene_baked.nb_TLAS_nodes)

        # TODO: Would be better with one radius per point, based on neighbours density
        point_inst = next((inst for inst in self.scene.instances if isinstance(inst.asset, PointsAsset)), None)
        radius = self._scene_baked.point_radius_by_asset.get(point_inst.asset.id, 0.1) if point_inst else 0.1

        glUniform1f(self.raytrace_shader.get_loc('point_radius'), radius)

        glUniform1i(self.raytrace_shader.get_loc('nb_samples'), self.samples_per_ommatidium)
        glUniform1f(self.raytrace_shader.get_loc('time'), float(self._time_counter))

        # Set camera uniforms for transforming rays into world space
        c2w_mat = glm.inverse(agent.view)
        glUniformMatrix4fv(self.raytrace_shader.get_loc('cam_to_world'), 1, False,
                           glm.value_ptr(c2w_mat))

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
        # Binding 0: Input for this pass - the raw ray results from Pass 1
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.ray_results_ssbo)
        # Binding 1: Output for this pass - the final averaged color for each ommatidium
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.final_colors_ssbo)

        # Set uniforms
        glUniform1i(self.reduction_shader.get_loc('nb_samples'), self.samples_per_ommatidium)
        glUniform1i(self.reduction_shader.get_loc('nb_ommatidia'), self.num_ommatidia)

        # Dispatch compute shader
        # Calculate workgroups needed to process all ommatidia
        work_groups = (self.num_ommatidia + 63) // 64  # Workgroup size is 64 in the shader
        glDispatchCompute(work_groups, 1, 1)

        # Another critical barrier: ensures the final_colors_ssbo is fully written
        # before the CPU or the drawing shader tries to read from it in the next frame
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        self.reduction_shader.stop()

    def _compute_colors(self, agent):
        """ The core ommatidia rendering logic """

        # First refresh the scene for anything that moved / changed
        self._scene_baked.update()

        # Pass 1: Ray-trace
        self._raytrace(agent)

        # Pass 2: Reduction
        self._reduction()

        # Unbind all resources
        for i in range(9):
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)

    def get_ommatidia_data(self, agent, to_cpu=False):

        self._compute_colors(agent)

        if self._time_dithering:
            self._time_counter += 1

        if to_cpu:
            self._fetch_to_cpu()

        return self.cpu_read_buffer

    def draw(self, view_mode: str, agent, tiled_mode: bool = False):
        """ Renders one of the rasterizer's supported views to the screen """

        if view_mode == 'compound_eye':
            super().draw(tiled_mode=tiled_mode)

        elif view_mode == 'panoramic':

            if self._pano_texture_id == 0 or self.pano_raytrace_shader is None:
                self._initialize_pano_resources()

            self._scene_baked.update()
            self._raytrace_panoramic(agent)
            self._texture_viewer.draw(self._pano_texture_id)

    def free(self):
        """ Frees all GPU resources, including shaders and all buffers """

        glDeleteBuffers(1, [self.ray_results_ssbo])
        if self.raytrace_shader: self.raytrace_shader.free()
        if self.reduction_shader: self.reduction_shader.free()

        if self.pano_raytrace_shader: self.pano_raytrace_shader.free()
        if self._pano_texture_id != 0: glDeleteTextures(1, [self._pano_texture_id])
        if self._texture_viewer: self._texture_viewer.free()

        self._scene_baked.free()
        super().free()