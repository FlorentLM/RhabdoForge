import OpenGL

OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *

from typing import Tuple, List, Dict, Optional
import numpy as np
from PIL import Image
from pyglm import glm
from pytinybvh import BVH, instance_dtype, Layout, supports_layout

from graphics.agent import Agent
from graphics.renderers.base import EyeRendererBase
from graphics.scene import Scene, MeshAsset, PointsAsset
from graphics.utils import VEC_DTYPE, ShaderProgram, write_pytinybvh_preamble
from graphics.renderers.panoramic import TextureViewer

# Custom more detailed dtype for the GPU SSBO
gpu_instance_dtype = np.dtype([
    ('transform', np.float32, (4, 4)),
    ('inverse_transform', np.float32, (4, 4)),
    ('blas_node_offset', np.uint32),
    ('vertex_or_point_offset', np.uint32),
    ('index_offset', np.uint32),
    ('material_id', np.uint32),
    ('point_mode', np.uint32),  # 0 for non-points (i.e. triangles asset), 1 for points, 2 for points blocks
    ('prim_index_offset', np.uint32),
    ('radius_factor', np.float32),
    ('padding', np.uint32, 1),  # 4 bytes (1 * uint) of padding
])  # total 160 bytes


class RaytracingSceneBaker:
    """
    Manages the creation, ownership, and updating of BVH structures
    and GPU buffers from a logical Scene object.
    """

    def __init__(self, scene: Scene):
        self.scene = scene

        self.TLAS: Optional[BVH] = None
        self.BLASes: List[BVH] = []

        # Maps the ID of a Scene.Instance to its index in the TLAS array
        self.dynamic_instance_map: Dict[int, int] = {}

        # OpenGL handles
        self.skybox_texture = 0
        self.materials_ssbo, self.tex_array = 0, 0
        self.triangles_ssbo, self.points_ssbo = 0, 0
        self.tlas_nodes_ssbo, self.blas_nodes_ssbo = 0, 0
        self.instances_info_ssbo, self.tlas_indices_ssbo = 0, 0
        self.point_blocks_first_ssbo, self.point_blocks_count_ssbo = 0, 0  # Buffers for point blocks

        # CPU-side data for building and updating
        self.gpu_instances_info: Optional[np.ndarray] = None
        self.material_map: Dict[int, int] = {}
        self.asset_to_blas_map: Dict[int, Dict] = {}

        print("Baking ray-tracing scene...")
        if not self.scene.instances:
            print("Warning: Scene is empty, nothing to bake.")
            return

        self.skybox_texture = self.scene.skybox.texture_id if self.scene.skybox else 0

        self._pack_materials()

        self._build_BLASes()
        self._build_TLAS()

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
        # Determine the target dimensions from the first texture
        # All other textures will be resized to match this one
        try:
            with Image.open(tex_paths[0]) as first_img:
                target_w, target_h = first_img.size
                print(f"Using target texture size: {target_w}x{target_h} (from {tex_paths[0]})")
        except FileNotFoundError:
            print(f"Error: Could not open base texture {tex_paths[0]}")
            return

        tex_ids = []

        for path in tex_paths:
            try:
                img = Image.open(path).convert("RGBA")

                # Check if the image needs resizing
                if img.size != (target_w, target_h):
                    print(f"Warning: Resizing texture '{path}' from {img.size} to {(target_w, target_h)}.")
                    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

                # Get raw image data for OpenGL
                img_data = img.tobytes()

                # Create an OpenGL texture from the (potentially resized) image data
                tex_id = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, tex_id)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, target_w, target_h, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)

                tex_ids.append(tex_id)

            except Exception as e:
                print(f"Error processing texture {path}: {e}")
                # Add a placeholder texture ID (0) if loading fails
                tex_ids.append(0)

        if tex_ids:
            self.tex_array = self._create_texture_array(tex_ids)
            # We can delete the individual 2D textures now that their data is in the array
            glDeleteTextures(len(tex_ids), tex_ids)

        mat_data = np.zeros((len(assets), 4), dtype=np.uint32)
        for mesh in assets:
            mat_data[self.material_map[mesh.id], 0] = tex_map[mesh.texture_path]

        self.materials_ssbo = glGenBuffers(1)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.materials_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, mat_data.nbytes, mat_data, GL_STATIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def _build_BLASes(self):
        """
        Produces/updates:
            self.BLASes: List[BVH]
            self.asset_to_blas_map[asset.id]: {'id', 'prim_offset', 'node_offset', 'prim_index_offset'}
            self.cpu_AllBLAS_nodes: float32 (N_total_nodes, 8) for Standard layout
            self.cpu_triangles: packed triangles (if any)
            self.cpu_points: packed points (if any)
            self._blas_leaf_id_chunks: list of uint32 leaf_ids per BLAS (concatenated later in _build_TLAS)
            self.cpu_point_blocks_first: maps block to first point index
            self.cpu_point_blocks_count: maps block to point count
        """

        all_vertices, all_indices, all_points, all_blas_nodes = [], [], [], []
        all_point_blocks_first, all_point_blocks_count = [], []
        current_vert_offset, current_idx_offset, current_point_offset, current_node_offset = 0, 0, 0, 0

        # collect BLAS leaf_ids so the GPU can do leaf -> primitive mapping
        self._blas_leaf_id_chunks = []
        current_blas_leaf_offset = 0  # running offset into the concatenated leaf_ids buffer

        print(f"Building BLASes for {len(self.scene.assets)} unique assets...")
        for asset in self.scene.assets.values():

            if asset.id in self.asset_to_blas_map:
                continue

            blas_id = len(self.BLASes)
            bundle = None  # will store the SSBO bundle from the BVH

            if isinstance(asset, MeshAsset):

                positions = asset.vertices[:, :3].astype(np.float32)
                verts4 = np.pad(positions, ((0, 0), (0, 1)), 'constant', constant_values=0)
                indices = asset.indices.astype(np.uint32)

                blas = BVH.from_indexed_mesh(verts4, indices)

                # Append mesh data to global lists
                all_vertices.append(asset.vertices)
                all_indices.append(asset.indices.flatten())

                # Record offsets for this asset, to be used by instances
                self.asset_to_blas_map[asset.id] = {
                    'id': blas_id,
                    'vert_offset': current_vert_offset,
                    'idx_offset': current_idx_offset,
                    'point_mode': 0  # Mesh asset, not a point cloud
                }
                current_vert_offset += len(asset.vertices)
                current_idx_offset += len(asset.indices.flatten())

            elif isinstance(asset, PointsAsset):

                points = asset.points.astype(np.float32)
                radii = asset.radii.astype(np.float32)

                # TODO: This should be based on an asset field
                USE_BLOCKS = False

                if USE_BLOCKS:
                    block_size = 32

                    blas = BVH.from_points_blocks(
                        points,
                        radius=radii,
                        block_size=block_size,
                        morton_bits=10,     # Max precision for sorting
                        traversal_cost=1.0,
                        intersection_cost=1.0
                    )
                    point_mode = 2  # Packed points BVH

                else:
                    blas = BVH.from_points(
                        points,
                        radius=radii,
                        traversal_cost=1.0,
                        intersection_cost=1.0
                    )
                    point_mode = 1  # Regular points BVH

                # The bundle contains re-ordered points and block info
                bundle = blas.get_SSBO_bundle(flatten_nodes=False)

                # Use compacted points if they exist (from blocked builder)
                if 'point_order' in bundle:
                    point_order = bundle['point_order']
                    compacted_points = asset.points[point_order]
                    compacted_radii = asset.radii[point_order]
                    compacted_normals = asset.normals[point_order]
                    compacted_colors = asset.colors[point_order]

                else:
                    compacted_points = asset.points
                    compacted_radii = asset.radii
                    compacted_normals = asset.normals
                    compacted_colors = asset.colors

                # Pack point data for the shader
                packed_points = np.zeros((len(compacted_points), 12), dtype=VEC_DTYPE)
                packed_points[:, 0:3] = compacted_points
                packed_points[:, 3] = compacted_radii
                packed_points[:, 4:7] = compacted_normals
                packed_points[:, 7:10] = compacted_colors
                all_points.append(packed_points)

                self.asset_to_blas_map[asset.id] = {
                    'id': blas_id,
                    'point_offset': current_point_offset,
                    'point_mode': point_mode
                }

                # If we used blocks, process the block data
                if point_mode == 2:
                    # The indices must be offset to be global
                    blocks_first = bundle['point_blocks_first'].astype(np.uint32) + current_point_offset
                    blocks_count = bundle['point_blocks_count'].astype(np.uint32)

                    all_point_blocks_first.append(blocks_first)
                    all_point_blocks_count.append(blocks_count)

                # Update the global point offset AFTER processing the current asset
                current_point_offset += len(compacted_points)

            else:
                # Skip unsupported asset types
                continue

            # target_layout = Layout.BVH_GPU
            target_layout = Layout.Standard

            if supports_layout(target_layout) and target_layout != blas.layout:
                blas.convert_to(target_layout, compact=True)
            elif target_layout != blas.layout:
                print(f"Warning: Layout {target_layout.name} not supported. Falling back to Standard.")
                blas.convert_to(Layout.Standard, compact=True)

            if bundle is None:
                bundle = blas.get_SSBO_bundle(flatten_nodes=False)

            nodes = bundle['nodes']
            prim_indices = bundle['leaf_ids'].astype(np.uint32)

            all_blas_nodes.append(nodes)
            self._blas_leaf_id_chunks.append(prim_indices)

            self.asset_to_blas_map[asset.id].update({
                'node_offset': current_node_offset,
                'prim_index_offset': current_blas_leaf_offset
            })
            self.BLASes.append(blas)
            current_node_offset += nodes.shape[0]
            current_blas_leaf_offset += prim_indices.size

        # Concatenate all CPU-side lists into single numpy arrays for uploading
        self.cpu_vertices = np.concatenate(all_vertices).ravel() if all_vertices else None
        self.cpu_indices = np.concatenate(all_indices).ravel() if all_indices else None
        self.cpu_points = np.concatenate(all_points).ravel() if all_points else None
        self.cpu_BLASes = np.concatenate(all_blas_nodes).astype(np.float32) if all_blas_nodes else None

        # Concatenate block data into final CPU buffers
        self.cpu_point_blocks_first = np.concatenate(all_point_blocks_first) if all_point_blocks_first else None
        self.cpu_point_blocks_count = np.concatenate(all_point_blocks_count) if all_point_blocks_count else None

    def _build_TLAS(self):

        if not self.BLASes:
            return

        num_instances = len(self.scene.instances)

        # pytinybvh input for TLAS build + our GPU-side per-instance struct
        tlas_build_data = np.zeros(num_instances, dtype=instance_dtype)
        self.gpu_instances_info = np.zeros(num_instances, dtype=gpu_instance_dtype)

        for i, inst in enumerate(self.scene.instances):
            blas_map = self.asset_to_blas_map[inst.asset.id]
            transform = np.asarray(inst.transform, dtype=np.float32)

            tlas_build_data[i]['transform'] = transform
            tlas_build_data[i]['blas_id'] = blas_map['id']
            tlas_build_data[i]['mask'] = 0xFFFFFFFF

            self.gpu_instances_info[i]['transform'] = transform
            self.gpu_instances_info[i]['inverse_transform'] = np.asarray(glm.inverse(inst.transform), dtype=np.float32)
            self.gpu_instances_info[i]['blas_node_offset'] = blas_map['node_offset']
            self.gpu_instances_info[i]['prim_index_offset'] = blas_map['prim_index_offset']
            self.gpu_instances_info[i]['point_mode'] = blas_map['point_mode']

            if isinstance(inst.asset, MeshAsset):
                self.gpu_instances_info[i]['vertex_or_point_offset'] = blas_map['vert_offset']
                self.gpu_instances_info[i]['index_offset'] = blas_map['idx_offset']
                self.gpu_instances_info[i]['material_id'] = self.material_map.get(inst.asset.id, 0)

            elif isinstance(inst.asset, PointsAsset):
                self.gpu_instances_info[i]['vertex_or_point_offset'] = blas_map['point_offset']
                self.gpu_instances_info[i]['index_offset'] = 0
                self.gpu_instances_info[i]['radius_factor'] = inst.properties.get('radius_factor', 1.0)

            # Track dynamic instances
            if getattr(inst, "dynamic", False):
                self.dynamic_instance_map[inst.id] = i

        # Build TLAS from instances and child BLAS list
        self.TLAS = BVH.build_tlas(tlas_build_data, self.BLASes)

        t = self.TLAS.get_SSBO_bundle(flatten_nodes=False)

        self.cpu_TLAS_nodes = t['nodes'].astype(np.float32)
        self.cpu_TLAS_prim_indices = t['leaf_ids'].astype(np.uint32)

        # Make the GLSL #defines visible to shaders
        write_pytinybvh_preamble(str(t.get('preamble', '')))

        # Concatenate all BLAS leaf_ids (used by shader to map leaf -> primitive)
        self.cpu_BLAS_prim_indices = np.concatenate(self._blas_leaf_id_chunks).astype(np.uint32)

    def _upload_buffers(self):

        def upload(buffer_id, data, usage, dtype, min_elms: int = 1):
            if data is None or getattr(data, "nbytes", 0) == 0:
                data = np.zeros((min_elms,), dtype=dtype)

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, buffer_id)
            glBufferData(GL_SHADER_STORAGE_BUFFER, data.nbytes, data, usage)

        # vertices, indices, points, BLAS nodes, TLAS nodes, instances, TLAS indices, BLAS indices
        (self.vertices_ssbo, self.indices_ssbo, self.points_ssbo, self.blas_nodes_ssbo,
         self.tlas_nodes_ssbo, self.instances_info_ssbo, self.tlas_indices_ssbo,
         self.blas_indices_ssbo, self.point_blocks_first_ssbo,
         self.point_blocks_count_ssbo) = glGenBuffers(10)

        # Primitive streams
        upload(self.vertices_ssbo, self.cpu_vertices, GL_STATIC_DRAW, np.float32, 5)  # (pos, uv)
        upload(self.indices_ssbo, self.cpu_indices, GL_STATIC_DRAW, np.uint32, 3)
        upload(self.points_ssbo, self.cpu_points, GL_STATIC_DRAW, np.float32, 12)

        # Upload point block data
        upload(self.point_blocks_first_ssbo, self.cpu_point_blocks_first, GL_STATIC_DRAW, np.uint32, 1)
        upload(self.point_blocks_count_ssbo, self.cpu_point_blocks_count, GL_STATIC_DRAW, np.uint32, 1)

        # Nodes
        upload(self.blas_nodes_ssbo, self.cpu_BLASes, GL_STATIC_DRAW, np.float32, 1)
        upload(self.tlas_nodes_ssbo, self.cpu_TLAS_nodes, GL_DYNAMIC_DRAW, np.float32, 1)

        # Leaf -> thing mappings
        upload(self.tlas_indices_ssbo, self.cpu_TLAS_prim_indices, GL_STATIC_DRAW, np.uint32, 1)  # TLAS: leaf -> instance
        upload(self.blas_indices_ssbo, self.cpu_BLAS_prim_indices, GL_STATIC_DRAW, np.uint32, 1)  # BLAS: leaf -> primitive

        # Per-instance info (updates with animation)
        upload(self.instances_info_ssbo, self.gpu_instances_info, GL_DYNAMIC_DRAW, gpu_instance_dtype, 1)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def update(self):
        """ Pulls transforms from dynamic scene instances, updates and refits the TLAS, and uploads to GPU """

        if not self.dynamic_instance_map or self.TLAS is None:
            return

        # Find the scene instances that are marked as dynamic
        dynamic_instances = {inst.id: inst for inst in self.scene.instances if inst.id in self.dynamic_instance_map}

        if not dynamic_instances:
            return

        for inst_id, tlas_idx in self.dynamic_instance_map.items():
            instance = dynamic_instances.get(inst_id)

            if instance is None:
                continue

            transform = np.asarray(instance.transform, dtype=np.float32)

            self.TLAS.set_instance_transform(tlas_idx, transform)

            # Update the CPU-side buffer destined for the GPU
            self.gpu_instances_info[tlas_idx]['transform'] = transform
            self.gpu_instances_info[tlas_idx]['inverse_transform'] = np.asarray(glm.inverse(instance.transform), dtype=np.float32)

        # Refit the TLAS in C++ after all transforms are set
        self.TLAS.refit_tlas()

        # Re-upload the updated buffers to the GPU
        new_tlas_nodes = self.TLAS.get_buffers()['nodes']

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.tlas_nodes_ssbo)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, new_tlas_nodes.nbytes, new_tlas_nodes)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.instances_info_ssbo)
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.gpu_instances_info.nbytes, self.gpu_instances_info)

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
            self.vertices_ssbo, self.indices_ssbo, self.points_ssbo, self.materials_ssbo,
            self.tlas_nodes_ssbo, self.blas_nodes_ssbo, self.instances_info_ssbo,
            self.tlas_indices_ssbo, self.blas_indices_ssbo,
            self.point_blocks_first_ssbo, self.point_blocks_count_ssbo
        ] if b != 0]

        if buffers:
            glDeleteBuffers(len(buffers), buffers)

        if self.tex_array != 0:
            glDeleteTextures(1, [self.tex_array])

class EyeRendererRay(EyeRendererBase):
    def __init__(self, eye_model, scene: Scene,
                 time_dithering: bool = True,
                 nb_samples: int = 256,
                 pano_res: Tuple[int, int] = (1024, 512),
                 batch_size: int = 1):

        # Store a reference to the scene manager
        self.scene = scene   # just for convenience
        self._scene_baked = RaytracingSceneBaker(scene)

        # we call super().__init__ *after* baking the scene so we can estimate VRAM
        super().__init__(eye_model, time_dithering=time_dithering, nb_samples=nb_samples, batch_size=batch_size)

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

    def estimate_vram_usage(self) -> float:
        """ Override base method to provide a more accurate VRAM estimate for the raytracer """
        baker = self._scene_baked
        total_bytes = 0
        buffers = [
            baker.cpu_vertices, baker.cpu_indices, baker.cpu_points, baker.cpu_BLASes,
            baker.cpu_TLAS_nodes, baker.cpu_TLAS_prim_indices, baker.cpu_BLAS_prim_indices,
            baker.gpu_instances_info,
            baker.cpu_point_blocks_first, baker.cpu_point_blocks_count
        ]
        for buf in buffers:
            if buf is not None:
                total_bytes += buf.nbytes

        # Add intermediate ray results buffer
        total_bytes += getattr(self, 'total_samples', 0) * 16

        return total_bytes / (1024 * 1024)  # Convert to MB

    def _initialize_pano_resources(self):
        """ Creates all resources needed for the panoramic view """

        print("Lazy-loading panoramic ray-tracer resources...")

        if self.pano_raytrace_shader is None:
            self.pano_raytrace_shader = ShaderProgram(comp_path='shaders/panoramic_raytracing.comp')

        if self._pano_texture_id == 0:

            texture_id = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, texture_id)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, self._pano_res[0], self._pano_res[1], 0, GL_RGBA, GL_FLOAT, None)

            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glBindTexture(GL_TEXTURE_2D, 0)

            self._pano_texture_id = texture_id

        if self._texture_viewer is None:
            self._texture_viewer = TextureViewer()

    def _bind_scene_resources(self, shader: ShaderProgram):

        # Bind Textures
        if self._scene_baked.scene.skybox:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_CUBE_MAP, self._scene_baked.skybox_texture)
            glUniform1i(shader.get_loc('skybox'), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self._scene_baked.tex_array)
        glUniform1i(shader.get_loc('scene_textures'), 1)

        # Bind SSBOs
        # bindings 0 and 1 are for the ommatidia data and rays outputs
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 2, self._scene_baked.vertices_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, self._scene_baked.indices_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 4, self._scene_baked.materials_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 5, self._scene_baked.points_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 6, self._scene_baked.blas_nodes_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 7, self._scene_baked.tlas_nodes_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 8, self._scene_baked.instances_info_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 9, self._scene_baked.tlas_indices_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 10, self._scene_baked.blas_indices_ssbo)
        # Bind point block buffers
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 11, self._scene_baked.point_blocks_first_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 12, self._scene_baked.point_blocks_count_ssbo)

        # Set Uniforms
        glUniform1ui(shader.get_loc('nb_tlas_nodes'), len(self._scene_baked.cpu_TLAS_nodes))

        glUniform1i(shader.get_loc('use_skybox'), int(self.scene.skybox is not None))
        bg = self.scene.background_color
        glUniform3f(shader.get_loc('background_color'), bg[0], bg[1], bg[2])

    def _raytrace_panoramic(self, agent):
        """ Dispatches a compute shader to generate a ray-traced panoramic image """

        self.pano_raytrace_shader.use()

        # Bind the output texture to image unit 0 for writing
        glBindImageTexture(0, self._pano_texture_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F)

        self._bind_scene_resources(self.pano_raytrace_shader)

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

        # Bind input/output buffers for this pass
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.input_om_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.ray_results_ssbo)

        # Bind all shared scene data (textures, geometry, BVHs)
        self._bind_scene_resources(self.raytrace_shader)

        # Set Uniforms
        glUniform1i(self.raytrace_shader.get_loc('nb_samples'), self.samples_per_ommatidium)
        glUniform1f(self.raytrace_shader.get_loc('time'), float(self._time_counter))

        # Set camera uniforms for transforming rays into world space
        c2w_mat = glm.inverse(agent.view)
        glUniformMatrix4fv(self.raytrace_shader.get_loc('cam_to_world'), 1, False, glm.value_ptr(c2w_mat))

        # Dispatch compute shader
        # Calculate the number of workgroups needed to process all rays
        workgroup_size = 64  # workgroup size is 64 in the shader
        work_groups = (self.total_samples + (workgroup_size - 1)) // workgroup_size
        glDispatchCompute(work_groups, 1, 1)

        # This barrier is critical: it ensures all writes to the ray_results_ssbo
        # from this pass are complete before the next pass tries to read from it
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        self.raytrace_shader.stop()

    def _reduction(self):
        """ Pass 2: Reduction """

        self.reduction_shader.use()

        # Bind buffers
        # Binding 0: Input for this pass - the raw ray results from Pass 1
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 0, self.ray_results_ssbo)
        # Binding 1: Output for this pass - the final averaged color for each ommatidium
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, self.final_colors_ssbo)

        # Set uniforms
        glUniform1i(self.reduction_shader.get_loc('nb_samples'), self.samples_per_ommatidium)
        glUniform1i(self.reduction_shader.get_loc('nb_ommatidia'), self.num_ommatidia)

        # Write into the history buffer circularly
        frame_offset = self._current_frame_index % self._batch_size
        glUniform1i(self.reduction_shader.get_loc('frame_index'), frame_offset)

        # Dispatch compute shader
        # Calculate workgroups needed to process all ommatidia
        workgroup_size = 64     # workgroup size is 64 in the shader
        work_groups = (self.num_ommatidia + (workgroup_size - 1)) // workgroup_size
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
        for i in range(13):
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)

    def draw(self, view_mode: str, agent: Agent, tiled_mode: bool = False):
        """ Renders one of the rasterizer's supported views to the screen """

        if view_mode == 'compound_eye':
            self._draw_voronoi(tiled_mode=tiled_mode)

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