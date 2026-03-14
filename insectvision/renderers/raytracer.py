import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *

from typing import Tuple, List, Dict, Optional, Set
import numpy as np
from PIL import Image
from pyglm import glm
from pytinybvh import BVH, instance_dtype, Layout, supports_layout

from insectvision.engine.agent import Agent
from insectvision.engine.scene import Scene, AssetType
from insectvision.engine.lights import DIR_LIGHT_DTYPE, POINT_LIGHT_DTYPE, AREA_LIGHT_DTYPE
from insectvision.engine.shader_utils import write_pytinybvh_preamble, ShaderProgram
from insectvision.interactive.utils import DisplayMode

from .commons import BaseRenderer, TextureViewer,BINDING_RECEPTORS, BINDING_COLORS, BINDING_STATE, BINDING_RAYS_INTERMEDIATE


# Scene geometry bindings
BINDING_VERTICES     = 5
BINDING_INDICES      = 6
BINDING_MATERIALS    = 7
BINDING_POINTS       = 8
BINDING_BLAS_NODES   = 9
BINDING_TLAS_NODES   = 10
BINDING_INSTANCES    = 11
BINDING_TLAS_INDICES = 12
BINDING_BLAS_INDICES = 13

# Light bindings
BINDING_LIGHT_DIR    = 14
BINDING_LIGHT_POINT  = 15
BINDING_LIGHT_AREA   = 16


RENDERABLE_INST_DTYPE = np.dtype([
    ('transform', np.float32, (4, 4)),
    ('inverse_transform', np.float32, (4, 4)),
    ('blas_node_offset', np.uint32),
    ('vertex_or_point_offset', np.uint32),
    ('index_offset', np.uint32),
    ('material_id', np.uint32),
    ('is_points', np.uint32),  # 0 for non-points (triangles asset), 1 for points
    ('prim_index_offset', np.uint32),
    ('radius_factor', np.float32),
    ('padding', np.uint32, 1),  # 4 bytes (1 * uint) of padding
])  # total 160 bytes
# TODO: This struct can probably be optimised more


class RaytracingSceneBaker:
    """
    Manages the creation, ownership, and updating of BVH structures
    and GPU buffers from a logical Scene object.

    Also packs lights from the scene into GPU-ready SSBOs.
    """

    def __init__(self, scene: Scene):
        self.scene = scene

        self.TLAS: Optional[BVH] = None
        self.BLASes: List[BVH] = []

        # Maps the ID of a Scene.Instance to its index in the TLAS array
        self.dynamic_instance_map: Dict[int, int] = {}

        # OpenGL handles for geometry
        self.skybox_texture = 0
        self.materials_ssbo, self.tex_array = 0, 0
        self.triangles_ssbo, self.points_ssbo = 0, 0
        self.vertices_ssbo, self.indices_ssbo = 0, 0
        self.tlas_nodes_ssbo, self.blas_nodes_ssbo = 0, 0
        self.instances_info_ssbo, self.tlas_indices_ssbo = 0, 0
        self.blas_indices_ssbo = 0

        # OpenGL handles for lights
        self.directional_lights_ssbo = 0
        self.point_lights_ssbo = 0
        self.area_lights_ssbo = 0

        # CPU-side data for building (and updating)
        self.gpu_instances_info: Optional[np.ndarray] = None
        self.material_map: Dict[int, int] = {}
        self.asset_to_blas_map: Dict[int, Dict] = {}

        # Light counts (for uniform upload)
        self.num_directional_lights = 0
        self.num_point_lights = 0
        self.num_area_lights = 0

        print("Baking ray-tracing scene...")
        if not self.scene.instances:
            print("Warning: Scene is empty, nothing to bake.")
            return

        self.skybox_texture = self.scene.skybox.texture_id if self.scene.skybox else 0

        self._pack_materials()
        self._build_BLASes()
        self._build_TLAS()
        self._upload_buffers()
        self._pack_lights()

        print("Ray-tracing scene baking complete.")

    def _filter_active_lights(self):
        """Returns active (enabled, intensity > 0) lights grouped by type."""

        directional = [l for l in self.scene.directional_lights if l.enabled and l.intensity > 0]
        point = [l for l in self.scene.point_lights if l.enabled and l.intensity > 0]
        area = [l for l in self.scene.area_lights if l.enabled and l.intensity > 0]

        return directional, point, area

    def _pack_lights(self):

        directional_lights, point_lights, area_lights = self._filter_active_lights()

        self.num_directional_lights = len(directional_lights)
        self.num_point_lights = len(point_lights)
        self.num_area_lights = len(area_lights)

        # Pack directional lights
        if directional_lights:
            packed = np.concatenate([l.pack() for l in directional_lights])
        else:
            packed = np.zeros(1, dtype=DIR_LIGHT_DTYPE)

        self.directional_lights_ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.directional_lights_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, packed.nbytes, packed, GL_DYNAMIC_DRAW)

        # Pack point lights
        if point_lights:
            packed = np.concatenate([l.pack() for l in point_lights])
        else:
            packed = np.zeros(1, dtype=POINT_LIGHT_DTYPE)

        self.point_lights_ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.point_lights_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, packed.nbytes, packed, GL_DYNAMIC_DRAW)

        # Pack area lights
        if area_lights:
            packed = np.concatenate([l.pack() for l in area_lights])
        else:
            packed = np.zeros(1, dtype=AREA_LIGHT_DTYPE)

        self.area_lights_ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.area_lights_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, packed.nbytes, packed, GL_DYNAMIC_DRAW)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

        total = self.num_directional_lights + self.num_point_lights + self.num_area_lights
        if total > 0:
            print(f"Packed {self.num_directional_lights} directional, "
                  f"{self.num_point_lights} point, {self.num_area_lights} area lights")

    def update_lights(self):
        """Re-packs lights if they have changed."""

        directional_lights, point_lights, area_lights = self._filter_active_lights()

        self.num_directional_lights = len(directional_lights)
        self.num_point_lights = len(point_lights)
        self.num_area_lights = len(area_lights)

        if directional_lights:
            packed = np.concatenate([l.pack() for l in directional_lights])
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.directional_lights_ssbo)
            glBufferData(GL_SHADER_STORAGE_BUFFER, packed.nbytes, packed, GL_DYNAMIC_DRAW)

        if point_lights:
            packed = np.concatenate([l.pack() for l in point_lights])
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.point_lights_ssbo)
            glBufferData(GL_SHADER_STORAGE_BUFFER, packed.nbytes, packed, GL_DYNAMIC_DRAW)

        if area_lights:
            packed = np.concatenate([l.pack() for l in area_lights])
            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.area_lights_ssbo)
            glBufferData(GL_SHADER_STORAGE_BUFFER, packed.nbytes, packed, GL_DYNAMIC_DRAW)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def _pack_materials(self):
        """Packs material data for all mesh assets into GPU buffers."""

        mesh_assets = {inst.asset for inst in self.scene.mesh_instances}

        if not mesh_assets:
            return

        self.material_map = {asset.id: i for i, asset in enumerate(mesh_assets)}

        # Collect textures from assets that have them
        texture_images = []
        asset_to_tex_idx = {}

        for asset in mesh_assets:
            if asset.has_texture:
                img = asset.texture_image  # lazy loads if needed
                if img is not None:
                    asset_to_tex_idx[asset.id] = len(texture_images)
                    texture_images.append(img)
                else:
                    asset_to_tex_idx[asset.id] = None
            else:
                asset_to_tex_idx[asset.id] = None

        # Create texture array if we have any
        if texture_images:
            target_w, target_h = texture_images[0].size
            print(f"Creating texture array: {len(texture_images)} textures at {target_w}x{target_h}")

            tex_ids = []
            for i, img in enumerate(texture_images):
                if img.size != (target_w, target_h):
                    print(f"  Resizing texture {i} from {img.size}")
                    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)

                img_data = img.convert("RGBA").tobytes()

                tex_id = glGenTextures(1)
                glBindTexture(GL_TEXTURE_2D, tex_id)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB_ALPHA, target_w, target_h, 0,
                             GL_RGBA, GL_UNSIGNED_BYTE, img_data)
                tex_ids.append(tex_id)

            self.tex_array = self._create_texture_array(tex_ids)
            glDeleteTextures(len(tex_ids), tex_ids)

        # Pack material buffer
        mat_data = np.zeros((len(mesh_assets), 4), dtype=np.uint32)

        for asset in mesh_assets:
            idx = self.material_map[asset.id]
            tex_idx = asset_to_tex_idx[asset.id]

            # Texture index (0xFFFFFFFF = no texture)
            mat_data[idx, 0] = tex_idx if tex_idx is not None else 0xFFFFFFFF

            # Pack base_color as RGBA8
            c = asset.material.base_color
            r = int(np.clip(c[0], 0, 1) * 255) & 0xFF
            g = int(np.clip(c[1], 0, 1) * 255) & 0xFF
            b = int(np.clip(c[2], 0, 1) * 255) & 0xFF
            a = int(np.clip(c[3], 0, 1) * 255) & 0xFF
            mat_data[idx, 1] = (a << 24) | (b << 16) | (g << 8) | r

        self.materials_ssbo = glGenBuffers(1)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.materials_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, mat_data.nbytes, mat_data, GL_STATIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def _build_BLASes(self):

        all_vertices, all_indices, all_points, all_blas_nodes = [], [], [], []
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

            if asset.asset_type == AssetType.Mesh:

                positions = asset.vertices[:, :3].astype(np.float32)
                verts4 = np.pad(positions, ((0, 0), (0, 1)), 'constant', constant_values=0)
                indices = asset.indices.astype(np.uint32)

                blas = BVH.from_indexed_mesh(verts4, indices)

                # All mesh data goes to a global list
                all_vertices.append(asset.vertices)
                all_indices.append(asset.indices.flatten())

                # Record offsets for this asset, to be used by instances
                self.asset_to_blas_map[asset.id] = {
                    'id': blas_id,
                    'vert_offset': current_vert_offset,
                    'idx_offset': current_idx_offset,
                    'is_points': 0  # Mesh asset, not a point cloud
                }

                current_vert_offset += len(asset.vertices)
                current_idx_offset += len(asset.indices.flatten())

            elif asset.asset_type == AssetType.Points:

                points = asset.points.astype(np.float32)
                radii = asset.radii.astype(np.float32)

                blas = BVH.from_points(
                    points,
                    radius=radii,
                    traversal_cost=1.0,
                    intersection_cost=1.0
                )

                bundle = blas.get_SSBO_bundle(flatten_nodes=False)

                # Pack point data for the shader
                nb_points = len(asset.points)

                packed_points = np.zeros((nb_points, 12), dtype=np.float32)
                packed_points[:, 0:3] = asset.points
                packed_points[:, 3] = asset.radii
                packed_points[:, 4:7] = asset.normals
                packed_points[:, 7:10] = asset.colors
                all_points.append(packed_points)

                self.asset_to_blas_map[asset.id] = {
                    'id': blas_id,
                    'point_offset': current_point_offset,
                }

                current_point_offset += nb_points

            else:
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

    def _build_TLAS(self):

        if not self.BLASes:
            return

        all_instances = self.scene.instances
        num_instances = len(all_instances)

        # pytinybvh input for TLAS build + our GPU-side per-instance struct
        tlas_build_data = np.zeros(num_instances, dtype=instance_dtype)
        self.gpu_instances_info = np.zeros(num_instances, dtype=RENDERABLE_INST_DTYPE)

        for i, inst in enumerate(all_instances):
            blas_map = self.asset_to_blas_map[inst.asset.id]

            if inst.visible:
                transform = np.asarray(inst.transform, dtype=np.float32)
                inv_transform = np.asarray(glm.inverse(inst.transform), dtype=np.float32)
            else:
                hidden = glm.translate(glm.mat4(1.0), glm.vec3(1e6, 1e6, 1e6))
                transform = np.asarray(hidden, dtype=np.float32)
                inv_transform = np.asarray(glm.inverse(hidden), dtype=np.float32)

            tlas_build_data[i]['transform'] = transform
            tlas_build_data[i]['blas_id'] = blas_map['id']
            tlas_build_data[i]['mask'] = 0xFFFFFFFF

            self.gpu_instances_info[i]['transform'] = transform
            self.gpu_instances_info[i]['inverse_transform'] = inv_transform
            self.gpu_instances_info[i]['blas_node_offset'] = blas_map['node_offset']
            self.gpu_instances_info[i]['prim_index_offset'] = blas_map['prim_index_offset']

            if inst.asset.asset_type == AssetType.Mesh:
                self.gpu_instances_info[i]['vertex_or_point_offset'] = blas_map['vert_offset']
                self.gpu_instances_info[i]['index_offset'] = blas_map['idx_offset']
                self.gpu_instances_info[i]['material_id'] = self.material_map.get(inst.asset.id, 0)

            elif inst.asset.asset_type == AssetType.Points:
                self.gpu_instances_info[i]['vertex_or_point_offset'] = blas_map['point_offset']
                self.gpu_instances_info[i]['index_offset'] = 0
                self.gpu_instances_info[i]['radius_factor'] = inst.properties.get('radius_factor', 1.0)
                self.gpu_instances_info[i]['is_points'] = 1

            # Track dynamic instances
            if inst.dynamic:
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
         self.blas_indices_ssbo) = glGenBuffers(8)

        # Primitive streams
        upload(self.vertices_ssbo, self.cpu_vertices, GL_STATIC_DRAW, np.float32, 5)  # (pos, uv)
        upload(self.indices_ssbo, self.cpu_indices, GL_STATIC_DRAW, np.uint32, 3)
        upload(self.points_ssbo, self.cpu_points, GL_STATIC_DRAW, np.float32, 12)

        # Nodes
        upload(self.blas_nodes_ssbo, self.cpu_BLASes, GL_STATIC_DRAW, np.float32, 1)
        upload(self.tlas_nodes_ssbo, self.cpu_TLAS_nodes, GL_DYNAMIC_DRAW, np.float32, 1)

        # Leaf -> thing mappings
        upload(self.tlas_indices_ssbo, self.cpu_TLAS_prim_indices,
               GL_STATIC_DRAW, np.uint32, 1)  # TLAS: leaf -> instance
        upload(self.blas_indices_ssbo, self.cpu_BLAS_prim_indices,
               GL_STATIC_DRAW, np.uint32, 1)  # BLAS: leaf -> primitive

        # Per-instance info (updates with animation)
        upload(self.instances_info_ssbo, self.gpu_instances_info, GL_DYNAMIC_DRAW, RENDERABLE_INST_DTYPE, 1)

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def update(self):
        """ Pulls transforms from dynamic scene instances, updates and refits the TLAS, and uploads to GPU """

        self.update_lights()

        if not self.dynamic_instance_map or self.TLAS is None:
            return

        updated = False

        for inst in self.scene._dynamic_instances:
            tlas_idx = self.dynamic_instance_map.get(inst.id)

            if tlas_idx is None:
                continue

            if inst.visible:
                transform = np.asarray(inst.transform, dtype=np.float32)
                inv_transform = np.asarray(glm.inverse(inst.transform), dtype=np.float32)
            else:
                hidden = glm.translate(glm.mat4(1.0), glm.vec3(1e6, 1e6, 1e6))
                transform = np.asarray(hidden, dtype=np.float32)
                inv_transform = np.asarray(glm.inverse(hidden), dtype=np.float32)

            self.TLAS.set_instance_transform(tlas_idx, transform)

            # Update GPU info
            self.gpu_instances_info[tlas_idx]['transform'] = transform
            self.gpu_instances_info[tlas_idx]['inverse_transform'] = inv_transform
            updated = True

        if updated:
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
        glTexStorage3D(GL_TEXTURE_2D_ARRAY, 1, GL_SRGB8_ALPHA8, tex_w, tex_h, layer_count)

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
        """Deletes all OpenGL resources managed by this class."""

        buffers = [b for b in [
            self.vertices_ssbo, self.indices_ssbo, self.points_ssbo, self.materials_ssbo,
            self.tlas_nodes_ssbo, self.blas_nodes_ssbo, self.instances_info_ssbo,
            self.tlas_indices_ssbo, self.blas_indices_ssbo,
            self.directional_lights_ssbo, self.point_lights_ssbo, self.area_lights_ssbo,
        ] if b != 0]

        if buffers:
            glDeleteBuffers(len(buffers), buffers)

        if self.tex_array != 0:
            glDeleteTextures(1, [self.tex_array])


class Raytracer(BaseRenderer):
    """
    Raytracer for compound eye rendering.
    """

    def __init__(self, receptor_array, scene: Scene,
                 time_dithering: bool = True,
                 nb_samples: int = 256,
                 quasi_random: bool = False,
                 pano_res: Tuple[int, int] = (1024, 512),
                 batch_size: int = 1,
                 enable_direct: bool = True,
                 enable_shadows: bool = True,
                 enable_ambient: bool = True,
                 ):

        self.scene = scene  # just for convenience
        self._scene_baked = RaytracingSceneBaker(scene)
        # TODO: Add properties to access the baked BLASes and TLAS

        # Global lighting controls
        self.enable_direct = enable_direct
        self.enable_shadows = enable_shadows
        self.enable_ambient = enable_ambient
        self.ambient_intensity = 1.0
        self.sky_intensity = 1.0

        # just to keep track of which defines were used for shader compilation
        # (so we can change and recompile when needed)
        self._active_light_defines: Set[str] = set()

        # super().__init__ *after* baking the scene to estimate VRAM
        super().__init__(
            receptor_array,
            time_dithering=time_dithering,
            nb_samples=nb_samples,
            quasi_random=quasi_random,
            batch_size=batch_size
        )

        self._compile_shaders()

        self.panoramic_shader = None  # lazily-loaded

        # standard perspective view shader (also lazy-loaded)
        self._persp_res = None
        self._persp_texture_id = 0
        self.perspective_shader = None

        self._pano_res = pano_res
        self._pano_texture_id = 0
        self._texture_viewer = TextureViewer()

        # Intermediate Ray result SSBO
        self.ray_results_ssbo = glGenBuffers(1)

        # Default number of samples
        self._samples_per_ommatidium = 0
        self.samples_per_receptor = nb_samples  # via the setter to allocate the SSBO

    @property
    def samples_per_receptor(self):
        return self._samples_per_ommatidium

    @samples_per_receptor.setter
    def samples_per_receptor(self, value):
        bytes_per_sample = 16
        max_total_samples = self._max_ssbo_size // bytes_per_sample
        max_samples_per_om = max(1, max_total_samples // self.total_receptors)
        new_value = int(np.clip(value, 1, max_samples_per_om))

        if new_value != value:
            print(f"Warning: Clamped samples per ommatidium to {new_value} (HW limit is {max_samples_per_om}).")

        if new_value == self._samples_per_ommatidium:
            return

        self.samples_per_pixel = 1
        self._samples_per_ommatidium = new_value
        self.total_samples = self.total_receptors * self._samples_per_ommatidium
        required_buffer_size = self.total_samples * bytes_per_sample

        print(f"Allocating ray result buffer for {self.total_samples:,} "
              f"total samples ({required_buffer_size / (1024 * 1024):.2f} MB).")

        glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.ray_results_ssbo)
        glBufferData(GL_SHADER_STORAGE_BUFFER, required_buffer_size, None, GL_DYNAMIC_DRAW)
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    def _get_light_defines(self) -> Set[str]:
        """
        Determines which #defines to inject based on the current light counts.
        """
        baker = self._scene_baked
        defines = set()

        if baker.num_directional_lights == 1:
            defines.add('HAS_DIRECTIONAL_LIGHT')
        elif baker.num_directional_lights > 1:
            defines.add('HAS_DIRECTIONAL_LIGHT')
            defines.add('MULTI_DIRECTIONAL')

        if baker.num_point_lights == 1:
            defines.add('HAS_POINT_LIGHT')
        elif baker.num_point_lights > 1:
            defines.add('HAS_POINT_LIGHT')
            defines.add('MULTI_POINT')

        if baker.num_area_lights == 1:
            defines.add('HAS_AREA_LIGHT')
        elif baker.num_area_lights > 1:
            defines.add('HAS_AREA_LIGHT')
            defines.add('MULTI_AREA')

        return defines

    def _compile_shaders(self):

        defines = self._get_light_defines()
        self._active_light_defines = defines

        define_names = sorted(defines) if defines else ['(none)']
        print(f"Compiling ray-tracing shaders with defines: {', '.join(define_names)}")

        self.raytrace_shader = ShaderProgram(comp_path='shaders/raytracing/ommatidiaRaytracing.comp', defines=defines)
        self.reduction_shader = ShaderProgram(comp_path='shaders/raytracing/raysReduction.comp')

    def _check_recompile(self):
        """Check if light configuration changed and recompile shaders if needed."""

        new_defines = self._get_light_defines()
        if new_defines != self._active_light_defines:
            print("Light configuration changed, recompiling shaders...")

            if self.raytrace_shader:
                self.raytrace_shader.free()

            if self.panoramic_shader:
                self.panoramic_shader.free()
                self.panoramic_shader = None

            if self.perspective_shader:
                self.perspective_shader.free()
                self.perspective_shader = None

            self._compile_shaders()

    @property
    def lights_count(self) -> int:
        baker = self._scene_baked
        return baker.num_directional_lights + baker.num_point_lights + baker.num_area_lights

    def estimate_vram_usage(self) -> float:
        """Override base method to provide a more accurate VRAM estimate for the raytracer."""

        baker = self._scene_baked
        total_bytes = 0
        buffers = [
            baker.cpu_vertices, baker.cpu_indices, baker.cpu_points, baker.cpu_BLASes,
            baker.cpu_TLAS_nodes, baker.cpu_TLAS_prim_indices, baker.cpu_BLAS_prim_indices,
            baker.gpu_instances_info,
        ]
        for buf in buffers:
            if buf is not None:
                total_bytes += buf.nbytes

        # Add intermediate ray results buffer
        total_bytes += getattr(self, 'total_samples', 0) * 16

        return total_bytes / (1024 * 1024)  # Convert to MB

    def _initialize_pano_resources(self):
        """Creates all resources needed for the panoramic view."""

        if self.panoramic_shader is None:
            defines = self._get_light_defines()
            self.panoramic_shader = ShaderProgram(comp_path='shaders/raytracing/panoramicRaytracing.comp', defines=defines)

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

    def _initialize_persp_resources(self):

        if self.perspective_shader is None:
            defines = self._get_light_defines()
            self.perspective_shader = ShaderProgram(comp_path='shaders/raytracing/perspectiveRaytracing.comp', defines=defines)

        if self._persp_res is None:
            viewport = glGetIntegerv(GL_VIEWPORT)
            self._persp_res = (viewport[2], viewport[3])

        if self._persp_texture_id == 0:
            tex = glGenTextures(1)
            glBindTexture(GL_TEXTURE_2D, tex)
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F, self._persp_res[0], self._persp_res[1], 0, GL_RGBA, GL_FLOAT,
                         None)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
            glBindTexture(GL_TEXTURE_2D, 0)

            self._persp_texture_id = tex

    def _bind_ssbos(self):
        """Binds all scene geometry and light SSBOs to their fixed slots."""

        # Bindings for the receptors data and rays outputs are handled in passes.

        # Bindings for scene geometry
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_VERTICES, self._scene_baked.vertices_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_INDICES, self._scene_baked.indices_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_MATERIALS, self._scene_baked.materials_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_POINTS, self._scene_baked.points_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_BLAS_NODES, self._scene_baked.blas_nodes_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_TLAS_NODES, self._scene_baked.tlas_nodes_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_INSTANCES, self._scene_baked.instances_info_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_TLAS_INDICES, self._scene_baked.tlas_indices_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_BLAS_INDICES, self._scene_baked.blas_indices_ssbo)

        # Bindings for lights
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_LIGHT_DIR, self._scene_baked.directional_lights_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_LIGHT_POINT, self._scene_baked.point_lights_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_LIGHT_AREA, self._scene_baked.area_lights_ssbo)

    def _bind_textures(self, shader: ShaderProgram):
        """Binds skybox and material textures."""

        if self._scene_baked.scene.skybox:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_CUBE_MAP, self._scene_baked.skybox_texture)
            glUniform1i(shader.get_loc('skybox'), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self._scene_baked.tex_array)
        glUniform1i(shader.get_loc('scene_textures'), 1)

    def _set_common_uniforms(self, shader: ShaderProgram):

        # Scene structure
        glUniform1ui(shader.get_loc('nb_tlas_nodes'), len(self._scene_baked.cpu_TLAS_nodes))

        # Background / sky
        glUniform1i(shader.get_loc('use_skybox'), int(self.scene.skybox is not None))
        bg = self.scene.background_color
        glUniform3f(shader.get_loc('background_color'), bg[0], bg[1], bg[2])
        glUniform1f(shader.get_loc('sky_intensity'), self.sky_intensity)

        # Dither counter (for RNG seeding)
        glUniform1ui(shader.get_loc('dither_counter'), int(self._dither_counter))

        # Global lighting controls
        glUniform1i(shader.get_loc('enable_ambient'), int(self.enable_ambient))
        glUniform1i(shader.get_loc('enable_direct'), int(self.enable_direct))
        glUniform1i(shader.get_loc('enable_shadows'), int(self.enable_shadows))
        glUniform1f(shader.get_loc('ambient_intensity'), self.ambient_intensity)

        # Light counts
        glUniform1i(shader.get_loc('directional_lights_count'), self._scene_baked.num_directional_lights)
        glUniform1i(shader.get_loc('point_lights_count'), self._scene_baked.num_point_lights)
        glUniform1i(shader.get_loc('area_lights_count'), self._scene_baked.num_area_lights)

    def _bind_resources(self, shader: ShaderProgram):
        """
        Convenience method: binds SSBOs, textures, and sets common uniforms.
        """
        self._bind_ssbos()
        self._bind_textures(shader)
        self._set_common_uniforms(shader)

    def _raytrace_panoramic(self, agent):
        """Dispatches a compute shader to generate a ray-traced panoramic image."""

        self.panoramic_shader.use()

        # Bind the output texture to image unit 0 for writing
        glBindImageTexture(0, self._pano_texture_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F)

        # Bind scene resources and common uniforms
        self._bind_resources(self.panoramic_shader)

        # TODO: Maybe move this in the common uniforms
        glUniform1i(self.panoramic_shader.get_loc('nb_samples'), self.samples_per_pixel)

        # Panoramic-specific uniforms
        c2w_mat = glm.inverse(agent.view)
        glUniformMatrix4fv(self.panoramic_shader.get_loc('cam_to_world'), 1, False, glm.value_ptr(c2w_mat))

        # Dispatch compute shader
        work_groups_x = (self._pano_res[0] + 15) // 16
        work_groups_y = (self._pano_res[1] + 15) // 16
        glDispatchCompute(work_groups_x, work_groups_y, 1)

        # ensure imageStore writes are complete before the texture is used for drawing
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        self.panoramic_shader.stop()

    def _raytrace_perspective(self, agent):

        self.perspective_shader.use()

        # Output image
        glBindImageTexture(0, self._persp_texture_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F)

        # Bind scene resources and common uniforms
        self._bind_resources(self.perspective_shader)

        # TODO: Maybe move this in the common uniforms
        glUniform1i(self.perspective_shader.get_loc('nb_samples'), self.samples_per_pixel)

        # Perspective-specific uniforms
        c2w = glm.inverse(agent.view)
        glUniformMatrix4fv(self.perspective_shader.get_loc('cam_to_world'), 1, False, glm.value_ptr(c2w))

        inverse_proj = glm.inverse(agent.projection)
        glUniformMatrix4fv(self.perspective_shader.get_loc('inv_projection'), 1, False, glm.value_ptr(inverse_proj))

        # Dispatch
        wg_x = (self._persp_res[0] + 15) // 16
        wg_y = (self._persp_res[1] + 15) // 16
        glDispatchCompute(wg_x, wg_y, 1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

        self.perspective_shader.stop()

    def _raytrace(self, agent):
        """Pass 1: Ray-tracing each receptor"""

        self.raytrace_shader.use()

        # Bind receptors-specific input/output buffers
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_RECEPTORS, self.receptors_data_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_RAYS_INTERMEDIATE, self.ray_results_ssbo)

        # Bind scene resources and common uniforms
        self._bind_resources(self.raytrace_shader)

        # Ommatidia-specific uniforms
        glUniform1i(self.raytrace_shader.get_loc('nb_samples'), self.samples_per_receptor)

        # Halton sampling
        glUniform1i(self.raytrace_shader.get_loc('use_quasi_random'), int(self._quasi_random))

        # Camera uniforms for transforming rays into world space
        c2w_mat = glm.inverse(agent.view)
        glUniformMatrix4fv(self.raytrace_shader.get_loc('cam_to_world'), 1, False, glm.value_ptr(c2w_mat))

        # Dispatch compute shader
        workgroup_size = 64  # TODO: maybe tweak workgroup size
        work_groups = (self.total_samples + (workgroup_size - 1)) // workgroup_size
        glDispatchCompute(work_groups, 1, 1)

        # make sure writes to ray_results_ssbo are complete before the next pass tries to read from it
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        self.raytrace_shader.stop()

    def _reduction(self):
        """Pass 2: Reduction."""

        self.reduction_shader.use()

        # Bind buffers
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_RAYS_INTERMEDIATE, self.ray_results_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_COLORS, self.final_colors_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_STATE, self.receptor_state_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_RECEPTORS, self.receptors_data_ssbo)

        # Set uniforms
        glUniform1i(self.reduction_shader.get_loc('nb_samples'), self.samples_per_receptor)
        glUniform1i(self.reduction_shader.get_loc('nb_receptors'), self.total_receptors)
        glUniform1f(self.reduction_shader.get_loc('dt'), self._dt)

        # Write into the history buffer circularly
        frame_offset = self._frame_index % self._batch_size
        glUniform1i(self.reduction_shader.get_loc('frame_offset'), frame_offset)

        # Dispatch compute shader
        workgroup_size = 64  # TODO: maybe tweak workgroup size
        work_groups = (self.total_receptors + (workgroup_size - 1)) // workgroup_size
        glDispatchCompute(work_groups, 1, 1)

        # final_colors_ssbo must be fully written before the CPU or the drawing shader tries to read from it
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

        self.reduction_shader.stop()

    def _compute_colors(self, agent):
        """The core receptors rendering logic."""

        self._tick()

        # First refresh the scene for anything that moved / changed
        self._scene_baked.update()

        # Pass 1: Ray-trace
        self._raytrace(agent)

        # Pass 2: Reduction
        self._reduction()

        # Unbind all resources
        for i in range(17):
            glBindBufferBase(GL_SHADER_STORAGE_BUFFER, i, 0)

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, 0)
        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)

    def draw(self, view_mode: DisplayMode, point_of_view: Agent, agent: Agent = None):
        """Renders one of the rasterizer's supported views to the screen."""

        if view_mode == DisplayMode.Compound:
            self._draw_voronoi()

        elif view_mode == DisplayMode.Panoramic:

            if self._pano_texture_id == 0 or self.panoramic_shader is None:
                self._initialize_pano_resources()

            self._scene_baked.update()
            self._raytrace_panoramic(point_of_view)
            self._texture_viewer.draw(self._pano_texture_id)

        elif view_mode == DisplayMode.Perspective or view_mode == DisplayMode.Third_person:

            if self._persp_texture_id == 0 or self.perspective_shader is None:
                self._initialize_persp_resources()

            self._scene_baked.update()
            self._raytrace_perspective(point_of_view)
            self._texture_viewer.draw(self._persp_texture_id)

        if view_mode == DisplayMode.Third_person:
            self._draw_eye_model(point_of_view, agent)

    def free(self):
        """Frees all GPU resources."""

        if self.ray_results_ssbo: glDeleteBuffers(1, [self.ray_results_ssbo])
        if self.raytrace_shader: self.raytrace_shader.free()
        if self.reduction_shader: self.reduction_shader.free()

        if self.panoramic_shader: self.panoramic_shader.free()
        if self._pano_texture_id != 0: glDeleteTextures(1, [self._pano_texture_id])

        if self.perspective_shader: self.perspective_shader.free()
        if self._persp_texture_id != 0: glDeleteTextures(1, [self._persp_texture_id])

        if self._texture_viewer: self._texture_viewer.free()

        self._scene_baked.free()
        super().free()


class Pathtracer(Raytracer):
    """
    Path tracer: multiple bounces with Monte Carlo integration.
    """
    def __init__(self, receptor_array, scene: Scene,
                 time_dithering: bool = True,
                 nb_samples: int = 256,
                 quasi_random: bool = False,
                 pano_res: Tuple[int, int] = (1024, 512),
                 batch_size: int = 1,
                 enable_shadows: bool = True,
                 enable_ambient: bool = True,
                 enable_direct: bool = True,
                 max_bounces: int = 3,
                 ):

        self.max_bounces = max_bounces

        super().__init__(
            receptor_array=receptor_array,
            scene=scene,
            time_dithering=time_dithering,
            nb_samples=nb_samples,
            quasi_random=quasi_random,
            pano_res=pano_res,
            batch_size=batch_size,
            enable_shadows=enable_shadows,
            enable_ambient=enable_ambient,
            enable_direct=enable_direct
        )

    def _get_light_defines(self) -> Set[str]:
        defines = super()._get_light_defines()
        defines.add('PATH_TRACING')
        return defines

    def _set_common_uniforms(self, shader: ShaderProgram):
        super()._set_common_uniforms(shader)
        glUniform1i(shader.get_loc('max_bounces'), self.max_bounces)