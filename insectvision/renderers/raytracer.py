import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *

from typing import Tuple, List, Dict, Optional, Set, Union
import numpy as np
from PIL import Image
from pyglm import glm
from pytinybvh import BVH, instance_dtype, Layout, supports_layout

from insectvision.interactive.utils import DisplayMode
from insectvision.compound_eyes import ReceptorArray
from insectvision.engine.agent import Agent
from insectvision.engine.scene import Scene, AssetType, Asset
from insectvision.engine.lights import DIR_LIGHT_DTYPE, POINT_LIGHT_DTYPE, AREA_LIGHT_DTYPE
from insectvision.engine.shader_utils import write_pytinybvh_preamble, ShaderProgram

from .commons import (
    BaseRenderer, TextureViewer, _create_ssbo, BINDING_RCPT_STATIC,
    BINDING_LENS_STATIC, BINDING_COLOR, BINDING_EMA_HIST, BINDING_RCPT_DYNAMIC, BINDING_RAYS_INTERMEDIATE, BINDING_LENS_DYNAMIC
)


# Scene geometry bindings
BINDING_VERTICES     = 7 #5
BINDING_INDICES      = 8 #6
BINDING_MATERIALS    = 9 #7
BINDING_POINTS       = 10 #8
BINDING_BLAS_NODES   = 11 #9
BINDING_TLAS_NODES   = 12 #10
BINDING_INSTANCES    = 13 #11
BINDING_TLAS_INDICES = 14 #12
BINDING_BLAS_INDICES = 15 #13

# Light bindings
BINDING_LIGHT_DIR    = 16 #14
BINDING_LIGHT_POINT  = 17 #15
BINDING_LIGHT_AREA   = 18 #16

RENDERABLE_INST_DTYPE = np.dtype([
    ('transform', np.float32, (4, 4)),
    ('inverse_transform', np.float32, (4, 4)),
    ('blas_node_offset', np.uint32),
    ('vertex_or_point_offset', np.uint32),
    ('index_offset', np.uint32),
    ('material_id', np.uint32),
    ('is_points', np.uint32),
    ('prim_index_offset', np.uint32),
    ('radius_factor', np.float32),
    ('padding', np.uint32, 1),
])  # 160 bytes


##

def _create_texture(width, height, image_data: Optional[np.ndarray] = None, repeat: bool = False, dtype=float):
    texture_id = glGenTextures(1)

    glBindTexture(GL_TEXTURE_2D, texture_id)

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT if repeat else GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT if repeat else GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    # TODO: This should probably use the image_data dtpye
    if dtype == float:
        bitdepth = GL_RGBA32F
        typ = GL_FLOAT
    elif dtype == int:
        bitdepth = GL_SRGB_ALPHA
        typ = GL_UNSIGNED_BYTE
    else:
        raise AttributeError(f'Unsupported data type {dtype}')

    glTexImage2D(GL_TEXTURE_2D, 0, bitdepth, width, height, 0, GL_RGBA, typ, image_data)

    return texture_id


def _create_tex_array(texture_ids):

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

    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

    glBindTexture(GL_TEXTURE_2D_ARRAY, 0)

    return tex_array_id


# Scene baker


class RaytraceBaker:
    """
    Manages BVH structures and GPU buffers for a Scene.
    """

    def __init__(self, scene: Scene):
        self.scene = scene

        self._tlas: Optional[BVH] = None
        self._blases: List[BVH] = []
        self._dynamic_map: Dict[int, int] = {}

        # Geometry
        self.skybox_tex = self.scene.skybox.texture_id if self.scene.skybox else 0
        self.materials_ssbo = None
        self.tex_array = 0
        self.points_ssbo = None
        self.verts_ssbo = None
        self.indices_ssbo = None
        self.tlas_nodes_ssbo = None
        self.blas_nodes_ssbo = None
        self.inst_info_ssbo = None
        self.tlas_indices_ssbo = None
        self.blas_indices_ssbo = None

        # Lights
        self.dir_lights_ssbo = None
        self.point_lights_ssbo = None
        self.area_lights_ssbo = None
        self._nb_dir_lights = 0
        self._nb_point_lights = 0
        self._nb_area_lights = 0

        # CPU-side data
        self.gpu_inst_info: Optional[np.ndarray] = None
        self._material_map: Dict[int, int] = {}
        self._asset_blas_map: Dict[int, Dict] = {}

        print("Baking ray-tracing scene...")
        if not self.scene.instances:
            print("Warning: Scene is empty, nothing to bake.")
            return

        # Pack scene materials and lights
        self._pack_materials()
        self._pack_lights()

        # Pack geometry into BVH
        self._build_blases()
        self._build_tlas()

        # Upload everything to GPU
        self._push_to_gpu()

    # Main packing methods

    def _pack_lights(self):

        dir_l = [l for l in self.scene.directional_lights if l.active]
        point_l = [l for l in self.scene.point_lights if l.active]
        area_l = [l for l in self.scene.area_lights if l.active]

        self._nb_dir_lights = len(dir_l)
        self._nb_point_lights = len(point_l)
        self._nb_area_lights = len(area_l)

        def _pack(lights, dtype):
            return np.concatenate([l.pack() for l in lights]) if lights else np.zeros(1, dtype=dtype)

        # Free previous buffers if re-packing
        for buf in (self.dir_lights_ssbo, self.point_lights_ssbo, self.area_lights_ssbo):
            if buf:
                glDeleteBuffers(1, [buf])

        self.dir_lights_ssbo = _create_ssbo(data=_pack(dir_l, DIR_LIGHT_DTYPE), usage=GL_DYNAMIC_DRAW)
        self.point_lights_ssbo = _create_ssbo(data=_pack(point_l, POINT_LIGHT_DTYPE), usage=GL_DYNAMIC_DRAW)
        self.area_lights_ssbo = _create_ssbo(data=_pack(area_l, AREA_LIGHT_DTYPE), usage=GL_DYNAMIC_DRAW)

    def _pack_materials(self):
        """Packs material data for all mesh assets into GPU buffers."""

        mesh_assets = {inst.asset for inst in self.scene.mesh_instances}
        if not mesh_assets:
            return

        self._material_map = {asset.id: i for i, asset in enumerate(mesh_assets)}

        texture_images = []
        self._asset_tex_map = {}

        for asset in mesh_assets:
            if asset.has_texture:
                img = asset.texture_image
                if img is not None:
                    self._asset_tex_map[asset.id] = len(texture_images)
                    texture_images.append(img)
                else:
                    self._asset_tex_map[asset.id] = None
            else:
                self._asset_tex_map[asset.id] = None

        if texture_images:
            self.tex_w, self.tex_h = texture_images[0].size
            print(f"Creating texture array: {len(texture_images)} textures at {self.tex_w}x{self.tex_h}")

            tex_ids = []
            for i, img in enumerate(texture_images):
                if img.size != (self.tex_w, self.tex_h):
                    print(f"  Resizing texture {i} from {img.size}")
                    img = img.resize((self.tex_w, self.tex_h), Image.Resampling.LANCZOS)

                img_data = img.convert('RGBA').tobytes()
                tex_id = _create_texture(self.tex_w, self.tex_h, image_data=img_data, repeat=True, dtype=int)
                tex_ids.append(tex_id)

            self.tex_array = _create_tex_array(tex_ids)
            glDeleteTextures(len(tex_ids), tex_ids)

        mat_data = np.zeros((len(mesh_assets), 4), dtype=np.uint32)

        for asset in mesh_assets:
            idx = self._material_map[asset.id]
            tex_idx = self._asset_tex_map[asset.id]

            mat_data[idx, 0] = tex_idx if tex_idx is not None else 0xFFFFFFFF

            c = asset.material.base_color
            r = int(np.clip(c[0], 0, 1) * 255) & 0xFF
            g = int(np.clip(c[1], 0, 1) * 255) & 0xFF
            b = int(np.clip(c[2], 0, 1) * 255) & 0xFF
            a = int(np.clip(c[3], 0, 1) * 255) & 0xFF
            mat_data[idx, 1] = (a << 24) | (b << 16) | (g << 8) | r

        self.materials_ssbo = _create_ssbo(data=mat_data, usage=GL_STATIC_DRAW)

    # BVH construction

    @staticmethod
    def _inst_transforms(inst):
        """Return (transform, inv_transform) arrays for a scene instance."""
        if inst.visible:
            transform = np.asarray(inst.transform, dtype=np.float32)
            inv_transform = np.asarray(glm.inverse(inst.transform), dtype=np.float32)
        else:
            hidden = glm.translate(glm.mat4(1.0), glm.vec3(1e6, 1e6, 1e6))
            transform = np.asarray(hidden, dtype=np.float32)
            inv_transform = np.asarray(glm.inverse(hidden), dtype=np.float32)

        return transform, inv_transform

    def _build_blases(self):

        all_verts, all_idxs, all_pts, all_nodes = [], [], [], []
        v_off, idx_off, pt_off, n_off = 0, 0, 0, 0

        self._blas_leaf_chunks = []
        l_off = 0

        print(f"Building BLASes for {len(self.scene.assets)} unique assets...")
        for asset in self.scene.assets.values():

            if asset.id in self._asset_blas_map:
                continue

            blas_id = len(self._blases)
            bundle = None

            if asset.asset_type == AssetType.Mesh:

                positions = asset.vertices[:, :3].astype(np.float32)
                verts4 = np.pad(positions, ((0, 0), (0, 1)), 'constant', constant_values=0)
                indices = asset.indices.astype(np.uint32)

                blas = BVH.from_indexed_mesh(verts4, indices)

                all_verts.append(asset.vertices)
                all_idxs.append(asset.indices.flatten())

                self._asset_blas_map[asset.id] = {
                    'id': blas_id,
                    'v_off': v_off,
                    'idx_off': idx_off,
                    'is_points': 0
                }

                v_off += len(asset.vertices)
                idx_off += len(asset.indices.flatten())

            elif asset.asset_type == AssetType.Points:

                points = asset.points.astype(np.float32)
                radii = asset.radii.astype(np.float32)

                blas = BVH.from_points(
                    points, radius=radii,
                    traversal_cost=1.0, intersection_cost=1.0
                )

                bundle = blas.get_SSBO_bundle(flatten_nodes=False)

                nb_points = len(asset.points)
                packed_points = np.zeros((nb_points, 12), dtype=np.float32)
                packed_points[:, 0:3] = asset.points
                packed_points[:, 3] = asset.radii
                packed_points[:, 4:7] = asset.normals
                packed_points[:, 7:10] = asset.colors
                all_pts.append(packed_points)

                self._asset_blas_map[asset.id] = {
                    'id': blas_id,
                    'pt_off': pt_off,
                }

                pt_off += nb_points

            else:
                continue

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

            all_nodes.append(nodes)
            self._blas_leaf_chunks.append(prim_indices)

            self._asset_blas_map[asset.id].update({'n_off': n_off, 'l_off': l_off})
            self._blases.append(blas)
            n_off += nodes.shape[0]
            l_off += prim_indices.size

        self.cpu_verts = np.concatenate(all_verts).ravel() if all_verts else None
        self.cpu_idx = np.concatenate(all_idxs).ravel() if all_idxs else None
        self.cpu_pts = np.concatenate(all_pts).ravel() if all_pts else None
        self.cpu_blas = np.concatenate(all_nodes).astype(np.float32) if all_nodes else None

    def _build_tlas(self):

        if not self._blases:
            return

        all_instances = self.scene.instances
        num_instances = len(all_instances)

        tlas_build_data = np.zeros(num_instances, dtype=instance_dtype)
        self.gpu_inst_info = np.zeros(num_instances, dtype=RENDERABLE_INST_DTYPE)

        for i, inst in enumerate(all_instances):
            blas_map = self._asset_blas_map[inst.asset.id]
            transform, inv_transform = self._inst_transforms(inst)

            tlas_build_data[i]['transform'] = transform
            tlas_build_data[i]['blas_id'] = blas_map['id']
            tlas_build_data[i]['mask'] = 0xFFFFFFFF

            self.gpu_inst_info[i]['transform'] = transform
            self.gpu_inst_info[i]['inverse_transform'] = inv_transform
            self.gpu_inst_info[i]['blas_node_offset'] = blas_map['n_off']
            self.gpu_inst_info[i]['prim_index_offset'] = blas_map['l_off']

            if inst.asset.asset_type == AssetType.Mesh:
                self.gpu_inst_info[i]['vertex_or_point_offset'] = blas_map['v_off']
                self.gpu_inst_info[i]['index_offset'] = blas_map['idx_off']
                self.gpu_inst_info[i]['material_id'] = self._material_map.get(inst.asset.id, 0)

            elif inst.asset.asset_type == AssetType.Points:
                self.gpu_inst_info[i]['vertex_or_point_offset'] = blas_map['pt_off']
                self.gpu_inst_info[i]['index_offset'] = 0
                self.gpu_inst_info[i]['radius_factor'] = inst.properties.get('radius_factor', 1.0)
                self.gpu_inst_info[i]['is_points'] = 1

            if inst.dynamic:
                self._dynamic_map[inst.id] = i

        self._tlas = BVH.build_tlas(tlas_build_data, self._blases)

        t = self._tlas.get_SSBO_bundle(flatten_nodes=False)

        self.cpu_tlas_nodes = t['nodes'].astype(np.float32)
        self.cpu_tlas_idx = t['leaf_ids'].astype(np.uint32)

        write_pytinybvh_preamble(str(t.get('preamble', '')))

        self.cpu_blas_idx = np.concatenate(self._blas_leaf_chunks).astype(np.uint32)

    def _push_to_gpu(self):

        def _safe(data, dtype, min_elms=1):
            if data is None or getattr(data, 'nbytes', 0) == 0:
                return np.zeros((min_elms,), dtype=dtype)
            return data

        self.verts_ssbo = _create_ssbo(data=_safe(self.cpu_verts, np.float32, 5), usage=GL_STATIC_DRAW)
        self.indices_ssbo = _create_ssbo(data=_safe(self.cpu_idx, np.uint32, 3), usage=GL_STATIC_DRAW)
        self.points_ssbo = _create_ssbo(data=_safe(self.cpu_pts, np.float32, 12), usage=GL_STATIC_DRAW)

        self.blas_nodes_ssbo = _create_ssbo(data=_safe(self.cpu_blas, np.float32, 1), usage=GL_STATIC_DRAW)
        self.tlas_nodes_ssbo = _create_ssbo(data=_safe(self.cpu_tlas_nodes, np.float32, 1), usage=GL_DYNAMIC_DRAW)

        self.tlas_indices_ssbo = _create_ssbo(data=_safe(self.cpu_tlas_idx, np.uint32, 1), usage=GL_STATIC_DRAW)
        self.blas_indices_ssbo = _create_ssbo(data=_safe(self.cpu_blas_idx, np.uint32, 1), usage=GL_STATIC_DRAW)

        self.inst_info_ssbo = _create_ssbo(data=_safe(self.gpu_inst_info, RENDERABLE_INST_DTYPE, 1), usage=GL_DYNAMIC_DRAW)

    # Dynamic updates

    def update_texture(self, asset: 'Asset'):
        """Update a texture on the GPU for a given Asset."""

        if self.tex_array == 0:
            return

        tex_idx = self._asset_tex_map.get(asset.id)
        if tex_idx is None:
            print(f"Warning: Asset '{asset.name}' did not have a texture when the scene was baked. "
                  f"Texture updates need the asset to be initialised with a texture.")
            return

        img = asset.texture_image
        if img is None:
            return

        if img.size != (self.tex_w, self.tex_h):
            img = img.resize((self.tex_w, self.tex_h), Image.Resampling.LANCZOS)

        glBindTexture(GL_TEXTURE_2D_ARRAY, self.tex_array)
        glTexSubImage3D(
            GL_TEXTURE_2D_ARRAY, 0,
            0, 0, tex_idx,
            self.tex_w, self.tex_h, 1,
            GL_RGBA, GL_UNSIGNED_BYTE, img.convert("RGBA").tobytes()
        )
        glBindTexture(GL_TEXTURE_2D_ARRAY, 0)

    def update(self):
        """Pulls transforms from dynamic instances, refits the TLAS, uploads to GPU."""

        self._pack_lights()

        if not self._dynamic_map or self._tlas is None:
            return

        updated = False

        for inst in self.scene._dynamic_instances:
            tlas_idx = self._dynamic_map.get(inst.id)
            if tlas_idx is None:
                continue

            transform, inv_transform = self._inst_transforms(inst)

            self._tlas.set_instance_transform(tlas_idx, transform)
            self.gpu_inst_info[tlas_idx]['transform'] = transform
            self.gpu_inst_info[tlas_idx]['inverse_transform'] = inv_transform
            updated = True

        if updated:
            self._tlas.refit_tlas()

            new_tlas_nodes = self._tlas.get_buffers()['nodes']

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.tlas_nodes_ssbo)
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, new_tlas_nodes.nbytes, new_tlas_nodes)

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, self.inst_info_ssbo)
            glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, self.gpu_inst_info.nbytes, self.gpu_inst_info)

            glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0)

    # Cleanup

    def free(self):
        for buf in (
            self.verts_ssbo, self.indices_ssbo, self.points_ssbo, self.materials_ssbo,
            self.tlas_nodes_ssbo, self.blas_nodes_ssbo, self.inst_info_ssbo,
            self.tlas_indices_ssbo, self.blas_indices_ssbo,
            self.dir_lights_ssbo, self.point_lights_ssbo, self.area_lights_ssbo,
        ):
            if buf:
                glDeleteBuffers(1, [buf])

        if self.tex_array:
            glDeleteTextures(1, [self.tex_array])


##
# Raytracer


class Raytracer(BaseRenderer):
    """
    Raytracer for compound eye rendering.
    """

    def __init__(
            self,
            receptor_array: 'ReceptorArray',
            scene: 'Scene',
            agent: 'Agent',
            time_dithering: bool = True,
            nb_samples: int = 256,
            quasi_random: bool = False,
            pano_res: Optional[Tuple[int, int]] = (1024, 512),
            batch_size: int = 1,
            enable_actuation: bool = False,
            enable_direct: bool = True,
            enable_shadows: bool = True,
            enable_ambient: bool = True,
        ):

        # Bake first so VRAM estimation is correct
        self.scene = scene
        self._baker = RaytraceBaker(scene)

        super().__init__(
            receptor_array=receptor_array,
            agent=agent,
            time_dithering=time_dithering,
            nb_samples=nb_samples,
            quasi_random=quasi_random,
            batch_size=batch_size,
            enable_actuation=enable_actuation
        )

        # Global lighting controls
        self.enable_direct = enable_direct
        self.enable_shadows = enable_shadows
        self.enable_ambient = enable_ambient
        self.ambient_intensity = 1.0
        self.sky_intensity = 1.0

        # Lazy resource handles
        self._active_defines: Set[str] = set()
        self._raytrace_shader: Optional[ShaderProgram] = None

        self._view_shaders: Dict[str, ShaderProgram] = {}
        self._view_textures: Dict[str, int] = {}
        self._pano_res = pano_res

        self._tex_viewer = TextureViewer()

    # Internal properties for lazy-loaded resources

    @property
    def raytrace_shader(self) -> ShaderProgram:
        self._ensure_defines()

        if self._raytrace_shader is None:
            self._raytrace_shader = ShaderProgram(
                comp_path='shaders/raytracing/ommatidiaRaytracing.comp',
                defines=self._active_defines
            )
        return self._raytrace_shader

    def _get_view_shader(self, view_name: str) -> ShaderProgram:
        self._ensure_defines()

        if view_name not in self._view_shaders:

            if view_name == 'panoramic':
                shader_path = 'shaders/raytracing/panoramicRaytracing.comp'
            elif view_name == 'perspective':
                shader_path = 'shaders/raytracing/perspectiveRaytracing.comp'

            self._view_shaders[view_name] = ShaderProgram(
                comp_path=shader_path,
                defines=self._active_defines
            )
        return self._view_shaders[view_name]

    def _get_view_texture(self, view_name: str) -> Tuple[int, Tuple[int, int]]:

        if view_name == 'panoramic':
            target_res = self._pano_res

        else:  # perspective
            viewport = glGetIntegerv(GL_VIEWPORT)
            target_res = (viewport[2], viewport[3])

        if view_name not in self._view_textures:
            self._view_textures[view_name] = _create_texture(*target_res, dtype=float)
            return self._view_textures[view_name], target_res

        return self._view_textures[view_name], target_res

    # Various internal helpers

    def _estim_vram_use(self) -> float:
        total_bytes = 0

        for buf in (
            self._baker.cpu_verts, self._baker.cpu_idx,
            self._baker.cpu_pts, self._baker.cpu_blas,
            self._baker.cpu_tlas_nodes, self._baker.cpu_tlas_idx,
            self._baker.cpu_blas_idx, self._baker.gpu_inst_info,
        ):
            if buf is not None:
                total_bytes += buf.nbytes

        total_bytes += getattr(self, '_total_samples', 0) * 16
        return total_bytes / (1024 * 1024)

    def _get_defines(self) -> Set[str]:
        """Determines which #defines to inject based on current baker light counts."""

        defines = set()

        for count, name in [
            (self._baker._nb_dir_lights, 'DIRECTIONAL'),
            (self._baker._nb_point_lights, 'POINT'),
            (self._baker._nb_area_lights, 'AREA')
        ]:
            if count >= 1:
                defines.add(f'HAS_{name}_LIGHT')

            if count > 1:
                defines.add(f'MULTI_{name}')

        return defines

    def _invalidate_shaders(self):
        """Invalidates all shaders that need be when defines change."""

        if self._raytrace_shader:
            self._raytrace_shader.free()

        for s in self._view_shaders.values():
            s.free()

        self._raytrace_shader = None
        self._view_shaders.clear()

    def _ensure_defines(self):

        new_defines = self._get_defines()

        if new_defines != self._active_defines:
            self._invalidate_shaders()
            self._active_defines = new_defines

    # Internal helpers for GL resource binding

    def _bind_scene_ssbos(self):
        """Bind scene geometry and light SSBOs."""

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_VERTICES, self._baker.verts_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_INDICES, self._baker.indices_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_MATERIALS, self._baker.materials_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_POINTS, self._baker.points_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_BLAS_NODES, self._baker.blas_nodes_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_TLAS_NODES, self._baker.tlas_nodes_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_INSTANCES, self._baker.inst_info_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_TLAS_INDICES, self._baker.tlas_indices_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_BLAS_INDICES, self._baker.blas_indices_ssbo)

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_LIGHT_DIR, self._baker.dir_lights_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_LIGHT_POINT, self._baker.point_lights_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_LIGHT_AREA, self._baker.area_lights_ssbo)

    def _bind_scene_textures(self, shader: ShaderProgram):
        """Bind skybox cubemap and material texture array."""

        if self._baker.scene.skybox:
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_CUBE_MAP, self._baker.skybox_tex)
            glUniform1i(shader.get_loc('skybox'), 0)

        glActiveTexture(GL_TEXTURE1)
        glBindTexture(GL_TEXTURE_2D_ARRAY, self._baker.tex_array)
        glUniform1i(shader.get_loc('scene_textures'), 1)

    def _set_scene_uniforms(self, shader: ShaderProgram):
        """Set uniforms shared by all ray-tracing compute shaders."""

        glUniform1ui(shader.get_loc('nb_tlas_nodes'), len(self._baker.cpu_tlas_nodes))

        glUniform1i(shader.get_loc('use_skybox'), int(self.scene.skybox is not None))

        r, g, b = self.scene.background_color
        glUniform3f(shader.get_loc('background_color'), r, g, b)
        glUniform1f(shader.get_loc('sky_intensity'), self.sky_intensity)

        glUniform1ui(shader.get_loc('dither_counter'), int(self._dither_counter))

        glUniform1i(shader.get_loc('enable_ambient'), int(self.enable_ambient))
        glUniform1i(shader.get_loc('enable_direct'), int(self.enable_direct))
        glUniform1i(shader.get_loc('enable_shadows'), int(self.enable_shadows))
        glUniform1f(shader.get_loc('ambient_intensity'), self.ambient_intensity)

        glUniform1i(shader.get_loc('directional_lights_count'), self._baker._nb_dir_lights)
        glUniform1i(shader.get_loc('point_lights_count'), self._baker._nb_point_lights)
        glUniform1i(shader.get_loc('area_lights_count'), self._baker._nb_area_lights)

    # Internal rendering logic and draw calls

    def _raytrace_thirdperson(self, view_name: str, pov: Union['Agent', 'OrbitCamera']):
        """Shared dispatch for panoramic / perspective ray-traced views."""

        shader = self._get_view_shader(view_name)
        shader.use()

        tex_id, res = self._get_view_texture(view_name)

        glBindImageTexture(0, tex_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F)

        self._bind_scene_ssbos()
        self._bind_scene_textures(shader)
        self._set_scene_uniforms(shader)

        glUniform1i(shader.get_loc('nb_samples'), self._samples_per_pixel)
        c2w = glm.inverse(pov.view)
        glUniformMatrix4fv(shader.get_loc('cam_to_world'), 1, False, glm.value_ptr(c2w))

        if view_name == 'perspective':
            inv_proj = glm.inverse(self.agent.projection)
            glUniformMatrix4fv(shader.get_loc('inv_projection'), 1, False, glm.value_ptr(inv_proj))

        glDispatchCompute((res[0] + 15) // 16, (res[1] + 15) // 16, 1)
        glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)
        shader.stop()

    def _raytrace_receptors(self):
        """Pass 1: ray-trace each receptor."""

        shader = self.raytrace_shader
        shader.use()

        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_RCPT_STATIC, self._rcpt_static_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_LENS_STATIC, self._lens_static_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_RCPT_DYNAMIC, self._rcpt_dynamic_ssbo)
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, BINDING_RAYS_INTERMEDIATE, self.sampling_results_ssbo)

        self._bind_scene_ssbos()
        self._bind_scene_textures(shader)
        self._set_scene_uniforms(shader)

        glUniform1i(shader.get_loc('nb_samples'), self.nb_samples)
        glUniform1i(shader.get_loc('use_quasi_random'), int(self._quasi_random))

        c2w_mat = glm.inverse(self.agent.view)
        glUniformMatrix4fv(shader.get_loc('cam_to_world'), 1, False, glm.value_ptr(c2w_mat))

        work_groups = (len(self._ra) * self._nb_samples + 63) // 64
        glDispatchCompute(work_groups, 1, 1)
        glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)
        shader.stop()

    def _sample_scene(self):
        self._baker.update()
        self._raytrace_receptors()

    # Main public methods

    def draw(self, view_mode: 'DisplayMode', point_of_view: Union['Agent', 'OrbitCamera']):

        if view_mode == DisplayMode.Compound:
            self._draw_eye_firstperson()

        elif view_mode in (DisplayMode.Panoramic, DisplayMode.Perspective, DisplayMode.Third_person):
            view_name = 'panoramic' if view_mode == DisplayMode.Panoramic else 'perspective'

            self._baker.update()
            self._raytrace_thirdperson(view_name, point_of_view)

            tex_id, _ = self._get_view_texture(view_name)
            self._tex_viewer.draw(tex_id, self.simulate_insect_vision, self.uv_encoded_textures)

        if view_mode == DisplayMode.Third_person:
            self._draw_eye_thirdperson(point_of_view)

    # Public properties and methods

    def update_texture(self, asset: 'Asset'):
        """Update a texture on the GPU for given Asset."""

        self._baker.update_texture(asset)

    # Cleanup

    def free(self):
        self._invalidate_shaders()

        if self.reduction_shader:
            self.reduction_shader.free()

        tex_ids = list(self._view_textures.values())
        if tex_ids:
            glDeleteTextures(len(tex_ids), tex_ids)

        self._tex_viewer.free()
        self._baker.free()
        super().free()


##
# Pathtracer


class Pathtracer(Raytracer):
    """
    Path tracer: multiple bounces with Monte Carlo integration.
    """

    def __init__(self,
            receptor_array: 'ReceptorArray',
            scene: 'Scene',
            agent: 'Agent',
            time_dithering: bool = True,
            nb_samples: int = 256,
            quasi_random: bool = False,
            pano_res: Tuple[int, int] = (1024, 512),
            batch_size: int = 1,
            enable_actuation: bool = False,
            enable_shadows: bool = True,
            enable_ambient: bool = True,
            enable_direct: bool = True,
            max_bounces: int = 3,
        ):

        self.max_bounces = max_bounces

        super().__init__(
            receptor_array=receptor_array,
            scene=scene,
            agent=agent,
            time_dithering=time_dithering,
            nb_samples=nb_samples,
            quasi_random=quasi_random,
            pano_res=pano_res,
            batch_size=batch_size,
            enable_actuation=enable_actuation,
            enable_shadows=enable_shadows,
            enable_ambient=enable_ambient,
            enable_direct=enable_direct
        )

    def _get_defines(self) -> Set[str]:
        defines = super()._get_defines()
        defines.add('PATH_TRACING')
        return defines

    def _set_scene_uniforms(self, shader: ShaderProgram):
        super()._set_scene_uniforms(shader)
        glUniform1i(shader.get_loc('max_bounces'), self.max_bounces)