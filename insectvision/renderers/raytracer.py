import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *

from typing import TYPE_CHECKING, Tuple, List, Dict, Optional, Union, Any
import numpy as np
from PIL import Image
from pyglm import glm
from pytinybvh import BVH, instance_dtype, Layout, supports_layout

from insectvision.utils import DisplayMode, RandomnessMode, SamplingMode
from insectvision.engine.agent import Agent
from insectvision.engine.scene import Scene, AssetType, Asset
from insectvision.engine.lights import DIR_LIGHT_DTYPE, POINT_LIGHT_DTYPE, AREA_LIGHT_DTYPE
from insectvision.engine.resources import (
    write_pytinybvh_preamble, ShaderProgram, GPUResourceManager,
    BufferRegistry, TextureRegistry, UniformRegistry
)
from insectvision.renderers.commons import BaseRenderer

if TYPE_CHECKING:
    from insectvision.compound_eyes import ReceptorArray
    from insectvision.engine.context import Context

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


class RaytraceBaker:
    """
    Manages BVH structures and GPU buffers for a Scene.
    """

    def __init__(self, scene: Scene, resource_manager: GPUResourceManager):
        self.scene = scene

        self._tlas: Optional[BVH] = None
        self._blases: List[BVH] = []
        self._dynamic_map: Dict[int, int] = {}

        self.resource_manager = resource_manager
        self.bvh_buffers = BufferRegistry(self.resource_manager)
        self.light_buffers = BufferRegistry(self.resource_manager)
        self.scene_textures = TextureRegistry(self.resource_manager)

        self._nb_dir_lights = 0
        self._nb_point_lights = 0
        self._nb_area_lights = 0

        # CPU-side data
        self.gpu_inst_info: Optional[np.ndarray] = None
        self._material_map: Dict[int, int] = {}
        self._asset_blas_map: Dict[int, Dict] = {}
        self._asset_tex_map = {}

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

        if self.scene.skybox:
            self.scene_textures.register_existing('skybox', self.scene.skybox.texture_id, GL_TEXTURE_CUBE_MAP)

    # Main packing methods

    def _pack_lights(self):

        dir_l = [l for l in self.scene.directional_lights if l.active]
        point_l = [l for l in self.scene.point_lights if l.active]
        area_l = [l for l in self.scene.area_lights if l.active]

        self._nb_dir_lights = len(dir_l)
        self._nb_point_lights = len(point_l)
        self._nb_area_lights = len(area_l)

        def _pack_or_update(name, lights, dtype):
            data = np.concatenate([l.pack() for l in lights]) if lights else np.zeros(1, dtype=dtype)
            if name in self.light_buffers:
                self.light_buffers[name].resize(len(data), data=data)
            else:
                self.light_buffers.allocate(name,
                                            dtype=dtype,
                                            count=len(data),
                                            data=data,
                                            usage=GL_DYNAMIC_DRAW)

        _pack_or_update('dir', dir_l, DIR_LIGHT_DTYPE)
        _pack_or_update('point', point_l, POINT_LIGHT_DTYPE)
        _pack_or_update('area', area_l, AREA_LIGHT_DTYPE)

    def _pack_materials(self):
        """Packs material data for all mesh assets into GPU buffers."""

        mesh_assets = {inst.asset for inst in self.scene.mesh_instances}
        if not mesh_assets:
            return

        self._material_map = {asset.id: i for i, asset in enumerate(mesh_assets)}

        texture_images = []

        for asset in mesh_assets:
            if asset.has_texture and asset.texture_image is not None:
                self._asset_tex_map[asset.id] = len(texture_images)
                texture_images.append(asset.texture_image)
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

                # Create a temporary standalone texture to push in the array
                temp_tex = self.scene_textures.allocate_2d('temp',
                                                           self.tex_w,
                                                           self.tex_h,
                                                           image_data=img.convert('RGBA').tobytes(),
                                                           repeat=True,
                                                           dtype=int)
                tex_ids.append(temp_tex.handle)

            self.scene_textures.allocate_array('materials', tex_ids)

            # remove the temporary IDs now that they've been copied to the array
            glDeleteTextures(len(tex_ids), tex_ids)
            del self.scene_textures._textures['temp']

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

        if len(mesh_assets) > 0:
            self.bvh_buffers.allocate('materials',
                                      dtype=np.uint32,
                                      count=mat_data.size,
                                      data=mat_data,
                                      usage=GL_STATIC_DRAW)

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

                self._asset_blas_map[asset.id] = {'id': blas_id, 'v_off': v_off, 'idx_off': idx_off, 'is_points': 0}
                v_off += len(asset.vertices)
                idx_off += len(asset.indices.flatten())

            elif asset.asset_type == AssetType.Points:

                points = asset.points.astype(np.float32)
                radii = asset.radii.astype(np.float32)

                blas = BVH.from_points(points,
                                       radius=radii,
                                       traversal_cost=1.0,
                                       intersection_cost=1.0)

                bundle = blas.get_SSBO_bundle(flatten_nodes=False)

                nb_points = len(asset.points)
                packed_points = np.zeros((nb_points, 12), dtype=np.float32)
                packed_points[:, 0:3] = asset.points
                packed_points[:, 3] = asset.radii
                packed_points[:, 4:7] = asset.normals
                packed_points[:, 7:10] = asset.colors
                all_pts.append(packed_points)

                self._asset_blas_map[asset.id] = {'id': blas_id, 'pt_off': pt_off}
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

        def _data_or_default(data, dtype, min_elems=1):
            if data is None or getattr(data, 'nbytes', 0) == 0:
                return np.zeros((min_elems,), dtype=dtype)
            return data

        verts = _data_or_default(data=self.cpu_verts, dtype=np.float32, min_elems=5)
        self.bvh_buffers.allocate('verts',
                                  dtype=np.float32,
                                  count=verts.size,
                                  data=verts)

        indices = _data_or_default(data=self.cpu_idx, dtype=np.uint32, min_elems=3)
        self.bvh_buffers.allocate('indices',
                                  dtype=np.uint32,
                                  count=indices.size,
                                  data=indices)

        points = _data_or_default(data=self.cpu_pts, dtype=np.float32, min_elems=12)
        self.bvh_buffers.allocate('points',
                                  dtype=np.float32,
                                  count=points.size,
                                  data=points)

        blas_nodes = _data_or_default(data=self.cpu_blas, dtype=np.float32, min_elems=1)
        self.bvh_buffers.allocate('blas_nodes',
                                  dtype=np.float32,
                                  count=blas_nodes.size,
                                  data=blas_nodes)

        tlas_nodes = _data_or_default(data=self.cpu_tlas_nodes, dtype=np.float32, min_elems=1)
        self.bvh_buffers.allocate('tlas_nodes',
                                  dtype=np.float32,
                                  count=tlas_nodes.size,
                                  data=tlas_nodes,
                                  usage=GL_DYNAMIC_DRAW)

        tlas_indices = _data_or_default(data=self.cpu_tlas_idx, dtype=np.uint32, min_elems=1)
        self.bvh_buffers.allocate('tlas_indices',
                                  dtype=np.uint32,
                                  count=tlas_indices.size,
                                  data=tlas_indices)

        blas_indices = _data_or_default(data=self.cpu_blas_idx, dtype=np.uint32, min_elems=1)
        self.bvh_buffers.allocate('blas_indices',
                                  dtype=np.uint32,
                                  count=blas_indices.size,
                                  data=blas_indices)

        inst_info = _data_or_default(data=self.gpu_inst_info, dtype=RENDERABLE_INST_DTYPE, min_elems=1)
        self.bvh_buffers.allocate('inst_info',
                                  dtype=RENDERABLE_INST_DTYPE,
                                  count=inst_info.size,
                                  data=inst_info,
                                  usage=GL_DYNAMIC_DRAW)

    # Dynamic updates

    def update_texture(self, asset: 'Asset'):
        """Update a texture on the GPU for a given Asset."""

        if 'materials' not in self.scene_textures:
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

        glBindTexture(GL_TEXTURE_2D_ARRAY, self.scene_textures['materials'].handle)
        glTexSubImage3D(
            GL_TEXTURE_2D_ARRAY, 0, 0, 0, tex_idx,
            self.tex_w, self.tex_h, 1, GL_RGBA, GL_UNSIGNED_BYTE, img.convert("RGBA").tobytes()
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

            self.bvh_buffers['tlas_nodes'].write(new_tlas_nodes)
            self.bvh_buffers['inst_info'].write(self.gpu_inst_info)

    # Cleanup

    def free(self):
        self.bvh_buffers.free()
        self.light_buffers.free()
        self.scene_textures.free()


##
# Raytracer


class Raytracer(BaseRenderer):
    """
    Raytracer for compound eye rendering.
    """

    def __init__(
            self,
            model: 'ReceptorArray',
            scene: 'Scene',
            agent: 'Agent',
            time_dithering: bool = True,
            nb_samples: int = 256,
            randomness_mode: Union[int, str, RandomnessMode] = RandomnessMode.Pseudo,
            sampling_mode: Union[int, str, SamplingMode] = SamplingMode.Gaussian,
            pano_res: Optional[Tuple[int, int]] = (1024, 512),
            batch_size: int = 1,
            enable_actuation: bool = False,
            enable_direct: bool = True,
            enable_shadows: bool = True,
            enable_ambient: bool = True,
            context: Optional['Context'] = None,
        ):

        # Bake first so VRAM estimation is correct
        self.scene = scene
        self.resource_manager = GPUResourceManager()
        self._baker = RaytraceBaker(scene, self.resource_manager)

        super().__init__(
            model=model,
            agent=agent,
            time_dithering=time_dithering,
            nb_samples=nb_samples,
            randomness_mode=randomness_mode,
            sampling_mode=sampling_mode,
            batch_size=batch_size,
            enable_actuation=enable_actuation,
            resource_manager=self.resource_manager,
            context=context,
        )

        # Global lighting controls
        self.enable_direct = enable_direct
        self.enable_shadows = enable_shadows
        self.enable_ambient = enable_ambient
        self.ambient_intensity = 1.0
        self.sky_intensity = 1.0

        # Lazy resource handles
        self._active_defines: Dict[str, Any] = {}
        self._raytrace_shader: Optional[ShaderProgram] = None

        self._view_shaders: Dict[str, ShaderProgram] = {}
        self.view_textures = TextureRegistry(self.resource_manager)
        self._pano_res = pano_res

        self._scene_uniforms = UniformRegistry(
            nb_tlas_nodes=len(self._baker.cpu_tlas_nodes),

            use_skybox=self.scene.skybox is not None,
            background_color=self.scene.background_color
        )

        if 'skybox' in self._baker.scene_textures:
            self._scene_uniforms.update(skybox=self._baker.scene_textures['skybox'].unit)

        if 'materials' in self._baker.scene_textures:
            self._scene_uniforms.update(scene_textures=self._baker.scene_textures['materials'].unit)

        self._lights_uniforms = UniformRegistry(

            enable_ambient=self.enable_ambient,
            enable_direct=self.enable_direct,
            enable_shadows=self.enable_shadows,

            sky_intensity=self.sky_intensity,
            ambient_intensity=self.ambient_intensity,

            directional_lights_count=self._baker._nb_dir_lights,
            point_lights_count=self._baker._nb_point_lights,
            area_lights_count=self._baker._nb_area_lights
        )

    # Internal properties for lazy-loaded resources

    @property
    def raytrace_shader(self) -> ShaderProgram:
        self._ensure_defines()

        if self._raytrace_shader is None:
            shader_path = 'shaders/raytracing/ommatidiaRaytracing.comp'

            self._raytrace_shader = ShaderProgram(comp_path=shader_path,
                                                  defines=self._active_defines)
        return self._raytrace_shader

    def _get_view_shader(self, view_name: str) -> ShaderProgram:
        self._ensure_defines()

        if view_name not in self._view_shaders:

            if view_name == 'panoramic':
                shader_path = 'shaders/raytracing/panoramicRaytracing.comp'
            elif view_name == 'perspective':
                shader_path = 'shaders/raytracing/perspectiveRaytracing.comp'

            self._view_shaders[view_name] = ShaderProgram(comp_path=shader_path,
                                                          defines=self._active_defines)
        return self._view_shaders[view_name]

    def _get_view_texture(self, view_name: str) -> Tuple[int, Tuple[int, int]]:

        if view_name == 'panoramic':
            target_res = self._pano_res

        else:  # perspective
            viewport = glGetIntegerv(GL_VIEWPORT)
            target_res = (viewport[2], viewport[3])

        if view_name not in self.view_textures:
            self.view_textures.allocate_2d(view_name, target_res[0], target_res[1], dtype=float)

        return self.view_textures[view_name].handle, target_res

    def _estim_vram_use(self) -> float:

        total_bytes = sum(buf.nbytes for buf in
            (self._baker.cpu_verts, self._baker.cpu_idx, self._baker.cpu_pts, self._baker.cpu_blas,
            self._baker.cpu_tlas_nodes, self._baker.cpu_tlas_idx, self._baker.cpu_blas_idx,
            self._baker.gpu_inst_info)
                          if buf is not None)
        total_bytes += getattr(self, '_total_samples', 0) * 16

        return total_bytes / (1024 * 1024)

    def _get_defines(self) -> Dict[str, Any]:
        """Determines which #defines to inject based on current baker light counts."""

        defines = {}
        defines.update(self.eye_buffers.shader_defines)
        defines.update(self._baker.bvh_buffers.shader_defines)
        defines.update(self._baker.light_buffers.shader_defines)

        for count, name in [
            (self._baker._nb_dir_lights, 'DIRECTIONAL'),
            (self._baker._nb_point_lights, 'POINT'),
            (self._baker._nb_area_lights, 'AREA')
        ]:
            if count >= 1:
                defines[f'HAS_{name}_LIGHT'] = 1
            if count > 1:
                defines[f'MULTI_{name}'] = 1

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

    def _update_uniforms(self):

        self._lights_uniforms.update(
            sky_intensity=self.sky_intensity,
            ambient_intensity=self.ambient_intensity,
            # directional_lights_count=self._baker._nb_dir_lights,
            # point_lights_count=self._baker._nb_point_lights,
            # area_lights_count=self._baker._nb_area_lights
        )

        # if 'materials' in self._baker.scene_textures:
        #     self._scene_uniforms.update(scene_textures=self._baker.scene_textures['materials'].unit)

        super()._update_uniforms()

    # Internal rendering logic and draw calls

    def _raytrace_thirdperson(self, view_name: str, pov: Union['Agent', 'OrbitCamera']):
        """Shared dispatch for panoramic / perspective ray-traced views."""

        tex_id, res = self._get_view_texture(view_name)

        bvh = self._baker.bvh_buffers
        lights = self._baker.light_buffers

        with self._get_view_shader(view_name) as shader:
            glBindImageTexture(0, tex_id, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_RGBA32F)

            with bvh.grouped_bind(), lights.grouped_bind():
                with self._baker.scene_textures.bind_all():

                    # Override number of samples for third person
                    self._eye_uniforms.update(nb_samples=self._samples_per_px)

                    self._eye_uniforms.update(cam_to_world=glm.inverse(pov.view))

                    if view_name == 'perspective':
                        self._eye_uniforms.update(inv_projection=glm.inverse(self.agent.projection))

                    self._eye_uniforms.apply(shader)
                    self._scene_uniforms.apply(shader)
                    self._lights_uniforms.apply(shader)

                    glDispatchCompute((res[0] + 15) // 16, (res[1] + 15) // 16, 1)
                    glMemoryBarrier(GL_SHADER_IMAGE_ACCESS_BARRIER_BIT)

    def _raytrace_receptors(self):

        eye = self.eye_buffers
        bvh = self._baker.bvh_buffers
        lights = self._baker.light_buffers

        with self.raytrace_shader as shader:
            with bvh.grouped_bind(), lights.grouped_bind(), eye.grouped_bind(['rays_intermediate', 'rcpt_static', 'lens_static', 'rcpt_dynamic']):
                with self._baker.scene_textures.bind_all():

                    self._eye_uniforms.update(cam_to_world=glm.inverse(self.agent.view))

                    self._eye_uniforms.apply(shader)
                    self._scene_uniforms.apply(shader)
                    self._lights_uniforms.apply(shader)

                    N = self._model.total_receptors
                    total_work = N * self._samples_per_rcpt
                    work_groups = (total_work + 63) // 64

                    glDispatchCompute(work_groups, 1, 1)
                    glMemoryBarrier(GL_SHADER_STORAGE_BARRIER_BIT)

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
            self.screen_surface.draw(
                tex_id,
                is_cubemap=False,
                simulate_insect_vision=self.simulate_insect_vision,
                uv_encoded_textures=self.uv_encoded_textures
            )

        if view_mode == DisplayMode.Third_person:
            self._draw_eye_thirdperson(point_of_view)

    # Public properties and methods

    def update_texture(self, asset: 'Asset'):
        """Update a texture on the GPU for given Asset."""
        self._baker.update_texture(asset)

    # Cleanup

    def free(self):
        self._invalidate_shaders()
        if self.reduction_shader: self.reduction_shader.free()
        self.view_textures.free()
        if self._screen_surface: self._screen_surface.free()
        self._baker.free()
        super().free()

    @property
    def TLAS(self) -> Optional[BVH]:
        return self._baker._tlas

    @property
    def BLASes(self) -> List[BVH]:
        return self._baker._blases


##
# Pathtracer


class Pathtracer(Raytracer):
    """
    Path tracer: multiple bounces with Monte Carlo integration.
    """

    def __init__(self,
                 model: 'ReceptorArray',
                 scene: 'Scene',
                 agent: 'Agent',
                 time_dithering: bool = True,
                 nb_samples: int = 256,
                 randomness_mode: Union[int, str, RandomnessMode] = RandomnessMode.Pseudo,
                 sampling_mode: Union[int, str, SamplingMode] = SamplingMode.Gaussian,
                 pano_res: Tuple[int, int] = (1024, 512),
                 batch_size: int = 1,
                 enable_actuation: bool = False,
                 enable_shadows: bool = True,
                 enable_ambient: bool = True,
                 enable_direct: bool = True,
                 max_bounces: int = 3,
                 context: Optional['Context'] = None
                 ):

        self.max_bounces = max_bounces

        super().__init__(
            model=model,
            scene=scene,
            agent=agent,
            context=context,
            time_dithering=time_dithering,
            nb_samples=nb_samples,
            randomness_mode=randomness_mode,
            sampling_mode=sampling_mode,
            pano_res=pano_res,
            batch_size=batch_size,
            enable_actuation=enable_actuation,
            enable_shadows=enable_shadows,
            enable_ambient=enable_ambient,
            enable_direct=enable_direct
        )

        self._scene_uniforms.update(max_bounces=self.max_bounces)
        # TODO: Would be nice to control this during runtime

    def _get_defines(self) -> Dict[str, Any]:
        defines = super()._get_defines()
        defines['PATH_TRACING'] = 1
        return defines