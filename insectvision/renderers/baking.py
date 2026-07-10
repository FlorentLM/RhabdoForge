import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *

from typing import TYPE_CHECKING, List, Dict, Optional
import numpy as np
from PIL import Image
from pyglm import glm
from pytinybvh import BVH, instance_dtype, Layout, supports_layout

from insectvision.types import RENDERABLE_DTYPE, DIR_LIGHT_DTYPE, POINT_LIGHT_DTYPE, AREA_LIGHT_DTYPE, AssetType
from insectvision.engine.scene import MeshAsset, PointsAsset
from insectvision.engine.resources import write_pytinybvh_preamble, GPUResourceManager, BufferRegistry, TextureRegistry

if TYPE_CHECKING:
    from insectvision.engine.scene import Scene


class SceneBaker:
    """
    Manages BVH structures and GPU buffers for a Scene.
    """

    def __init__(self, scene: 'Scene', resource_manager: 'GPUResourceManager'):
        self.scene = scene

        self._tlas: Optional['BVH'] = None
        self._blases: List['BVH'] = []
        self._dynamic_map: Dict[int, int] = {}

        self.resource_manager: 'GPUResourceManager' = resource_manager

        self.bvh_buffers: 'BufferRegistry' = BufferRegistry(self.resource_manager)
        self.light_buffers: 'BufferRegistry' = BufferRegistry(self.resource_manager)
        self.scene_textures: 'TextureRegistry' = TextureRegistry(self.resource_manager)

        self._nb_dir_lights: int = 0
        self._nb_point_lights: int = 0
        self._nb_area_lights: int = 0

        # CPU-side data
        self.gpu_inst_info: Optional[np.ndarray] = None
        self._material_map: Dict[int, int] = {}
        self._asset_blas_map: Dict[int, Dict] = {}
        self._asset_tex_map = {}

        if not self.scene.instances:
            print('Warning: Scene is empty, nothing to bake.')
            return

        # Pack scene materials and lights
        self._pack_materials()
        self._pack_lights()

        # Pack geometry into BVH
        self._build_blases()
        self._build_tlas()

        # Upload everything to GPU
        self._push_to_gpu()

        if self.scene.sky:
            self.scene_textures.register_existing('sky_texture', self.scene.sky.texture_id, GL_TEXTURE_2D)

    def __repr__(self):
        return (f"<SceneBaker | {len(self._blases)} BLAS"
                f" | lights {self._nb_dir_lights}D/{self._nb_point_lights}P/{self._nb_area_lights}A"
                f" | TLAS {'ready' if self._tlas is not None else 'empty'}>")

    # Main packing methods

    def _pack_lights(self):

        self._dir_order = [l for l in self.scene.directional_lights if l.active]
        self._point_order = [l for l in self.scene.point_lights if l.active]
        self._area_order = [l for l in self.scene.area_lights if l.active]

        self._nb_dir_lights = len(self._dir_order)
        self._nb_point_lights = len(self._point_order)
        self._nb_area_lights = len(self._area_order)

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

        _pack_or_update('dir', self._dir_order, DIR_LIGHT_DTYPE)
        _pack_or_update('point', self._point_order, POINT_LIGHT_DTYPE)
        _pack_or_update('area', self._area_order, AREA_LIGHT_DTYPE)

        self._light_last_rev = {
            id(l): l.revision for l in (*self._dir_order, *self._point_order, *self._area_order)
        }

    @staticmethod
    def pack_rgba8(rgba):
        # Clamps and packs 4 floats into one 32-bit uint
        r, g, b, a = [int(np.clip(x, 0, 1) * 255) & 0xFF for x in rgba]
        return (a << 24) | (b << 16) | (g << 8) | r

    def _pack_material_row(self, asset) -> np.ndarray:
        row = np.zeros(4, dtype=np.uint32)

        tex_idx = self._asset_tex_map.get(asset.id)
        row[0] = tex_idx if tex_idx is not None else 0xFFFFFFFF

        row[1] = self.pack_rgba8(asset.material.base_color)

        return row

    def _prepare_texture_array(self, mesh_assets):
        """Gathers images from assets and creates the GL_TEXTURE_2D_ARRAY."""

        texture_images = []

        # Identify which assets need a slot in the array
        for asset in mesh_assets:
            if asset.has_texture and asset.texture_image is not None:
                self._asset_tex_map[asset.id] = len(texture_images)
                texture_images.append(asset.texture_image)
            else:
                self._asset_tex_map[asset.id] = None

        if not texture_images:
            return

        # Determine dimensions (based on first texture)
        # TODO: This is kinda crap
        self.tex_w, self.tex_h = texture_images[0].size
        tex_ids = []

        # Create temporary textures for each
        for img in texture_images:
            if img.size != (self.tex_w, self.tex_h):
                img = img.resize((self.tex_w, self.tex_h), Image.Resampling.LANCZOS)

            # Upload to a temporary handle to allow the TextureRegistry to 'array' them
            temp_tex = self.scene_textures.allocate_2d(
                'temp', self.tex_w, self.tex_h,
                image_data=img.convert('RGBA').tobytes(),
                repeat=True, dtype=int
            )
            tex_ids.append(temp_tex.handle)

        # Collapse into final array
        self.scene_textures.allocate_array('materials', tex_ids)

        # Cleanup temporary handles
        glDeleteTextures(len(tex_ids), tex_ids)
        if 'temp' in self.scene_textures._textures:
            del self.scene_textures._textures['temp']

    def _pack_materials(self):
        """Initial material data packing for all mesh assets into GPU buffers."""

        mesh_assets = {inst.asset for inst in self.scene.mesh_instances}

        # allocate a dummy buffer when there are no meshes
        if not mesh_assets:
            self._material_assets = []
            dummy_data = np.zeros((1, 4), dtype=np.uint32)
            self.bvh_buffers.allocate('materials',
                                      dtype=np.uint32,
                                      count=dummy_data.size,
                                      data=dummy_data,
                                      usage=GL_STATIC_DRAW)
            return

        self._material_assets = list(mesh_assets)
        self._mat_last_rev = {a.id: a.material_revision for a in mesh_assets}
        self._tex_last_rev = {a.id: a.texture_revision for a in mesh_assets}
        self._material_map = {a.id: i for i, a in enumerate(mesh_assets)}

        self._prepare_texture_array(mesh_assets)        # populates self._asset_tex_map

        mat_data = np.zeros((len(mesh_assets), 4), dtype=np.uint32)

        for asset in mesh_assets:
            idx = self._material_map[asset.id]
            mat_data[idx] = self._pack_material_row(asset)

        self.bvh_buffers.allocate('materials',
                                  dtype=np.uint32,
                                  count=mat_data.size,
                                  data=mat_data,
                                  usage=GL_STATIC_DRAW)
    # BVH construction

    def _build_blases(self):

        all_verts, all_idxs, all_pts, all_nodes = [], [], [], []
        v_off, idx_off, pt_off, n_off = 0, 0, 0, 0

        self._blas_leaf_chunks = []
        l_off = 0

        for asset in self.scene.assets.values():

            if asset.id in self._asset_blas_map:
                continue

            blas_id = len(self._blases)
            bundle = None

            if isinstance(asset, MeshAsset):
                blas = BVH.from_indexed_mesh(asset.vertices4, asset.indices)

                all_verts.append(asset.shading_vertices())
                all_idxs.append(asset.indices.flatten())

                self._asset_blas_map[asset.id] = {'id': blas_id, 'v_off': v_off, 'idx_off': idx_off, 'is_points': 0}

                v_off += len(asset.vertices4)
                idx_off += len(asset.indices.flatten())

            elif isinstance(asset, PointsAsset):
                blas = BVH.from_points(asset.points, radius=asset.radii)

                bundle = blas.get_SSBO_bundle(flatten_nodes=False)

                all_pts.append(asset.packed_points())

                self._asset_blas_map[asset.id] = {'id': blas_id, 'pt_off': pt_off, 'is_points': 1}

                pt_off += asset.nb_points

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

        self._geom_last_rev = {a.id: a.geometry_revision for a in self.scene.assets.values()}

        self.cpu_verts = np.concatenate(all_verts).ravel() if all_verts else None
        self.cpu_idx = np.concatenate(all_idxs).ravel() if all_idxs else None
        self.cpu_pts = np.concatenate(all_pts).ravel() if all_pts else None
        self.cpu_blas_nodes = np.concatenate(all_nodes).astype(np.float32) if all_nodes else None

    def _build_tlas(self):

        if not self._blases:
            return

        all_instances = self.scene.instances
        num_instances = len(all_instances)

        tlas_build_data = np.zeros(num_instances, dtype=instance_dtype)
        self.gpu_inst_info = np.zeros(num_instances, dtype=RENDERABLE_DTYPE)

        for i, inst in enumerate(all_instances):
            blas_map = self._asset_blas_map[inst.asset.id]

            transform = np.asarray(inst.transform, dtype=np.float32)
            inv_transform = np.asarray(glm.inverse(inst.transform), dtype=np.float32)

            tlas_build_data[i]['transform'] = transform
            tlas_build_data[i]['blas_id'] = blas_map['id']
            tlas_build_data[i]['mask'] = 0xFFFFFFFF if inst.visible else 0x00000000

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

            self.gpu_inst_info[i]['is_srgb'] = inst.asset.is_srgb

            if inst.dynamic:
                self._dynamic_map[inst.id] = i

        self._dynamic_last_rev = {inst.id: inst.transform_revision for inst in all_instances if inst.dynamic}
        self._vis_last_rev = {inst.id: inst.visibility_revision for inst in all_instances}
        self.cpu_inst_visible = np.array([1 if inst.visible else 0 for inst in all_instances], dtype=np.uint32)
        self._inst_row = {inst.id: i for i, inst in enumerate(all_instances)}  # any instance -> TLAS row

        self._tlas = BVH.build_tlas(tlas_build_data, self._blases)

        ssbo_dat = self._tlas.get_SSBO_bundle(flatten_nodes=False)

        self.cpu_tlas_nodes = ssbo_dat['nodes'].astype(np.float32)
        self.cpu_tlas_indices = ssbo_dat['leaf_ids'].astype(np.uint32)

        write_pytinybvh_preamble(str(ssbo_dat.get('preamble', '')))

        self.cpu_blas_indices = np.concatenate(self._blas_leaf_chunks).astype(np.uint32)

    def _push_to_gpu(self):

        v_count = self.cpu_verts.size if self.cpu_verts is not None else 0
        i_count = self.cpu_idx.size if self.cpu_idx is not None else 0
        p_count = self.cpu_pts.size if self.cpu_pts is not None else 0
        bn_count = self.cpu_blas_nodes.size if self.cpu_blas_nodes is not None else 0
        bi_count = self.cpu_blas_indices.size if self.cpu_blas_indices is not None else 0
        tn_count = self.cpu_tlas_nodes.size if self.cpu_tlas_nodes is not None else 0
        ti_count = self.cpu_tlas_indices.size if self.cpu_tlas_indices is not None else 0
        ii_count = self.gpu_inst_info.size if self.gpu_inst_info is not None else 0
        iv_count = self.cpu_inst_visible.size if self.cpu_inst_visible is not None else 0

        self.bvh_buffers.allocate('verts', dtype=np.float32, count=v_count, data=self.cpu_verts, min_count=5)
        self.bvh_buffers.allocate('indices', dtype=np.uint32, count=i_count, data=self.cpu_idx, min_count=3)
        self.bvh_buffers.allocate('points', dtype=np.float32, count=p_count, data=self.cpu_pts, min_count=12)
        self.bvh_buffers.allocate('blas_nodes', dtype=np.float32, count=bn_count, data=self.cpu_blas_nodes, min_count=1)
        self.bvh_buffers.allocate('blas_indices', dtype=np.uint32, count=bi_count, data=self.cpu_blas_indices, min_count=1)
        self.bvh_buffers.allocate('tlas_nodes', dtype=np.float32, count=tn_count, data=self.cpu_tlas_nodes, min_count=1, usage=GL_DYNAMIC_DRAW)
        self.bvh_buffers.allocate('tlas_indices', dtype=np.uint32, count=ti_count, data=self.cpu_tlas_indices, min_count=1)
        self.bvh_buffers.allocate('inst_info', dtype=RENDERABLE_DTYPE, count=ii_count, data=self.gpu_inst_info, min_count=1, usage=GL_DYNAMIC_DRAW)
        self.bvh_buffers.allocate('inst_visible', dtype=np.uint32, count=iv_count, data= self.cpu_inst_visible, min_count=1, usage=GL_DYNAMIC_DRAW)

    # Dynamic updates

    def _sync_lights(self):

        new_dir = [l for l in self.scene.directional_lights if l.active]
        new_point = [l for l in self.scene.point_lights if l.active]
        new_area = [l for l in self.scene.area_lights if l.active]

        if (len(new_dir), len(new_point), len(new_area)) != \
                (self._nb_dir_lights, self._nb_point_lights, self._nb_area_lights):

            self._pack_lights()  # resize + re-pin order + re-snapshot revisions
            return  # shader recompile happens via _ensure_defines (counts changed)

        self._dir_order, self._point_order, self._area_order = new_dir, new_point, new_area

        for name, order in (('dir', self._dir_order),
                            ('point', self._point_order),
                            ('area', self._area_order)):

            for row, light in enumerate(order):
                if light.revision == self._light_last_rev.get(id(light)):
                    continue

                self.light_buffers[name].write(light.pack(), start=row)
                self._light_last_rev[id(light)] = light.revision

    def _sync_materials(self):

        if 'materials' not in self.bvh_buffers:
            return

        for asset in self._material_assets:
            if asset.material_revision == self._mat_last_rev.get(asset.id):
                continue

            self.bvh_buffers['materials'].write(self._pack_material_row(asset),
                                                start=self._material_map[asset.id] * 4)
            self._mat_last_rev[asset.id] = asset.material_revision

    def _sync_textures(self):

        if 'materials' not in self.scene_textures:
            return

        for asset in self._material_assets:
            if asset.texture_revision == self._tex_last_rev.get(asset.id):
                continue
            self._tex_last_rev[asset.id] = asset.texture_revision

            tex_idx = self._asset_tex_map.get(asset.id)
            if tex_idx is None:  # had no texture at bake but promotion needs an array realloc (TODO in future)
                continue
            img = asset.texture_image
            if img is None:
                continue

            if img.size != (self.tex_w, self.tex_h):
                img = img.resize((self.tex_w, self.tex_h), Image.Resampling.LANCZOS)

            glBindTexture(GL_TEXTURE_2D_ARRAY, self.scene_textures['materials'].handle)
            glTexSubImage3D(GL_TEXTURE_2D_ARRAY, 0, 0, 0, tex_idx,
                            self.tex_w, self.tex_h, 1,
                            GL_RGBA, GL_UNSIGNED_BYTE, img.convert('RGBA').tobytes())
            glBindTexture(GL_TEXTURE_2D_ARRAY, 0)

    def _sync_visibility(self):

        if self._tlas is None:
            return

        for inst in self.scene.instances:
            row = self._inst_row.get(inst.id)
            if row is None or inst.visibility_revision == self._vis_last_rev.get(inst.id):
                continue
            self._vis_last_rev[inst.id] = inst.visibility_revision

            vis = 1 if inst.visible else 0
            self.cpu_inst_visible[row] = vis
            self.bvh_buffers['inst_visible'].write(np.array([vis], dtype=np.uint32), start=row)  # GPU
            self._tlas.set_instance_mask(row, 0xFFFFFFFF if vis else 0x00000000)  # CPU collision

    def _sync_geometry(self) -> bool:
        """
        Refit BLASes whose geometry was edited in place (positions only).
        Returns True if anything changed.
        """

        any_changed = False

        for asset in self.scene.assets.values():

            bmap = self._asset_blas_map.get(asset.id)
            if bmap is None or asset.geometry_revision == self._geom_last_rev.get(asset.id):
                continue

            self._geom_last_rev[asset.id] = asset.geometry_revision

            blas = self._blases[bmap['id']]
            if not blas.is_refittable:
                print(f"Warning: BLAS for '{asset.name}' is not refittable; geometry edit needs a re-bake.")
                continue

            blas.refit()
            nodes = blas.get_buffers()['nodes'].astype(np.float32).ravel()
            self.bvh_buffers['blas_nodes'].write(nodes, start=bmap['n_off'] * 8)   # 8 floats / standard node

            if bmap['is_points'] == 0:
                self.bvh_buffers['verts'].write(
                    asset.shading_vertices().astype(np.float32).ravel(), start=bmap['v_off'] * 5)
            else:
                self.bvh_buffers['points'].write(
                    asset.packed_points().ravel(), start=bmap['pt_off'] * 12)
            any_changed = True

        return any_changed

    def _sync_transforms(self, refit_TLAS: bool = False):

        if self._tlas is None or not self._dynamic_map:
            return

        dirty_rows = []

        for inst in self.scene.dynamic_instances:
            row = self._dynamic_map.get(inst.id)
            if row is None or inst.transform_revision == self._dynamic_last_rev.get(inst.id):
                continue

            transform = np.asarray(inst.transform, dtype=np.float32)
            inv_transform = np.asarray(glm.inverse(inst.transform), dtype=np.float32)

            self._tlas.set_instance_transform(row, transform)   # TODO: Use pytinybvh's update_instances for batched updates if many changes

            self.gpu_inst_info[row]['transform'] = transform
            self.gpu_inst_info[row]['inverse_transform'] = inv_transform

            self._dynamic_last_rev[inst.id] = inst.transform_revision

            dirty_rows.append(row)

        if not dirty_rows and not refit_TLAS:
            return

        self._tlas.refit_tlas()
        self.bvh_buffers['tlas_nodes'].write(self._tlas.get_buffers()['nodes'].astype(np.float32))

        if dirty_rows:
            rows = np.unique(np.asarray(dirty_rows, dtype=np.int64))
            for block in np.split(rows, np.where(np.diff(rows) != 1)[0] + 1):
                s, n = int(block[0]), block.size
                self.bvh_buffers['inst_info'].write(self.gpu_inst_info[s:s + n], start=s)

    def update(self):
        """
        Consumes all scene change channels and pushes minimal updates to the GPU.
        """

        self._sync_lights()
        self._sync_materials()
        self._sync_textures()
        self._sync_visibility()

        geom_changed = self._sync_geometry()
        self._sync_transforms(refit_TLAS=geom_changed)

    @property
    def tlas(self) -> Optional['BVH']:
        return self._tlas

    @property
    def blases(self) -> List['BVH']:
        return self._blases

    # Cleanup

    def free(self):
        self.bvh_buffers.free()
        self.light_buffers.free()
        self.scene_textures.free()