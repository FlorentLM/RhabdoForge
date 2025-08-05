from pathlib import Path
from typing import Optional

import numpy as np
from pyglm import glm
from OpenGL.GL import *
from graphics.utils import load_shaders, load_texture, VEC_DTYPE
import open3d as o3d
import pytinybvh


class Mesh:
    """ Renderable object with its own shaders, texture, and vertex data """

    def __init__(self, vertex_data, vert_shader_path, frag_shader_path, texture_path):
        self.data = vertex_data
        self.draw_type = GL_TRIANGLES
        self.draw_start = 0
        self.draw_count = len(vertex_data) // 5  # 5 floats per vertex (x, y, z, u, v)

        # Compile GLSL files and load texture
        self.shaders = load_shaders(vert_shader_path, frag_shader_path)
        self.texture = load_texture(texture_path)

        # Create and bind a VAO
        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        # Create and bind a VBO
        self.vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo)

        # Send vertex data to VBO
        glBufferData(GL_ARRAY_BUFFER, self.data.nbytes, self.data, GL_STATIC_DRAW)

        # Configure vertex attributes
        glUseProgram(self.shaders)

        # Position attribute (3 floats)
        pos_loc = glGetAttribLocation(self.shaders, "pos")
        glEnableVertexAttribArray(pos_loc)
        glVertexAttribPointer(pos_loc,
                              3,
                              GL_FLOAT,
                              GL_FALSE,
                              5 * self.data.itemsize,
                              ctypes.c_void_p(0))

        # Texture coordinate attribute (2 floats)
        vertTexCoord_loc = glGetAttribLocation(self.shaders, "vertTexCoord")
        glEnableVertexAttribArray(vertTexCoord_loc)
        glVertexAttribPointer(vertTexCoord_loc,
                              2,
                              GL_FLOAT,
                              GL_FALSE,
                              5 * self.data.itemsize,
                              ctypes.c_void_p(3 * self.data.itemsize))

        # Unbind everything to be safe
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        glBindVertexArray(0)
        glUseProgram(0)

    def free(self):
        glDeleteVertexArrays(1, [self.vao])
        glDeleteBuffers(1, [self.vbo])
        glDeleteProgram(self.shaders)
        glDeleteTextures(1, [self.texture])


class Instance:
    """ A specific instance of a Mesh in the scene, with its own transform """

    def __init__(self, asset: Mesh, transform: Optional[glm.mat4] = None):
        self.asset = asset
        if transform is None:
            self.transform = glm.mat4(1.0)
        else:
            self.transform = transform


class PointCloud:
    """
    Container for point cloud data, which builds the BVH and packs the data for the GPU
    # TODO: this should probably move to the RayTracing scene now
    """

    def __init__(self, file_path: str, hit_radius: Optional[float] = None):

        self.source_path = Path(file_path)
        if not self.source_path.exists():
            raise FileNotFoundError(f"Could not find point cloud source file: {self.source_path}")

        print(f"Loading point cloud from {self.source_path}...")
        pcd = o3d.io.read_point_cloud(self.source_path)
        points = np.asarray(pcd.points, dtype=np.float32)

        if hit_radius is None:
            print("Automatically determining optimal hit radius...")
            if len(points) < 2:
                # Not enough points to determine density
                self.hit_radius = 0.1
            else:
                from scipy.spatial import KDTree
                tree = KDTree(points)
                sample_size = min(len(points), 1000)
                sample_indices = np.random.choice(len(points), sample_size, replace=False)
                distances, _ = tree.query(points[sample_indices], k=2)

                # average distance to nearest neighbour
                avg_dist = np.mean(distances[:, 1])
                # set the radius to a bit more than half this distance so that spheres from adjacent points touch
                self.hit_radius = avg_dist * 0.75
            print(f"Estimated hit radius: {self.hit_radius:.4f}")
        else:
            self.hit_radius = float(hit_radius)

        print("Building BVH from point data...")
        self.bvh_nodes, prim_indices = pytinybvh.from_points(points, radius=self.hit_radius)

        print("Reordering primitive attributes according to BVH indices...")
        # Check for normals and colors, creating placeholders if they don't exist
        if pcd.has_normals():
            normals = np.asarray(pcd.normals, dtype=np.float32)[prim_indices]
        else:
            print("Warning: Point cloud has no normals. Creating placeholders.")
            normals = np.zeros_like(points)

        if pcd.has_colors():
            colors = np.asarray(pcd.colors, dtype=np.float32)[prim_indices]
        else:
            print("Warning: Point cloud has no colors. Defaulting to white.")
            colors = np.ones_like(points)

        # Reorder the base points using the new indices
        reordered_points = points[prim_indices]

        # Pack the reordered attributes for the GPU
        # Matches the 'Point' struct layout in commons.glsl (vec4 pos, vec4 normal, vec4 color)
        self.num_points = len(reordered_points)
        self.point_attributes = np.zeros((self.num_points, 12), dtype=VEC_DTYPE)

        self.point_attributes[:, 0:3] = reordered_points
        self.point_attributes[:, 4:7] = normals
        self.point_attributes[:, 8:11] = colors
        # The 'w' components are left as 0.0 for padding

        self.num_nodes = len(self.bvh_nodes)
        print(f"Loaded and processed point cloud with {self.num_points} points and a BVH with {self.num_nodes} nodes.")

    def free(self):
        # Data is just in numpy arrays, nothing to free
        pass


class Scene:
    """ Container for all objects in the world """

    def __init__(self):
        self.instances = []
        self.assets = {}
        self.point_cloud: Optional[PointCloud] = None

    def add_instance(self, instance: Instance):
        self.instances.append(instance)

    def add_point_cloud(self, point_cloud: PointCloud):
        self.point_cloud = point_cloud

    def free(self):
        for asset in self.assets.values():
            asset.free()
        if self.point_cloud:
            self.point_cloud.free()
        self.instances.clear()
        self.assets.clear()


class RaytracingScene:
    def __init__(self, scene: Scene):
        """ Initializes and packs the scene, and builds acceleration structures """

        # Common data packing
        self.materials = None
        self.texture_ids = []
        self._pack_materials(scene.assets.values())

        # Triangle data and BVH
        self.triangles = None
        self.triangle_bvh_nodes = None
        self.num_triangles = 0

        # State for dynamic updates
        self._base_positions = None         # untransformed vertex positions
        self._base_uvs = None               # vertex UVs
        self._material_indices = None       # Per-triangle material index
        self._vert_counts = []
        self._transforms_stack = None

        # Pack static geometry and build the initial BVH
        self._pack_and_build_bvh(scene.instances)

        # Populate triangles buffer with initial (reordered) state of the scene
        self.update(scene.instances)

    def _pack_and_build_bvh(self, instances):
        """ Gathers all geometry, builds a BVH for all triangles and reorders them """

        all_verts_pos = []
        all_verts_uv = []
        all_material_indices = []

        if not instances:
            print("No mesh instances in the scene to build BVH from.")
            return

        print("Gathering mesh data for BVH construction...")
        for instance in instances:
            if not isinstance(instance.asset, Mesh):
                continue

            mesh = instance.asset
            interleaved_data = mesh.data.reshape(-1, 5)

            # Apply instance transform directly to vertices
            # (this 'flattens' the scene into a single large mesh for the BVH)
            positions = interleaved_data[:, :3]
            positions_h = np.hstack([positions, np.ones((len(positions), 1), dtype=VEC_DTYPE)])

            # Convert glm.mat4 to numpy array for batch multiplication.
            # np.asarray creates a column-major numpy array from the glm matrix.
            np_transform = np.asarray(instance.transform)
            transformed_pos_h = (np_transform @ positions_h.T).T

            all_verts_pos.append(transformed_pos_h[:, :3])
            all_verts_uv.append(interleaved_data[:, 3:])

            material_idx = self.material_map[mesh]
            num_triangles_in_instance = len(interleaved_data) // 3
            all_material_indices.append(np.full(num_triangles_in_instance, material_idx, dtype=np.uint32))

        if not all_verts_pos:
            self.num_triangles = 0
            return

        # Concatenate all transformed vertices into one large array
        all_verts_pos_np = np.concatenate(all_verts_pos, axis=0)
        all_verts_uv_np = np.concatenate(all_verts_uv, axis=0)
        self.material_indices = np.concatenate(all_material_indices)

        # Reshape for pytinybvh which expects (N, 9) for triangles
        triangles_for_bvh = all_verts_pos_np.reshape(-1, 9)
        self.num_triangles = len(triangles_for_bvh)

        print(f"Building BVH for {self.num_triangles:,} total triangles...")
        self.triangle_bvh_nodes, prim_indices = pytinybvh.from_triangles(triangles_for_bvh)
        print(f"Triangle BVH built with {len(self.triangle_bvh_nodes)} nodes.")

        # Reorder triangle attributes based on BVH primitive indices
        reordered_verts_pos = all_verts_pos_np.reshape(-1, 3)[
            np.repeat(prim_indices * 3, 3) + np.tile([0, 1, 2], len(prim_indices))]

        reordered_verts_uv = all_verts_uv_np.reshape(-1, 2)[
            np.repeat(prim_indices * 3, 3) + np.tile([0, 1, 2], len(prim_indices))]

        self.material_indices = self.material_indices[prim_indices]

        # Store reordered untransformed data for dynamic updates
        # TODO: TLAS for animations
        self._base_positions = reordered_verts_pos
        self._base_uvs = reordered_verts_uv

        # Allocate buffer for the GPU
        # GLSL Triangle struct is 80 bytes (so 20 floats) with final std430 padding
        self.triangles = np.zeros(self.num_triangles * 20, dtype=VEC_DTYPE)
        self._fill_gpu_triangle_buffer()

    def _fill_gpu_triangle_buffer(self):

        if self.num_triangles == 0:
            return

        flat_view = self.triangles.reshape(self.num_triangles, 20)

        v = self._base_positions.reshape(self.num_triangles, 3, 3)
        uv = self._base_uvs.reshape(self.num_triangles, 3, 2)

        flat_view[:, 0:3] = v[:, 0, :]
        flat_view[:, 4:7] = v[:, 1, :]
        flat_view[:, 8:11] = v[:, 2, :]
        flat_view[:, 12:14] = uv[:, 0, :]
        flat_view[:, 14:16] = uv[:, 1, :]
        flat_view[:, 16:18] = uv[:, 2, :]
        flat_view[:, 18] = self.material_indices.view(VEC_DTYPE)

    def update(self, instances):
        # TODO: TLAS and/or BVH refitting (need to add this in pytinybvh)
        pass

    def _pack_materials(self, assets):
        """ Packs materials and unique textures """

        material_list = []
        texture_map = {}
        self.material_map = {}  # maps Mesh asset to material_idx

        for asset in assets:
            if isinstance(asset, Mesh) and asset not in self.material_map:
                tex_id = asset.texture
                if tex_id not in texture_map:
                    texture_idx = len(self.texture_ids)
                    self.texture_ids.append(tex_id)
                    texture_map[tex_id] = texture_idx
                else:
                    texture_idx = texture_map[tex_id]

                material_idx = len(material_list)
                self.material_map[asset] = material_idx
                material_list.append(texture_idx)

        # GLSL Material struct is 16 bytes (so 4 floats)
        num_materials = len(material_list)
        self.materials = np.zeros(num_materials * 4, dtype=VEC_DTYPE)

        # View as uint32 to place the texture index correctly
        materials_u32 = self.materials.view(np.uint32)
        materials_u32[0::4] = material_list