from pathlib import Path
from typing import Optional

import numpy as np
from OpenGL.GL import *
from graphics.utils import load_shaders, load_texture, VEC_DTYPE


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

    def __init__(self, asset: Mesh, transform=None):
        self.asset = asset
        if transform is None:
            self.transform = np.eye(4, dtype=VEC_DTYPE)
        else:
            self.transform = transform


class PointCloud:
    """
    Container for point cloud data accelerated by a BVH
    """
    def __init__(self, file_prefix: str):
        bvh_path = Path(f"{file_prefix}.bvh.npy")
        primitives_path = Path(f"{file_prefix}.primitives.npy")

        # TODO: get rid of this file loading - should be done by the scene class

        if not bvh_path.exists() or not primitives_path.exists():
            raise FileNotFoundError(
                f"Could not find pre-processed BVH data. "
                f"Please run pointcloud_BVH.py to generate '{bvh_path.name}' and '{primitives_path.name}'."
            )

        print(f"Loading point cloud BVH from {bvh_path}")
        self.bvh_nodes = np.load(bvh_path)
        print(f"Loading reordered point cloud primitives from {primitives_path}")
        self.point_attributes = np.load(primitives_path)

        self.num_nodes = len(self.bvh_nodes)
        self.num_points = len(self.point_attributes)
        print(f"Loaded point cloud with {self.num_points} points and BVH with {self.num_nodes} nodes.")

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
        self.instances.clear()
        self.assets.clear()


class RaytracingScene:
    def __init__(self, scene: Scene):
        """ Initializes and packs the scene """

        # Material and Texture packing
        self.materials = None
        self.texture_ids = []
        self._pack_materials(scene.assets.values())

        # Static geometry packing
        self.base_positions = None      # Untransformed vertex positions
        self.base_uvs = None            # Vertex UVs
        self.material_indices = None    # Per-triangle material index

        # State for the update() method
        self._vert_counts = []
        self._transforms_stack = None

        self._pack_static_geometry(scene.instances)

        # Buffer allocation
        num_triangles = len(self.base_positions) // 3 if self.base_positions is not None else 0

        # GLSL Triangle struct is 80 bytes (so 20 floats) with final std430 padding
        self.triangles = np.zeros(num_triangles * 20, dtype=VEC_DTYPE)

        # Populate the .triangles buffer with the initial state of the scene
        self.update(scene.instances)

    def _pack_materials(self, assets):
        """ Packs materials and unique textures once """

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

    def _pack_static_geometry(self, instances):
        """ Gathers all untransformed geometry (positions, UVs) from instances """
        all_verts_pos = []
        all_verts_uv = []
        all_material_indices = []

        for instance in instances:
            if not isinstance(instance.asset, Mesh):
                continue

            mesh = instance.asset
            interleaved_data = mesh.data.reshape(-1, 5)
            num_vertices = len(interleaved_data)

            all_verts_pos.append(interleaved_data[:, :3])
            all_verts_uv.append(interleaved_data[:, 3:])

            material_idx = self.material_map[mesh]

            # One material index per triangle
            all_material_indices.append(np.full(num_vertices // 3, material_idx, dtype=np.uint32))

            self._vert_counts.append(num_vertices)

        if not all_verts_pos:
            print("Packed scene for ray tracing: 0 triangles.")
            self.num_vertices = 0
            self.num_triangles = 0
            return

        # Concat into large untransformed scene data array
        self.base_positions = np.concatenate(all_verts_pos, axis=0)
        self.base_uvs = np.concatenate(all_verts_uv, axis=0)
        self.material_indices = np.concatenate(all_material_indices)

        self.num_vertices = len(self.base_positions)
        self.num_triangles = self.num_vertices // 3

    def update(self, instances):
        """
        Updates the ray-tracing scene by only re-transforming vertex positions
        This is called every frame if objects are dynamic
        """

        if self.base_positions is None:
            return  # nothing to update

        # Get current transforms from instances
        self._transforms_stack = np.array([inst.transform for inst in instances], dtype=VEC_DTYPE)

        # Apply all transformations
        # convert base positions to homogeneous coordinates for matrix multiplication
        positions_h = np.hstack([self.base_positions, np.ones((self.base_positions.shape[0], 1), dtype=VEC_DTYPE)])
        repeated_transforms = np.repeat(self._transforms_stack, self._vert_counts, axis=0)
        transformed_pos_h = np.einsum('aij,aj->ai', repeated_transforms, positions_h)

        # Fill the final flat buffer with all data (static and dynamic)
        num_triangles = len(transformed_pos_h) // 3

        # Reshape data for assignment
        v = transformed_pos_h[:, :3].reshape(num_triangles, 3, 3)   # (N, 3 verts, 3 coords)
        uv = self.base_uvs.reshape(num_triangles, 3, 2)             # (N, 3 verts, 2 coords)

        # view of flat buffer reshaped for easy triangle-wise assignment
        flat_view = self.triangles.reshape(num_triangles, 20)   # stride 20 (80 bytes total, 4 bytes per float)

        # Assign the data in chunks
        flat_view[:, 0:3] = v[:, 0, :]      # v0 positions
        flat_view[:, 4:7] = v[:, 1, :]      # v1 positions
        flat_view[:, 8:11] = v[:, 2, :]     # v2 positions
        flat_view[:, 12:14] = uv[:, 0, :]   # uv0s
        flat_view[:, 14:16] = uv[:, 1, :]   # uv1s
        flat_view[:, 16:18] = uv[:, 2, :]   # uv2s

        # Bit-cast and assign the static material indices
        flat_view[:, 18] = self.material_indices.view(VEC_DTYPE)