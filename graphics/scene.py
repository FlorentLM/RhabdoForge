import OpenGL

OpenGL.ERROR_CHECKING = False

from typing import Dict, List, Optional, Union, Sequence
from numpy.typing import ArrayLike
from enum import Enum, auto

from pathlib import Path
import numpy as np
import open3d as o3d

from OpenGL.GL import *
from pyglm import glm
from geometry.primitives import CUBE_VERTICES, CUBE_INDICES
from graphics.utils import load_shaders, load_cubemap, WORLD_UP, WORLD_RIGHT, WORLD_FORWARD, DeltaTimeTransformer


class AssetType(Enum):
    """ Distinguishes between different types of geometry assets """
    Mesh = auto()
    Points = auto()


class Asset:
    """
    A container for a renderable asset which can be either a mesh or a point cloud.

    To create a Mesh:
    - From file: Asset(name="my_mesh", file_path="path/to/mesh.obj", texture_path="path/to/tex.jpg")
    - From data: Asset(name="my_mesh", vertices=verts_array, indices=indices_array)

    To create a Point Cloud:
    - From file: Asset(name="my_pcd", file_path="path/to/points.ply", radii=0.01)
    - From data: Asset(name="my_pcd", points=points_array, colors=colors_array, radii=0.01)
    """

    def __init__(self,
                 name: str,
                 file_path: Optional[Union[Path, str]] = None,

                 # Mesh-specific data
                 vertices: Optional[np.ndarray] = None,
                 indices: Optional[np.ndarray] = None,
                 texture_path: Union[Path, str] = 'textures/wood.jpg',

                 # Point cloud-specific data
                 points: Optional[ArrayLike] = None,
                 colors: Optional[ArrayLike] = None,
                 normals: Optional[ArrayLike] = None,
                 radii: Optional[Union[float, ArrayLike]] = None
                 ):
        self.id = id(self)
        self.name = name
        self.asset_type: Optional[AssetType] = None

        # Determine asset type and load data
        if file_path is not None:
            self._load_from_file(Path(file_path), texture_path=texture_path, radii=radii)

        elif vertices is not None and indices is not None:
            self._create_mesh_from_data(vertices, indices, texture_path)

        elif points is not None:
            self._create_points_from_data(points, colors, normals, radii)

        else:
            raise ValueError(
                "Insufficient data to create an Asset. Provide either 'file_path', "
                "('vertices' and 'indices' for a mesh), or 'points' for a point cloud."
            )

    def _load_from_file(self, path: Path, texture_path: str, radii: Optional[Union[float, ArrayLike]]):
        """
        Loads an asset from a file, determining whether it's a mesh or a point cloud based on its contents
        """
        if not path.exists():
            raise FileNotFoundError(f"Could not find asset file: {path}")

        print(f"Attempting to load asset from {path}...")

        # First, try to load the file as a TriangleMesh
        mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=True)

        # **IMPROVEMENT**: Decide type based on content, not file extension.
        if mesh and mesh.has_triangles():
            print(f"File '{path}' contains triangles. Loading as a Mesh asset.")
            self._process_mesh_data(mesh, texture_path)

        elif mesh and mesh.has_vertices():
            print(f"File '{path}' has vertices but no triangles. Loading as a Points asset.")

            # Extract point data from the mesh structure
            points = np.asarray(mesh.vertices, dtype=np.float32)
            colors = np.asarray(mesh.vertex_colors, dtype=np.float32) if mesh.has_vertex_colors() else None
            normals = np.asarray(mesh.vertex_normals, dtype=np.float32) if mesh.has_vertex_normals() else None
            self._create_points_from_data(points, colors, normals, radii)
            print(f"Loaded points Asset '{self.name}' with {self.num_points} points from {path}.")

        else:
            # Fallback for formats that read_triangle_mesh might fail on
            print(f"Could not read triangles from '{path}'. Attempting to load as a Point Cloud.")
            try:
                pcd = o3d.io.read_point_cloud(path)
                if pcd and pcd.has_points():
                    self._process_points_data(pcd, radii)
                    print(f"Loaded points Asset '{self.name}' with {self.num_points} points from {path}.")

                else:
                    raise IOError(f"File '{path}' could not be loaded as a mesh or a point cloud.")

            except Exception as e:
                raise IOError(f"Failed to load file '{path}'. Could not interpret as mesh or point cloud. Error: {e}")

    def _process_mesh_data(self, mesh: o3d.geometry.TriangleMesh, texture_path: str):
        """ Processes geometry from an already-loaded Open3D mesh object """

        # Handle meshes without texture coordinates
        if not mesh.has_triangle_uvs():
            print(f"Warning: Mesh for asset '{self.name}' does not contain texture coordinates (UVs).")
            # Create placeholder UVs (texture mapping for this asset will be incorrect)

            num_triangles = len(mesh.triangles)
            mesh.triangle_uvs = o3d.utility.Vector2dVector(np.zeros((num_triangles * 3, 2), dtype=np.float64))

            if mesh.has_vertex_colors():
                print("Info: This mesh has vertex colors, which could be used with a custom shader.")

        o3d_verts = np.asarray(mesh.vertices, dtype=np.float32)
        o3d_uvs = np.asarray(mesh.triangle_uvs, dtype=np.float32)
        o3d_indices = np.asarray(mesh.triangles, dtype=np.uint32)

        unique_vert_map, vert_list, new_indices_flat = {}, [], []
        flat_uvs, flat_indices = o3d_uvs.reshape(-1, 2), o3d_indices.flatten()

        for i in range(len(flat_indices)):
            pos_idx, uv_tuple = flat_indices[i], tuple(flat_uvs[i])
            key = (pos_idx, uv_tuple)
            if key not in unique_vert_map:
                new_idx = len(vert_list)
                unique_vert_map[key] = new_idx
                pos = o3d_verts[pos_idx]
                vert_list.append(np.concatenate([pos, uv_tuple]))
            else:
                new_idx = unique_vert_map[key]
            new_indices_flat.append(new_idx)

        self.vertices = np.array(vert_list, dtype=np.float32)
        self.indices = np.array(new_indices_flat, dtype=np.uint32).reshape(-1, 3)
        self.texture_path = Path(texture_path)
        self.asset_type = AssetType.Mesh

        print(
            f"Finalized mesh Asset '{self.name}' with {len(self.vertices)} vertices and {len(self.indices)} triangles.")

    def _create_mesh_from_data(self, vertices: np.ndarray, indices: np.ndarray, texture_path: Union[Path, str]):
        """ Creates a mesh asset from numpy arrays """

        self.vertices = vertices.reshape(-1, 5)
        self.indices = indices.reshape(-1, 3)
        self.texture_path = Path(texture_path)
        self.asset_type = AssetType.Mesh

        print(f"Created mesh Asset '{self.name}' from data.")

    def _process_points_data(self, pcd: o3d.geometry.PointCloud, radii: Optional[Union[float, ArrayLike]]):
        """ Processes geometry from an already-loaded Open3D point cloud object """

        if not pcd.has_points():
            raise ValueError("Point cloud object contains no points.")

        points = np.asarray(pcd.points, dtype=np.float32)
        colors = np.asarray(pcd.colors, dtype=np.float32) if pcd.has_colors() else None
        normals = np.asarray(pcd.normals, dtype=np.float32) if pcd.has_normals() else None

        self._create_points_from_data(points, colors, normals, radii=radii)

    def _create_points_from_data(self, points: ArrayLike, colors: Optional[ArrayLike],
                                 normals: Optional[ArrayLike], radii: Optional[Union[float, ArrayLike]]):
        """ Creates a point cloud asset from numpy arrays """
        self.points = np.asarray(points, dtype=np.float32)
        self.num_points = len(self.points)

        self.colors = np.asarray(colors, dtype=np.float32) if colors is not None else np.ones_like(self.points,
                                                                                                   dtype=np.float32)
        self.normals = np.asarray(normals, dtype=np.float32) if normals is not None else np.zeros_like(self.points,
                                                                                                       dtype=np.float32)

        if isinstance(radii, (float, int)):
            self.radii = np.full(self.num_points, radii, dtype=np.float32)
        elif radii is not None:
            self.radii = np.asarray(radii, dtype=np.float32)
            if len(self.radii) != self.num_points:
                raise ValueError("Number of radii must match the number of points.")
        else:
            # Fallback to a default only if radii is explicitly None
            self.radii = np.full(self.num_points, 0.05, dtype=np.float32)

        self.asset_type = AssetType.Points

        print(f"Created points Asset '{self.name}' with {self.num_points} points from data.")

    @property
    def nb_triangles(self):
        if self.asset_type == AssetType.Mesh:
            return len(self.indices)
        return 0

    # TODO: more properties


class Instance:
    """
    Logical instance of an Asset in the scene. Renderer-agnostic.
    This class is the single source of truth for an instance's transform.
    """
    def __init__(self,
                 asset: Asset,
                 transform: Optional[Union[glm.mat4, ArrayLike]] = None,
                 dynamic: bool = False,
                 **kwargs):

        self.id = id(self)
        self.asset = asset
        self.dynamic = dynamic
        self.properties = kwargs

        if transform is None:
            self.transform = glm.mat4(1.0)
        else:
            transform_np = np.asarray(transform, dtype=np.float32)

            if transform_np.shape == (4, 4):
                self.transform = glm.mat4(transform_np)

            elif transform_np.shape == (3,):
                self.transform = glm.translate(glm.mat4(1.0), glm.vec3(transform_np))

            else:
                raise ValueError(
                    f"Unsupported shape for transform: {transform_np.shape}. "
                    "Expected a (4, 4) matrix or a (3,) position vector."
                )

    def dt(self, delta_time: float) -> DeltaTimeTransformer:
        """
        Enables framerate-independent transformations for a chain of method calls

        Example:
            # Rotates at 90 degrees per second
            my_instance.dt(delta_time).rotate_axis(90, 'y')
        """
        return DeltaTimeTransformer(self, delta_time)

    @property
    def position(self):
        return glm.vec3(self.transform[3])

    @position.setter
    def position(self, value: Union[glm.vec3, ArrayLike]):
        self.transform[3] = glm.vec4(glm.vec3(value), 1.0)

    def translate(self, translation: Union[glm.vec3, ArrayLike]):
        self.transform = glm.translate(self.transform, glm.vec3(translation))
        return self

    def rotate_axis(self, angle: float, axis: Union[str, glm.vec3, ArrayLike], degrees: bool = True):
        """
        Rotates the instance around a given axis
        """

        if isinstance(axis, str):
            axis_str = axis.lower()
            axis_map = {
                'x': WORLD_RIGHT,
                'y': WORLD_UP,
                'z': WORLD_FORWARD,
                'right': WORLD_RIGHT,
                'left': -WORLD_RIGHT,
                'up': WORLD_UP,
                'down': -WORLD_UP,
                'forward': WORLD_FORWARD,
                'backward': -WORLD_FORWARD,
                'yaw': WORLD_UP,
                'pitch': WORLD_RIGHT,
                'roll': WORLD_FORWARD,
            }
            try:
                rotation_axis = axis_map[axis_str]
            except KeyError:
                raise ValueError(f"Unknown axis identifier: '{axis}'. Valid options are: {list(axis_map.keys())}")
        else:
            rotation_axis = glm.vec3(axis)

        angle_rad = glm.radians(angle) if degrees else angle
        self.transform = glm.rotate(self.transform, angle_rad, rotation_axis)
        return self

    def rotate(self, yaw_delta: float = 0.0, pitch_delta: float = 0.0, roll_delta: float = 0.0, degrees: bool = True):
        """ Rotates the instance by given Euler angle deltas """

        if yaw_delta != 0.0:
            self.transform = glm.rotate(self.transform, glm.radians(yaw_delta) if degrees else yaw_delta, WORLD_UP)

        if pitch_delta != 0.0:
            self.transform = glm.rotate(self.transform, glm.radians(pitch_delta) if degrees else pitch_delta, WORLD_RIGHT)

        if roll_delta != 0.0:
            self.transform = glm.rotate(self.transform, glm.radians(roll_delta) if degrees else roll_delta, WORLD_FORWARD)

        return self

    def scale(self, scale_factors: Union[glm.vec3, ArrayLike]):
        self.transform = glm.scale(self.transform, glm.vec3(scale_factors))
        return self


class Skybox:
    def __init__(self, texture_path: Path | str = 'textures/bright_day'):

        self.texture_id = load_cubemap(texture_path)

        self.program = load_shaders('shaders/skybox.vert', 'shaders/skybox.frag')

        skybox_vertices = CUBE_VERTICES.reshape(-1, 5)[:, :3]

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, skybox_vertices.nbytes, skybox_vertices, GL_STATIC_DRAW)

        ebo = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, CUBE_INDICES.nbytes, CUBE_INDICES, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, False, 0, ctypes.c_void_p(0))

        glBindVertexArray(0)

    def draw(self, projection_matrix, view_matrix):

        glDepthFunc(GL_LEQUAL)  # changing depth function to LEQUAL is needed so the 1.0 depth passes
        glUseProgram(self.program)

        # We are inside the skybox so we need to see the back-faces
        glDisable(GL_CULL_FACE)

        glUniformMatrix4fv(glGetUniformLocation(self.program, "projection"), 1, False, glm.value_ptr(projection_matrix))
        glUniformMatrix4fv(glGetUniformLocation(self.program, "view"), 1, False, glm.value_ptr(view_matrix))

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, self.texture_id)
        glUniform1i(glGetUniformLocation(self.program, "skybox"), 0)

        glBindVertexArray(self.vao)
        glDrawElements(GL_TRIANGLES, 36, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

        glEnable(GL_CULL_FACE) # Re-enable culling for the rest of the scene
        glDepthFunc(GL_LESS) # restore default depth function


class Scene:
    """
    The logical scene representation. A simple container for assets and instances.
    """
    def __init__(self, background_color: Sequence[float] = (0.0, 0.0, 0.0)):
        self.assets: Dict[str, Asset] = {}
        self.instances: List[Instance] = []
        self.skybox: Optional[Skybox] = None
        self.background_color = background_color

    def add_instance(self, asset: Union[Asset, str], transform: Optional[Union[glm.mat4, ArrayLike]] = None, **kwargs) -> Instance:

        if isinstance(asset, Asset):
            asset_obj = asset

            if asset_obj.name not in self.assets:
                print(f"New {asset_obj.asset_type.name} asset '{asset_obj.name}' registered with the scene.")
                self.assets[asset_obj.name] = asset_obj

            elif self.assets[asset_obj.name].id != asset_obj.id:
                raise ValueError(
                    f"An asset named '{asset_obj.name}' already exists but is a different object. "
                    "Asset names must be unique.")

        elif isinstance(asset, str):
            if asset not in self.assets:
                raise ValueError(
                    f"Asset with name '{asset}' not found. "
                    "You must add an instance of the asset object itself first to register it."
                )
            asset_obj = self.assets[asset]

        else:
            raise TypeError(
                f"Invalid type for asset_or_name. Expected Asset or str, but got {type(asset).__name__}.")

        instance = Instance(asset_obj, transform, **kwargs)
        self.instances.append(instance)
        return instance

    def add_skybox(self, texture_path: str):
        """ Creates and loads a skybox from a directory of textures """
        self.skybox = Skybox(texture_path)

    @property
    def total_triangles(self) -> int:
        count = 0
        for instance in self.instances:
            if instance.asset.asset_type == AssetType.Mesh:
                count += instance.asset.nb_triangles
        return count

    @property
    def total_points(self) -> int:
        count = 0
        for instance in self.instances:
            if instance.asset.asset_type == AssetType.Points:
                count += instance.asset.num_points
        return count

    def free(self):
        self.assets.clear()
        self.instances.clear()
        # Note: GPU resources tied to skybox/assets are freed by the bakers/renderers