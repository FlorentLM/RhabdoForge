import OpenGL
OpenGL.ERROR_CHECKING = False

from pathlib import Path
from typing import Dict, List, Optional, Union, Sequence
from abc import ABC

import numpy as np
from numpy.typing import ArrayLike
import open3d as o3d

from OpenGL.GL import *
from pyglm import glm
from geometry.primitives import CUBE_VERTICES, CUBE_INDICES
from graphics.utils import VEC_DTYPE, load_shaders, load_cubemap, WORLD_UP, WORLD_RIGHT, WORLD_FORWARD, \
    DeltaTimeTransformer


class Asset(ABC):
    """
    Abstract base class for a renderable asset
    Contains the raw data (vertices, points, etc)
    """

    def __init__(self, name: str):
        self.id = id(self)
        self.name = name


class MeshAsset(Asset):
    """
    Pure data container for a mesh. Stores geometry as indexed vertices.
    Each vertex contains position and UV coordinates.
    """

    def __init__(self,
                 name: str,
                 file_path: Optional[Path | str] = None,
                 vertices: Optional[np.ndarray] = None,
                 indices: Optional[np.ndarray] = None,
                 texture_path: Path | str = 'textures/wood.jpg'):
        super().__init__(name)

        if file_path is None and (vertices is None or indices is None):
            raise ValueError("MeshAsset requires either 'file_path' or both 'vertices' and 'indices'.")

        self.texture_path = Path(texture_path)

        if file_path is not None:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Could not find mesh file: {path}")

            mesh = o3d.io.read_triangle_mesh(path, enable_post_processing=True)

            if not mesh.has_triangle_uvs():
                raise ValueError(f"Mesh file '{path}' does not contain texture coordinates (UVs).")

            # Un-weld the mesh to create a clean indexed buffer for rendering
            o3d_verts = np.asarray(mesh.vertices, dtype=np.float32)
            o3d_uvs = np.asarray(mesh.triangle_uvs, dtype=np.float32)
            o3d_indices = np.asarray(mesh.triangles, dtype=np.uint32)

            unique_vert_map = {}
            vert_list = []
            new_indices_flat = []

            flat_uvs = o3d_uvs.reshape(-1, 2)
            flat_indices = o3d_indices.flatten()

            for i in range(len(flat_indices)):
                pos_idx = flat_indices[i]
                uv_tuple = tuple(flat_uvs[i])
                key = (pos_idx, uv_tuple)

                if key not in unique_vert_map:
                    new_idx = len(vert_list)
                    unique_vert_map[key] = new_idx
                    pos = o3d_verts[pos_idx]
                    vert_list.append(np.concatenate([pos, uv_tuple]))
                else:
                    new_idx = unique_vert_map[key]

                new_indices_flat.append(new_idx)

            self.vertices = np.array(vert_list, dtype=VEC_DTYPE)
            self.indices = np.array(new_indices_flat, dtype=np.uint32).reshape(-1, 3)

        else:
            self.vertices = vertices.reshape(-1, 5)  # Ensure correct shape
            self.indices = indices.reshape(-1, 3)

    @property
    def num_triangles(self):
        return len(self.indices)


class PointsAsset(Asset):
    """
    Pure data container for a point cloud
    Contains geometry and material properties
    """

    def __init__(self,
                 name: str,
                 file_path: Optional[Path | str] = None,
                 points: Optional[np.ndarray] = None,
                 colors: Optional[np.ndarray] = None,
                 normals: Optional[np.ndarray] = None
                 ):
        super().__init__(name)

        if file_path is None and points is None:
            raise ValueError("PointsAsset requires either 'file_path' or 'points' data.")

        if file_path is not None and points is not None:
            raise ValueError("Provide either 'file_path' or 'points' data, not both.")

        if file_path is not None:
            path = Path(file_path)
            if not path.exists():
                raise FileNotFoundError(f"Could not find point cloud file: {path}")

            print(f"Loading point cloud data from {path}...")
            pcd = o3d.io.read_point_cloud(path)

            self.points = np.asarray(pcd.points, dtype=VEC_DTYPE)
            self.normals = np.zeros_like(self.points)
            self.colors = np.ones_like(self.points)

            if pcd.has_normals():
                self.normals = np.asarray(pcd.normals, dtype=VEC_DTYPE)

            if pcd.has_colors():
                self.colors = np.asarray(pcd.colors, dtype=VEC_DTYPE)

        else:
            self.points = points
            self.colors = colors if colors is not None else np.ones_like(points)
            self.normals = normals if normals is not None else np.zeros_like(points)

        self.num_points = len(self.points)
        print(f"Loaded {self.num_points} points for asset '{name}'.")


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
            transform_np = np.asarray(transform, dtype=VEC_DTYPE)

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

    def rotate_axis(self, angle_degrees: float, axis: Union[str, glm.vec3, ArrayLike]):
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

        self.transform = glm.rotate(self.transform, glm.radians(angle_degrees), rotation_axis)
        return self

    def rotate(self, yaw_delta: float = 0.0, pitch_delta: float = 0.0, roll_delta: float = 0.0):
        """
        Rotates the instance by given Euler angle deltas
        """
        if yaw_delta != 0.0:
            self.transform = glm.rotate(self.transform, glm.radians(yaw_delta), WORLD_UP)

        if pitch_delta != 0.0:
            self.transform = glm.rotate(self.transform, glm.radians(pitch_delta), WORLD_RIGHT)

        if roll_delta != 0.0:
            self.transform = glm.rotate(self.transform, glm.radians(roll_delta), glm.vec3(0, 0, -1))
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
                print(f"New {type(asset_obj).__name__} asset '{asset_obj.name}' registered with the scene.")
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
            if isinstance(instance.asset, MeshAsset):
                count += instance.asset.num_triangles
        return count

    @property
    def total_points(self) -> int:
        count = 0
        for instance in self.instances:
            if isinstance(instance.asset, PointsAsset):
                count += instance.asset.num_points
        return count

    def free(self):
        self.assets.clear()
        self.instances.clear()
        # Note: GPU resources tied to skybox/assets are freed by the bakers/renderers