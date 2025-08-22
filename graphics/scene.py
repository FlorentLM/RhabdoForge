from pathlib import Path
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

import numpy as np
import open3d as o3d
import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *
from pyglm import glm

from geometry.primitives import CUBE_VERTICES
from graphics.utils import VEC_DTYPE, load_shaders, load_cubemap


class Asset(ABC):
    """
    Abstract base class for a renderable asset
    Contains the raw data (vertices, points, etc)
    """

    def __init__(self, name: str):
        self.id = id(self)
        self.name = name


# TODO: Assets loading interface need to be unified. Either load a file, or vertex / indices data


class MeshAsset(Asset):
    """
    Pure data container for a mesh
    Contains geometry and material properties
    """

    def __init__(self, name: str, vertex_data: np.ndarray, texture_path: str):
        super().__init__(name)

        self.vertex_data = vertex_data
        self.texture_path = Path(texture_path)

    @property
    def num_triangles(self):
        # Each vertex has 5 components (pos, uv), 3 vertices per triangle
        return self.vertex_data.shape[0] // 3


class PointsAsset(Asset):
    """
    Pure data container for a point cloud
    Contains geometry and material properties
    """

    def __init__(self, name: str, file_path: str):
        super().__init__(name)
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

        self.num_points = len(self.points)
        print(f"Loaded {self.num_points} points for asset '{name}'.")


class Instance:
    """
    Logical instance of an Asset in the scene
    Renderer-agnostic
    """

    def __init__(self, asset: Asset, transform: glm.mat4 = None, **kwargs):
        self.asset = asset
        self.transform = transform or glm.mat4(1.0)
        self.properties = kwargs  # for example point_radius


class Skybox:
    def __init__(self):
        self.program = load_shaders('shaders/skybox.vert', 'shaders/skybox.frag')

        interleaved_2d = CUBE_VERTICES.reshape(-1, 5)
        skybox_vertices = interleaved_2d[:, :3].copy()

        self.vao = glGenVertexArrays(1)
        glBindVertexArray(self.vao)

        vbo = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, vbo)
        glBufferData(GL_ARRAY_BUFFER, skybox_vertices.nbytes, skybox_vertices, GL_STATIC_DRAW)

        glEnableVertexAttribArray(0)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, ctypes.c_void_p(0))

        glBindVertexArray(0)

    def draw(self, projection_matrix, view_matrix, cubemap_tex_id):

        glDepthFunc(GL_LEQUAL)  # changing depth function to LEQUAL is needed so the 1.0 depth passes
        glUseProgram(self.program)

        # We are inside the skybox so we need to see the back-faces
        glDisable(GL_CULL_FACE)

        glUniformMatrix4fv(glGetUniformLocation(self.program, "projection"), 1, False, glm.value_ptr(projection_matrix))
        glUniformMatrix4fv(glGetUniformLocation(self.program, "view"), 1, False, glm.value_ptr(view_matrix))

        glActiveTexture(GL_TEXTURE0)
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemap_tex_id)
        glUniform1i(glGetUniformLocation(self.program, "skybox"), 0)

        glBindVertexArray(self.vao)
        glDrawArrays(GL_TRIANGLES, 0, 36)  # cube has 36 vertices
        glBindVertexArray(0)

        glEnable(GL_CULL_FACE) # Re-enable culling for the rest of the scene
        glDepthFunc(GL_LESS) # restore default depth function


class Scene:
    """
    The logical scene representation
    Maintains a list of assets and instances
    """

    def __init__(self):
        self.assets: Dict[str, Asset] = {}
        self.instances: List[Instance] = []
        self.skybox: Optional[Skybox] = None
        self.skybox_texture_id: Optional[int] = None

    def load_mesh_asset(self, name: str, vertex_data, texture_path: str) -> MeshAsset:
        if name in self.assets:
            print(f"Warning: Asset with name '{name}' already exists. Overwriting.")

        asset = MeshAsset(name, vertex_data, texture_path)
        self.assets[name] = asset
        return asset

    def load_point_cloud_asset(self, name: str, file_path: str) -> PointsAsset:
        if name in self.assets:
            print(f"Warning: Asset with name '{name}' already exists. Overwriting.")

        asset = PointsAsset(name, file_path)
        self.assets[name] = asset
        return asset

    def add_instance(self, asset: Asset, transform: glm.mat4 = None, **kwargs) -> Instance:
        if asset.name not in self.assets:
            raise ValueError(f"Asset '{asset.name}' not loaded into the scene. Call a load_*_asset method first.")

        instance = Instance(asset, transform, **kwargs)
        self.instances.append(instance)
        return instance

    def add_skybox(self, texture_path: str):
        """ Creates and loads a skybox from a directory of textures """
        self.skybox = Skybox()
        self.skybox_texture_id = load_cubemap(texture_path)
        print(f"Loaded skybox from {texture_path}")

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