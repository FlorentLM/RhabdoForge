from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import open3d as o3d

import OpenGL
OpenGL.ERROR_CHECKING = False
from OpenGL.GL import *

from pyglm import glm

from geometry.primitives import CUBE_VERTICES
from graphics.utils import VEC_DTYPE, load_shaders


class MeshAsset:
    """ A pure data container for a mesh, contains geometry and material """

    def __init__(self, name: str, vertex_data: np.ndarray, texture_path: str):
        self.id = id(self)
        self.name = name
        self.vertex_data = vertex_data
        self.texture_path = Path(texture_path)


class PointCloudAsset:
    """ A pure data container for a point cloud """

    def __init__(self, name: str, file_path: str):
        self.id = id(self)
        self.name = name
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
    """ A logical instance of an asset in the scene, with its own transform. Renderer-agnostic. """

    def __init__(self, asset: MeshAsset | PointCloudAsset, transform: glm.mat4 = None):
        self.asset = asset
        self.transform = transform if transform is not None else glm.mat4(1.0)


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
    The logical scene representation. Maintains a list of assets and instances, completely renderer-agnostic
    """

    def __init__(self):
        self.assets: Dict[str, MeshAsset | PointCloudAsset] = {}
        self.instances: List[Instance] = []
        self.point_cloud: Optional[PointCloudAsset] = None
        self.skybox: Optional[Skybox] = None
        self.skybox_texture_id: Optional[int] = None

    def load_mesh(self, name: str, vertex_data, texture_path: str) -> MeshAsset:
        if name not in self.assets:
            self.assets[name] = MeshAsset(name, vertex_data, texture_path)
        return self.assets[name]

    def add_point_cloud(self, name: str, file_path: str) -> PointCloudAsset:
        if name not in self.assets:
            asset = PointCloudAsset(name, file_path)
            self.assets[name] = asset
            self.point_cloud = asset  # just one point cloud for now
        return self.assets[name]

    def add_instance(self, asset: MeshAsset, transform: glm.mat4 = None) -> Instance:
        instance = Instance(asset, transform)
        self.instances.append(instance)
        return instance

    def free(self):
        """ Clears all logical scene data """

        self.assets.clear()
        self.instances.clear()
        self.point_cloud = None

